"use client";

import { create } from "zustand";

import { DEFAULT_SCENARIO, normalizeScenario } from "@/lib/model/simulation";
import type { Scenario } from "@/lib/model/types";

type MarketState = {
  scenario: Scenario;
  secondarySeed: number;
  activeStep: number;
  playing: boolean;
  seedLocked: boolean;
  setScenario: (scenario: Scenario) => void;
  patchScenario: (patch: Partial<Scenario>) => void;
  setSecondarySeed: (seed: number) => void;
  setActiveStep: (step: number) => void;
  setPlaying: (playing: boolean) => void;
  toggleSeedLock: () => void;
  rerun: () => void;
  reset: () => void;
};

export const useMarketStore = create<MarketState>((set, get) => ({
  scenario: DEFAULT_SCENARIO,
  secondarySeed: 91,
  activeStep: 60,
  playing: false,
  seedLocked: false,
  setScenario: (scenario) =>
    set({ scenario: normalizeScenario(scenario), activeStep: Math.min(60, scenario.periods) }),
  patchScenario: (patch) =>
    set((state) => ({
      scenario: normalizeScenario({ ...state.scenario, ...patch }),
      activeStep: Math.min(state.activeStep, patch.periods ?? state.scenario.periods),
    })),
  setSecondarySeed: (secondarySeed) => set({ secondarySeed }),
  setActiveStep: (activeStep) => set({ activeStep }),
  setPlaying: (playing) => set({ playing }),
  toggleSeedLock: () => set((state) => ({ seedLocked: !state.seedLocked })),
  rerun: () => {
    const { scenario, seedLocked } = get();
    if (!seedLocked) {
      set({
        scenario: { ...scenario, seed: (scenario.seed + 1) >>> 0 },
        secondarySeed: (get().secondarySeed + 17) >>> 0,
        activeStep: Math.min(60, scenario.periods),
      });
    } else {
      set({ activeStep: Math.min(60, scenario.periods) });
    }
  },
  reset: () =>
    set({
      scenario: DEFAULT_SCENARIO,
      secondarySeed: 91,
      activeStep: 60,
      playing: false,
      seedLocked: false,
    }),
}));
