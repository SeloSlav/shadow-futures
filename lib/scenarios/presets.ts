import { DEFAULT_SCENARIO } from "@/lib/model/simulation";
import type { Scenario } from "@/lib/model/types";

const withDefaults = (scenario: Partial<Scenario> & Pick<Scenario, "name">): Scenario => ({
  ...DEFAULT_SCENARIO,
  ...scenario,
});

export const SCENARIO_PRESETS: Scenario[] = [
  withDefaults({
    name: "Two founders",
    n: 2,
    inputs: [[0.82], [0.58]],
    initialPositions: [0, 0],
    periods: 240,
    worlds: 128,
  }),
  withDefaults({ name: "Creator platform" }),
  withDefaults({
    name: "Gig marketplace",
    inputs: [[0.74], [0.65], [0.48], [0.31], [0.2]],
    rho: 1.2,
  }),
  withDefaults({
    name: "Scientific citations",
    inputs: [[0.76], [0.69], [0.55], [0.43], [0.32]],
    rho: 1.1,
    periods: 1_000,
  }),
  withDefaults({
    name: "App store",
    n: 7,
    inputs: [[0.82], [0.74], [0.61], [0.53], [0.38], [0.29], [0.17]],
    initialPositions: [1, 0, 0, 0, 0, 0, 0],
    periods: 800,
  }),
  withDefaults({ name: "No reinforcement", rho: 0 }),
  withDefaults({ name: "Linear boundary", rho: 1 }),
  withDefaults({ name: "Strong reinforcement", rho: 1.8 }),
  withDefaults({ name: "Randomized exposure", exploration: 0.08 }),
  withDefaults({ name: "Multiple independent channels", channels: 8, periods: 64 }),
];

export function findPreset(name: string): Scenario | undefined {
  return SCENARIO_PRESETS.find((preset) => preset.name === name);
}
