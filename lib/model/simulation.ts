import { allocationProbabilities, drawRecipient } from "./allocation";
import {
  fisherInformationTrace,
  herfindahl,
  residualContestability,
  squaredDiameter,
} from "./metrics";
import { deriveSeed, mulberry32 } from "./prng";
import type { Scenario, SimulationResult, WorldSummary } from "./types";

export const DEFAULT_SCENARIO: Scenario = {
  name: "Creator platform",
  n: 5,
  inputs: [[0.78], [0.6], [0.42], [0.22], [0.08]],
  initialPositions: [0, 0, 0, 0, 0],
  beta: [1],
  baseline: 1,
  rho: 1.35,
  periods: 500,
  seed: 42,
  worlds: 128,
  channels: 1,
  exploration: 0,
  resetCadence: 0,
};

export function normalizeScenario(candidate: Scenario): Scenario {
  const n = Math.min(10, Math.max(2, Math.round(candidate.n)));
  const dimensions = Math.min(2, Math.max(1, candidate.beta.length));
  const fallbackInput = (index: number) => [Math.max(0, 0.8 - index * 0.15)];
  const inputs = Array.from({ length: n }, (_, index) => {
    const input = candidate.inputs[index] ?? fallbackInput(index);
    return Array.from({ length: dimensions }, (_, dimension) =>
      Number.isFinite(input[dimension]) ? input[dimension] : 0,
    );
  });
  return {
    ...candidate,
    n,
    inputs,
    initialPositions: Array.from({ length: n }, (_, index) =>
      Math.max(0, candidate.initialPositions[index] ?? 0),
    ),
    beta: Array.from({ length: dimensions }, (_, dimension) =>
      Number.isFinite(candidate.beta[dimension]) ? candidate.beta[dimension] : 0,
    ),
    baseline: Math.max(0.05, candidate.baseline),
    rho: Math.min(2.5, Math.max(0, candidate.rho)),
    periods: Math.min(10_000, Math.max(10, Math.round(candidate.periods))),
    seed: Math.max(0, Math.round(candidate.seed)) >>> 0,
    worlds: Math.min(1_000, Math.max(2, Math.round(candidate.worlds))),
    channels: Math.min(100, Math.max(1, Math.round(candidate.channels))),
    exploration: Math.min(0.5, Math.max(0, candidate.exploration)),
    resetCadence: Math.max(0, Math.round(candidate.resetCadence)),
  };
}

export function simulateScenario(candidate: Scenario): SimulationResult {
  const scenario = normalizeScenario(candidate);
  const random = mulberry32(scenario.seed);
  const counts = Array.from({ length: scenario.n }, () => 0);
  const diameterSquared = squaredDiameter(scenario.inputs);
  const steps: SimulationResult["steps"] = [];
  let comparisonBudget = 0;
  let cumulativeInformation = 0;

  for (let t = 0; t < scenario.periods; t += 1) {
    if (scenario.resetCadence > 0 && t > 0 && t % scenario.resetCadence === 0) {
      counts.fill(0);
    }
    const probabilities = allocationProbabilities({
      inputs: scenario.inputs,
      beta: scenario.beta,
      counts,
      initialPositions: scenario.initialPositions,
      baseline: scenario.baseline,
      rho: scenario.rho,
      exploration: scenario.exploration,
    });
    const residual = residualContestability(probabilities);
    const information = fisherInformationTrace(probabilities, scenario.inputs);
    const informationBound = diameterSquared * residual;
    comparisonBudget += residual;
    cumulativeInformation += information;
    const recipient = drawRecipient(probabilities, random());
    counts[recipient] += 1;
    const leader = probabilities.indexOf(Math.max(...probabilities));
    steps.push({
      t: t + 1,
      probabilities,
      counts: [...counts],
      recipient,
      residualContestability: residual,
      comparisonBudget,
      information,
      informationBound,
      cumulativeInformation,
      leader,
    });
  }

  const total = counts.reduce((sum, count) => sum + count, 0) || 1;
  const shares = counts.map((count) => count / total);
  return {
    scenario,
    steps,
    finalCounts: counts,
    winner: counts.indexOf(Math.max(...counts)),
    concentration: herfindahl(shares),
    comparisonBudget,
    cumulativeInformation,
  };
}

export function summarizeWorld(result: SimulationResult): WorldSummary {
  const total = result.finalCounts.reduce((sum, count) => sum + count, 0) || 1;
  return {
    seed: result.scenario.seed,
    winner: result.winner,
    finalCounts: result.finalCounts,
    shares: result.finalCounts.map((count) => count / total),
    concentration: result.concentration,
    comparisonBudget: result.comparisonBudget,
    cumulativeInformation: result.cumulativeInformation,
  };
}

export function simulateWorlds(
  scenario: Scenario,
  count = scenario.worlds,
): WorldSummary[] {
  return Array.from({ length: count }, (_, index) => {
    const seededScenario = {
      ...scenario,
      seed: deriveSeed(scenario.seed, index),
    };
    return summarizeWorld(simulateScenario(seededScenario));
  });
}
