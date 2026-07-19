import { describe, expect, it } from "vitest";

import { allocationProbabilities } from "@/lib/model/allocation";
import {
  fisherInformationTrace,
  gaugeTransform,
  residualContestability,
  squaredDiameter,
} from "@/lib/model/metrics";
import { deriveSeed, mulberry32 } from "@/lib/model/prng";
import {
  DEFAULT_SCENARIO,
  simulateScenario,
  simulateWorlds,
} from "@/lib/model/simulation";
import type { Scenario } from "@/lib/model/types";

describe("allocation model", () => {
  it("produces positive probabilities that sum to one", () => {
    const probabilities = allocationProbabilities({
      inputs: [[0.8], [0.4], [-0.2]],
      beta: [1.4],
      counts: [20, 2, 0],
      initialPositions: [0, 0, 0],
      baseline: 1,
      rho: 1.35,
    });
    expect(probabilities.every((probability) => probability > 0)).toBe(true);
    expect(probabilities.reduce((sum, probability) => sum + probability, 0)).toBeCloseTo(1, 12);
  });

  it("is deterministic for a fixed seed", () => {
    const first = simulateScenario(DEFAULT_SCENARIO);
    const second = simulateScenario(DEFAULT_SCENARIO);
    expect(second.finalCounts).toEqual(first.finalCounts);
    expect(second.steps.map((step) => step.recipient)).toEqual(
      first.steps.map((step) => step.recipient),
    );
    expect(second.comparisonBudget).toBe(first.comparisonBudget);
  });

  it("computes residual contestability as one minus the largest probability", () => {
    const probabilities = [0.14, 0.52, 0.34];
    expect(residualContestability(probabilities)).toBeCloseTo(0.48, 12);
  });

  it("accumulates comparison budget from residual contestability", () => {
    const result = simulateScenario({ ...DEFAULT_SCENARIO, periods: 120 });
    const directSum = result.steps.reduce(
      (sum, step) => sum + step.residualContestability,
      0,
    );
    expect(result.comparisonBudget).toBeCloseTo(directSum, 12);
    expect(result.steps.at(-1)?.comparisonBudget).toBeCloseTo(directSum, 12);
  });

  it("satisfies the information bound for many bounded input profiles", () => {
    const random = mulberry32(7);
    for (let trial = 0; trial < 300; trial += 1) {
      const n = 2 + Math.floor(random() * 9);
      const dimensions = random() > 0.5 ? 2 : 1;
      const inputs = Array.from({ length: n }, () =>
        Array.from({ length: dimensions }, () => random() * 4 - 2),
      );
      const raw = Array.from({ length: n }, () => 0.01 + random());
      const total = raw.reduce((sum, value) => sum + value, 0);
      const probabilities = raw.map((value) => value / total);
      const information = fisherInformationTrace(probabilities, inputs);
      const bound =
        squaredDiameter(inputs) * residualContestability(probabilities);
      expect(information).toBeLessThanOrEqual(bound + 1e-10);
    }
  });

  it("no reinforcement preserves more comparison than strong reinforcement in the seeded illustration", () => {
    const base: Scenario = {
      ...DEFAULT_SCENARIO,
      periods: 600,
      inputs: [[0.72], [0.65], [0.54], [0.41], [0.3]],
    };
    const noFeedback = simulateScenario({ ...base, rho: 0 });
    const strong = simulateScenario({ ...base, rho: 1.9 });
    expect(noFeedback.comparisonBudget).toBeGreaterThan(strong.comparisonBudget);
  });

  it("stronger superlinear reinforcement typically accelerates concentration", () => {
    const weakConcentrations: number[] = [];
    const strongConcentrations: number[] = [];
    for (let index = 0; index < 36; index += 1) {
      const seed = deriveSeed(21, index);
      weakConcentrations.push(
        simulateScenario({ ...DEFAULT_SCENARIO, seed, periods: 450, rho: 1.05 })
          .concentration,
      );
      strongConcentrations.push(
        simulateScenario({ ...DEFAULT_SCENARIO, seed, periods: 450, rho: 1.9 })
          .concentration,
      );
    }
    const average = (values: number[]) =>
      values.reduce((sum, value) => sum + value, 0) / values.length;
    expect(average(strongConcentrations)).toBeGreaterThan(average(weakConcentrations));
  });

  it("parallel worlds use distinct deterministic seeds without mutating shared parameters", () => {
    const base = { ...DEFAULT_SCENARIO, worlds: 20 };
    const before = JSON.stringify(base);
    const first = simulateWorlds(base, 20);
    const second = simulateWorlds(base, 20);
    expect(new Set(first.map((world) => world.seed)).size).toBe(20);
    expect(first).toEqual(second);
    expect(JSON.stringify(base)).toBe(before);
  });

  it("gauge transformations preserve composite indices and allocation probabilities", () => {
    const inputs = [[0.9], [0.5], [0.2]];
    const beta = [1.1];
    const positions = [0.2, -0.1, 0.35];
    const displacement = [0.7];
    const transformed = gaugeTransform(beta, positions, inputs, displacement);
    const compositeBefore = inputs.map(
      (input, index) => input[0] * beta[0] + positions[index],
    );
    const compositeAfter = inputs.map(
      (input, index) =>
        input[0] * transformed.beta[0] + transformed.positions[index],
    );
    expect(compositeAfter).toEqual(compositeBefore);

    const common = {
      inputs,
      counts: [0, 0, 0],
      initialPositions: [0, 0, 0],
      baseline: 1,
      rho: 0,
    };
    const before = allocationProbabilities({
      ...common,
      beta,
      latentPositions: positions,
    });
    const after = allocationProbabilities({
      ...common,
      beta: transformed.beta,
      latentPositions: transformed.positions,
    });
    after.forEach((probability, index) =>
      expect(probability).toBeCloseTo(before[index], 14),
    );
  });

  it("exploration floors keep every alternative probability positive", () => {
    const exploration = 0.12;
    const probabilities = allocationProbabilities({
      inputs: [[10], [0], [-10]],
      beta: [10],
      counts: [10_000, 0, 0],
      initialPositions: [0, 0, 0],
      baseline: 1,
      rho: 2.5,
      exploration,
    });
    probabilities.forEach((probability) =>
      expect(probability).toBeGreaterThanOrEqual(exploration / 3),
    );
  });
});
