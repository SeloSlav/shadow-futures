import { dot } from "./metrics";
import type { InputVector } from "./types";

export type AllocationArguments = {
  inputs: InputVector[];
  beta: InputVector;
  counts: number[];
  initialPositions: number[];
  baseline: number;
  rho: number;
  exploration?: number;
  latentPositions?: number[];
};

export function allocationProbabilities({
  inputs,
  beta,
  counts,
  initialPositions,
  baseline,
  rho,
  exploration = 0,
  latentPositions,
}: AllocationArguments): number[] {
  const logits = inputs.map((input, index) => {
    const accumulatedPosition = initialPositions[index] + counts[index];
    const reinforcement =
      rho === 0 ? 0 : rho * Math.log(Math.max(Number.EPSILON, baseline + accumulatedPosition));
    return dot(input, beta) + reinforcement + (latentPositions?.[index] ?? 0);
  });
  const maximum = Math.max(...logits);
  const weights = logits.map((logit) => Math.exp(logit - maximum));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const base = weights.map((weight) => weight / total);
  const eta = Math.min(1, Math.max(0, exploration));
  return base.map((probability) => (1 - eta) * probability + eta / base.length);
}

export function drawRecipient(probabilities: number[], randomValue: number): number {
  let cumulative = 0;
  for (let index = 0; index < probabilities.length; index += 1) {
    cumulative += probabilities[index];
    if (randomValue < cumulative || index === probabilities.length - 1) {
      return index;
    }
  }
  return probabilities.length - 1;
}
