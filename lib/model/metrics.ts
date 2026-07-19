import type { InputVector } from "./types";

export function dot(left: InputVector, right: InputVector): number {
  return left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0);
}

export function residualContestability(probabilities: number[]): number {
  return 1 - Math.max(...probabilities);
}

export function fisherInformationTrace(
  probabilities: number[],
  inputs: InputVector[],
): number {
  const dimensions = Math.max(1, ...inputs.map((input) => input.length));
  const means = Array.from({ length: dimensions }, (_, dimension) =>
    inputs.reduce(
      (sum, input, index) => sum + probabilities[index] * (input[dimension] ?? 0),
      0,
    ),
  );

  return inputs.reduce((trace, input, index) => {
    const squaredDistance = means.reduce((sum, mean, dimension) => {
      const difference = (input[dimension] ?? 0) - mean;
      return sum + difference * difference;
    }, 0);
    return trace + probabilities[index] * squaredDistance;
  }, 0);
}

export function squaredDiameter(inputs: InputVector[]): number {
  let maximum = 0;
  for (let left = 0; left < inputs.length; left += 1) {
    for (let right = left + 1; right < inputs.length; right += 1) {
      const dimensions = Math.max(inputs[left].length, inputs[right].length);
      let distance = 0;
      for (let dimension = 0; dimension < dimensions; dimension += 1) {
        const difference =
          (inputs[left][dimension] ?? 0) - (inputs[right][dimension] ?? 0);
        distance += difference * difference;
      }
      maximum = Math.max(maximum, distance);
    }
  }
  return maximum;
}

export function herfindahl(shares: number[]): number {
  return shares.reduce((sum, share) => sum + share * share, 0);
}

export function quantile(sortedValues: number[], probability: number): number {
  if (sortedValues.length === 0) return 0;
  const index = (sortedValues.length - 1) * probability;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

export function gaugeTransform(
  beta: InputVector,
  positions: number[],
  inputs: InputVector[],
  displacement: InputVector,
): { beta: InputVector; positions: number[] } {
  const transformedBeta = beta.map(
    (value, index) => value + (displacement[index] ?? 0),
  );
  const transformedPositions = positions.map(
    (position, index) => position - dot(inputs[index], displacement),
  );
  return { beta: transformedBeta, positions: transformedPositions };
}
