export type InputVector = number[];

export type Scenario = {
  name: string;
  n: number;
  inputs: InputVector[];
  initialPositions: number[];
  beta: InputVector;
  baseline: number;
  rho: number;
  periods: number;
  seed: number;
  worlds: number;
  channels: number;
  exploration: number;
  resetCadence: number;
};

export type AllocationStep = {
  t: number;
  probabilities: number[];
  counts: number[];
  recipient: number;
  residualContestability: number;
  comparisonBudget: number;
  information: number;
  informationBound: number;
  cumulativeInformation: number;
  leader: number;
};

export type SimulationResult = {
  scenario: Scenario;
  steps: AllocationStep[];
  finalCounts: number[];
  winner: number;
  concentration: number;
  comparisonBudget: number;
  cumulativeInformation: number;
};

export type WorldSummary = {
  seed: number;
  winner: number;
  finalCounts: number[];
  shares: number[];
  concentration: number;
  comparisonBudget: number;
  cumulativeInformation: number;
};
