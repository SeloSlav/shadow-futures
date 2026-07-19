import { describe, expect, it } from "vitest";

import { DEFAULT_SCENARIO } from "@/lib/model/simulation";
import {
  deserializeScenario,
  serializeScenario,
} from "@/lib/scenarios/serialization";

describe("scenario URL state", () => {
  it("serializes and restores a scenario losslessly", () => {
    const scenario = {
      ...DEFAULT_SCENARIO,
      name: "Round trip",
      beta: [1.2, -0.3],
      inputs: [
        [0.8, 0.1],
        [0.6, 0.4],
        [0.3, 0.7],
        [0.2, 0.5],
        [0.1, 0.2],
      ],
      initialPositions: [1, 2, 3, 4, 5],
      exploration: 0.07,
      resetCadence: 120,
    };
    expect(deserializeScenario(serializeScenario(scenario))).toEqual(scenario);
  });
});
