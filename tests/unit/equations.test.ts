import { describe, expect, it } from "vitest";

import { EQUATIONS } from "@/lib/equations/registry";

describe("equation registry", () => {
  it("registers every displayed equation from the paper and appendix", () => {
    expect(EQUATIONS.filter((equation) => equation.source !== "inline")).toHaveLength(49);
    expect(EQUATIONS).toHaveLength(52);
    expect(new Set(EQUATIONS.map((equation) => equation.id)).size).toBe(52);
  });

  it("provides the explanatory fields required by the equation lab", () => {
    for (const equation of EQUATIONS) {
      expect(equation.latex.length).toBeGreaterThan(3);
      expect(equation.plainLanguage.length).toBeGreaterThan(20);
      expect(equation.variables.length).toBeGreaterThan(0);
      expect(equation.assumptions.length).toBeGreaterThan(0);
      expect(equation.derivationSteps.length).toBeGreaterThan(0);
      expect(equation.visualization.length).toBeGreaterThan(0);
    }
  });
});
