export type EquationKind =
  | "definition"
  | "identity"
  | "bound"
  | "condition"
  | "theorem"
  | "policy implication";

export type VariableDefinition = {
  symbol: string;
  name: string;
  definition: string;
};

export type ControlDefinition = {
  symbol: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
};

export type EquationDefinition = {
  id: string;
  title: string;
  section: string;
  equationNumber?: string;
  latex: string;
  plainLanguage: string;
  assumptions: string[];
  variables: VariableDefinition[];
  controls?: ControlDefinition[];
  visualization: string;
  derivationSteps: string[];
  kind: EquationKind;
  source?: "displayed" | "inline";
};
