import { AUTHOR, PAPER } from "@/lib/paper/citation";

export type ResearchPaper = {
  id: string;
  title: string;
  focus: string;
  description: string;
  pdfPath: string;
  pages: number;
  status: string;
};

export const RESEARCH_REVISION = "October 2026";
export const RESEARCH_PUBLISHED_DATE = "2026-10-09";
export const RESEARCH_AUTHOR = AUTHOR.name;

export const MAIN_RESEARCH_PAPER: ResearchPaper = {
  id: "shadow-futures",
  title: PAPER.title,
  focus: "The foundational paper",
  description:
    "Introduces the comparison budget and shows when one adaptive market history cannot consistently reveal contribution effects, even as forecasts converge. The result separates abundant transactions from the comparisons needed for attribution.",
  pdfPath: PAPER.pdfPath,
  pages: 22,
  status: "Author preprint · V5",
};

export const COMPANION_PAPERS: readonly ResearchPaper[] = [
  {
    id: "institutions-for-comparative-discovery",
    title: "Institutions for Comparative Discovery in Scaling Markets",
    focus: "Institutions and governance",
    description:
      "Examines how contracts, procurement, independent testing and shared infrastructure can assemble a worthwhile comparative experiment. A joint-permission model distinguishes participation, financing and coordination when separately controlled inputs are informative together.",
    pdfPath: "/papers/institutions-for-comparative-discovery.pdf",
    pages: 23,
    status: "Research draft",
  },
  {
    id: "endogenous-contestability-and-technological-scale",
    title:
      "Endogenous Contestability and Technological Scale: Entry, Selection, and the Value of Comparison",
    focus: "Entry and technological scale",
    description:
      "Models entry and access alongside productive scale. More entry need not produce more useful information: the analysis separates qualified challengers’ entry incentives, the value of their allocation signal, and the policy margins needed to preserve informative comparison.",
    pdfPath: "/papers/endogenous-contestability-and-technological-scale.pdf",
    pages: 23,
    status: "Research draft",
  },
  {
    id: "measuring-comparison-opportunities",
    title:
      "Measuring Comparison Opportunities in Adaptive Markets: Finite-Horizon Inference and Experimental Design",
    focus: "Measurement and experimental design",
    description:
      "Develops a finite-horizon reporting and design protocol for comparison budgets, parameter information and discrimination. Reproducible synthetic experiments evaluate replications, response channels and reset persistence, with inference conditional on a declared model and nuisance class.",
    pdfPath: "/papers/measuring-comparison-opportunities.pdf",
    pages: 29,
    status: "Research draft",
  },
  {
    id: "forecast-agreement-and-counterfactual-warrant",
    title: "Forecast Agreement and Counterfactual Warrant in Adaptive Markets",
    focus: "Counterfactuals and methodology",
    description:
      "Explains why agreeing forecasts can coexist with unresolved attribution. An evidence contract connects a claim to its intervention, observation channels, model restrictions and decision stakes, including a worked AI-procurement example that separates performance from contribution attribution.",
    pdfPath: "/papers/forecast-agreement-and-counterfactual-warrant.pdf",
    pages: 24,
    status: "Research draft",
  },
  {
    id: "competition-policy-for-informative-market-access",
    title: "Competition Policy for Informative Market Access",
    focus: "Competition policy",
    description:
      "Evaluates costed access remedies using a common decision rule across plausible models. Robust certificates and a hypothetical procurement comparison connect information to welfare, while retaining the statutory conditions of European competition and digital regulation.",
    pdfPath: "/papers/competition-policy-for-informative-market-access.pdf",
    pages: 30,
    status: "Research draft",
  },
  {
    id: "taxing-lost-comparison",
    title:
      "Taxing Lost Comparison: Decision Information, Private Testing, and Costed Corrective Instruments",
    focus: "Corrective instruments and public testing",
    description:
      "Derives a conditional corrective margin from a valued decision experiment. The model separates activity and private-testing incentives, shows when public testing crowds out private trials, and uses available replacement testing to cap the marginal welfare cost of lost organic comparison.",
    pdfPath: "/papers/taxing-lost-comparison.pdf",
    pages: 25,
    status: "Research draft",
  },
];
