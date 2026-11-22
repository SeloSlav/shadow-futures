import { AUTHOR, PAPER } from "@/lib/paper/citation";

export type ResearchPaper = {
  id: string;
  title: string;
  focus: string;
  description: string;
  pdfPath: string;
  pages: number;
  status: string;
  publishedDate?: string;
  evidencePath?: string;
  supplements?: readonly { label: string; path: string }[];
};

export const RESEARCH_PUBLISHED_DATE = "2026-10-09";
export const RESEARCH_UPDATED_DATE = "2026-11-22";
export const RESEARCH_REVISION = "November 2026";
export const RESEARCH_AUTHOR = AUTHOR.name;

const acquisitionPaperId = "product-continuation-and-independent-experimentation";
const acquisitionPaperPath = `/papers/${acquisitionPaperId}`;
export const ACQUISITION_PAPER = {
  id: acquisitionPaperId,
  publishedDate: "2026-11-21",
  pdfPath: `${acquisitionPaperPath}.pdf`,
  evidencePath: `${acquisitionPaperPath}/evidence`,
  ledgerPath: `${acquisitionPaperPath}/acquisition-pilot-evidence.json`,
  reviewPath: `${acquisitionPaperPath}/acquisition-independent-review.json`,
};

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
      "The National Oncologic PET Registry shows how a payer, imaging facilities and referring physicians assembled evidence under conditional coverage. A joint-permission model explains when a sponsor can finance the necessary coordination.",
    pdfPath: "/papers/institutions-for-comparative-discovery.pdf",
    pages: 24,
    status: "Research draft",
  },
  {
    id: "endogenous-contestability-and-technological-scale",
    title:
      "Endogenous Contestability and Technological Scale: Entry, Selection, and the Value of Comparison",
    focus: "Entry and technological scale",
    description:
      "When applicants privately know their prospects, entry itself conveys information. The paper derives its decision value and shows when tightening access lowers entry while raising the value of the comparisons that remain.",
    pdfPath: "/papers/endogenous-contestability-and-technological-scale.pdf",
    pages: 25,
    status: "Research draft",
  },
  {
    id: "measuring-comparison-opportunities",
    title:
      "Measuring Comparison Opportunities in Adaptive Markets: Finite-Horizon Inference and Experimental Design",
    focus: "Measurement and experimental design",
    description:
      "Two quality and reinforcement pairs yield the same long-run share. At 8,192 sales an optimal classifier still errs about 19% of the time under equal priors. A separate MusicLab application checks next-download forecasts in seven held-out worlds.",
    pdfPath: "/papers/measuring-comparison-opportunities.pdf",
    pages: 31,
    status: "Research draft",
    supplements: [{ label: "Replication package (ZIP)", path: "/papers/measuring-comparison-opportunities/replication.zip" }],
  },
  {
    id: "forecast-agreement-and-counterfactual-warrant",
    title: "Forecast Agreement and Counterfactual Warrant in Adaptive Markets",
    focus: "Counterfactuals and methodology",
    description:
      "A randomized procurement trial can justify buying a service while leaving its producer’s contribution to market rewards unresolved. A worked example specifies the outcomes, assumptions and evidence each claim requires.",
    pdfPath: "/papers/forecast-agreement-and-counterfactual-warrant.pdf",
    pages: 14,
    status: "Research draft",
  },
  {
    id: "competition-policy-for-informative-market-access",
    title: "Competition Policy for Informative Market Access",
    focus: "Competition policy",
    description:
      "A hypothetical fifteen-trial procurement pilot certifies a €103,207 gain under the stated signal law. Allowing 15% arbitrary contamination lowers the guarantee to −€3,024. The paper derives the accuracy and cost bounds needed to defend the decision.",
    pdfPath: "/papers/competition-policy-for-informative-market-access.pdf",
    pages: 29,
    status: "Research draft",
  },
  {
    id: "taxing-lost-comparison",
    title:
      "The Value of Replacing Lost Market Experiments",
    focus: "Corrective instruments and public testing",
    description:
      "When an equivalent independent trial can replace lost market evidence, its resource cost caps the marginal loss. A drinking-water assay illustrates how to test decision equivalence and measure full replacement costs.",
    pdfPath: "/papers/taxing-lost-comparison.pdf",
    pages: 26,
    status: "Research draft",
  },
  {
    id: ACQUISITION_PAPER.id,
    title: "Product Continuation and Independent Experimentation after Software Acquisitions",
    focus: "Acquisitions and experimentation",
    description:
      "Tracks 29 product lines across 22 UK merger cases. A second AI coding pass agreed on 27 shipment classifications. The revised study traces interim restrictions and substitute owners, and distinguishes evidence of a release from evidence of who authorized it.",
    pdfPath: ACQUISITION_PAPER.pdfPath,
    evidencePath: ACQUISITION_PAPER.evidencePath,
    pages: 28,
    status: "Research draft",
    publishedDate: ACQUISITION_PAPER.publishedDate,
    supplements: [
      { label: "Second AI coding and reconciliation (JSON)", path: ACQUISITION_PAPER.reviewPath },
      {
        label: "Evidence ledger (JSON)",
        path: ACQUISITION_PAPER.ledgerPath,
      },
      {
        label: "Replication code (Python)",
        path: `${acquisitionPaperPath}/summarize-acquisition-pilot.py`,
      },
    ],
  },
  {
    id: "learning-contribution-from-one-market-history",
    title: "Learning Contribution from One Market History",
    focus: "Learning from adaptive allocations",
    description:
      "Two models fit the same procurement awards and disagree about whether faster delivery pays. With five orders and the rival initially holding the badge, a vendor chooses standard delivery first under a fixed badge and fast delivery first under a transferable badge for priors between about 26.864% and 66.298% on the high-return model. Under transfer, every optimal policy preserves an uncertain vendor’s prior on all 32 award sequences exactly when that prior exceeds 64%. At 64%, the result depends on how the vendor breaks a last-order tie. The paper shows when an award history can identify the return to quality apart from brand.",
    pdfPath: "/papers/learning-contribution-from-one-market-history.pdf",
    pages: 32,
    status: "Research draft",
    publishedDate: "2026-11-22",
    supplements: [
      { label: "Dispatch decisions and Tables 1 and 3 (Python)", path: "/papers/learning-contribution-from-one-market-history/procurement-calculations.py" },
      { label: "AI review of the earlier full-memory proof", path: "/papers/learning-contribution-from-one-market-history/mathematical-review.md" },
      {
        label: "Badge calculations and Table 2 (Python)",
        path: "/papers/learning-contribution-from-one-market-history/badge-calculations.py",
      },
      {
        label: "Full-memory estimator (Python)",
        path: "/papers/learning-contribution-from-one-market-history/full-memory-estimator.py",
      },
    ],
  },
];
