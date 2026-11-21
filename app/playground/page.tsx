import type { Metadata } from "next";

import { ComparisonBudget } from "@/components/playground/comparison-budget";

const title = "Comparison Budget Playground";
const description = "Compare any two of six markets on one chart. Replay the same 4,000 sales to see how their comparison budgets grow under each shelf rule.";

export const metadata: Metadata = {
  title,
  description,
  alternates: { canonical: "/playground" },
  openGraph: {
    type: "website",
    title,
    description,
    url: "/playground",
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
  },
};

export default function ComparisonBudgetPage() {
  return <ComparisonBudget />;
}
