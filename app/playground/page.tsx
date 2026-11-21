import type { Metadata } from "next";

import { ComparisonBudget } from "@/components/playground/comparison-budget";

const title = "Comparison Budget Playground";
const description = "Try six markets, from AI to restaurants. Watch two rivals compete for 4,000 sales and see how the comparison budget changes when you rotate the shelf.";

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
