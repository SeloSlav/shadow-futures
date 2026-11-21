import type { Metadata } from "next";

import { ComparisonBudget } from "@/components/playground/comparison-budget";

const title = "The Comparison Budget";
const description = "Watch two products compete for 4,000 sales. Rotate the shelf to see how giving a rival a chance changes what those sales can teach us.";

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
