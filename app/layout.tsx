import type { Metadata } from "next";
import type { ReactNode } from "react";

import "@/app/globals.css";
import { SiteHeader } from "@/components/site-header";
import { PAPER } from "@/lib/paper/citation";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3010"),
  title: {
    default: "Shadow Futures — Contribution Uncertainty and the Self-Reinforcing Market",
    template: "%s — Shadow Futures",
  },
  description:
    "An interactive explanation of how early advantages compound for creators and firms, why winners change across replays, and how markets can lose the comparisons needed to measure contribution.",
  authors: [{ name: PAPER.author }],
  creator: PAPER.author,
  alternates: { canonical: "/" },
  openGraph: {
    type: "article",
    title: PAPER.title,
    description:
      "An early audience or customer can become a lasting market lead. Explore the careers, firms, and technologies hidden by self-reinforcing competition.",
    authors: [PAPER.author],
    siteName: "Shadow Futures",
  },
  twitter: {
    card: "summary_large_image",
    title: PAPER.title,
    description:
      "Animated graphs explain contribution uncertainty for creators and firms in self-reinforcing markets.",
  },
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
