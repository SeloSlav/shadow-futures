import type { Metadata } from "next";

import { ShadowPlayground } from "@/components/playground/shadow-playground";

export const metadata: Metadata = {
  title: "Comparison Playground",
  description:
    "Run the central Shadow Futures equation as an interactive 3D market and watch comparison paths disappear as self-reinforcement closes the contest.",
  alternates: { canonical: "/playground" },
  openGraph: {
    type: "website",
    title: "Shadow Futures Comparison Playground",
    description:
      "Run the central equation and watch comparison paths disappear as self-reinforcement closes a market.",
    url: "/playground",
  },
};

export default function PlaygroundPage() {
  return <ShadowPlayground />;
}
