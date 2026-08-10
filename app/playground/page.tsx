import type { Metadata } from "next";

import { ShadowPlayground } from "@/components/playground/shadow-playground";

export const metadata: Metadata = {
  title: "Five-Company Competition Playground",
  description:
    "Follow five companies competing for 360 customers and see how product quality, self-reinforcement, and guaranteed challenger discovery change the market.",
  alternates: { canonical: "/playground" },
  openGraph: {
    type: "website",
    title: "Shadow Futures Five-Company Competition Playground",
    description:
      "Follow five companies competing for 360 customers and watch early wins reshape later chances.",
    url: "/playground",
  },
};

export default function PlaygroundPage() {
  return <ShadowPlayground />;
}
