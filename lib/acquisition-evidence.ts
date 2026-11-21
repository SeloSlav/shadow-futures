import { z } from "zod";

import ledger from "@/public/papers/product-continuation-and-independent-experimentation/acquisition-pilot-evidence.json";

const source = z.url({ protocol: /^https$/ });
const optionalSource = source.nullish();
const dealSchema = z.object({
  name: z.string(),
  original_outcome: z.enum(["closed", "fell through"]),
  anchor_date: z.iso.date(),
  anchor_source: source,
  case_source: source,
  ownership_and_remedies: z.string(),
  ownership_sources: z.array(source),
});
const projectSchema = z.object({
  deal: z.string(),
  product: z.string(),
  overlap: z.enum(["direct", "potential", "no direct overlap established"]),
  overlap_reason: z.string(),
  overlap_source: source,
  outcome: z.enum(["confirmed shipment", "active only", "documented cessation", "unresolved"]),
  evidence_date: z.union([z.iso.date(), z.string().regex(/^\d{4}-(0[1-9]|1[0-2])$/)]).nullable(),
  evidence: z.string(),
  outcome_source: source,
  additional_source: optionalSource,
  review_source: optionalSource,
  review_additional_source: optionalSource,
  review_note: z.string().optional(),
  path: z.string(),
});

export type AcquisitionEvidenceRow = Omit<z.infer<typeof projectSchema>, "deal"> & {
  id: string;
  deal: z.infer<typeof dealSchema>;
};

export function loadAcquisitionEvidence() {
  const { projects, deals } = z.object({
    deals: z.record(z.string(), dealSchema),
    projects: z.array(projectSchema),
  }).parse(ledger);

  const rows: AcquisitionEvidenceRow[] = projects.map(({ deal: dealId, ...project }) => {
    const deal = deals[dealId];
    if (!deal) throw new Error(`Missing deal for ${project.product}`);
    return { ...project, id: `${dealId}:${project.product}`, deal };
  });

  return { rows, dealCount: Object.keys(deals).length };
}
