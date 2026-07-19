import { z } from "zod";

import { normalizeScenario } from "@/lib/model/simulation";
import type { Scenario } from "@/lib/model/types";

const scenarioSchema = z.object({
  name: z.string().max(80),
  n: z.number().int().min(2).max(10),
  inputs: z.array(z.array(z.number().finite()).min(1).max(2)).min(2).max(10),
  initialPositions: z.array(z.number().finite().nonnegative()).min(2).max(10),
  beta: z.array(z.number().finite()).min(1).max(2),
  baseline: z.number().finite().positive(),
  rho: z.number().finite().min(0).max(2.5),
  periods: z.number().int().min(10).max(10_000),
  seed: z.number().int().nonnegative(),
  worlds: z.number().int().min(2).max(1_000),
  channels: z.number().int().min(1).max(100),
  exploration: z.number().finite().min(0).max(0.5),
  resetCadence: z.number().int().nonnegative(),
});

function toBase64Url(value: string): string {
  if (typeof window === "undefined") {
    return Buffer.from(value, "utf8").toString("base64url");
  }
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
}

function fromBase64Url(value: string): string {
  if (typeof window === "undefined") {
    return Buffer.from(value, "base64url").toString("utf8");
  }
  const normalized = value.replaceAll("-", "+").replaceAll("_", "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  const binary = atob(padded);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

export function serializeScenario(scenario: Scenario): string {
  return toBase64Url(JSON.stringify(scenarioSchema.parse(normalizeScenario(scenario))));
}

export function deserializeScenario(serialized: string): Scenario {
  const parsed: unknown = JSON.parse(fromBase64Url(serialized));
  return normalizeScenario(scenarioSchema.parse(parsed));
}

export function scenarioFromSearch(search: string): Scenario | null {
  const encoded = new URLSearchParams(search).get("scenario");
  if (!encoded) return null;
  try {
    return deserializeScenario(encoded);
  } catch {
    return null;
  }
}
