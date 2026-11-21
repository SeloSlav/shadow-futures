import { allocationProbabilities, drawRecipient } from "./allocation";
import { residualContestability } from "./metrics";
import { deriveSeed, mulberry32 } from "./prng";

export const TOTAL_SALES = 4_000;
export const ROTATION_START = 3_000;
export const ROTATION_LENGTH = 300;
export const SHADOW_SALES = 1_000;

export const MARKET_PRESETS = [
  { name: "Big Tech", qualityEdge: 0.3, reinforcement: 2.4, description: "Two search engines compete to be the default. A larger user base helps the leader secure the next placement." },
  { name: "AI", qualityEdge: 0.3, reinforcement: 1.6, description: "Two AI assistants compete for teams. Existing integrations and familiar workflows give the more widely adopted assistant an edge with the next team." },
  { name: "Bestsellers", qualityEdge: 0.5, reinforcement: 1.1, description: "Two books compete for readers. A place on the bestseller list puts a book in front of more shoppers." },
  { name: "Airlines", qualityEdge: 0.3, reinforcement: 0.8, description: "Two airlines serve the same route. Frequent-flyer loyalty favors the busier carrier, while travelers can still switch flights." },
  { name: "Restaurants", qualityEdge: 0.3, reinforcement: 0.4, description: "Two restaurants compete for diners. Yesterday’s crowd draws some attention, but tonight’s diners still give both menus a look." },
  { name: "Commodities", qualityEdge: 0.1, reinforcement: 0.1, description: "Two suppliers sell similar grades of wheat. Buyers can switch easily, so the last order gives its seller little advantage on the next one." },
];

export type MarketPreset = (typeof MARKET_PRESETS)[number];
const products = { inputs: [[1], [0]], initialPositions: [0, 0], baseline: 1 };

export function goldOddsAtFullRanking(market: MarketPreset, goldSales: number, tealSales: number) {
  return allocationProbabilities({
    ...products, counts: [goldSales, tealSales], beta: [market.qualityEdge], rho: market.reinforcement,
  })[0];
}

export function shelfHistory(seed: number, { market = MARKET_PRESETS[0], rotate = false, rankFromStart = false } = {}) {
  const random = mulberry32(seed);
  const counts = [0, 0];
  const inputs = { ...products, counts, beta: [market.qualityEdge] };
  const frames = [];
  const horizon = rankFromStart ? SHADOW_SALES : TOTAL_SALES;
  let budget = 0;

  // Each frame describes completed sales and the odds BEFORE the next sale.
  for (let t = 0; t <= horizon; t++) {
    const ramp = rankFromStart ? 1 : Math.max(0, Math.min(1, (t - 150) / 1_050));
    const rotating = rotate && t >= ROTATION_START && t < ROTATION_START + ROTATION_LENGTH;
    const odds = allocationProbabilities({ ...inputs, rho: rotating ? 0 : market.reinforcement * ramp });
    const nextPoint = residualContestability(odds);
    frames.push({ budget, nextPoint, goldOdds: odds[0], goldSales: counts[0] });
    if (t === horizon) break;
    budget += nextPoint;
    counts[drawRecipient(odds, random())]++;
  }
  return frames;
}

export function shadowSales(seed: number, market: MarketPreset) {
  return Array.from({ length: 24 }, (_, i) => {
    const history = shelfHistory(deriveSeed(seed, i), { market, rankFromStart: true });
    return history[SHADOW_SALES].goldSales;
  });
}
