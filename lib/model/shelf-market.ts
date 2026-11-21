import { allocationProbabilities, drawRecipient } from "./allocation";
import { residualContestability } from "./metrics";
import { deriveSeed, mulberry32 } from "./prng";

export const TOTAL_SALES = 4_000;
export const ROTATION_START = 3_000;
export const ROTATION_LENGTH = 300;
export const SHADOW_SALES = 1_000;

const products = { inputs: [[1], [0]], beta: [0.3], initialPositions: [0, 0], baseline: 1 };
export const OPEN_SHELF_RATE = residualContestability(
  allocationProbabilities({ ...products, counts: [0, 0], rho: 0 }),
);

export function shelfHistory(seed: number, { rotate = false, rankFromStart = false } = {}) {
  const random = mulberry32(seed);
  const counts = [0, 0];
  const frames = [];
  const horizon = rankFromStart ? SHADOW_SALES : TOTAL_SALES;
  let budget = 0;

  // Each frame describes completed sales and the odds BEFORE the next sale.
  for (let t = 0; t <= horizon; t++) {
    const ramp = rankFromStart ? 1 : Math.max(0, Math.min(1, (t - 150) / 1_050));
    const rotating = rotate && t >= ROTATION_START && t < ROTATION_START + ROTATION_LENGTH;
    const odds = allocationProbabilities({ ...products, counts, rho: rotating ? 0 : 2.4 * ramp });
    const nextPoint = residualContestability(odds);
    frames.push({ budget, nextPoint, goldOdds: odds[0], goldSales: counts[0] });
    if (t === horizon) break;
    budget += nextPoint;
    counts[drawRecipient(odds, random())]++;
  }
  return frames;
}

export function shadowSales(seed: number) {
  return Array.from({ length: 24 }, (_, i) => {
    const history = shelfHistory(deriveSeed(seed, i), { rankFromStart: true });
    return history[SHADOW_SALES].goldSales;
  });
}
