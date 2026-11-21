import { describe, expect, it } from "vitest";
import { MARKET_PRESETS, ROTATION_LENGTH, ROTATION_START, TOTAL_SALES, goldOddsAtFullRanking, shadowSales, shelfHistory } from "../../lib/model/shelf-market";

describe("shelf comparison budget", () => {
  it("credits the probability before each sale, including the last one", () => {
    const history = shelfHistory(42);
    expect(history[0].budget).toBe(0);
    expect(history[150].budget).toBeCloseTo(150 / (1 + Math.exp(0.3)), 9);
    for (let t = 1; t < history.length; t++) {
      expect(history[t].budget - history[t - 1].budget).toBeCloseTo(history[t - 1].nextPoint, 10);
    }
  });

  it.each(MARKET_PRESETS)("uses the $name settings for ranking, equal placement, and the 300-sale rotation", market => {
    const ranked = shelfHistory(42, { market });
    const rotated = shelfHistory(42, { market, rotate: true });
    const openRate = 1 / (1 + Math.exp(market.qualityEdge));
    const end = ROTATION_START + ROTATION_LENGTH;
    expect(rotated.slice(0, ROTATION_START)).toEqual(ranked.slice(0, ROTATION_START));
    expect(rotated[ROTATION_START].budget).toBe(ranked[ROTATION_START].budget);
    expect(rotated[ROTATION_START].nextPoint).toBeCloseTo(openRate, 10);
    expect(rotated[end].budget - rotated[ROTATION_START].budget).toBeCloseTo(ROTATION_LENGTH * openRate, 8);
    const gold = ranked[TOTAL_SALES].goldSales;
    expect(ranked[TOTAL_SALES].goldOdds).toBeCloseTo(goldOddsAtFullRanking(market, gold, TOTAL_SALES - gold), 10);
    expect(rotated[end].nextPoint).toBeLessThan(openRate);
  });

  it("produces distinct budgets and shadow histories for each preset with identical draws", () => {
    const budgets = MARKET_PRESETS.map(market => shelfHistory(42, { market })[TOTAL_SALES].budget);
    expect(new Set(budgets).size).toBe(MARKET_PRESETS.length);
    const batches = MARKET_PRESETS.map(market => shadowSales(42, market).join(","));
    expect(new Set(batches).size).toBe(MARKET_PRESETS.length);
  });
});
