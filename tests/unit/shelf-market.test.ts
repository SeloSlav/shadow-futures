import { describe, expect, it } from "vitest";
import { MARKET_PRESETS, ROTATION_LENGTH, ROTATION_START, SHADOW_RUNS, SHADOW_STEPS, TOTAL_SALES, goldOddsAtFullRanking, shadowFutures, shelfHistory } from "../../lib/model/shelf-market";

describe("shelf comparison budget", () => {
  it("credits the probability before each sale, including the last one", () => {
    const history = shelfHistory(42);
    expect(history[0].budget).toBe(0);
    expect(history[150].budget).toBeCloseTo(150 / (1 + Math.exp(0.3)), 9);
    for (let t = 1; t < history.length; t++) {
      expect(history[t].budget - history[t - 1].budget).toBeCloseTo(history[t - 1].nextPoint, 10);
    }
  });

  it.each(MARKET_PRESETS.filter(market => !market.singleFirm))("uses the $name settings for ranking, equal placement, and the 300-sale rotation", market => {
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

  it.each([{}, { rotate: true }, { equalShelf: true }])("gives every sale to one firm in Singularity under rule %j", shelf => {
    const market = MARKET_PRESETS.find(preset => preset.name === "Singularity")!;
    const rule = { market, ...shelf };
    const history = shelfHistory(42, rule);
    expect(history).toHaveLength(TOTAL_SALES + 1);
    expect(history.every((frame, sales) => frame.goldSales === sales && frame.goldOdds === 1 && frame.nextPoint === 0 && frame.budget === 0)).toBe(true);
    const futures = shadowFutures(13, ROTATION_START, ROTATION_START, rule);
    expect(futures.litProbability).toBe(0);
    expect(futures.paths.flat().every(sale => sale.winner === 0 && !sale.rivalWon)).toBe(true);
  });

  it("produces distinct budgets for each preset with identical draws", () => {
    const budgets = MARKET_PRESETS.map(market => shelfHistory(42, { market })[TOTAL_SALES].budget);
    expect(new Set(budgets).size).toBe(MARKET_PRESETS.length);
  });

  it.each([false, true])("ranks AI below Big Tech and commodities highest at 4,000 sales (rotation %s)", rotate => {
    const ranked = MARKET_PRESETS.map(market => ({
      name: market.name,
      budget: shelfHistory(42, { market, rotate })[TOTAL_SALES].budget,
    })).sort((a, b) => a.budget - b.budget);
    expect(ranked.map(run => run.name)).toEqual(["Singularity", "AI", "Big Tech", "Bestsellers", "Airlines", "Restaurants", "Commodities"]);
  });

  it("uses each market’s current counts and updates its favorite before every branch sale", () => {
    for (const market of MARKET_PRESETS) {
      const { goldSales } = shelfHistory(42, { market })[TOTAL_SALES];
      const { paths } = shadowFutures(42, TOTAL_SALES, goldSales, { market });
      expect(paths).toHaveLength(SHADOW_RUNS);
      for (const path of paths) {
        let gold = goldSales;
        let teal = TOTAL_SALES - gold;
        expect(path).toHaveLength(SHADOW_STEPS);
        for (const sale of path) {
          const favorite = goldOddsAtFullRanking(market, gold, teal) >= 0.5 ? 0 : 1;
          expect(sale.rivalWon).toBe(sale.winner !== favorite);
          if (sale.winner === 0) gold++;
          else teal++;
        }
      }
    }
  });

  it("uses equal-placement odds through a rotation and updates them when ranking resumes", () => {
    const market = MARKET_PRESETS[0];
    const fullRule = { market, rotate: true };
    const lastRotatedSale = ROTATION_START + ROTATION_LENGTH - 1;
    const { goldSales, goldOdds } = shelfHistory(42, fullRule)[lastRotatedSale];
    const equal = shadowFutures(42, lastRotatedSale, goldSales, { market, equalShelf: true });
    expect(equal.litProbability).toBeCloseTo(1 - goldOdds ** SHADOW_STEPS, 10);
    const during = shadowFutures(42, ROTATION_START, 2500, fullRule);
    expect(during).toEqual(equal);
    const crossing = shadowFutures(42, lastRotatedSale, goldSales, fullRule);
    let allFavorite = goldOdds;
    for (let step = 1; step < SHADOW_STEPS; step++) {
      allFavorite *= goldOddsAtFullRanking(market, goldSales + step, lastRotatedSale - goldSales);
    }
    expect(crossing.litProbability).toBeCloseTo(1 - allFavorite, 10);
  });
});
