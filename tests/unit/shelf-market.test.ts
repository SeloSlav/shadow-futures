import { describe, expect, it } from "vitest";
import { OPEN_SHELF_RATE, ROTATION_LENGTH, ROTATION_START, shelfHistory } from "../../lib/model/shelf-market";

describe("shelf comparison budget", () => {
  it("credits the probability before each sale, including the last one", () => {
    const history = shelfHistory(42);
    expect(history[0].budget).toBe(0);
    expect(history[150].budget).toBeCloseTo(150 / (1 + Math.exp(0.3)), 9);
    for (let t = 1; t < history.length; t++) {
      expect(history[t].budget - history[t - 1].budget).toBeCloseTo(history[t - 1].nextPoint, 10);
    }
  });

  it("keeps the shared history and removes shelf advantage for exactly 300 sales", () => {
    const ranked = shelfHistory(42);
    const rotated = shelfHistory(42, { rotate: true });
    const end = ROTATION_START + ROTATION_LENGTH;
    expect(rotated.slice(0, ROTATION_START)).toEqual(ranked.slice(0, ROTATION_START));
    expect(rotated[ROTATION_START].budget).toBe(ranked[ROTATION_START].budget);
    expect(rotated[ROTATION_START].nextPoint).toBe(OPEN_SHELF_RATE);
    expect(rotated[end].budget - rotated[ROTATION_START].budget).toBeCloseTo(ROTATION_LENGTH * OPEN_SHELF_RATE, 8);
    expect(rotated[end].nextPoint).toBeLessThan(OPEN_SHELF_RATE);
  });
});
