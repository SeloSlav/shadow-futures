// Keep small, positive rival chances visible when a market is nearly locked in.
export function formatChance(chance: number) {
  const share = 100 * chance;
  const nearLimit = (share > 0 && share < 0.1) || (share > 99.9 && share < 100);
  return share.toFixed(nearLimit ? 3 : 1) + "%";
}
