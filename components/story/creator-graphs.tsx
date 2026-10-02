"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";

import { mulberry32 } from "@/lib/model/prng";

const CREATOR_COUNT = 24;
const RECOMMENDATIONS = 1_600;
const SAMPLE_EVERY = 40;
const FEEDBACK_STRENGTH = 1.55;
const RESET_INTERVAL = 400;
const COMPARISON_RESET_INTERVAL = 160;
const COMPARISON_RESET_COUNT = Math.floor((RECOMMENDATIONS - 1) / COMPARISON_RESET_INTERVAL);
const WORLD_SEEDS = [31, 97, 160, 174, 214];
const TOP_TEN_COLORS = [
  "var(--rust)",
  "var(--blue)",
  "var(--shadow)",
  "#6f7f54",
  "#936d8b",
  "#b47d32",
  "#4e8582",
  "#786b57",
  "#70787a",
  "#465f70",
] as const;
const MODELED_AUDIENCE_FIT: readonly number[] = [
  0.92, 1.08, 0.98, 1.15, 0.88, 1.04, 0.95, 1.12, 0.86, 1.01, 1.18, 0.9,
  1.06, 0.96, 1.1, 0.84, 1.03, 0.94, 1.14, 0.89, 1.07, 0.97, 1.16, 1,
];

type CreatorWorld = {
  comparison: number[];
  finalShare: number;
  series: number[][];
  winner: number;
};

function sharePercent(share: number) {
  return (share * 100).toFixed(1);
}

function simulateCreatorWorld(
  seed: number,
  clearScoresEvery = 0,
  comparisonFloor = 0,
): CreatorWorld {
  const random = mulberry32(seed);
  const scoreCounts = Array.from({ length: CREATOR_COUNT }, () => 0);
  const exposureCounts = Array.from({ length: CREATOR_COUNT }, () => 0);
  const series = Array.from({ length: CREATOR_COUNT }, () => [0]);
  const comparison = [0];
  let comparisonTotal = 0;

  for (
    let recommendation = 0;
    recommendation < RECOMMENDATIONS;
    recommendation += 1
  ) {
    if (
      clearScoresEvery > 0 &&
      recommendation > 0 &&
      recommendation % clearScoresEvery === 0
    ) {
      scoreCounts.fill(0);
    }

    const weights = scoreCounts.map(
      (count, index) =>
        MODELED_AUDIENCE_FIT[index] * (2 + count) ** FEEDBACK_STRENGTH,
    );
    const weightTotal = weights.reduce((sum, weight) => sum + weight, 0);
    let probabilities = weights.map((weight) => weight / weightTotal);
    const residualContestability = 1 - Math.max(...probabilities);
    if (comparisonFloor > residualContestability) {
      const uniformResidualContestability = 1 - 1 / CREATOR_COUNT;
      const uniformMix = Math.min(
        1,
        (comparisonFloor - residualContestability) /
          (uniformResidualContestability - residualContestability),
      );
      probabilities = probabilities.map(
        (probability) =>
          (1 - uniformMix) * probability + uniformMix / CREATOR_COUNT,
      );
    }
    comparisonTotal += 1 - Math.max(...probabilities);

    let draw = random();
    let recipient = 0;
    while (
      recipient < CREATOR_COUNT - 1 &&
      (draw -= probabilities[recipient]) > 0
    ) {
      recipient += 1;
    }
    scoreCounts[recipient] += 1;
    exposureCounts[recipient] += 1;

    if ((recommendation + 1) % SAMPLE_EVERY === 0) {
      const total = exposureCounts.reduce((sum, count) => sum + count, 0) || 1;
      exposureCounts.forEach((count, index) => series[index].push(count / total));
      comparison.push(comparisonTotal);
    }
  }

  const total = exposureCounts.reduce((sum, count) => sum + count, 0) || 1;
  const winner = exposureCounts.indexOf(Math.max(...exposureCounts));
  return {
    comparison,
    finalShare: exposureCounts[winner] / total,
    series,
    winner,
  };
}

function useGraphAnimation(duration = 2_800) {
  const prefersReducedMotion = useReducedMotion();
  const [progress, setProgress] = useState(0);
  const [runId, setRunId] = useState(0);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (runId === 0) return;
    let frame = 0;
    if (prefersReducedMotion) {
      frame = window.requestAnimationFrame(() => {
        setProgress(1);
        setRunning(false);
      });
      return () => window.cancelAnimationFrame(frame);
    }

    let startedAt: number | undefined;
    const tick = (now: number) => {
      startedAt ??= now;
      const next = Math.min(1, (now - startedAt) / duration);
      setProgress(next);
      if (next < 1) {
        frame = window.requestAnimationFrame(tick);
      } else {
        setRunning(false);
      }
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [duration, prefersReducedMotion, runId]);

  const play = useCallback(() => {
    setProgress(0);
    setRunning(true);
    setRunId((current) => current + 1);
  }, []);

  return {
    play,
    progress,
    running,
    state: progress >= 1 ? "complete" : running ? "running" : "idle",
  };
}

function linePath(
  values: number[],
  width: number,
  height: number,
  maxValue: number,
  inset = { top: 24, right: 28, bottom: 52, left: 54 },
) {
  const innerWidth = width - inset.left - inset.right;
  const innerHeight = height - inset.top - inset.bottom;
  return values
    .map((value, index) => {
      const x = inset.left + (index / Math.max(1, values.length - 1)) * innerWidth;
      const y =
        inset.top + (1 - value / Math.max(Number.EPSILON, maxValue)) * innerHeight;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export function BreakoutGraph() {
  const [seedIndex, setSeedIndex] = useState(0);
  const [hasPlayed, setHasPlayed] = useState(false);
  const animation = useGraphAnimation();
  const world = useMemo(
    () => simulateCreatorWorld(WORLD_SEEDS[seedIndex]),
    [seedIndex],
  );
  const resetWorld = useMemo(
    () => simulateCreatorWorld(WORLD_SEEDS[seedIndex], RESET_INTERVAL),
    [seedIndex],
  );
  const topTen = useMemo(
    () =>
      world.series
        .map((values, creator) => ({
          creator,
          share: values.at(-1) ?? 0,
        }))
        .sort((left, right) => right.share - left.share)
        .slice(0, 10),
    [world],
  );
  const runnerUpCounterfactuals = topTen.slice(1, 3).map((entry, index) => ({
    ...entry,
    color: TOP_TEN_COLORS[index + 1],
    resetSeries: resetWorld.series[entry.creator],
    resetShare: resetWorld.series[entry.creator].at(-1) ?? 0,
  }));
  const sampleIndex = Math.min(
    world.series[0].length - 1,
    Math.floor(animation.progress * (world.series[0].length - 1)),
  );
  const leaderShare = world.series[world.winner][sampleIndex] ?? 0;
  const chartWidth = 860;
  const chartHeight = 420;
  const x = 54 + (sampleIndex / Math.max(1, world.series[0].length - 1)) * 778;
  const y = 24 + (1 - leaderShare) * 344;
  const miniChartWidth = 300;
  const miniChartHeight = 132;
  const miniChartInset = { top: 12, right: 12, bottom: 18, left: 12 };
  const interventionSamples = Array.from(
    { length: RECOMMENDATIONS / RESET_INTERVAL - 1 },
    (_, index) => ((index + 1) * RESET_INTERVAL) / SAMPLE_EVERY,
  );

  const play = () => {
    setSeedIndex((current) =>
      hasPlayed ? (current + 1) % WORLD_SEEDS.length : current,
    );
    setHasPlayed(true);
    animation.play();
  };

  return (
    <div
      className="creator-graph"
      data-animation-state={animation.state}
      data-testid="breakout-graph"
    >
      <div className="creator-graph__head">
        <div>
          <div className="panel__meta">One platform chart in motion</div>
          <strong>
            Fixed audience appeal and past exposure both shape the next recommendation.
          </strong>
        </div>
        <button className="button button--small" type="button" onClick={play}>
          {animation.running
            ? "Recommendations are running…"
            : hasPlayed
              ? "Run new recommendations"
              : "Run the recommendations"}
        </button>
      </div>

      <div className="creator-graph__plot">
        <svg
          className="creator-line-chart"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          role="img"
          aria-label="Cumulative recommendation shares for the ten highest-ranked creators out of 24. All start together with fixed modeled appeal. Each past recommendation increases future recommendation odds."
        >
          <title>One creator’s early exposure becomes a runaway platform lead</title>
          <desc>
            Twenty-four creators have different levels of modeled audience response. Small early
            differences in exposure are amplified until one creator receives much more of the
            platform’s attention. The ten leading observed paths are shown.
          </desc>
          <text x="54" y="15" className="creator-chart-label comparison-axis-title">
            share of recommendations received so far
          </text>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const tickY = 24 + (1 - tick) * 344;
            return (
              <g key={tick}>
                <line
                  x1="54"
                  x2="832"
                  y1={tickY}
                  y2={tickY}
                  className="creator-chart-grid"
                />
                <text x="42" y={tickY + 5} textAnchor="end" className="creator-chart-label">
                  {Math.round(tick * 100)}%
                </text>
              </g>
            );
          })}
          {topTen.map((entry, rank) => (
            <motion.path
              key={entry.creator}
              d={linePath(world.series[entry.creator], chartWidth, chartHeight, 1)}
              fill="none"
              stroke={TOP_TEN_COLORS[rank]}
              strokeWidth={rank === 0 ? 5 : rank < 3 ? 3 : 1.7}
              strokeLinecap="round"
              initial={false}
              animate={{ pathLength: animation.progress }}
              transition={{ duration: 0.08, ease: "linear" }}
              opacity={rank < 3 ? 1 : 0.78}
            />
          ))}
          {animation.progress > 0 ? (
            <motion.circle
              cx={x}
              cy={y}
              r="7"
              fill="var(--rust)"
              initial={false}
              animate={{ cx: x, cy: y }}
              transition={{ duration: 0.08, ease: "linear" }}
            />
          ) : null}
          <text x="54" y="402" className="creator-chart-label">
            first recommendation
          </text>
          <text x="832" y="402" textAnchor="end" className="creator-chart-label">
            recommendation 1,600
          </text>
        </svg>

        <div className="creator-chart-key" role="list" aria-label="Top ten creator paths">
          {topTen.map((entry, rank) => {
            const counterfactual = runnerUpCounterfactuals.find(
              (candidate) => candidate.creator === entry.creator,
            );
            return (
              <div className="creator-chart-key__item" key={entry.creator} role="listitem">
                <span
                  className="creator-chart-key__swatch"
                  style={{ backgroundColor: TOP_TEN_COLORS[rank] }}
                  aria-hidden="true"
                />
                <strong>#{rank + 1}</strong>
                <span>Creator {entry.creator + 1}</span>
                <span>{sharePercent(entry.share)}%</span>
                <small>
                  Appeal {MODELED_AUDIENCE_FIT[entry.creator].toFixed(2)}×
                  {counterfactual ? " · compared below" : ""}
                </small>
              </div>
            );
          })}
        </div>

        <section className="creator-shadow-comparison" aria-labelledby="shadow-paths-title">
          <div className="creator-shadow-comparison__intro">
            <div>
              <span className="panel__meta">A policy experiment with two of the same creators</span>
              <h3 id="shadow-paths-title">What changes when the platform reopens discovery?</h3>
            </div>
            <p>
              Same creator, same modeled audience appeal and same random sequence. Only the
              accumulated ranking score resets after recommendations 400, 800 and 1,200.
            </p>
          </div>

          <div className="creator-shadow-comparison__grid">
            {runnerUpCounterfactuals.map((entry, index) => {
              const comparisonMax = Math.ceil(100 * Math.max(
                0.12,
                Math.max(...world.series[entry.creator], ...entry.resetSeries) * 1.08,
              )) / 100;
              const delta = (entry.resetShare - entry.share) * 100;
              return (
                <article className="creator-shadow-card" key={entry.creator}>
                  <header>
                    <div>
                      <span>Original #{index + 2}</span>
                      <strong>Creator {entry.creator + 1}</strong>
                    </div>
                    <span className="creator-shadow-card__delta">
                      {delta >= 0 ? "+" : ""}
                      {delta.toFixed(1)} points
                    </span>
                  </header>

                  <div className="creator-shadow-card__panels">
                    <div className="creator-shadow-card__panel">
                      <div className="creator-shadow-card__panel-head">
                        <span>Observed ranking</span>
                        <strong>{sharePercent(entry.share)}%</strong>
                      </div>
                      <svg
                        viewBox={`0 0 ${miniChartWidth} ${miniChartHeight}`}
                        role="img"
                        aria-label={`Creator ${entry.creator + 1} receives ${sharePercent(entry.share)} percent of recommendations under the observed ranking.`}
                      >
                        <line
                          x1={miniChartInset.left}
                          x2={miniChartWidth - miniChartInset.right}
                          y1={miniChartHeight - miniChartInset.bottom}
                          y2={miniChartHeight - miniChartInset.bottom}
                          className="creator-shadow-card__axis"
                        />
                        <motion.path
                          d={linePath(
                            world.series[entry.creator],
                            miniChartWidth,
                            miniChartHeight,
                            comparisonMax,
                            miniChartInset,
                          )}
                          fill="none"
                          stroke={entry.color}
                          strokeWidth="5"
                          strokeLinecap="round"
                          initial={false}
                          animate={{ pathLength: animation.progress }}
                          transition={{ duration: 0.08, ease: "linear" }}
                        />
                      </svg>
                      <small className="creator-mini-scale">
                        0–{Math.ceil(comparisonMax * 100)}% share · 0–1,600 recommendations
                      </small>
                    </div>

                    <div className="creator-shadow-card__panel creator-shadow-card__panel--reset">
                      <div className="creator-shadow-card__panel-head">
                        <span>Ranking reset</span>
                        <strong>{sharePercent(entry.resetShare)}%</strong>
                      </div>
                      <svg
                        viewBox={`0 0 ${miniChartWidth} ${miniChartHeight}`}
                        role="img"
                        aria-label={`Creator ${entry.creator + 1} receives ${sharePercent(entry.resetShare)} percent of recommendations when ranking scores reset every 400 recommendations.`}
                      >
                        <line
                          x1={miniChartInset.left}
                          x2={miniChartWidth - miniChartInset.right}
                          y1={miniChartHeight - miniChartInset.bottom}
                          y2={miniChartHeight - miniChartInset.bottom}
                          className="creator-shadow-card__axis"
                        />
                        {interventionSamples.map((interventionSample) => {
                          const interventionX =
                            miniChartInset.left +
                            (interventionSample /
                              Math.max(1, entry.resetSeries.length - 1)) *
                              (miniChartWidth -
                                miniChartInset.left -
                                miniChartInset.right);
                          return (
                            <line
                              key={interventionSample}
                              x1={interventionX}
                              x2={interventionX}
                              y1={miniChartInset.top}
                              y2={miniChartHeight - miniChartInset.bottom}
                              className="creator-shadow-card__reset-marker"
                            />
                          );
                        })}
                        <motion.path
                          d={linePath(
                            entry.resetSeries,
                            miniChartWidth,
                            miniChartHeight,
                            comparisonMax,
                            miniChartInset,
                          )}
                          fill="none"
                          stroke={entry.color}
                          strokeWidth="5"
                          strokeLinecap="round"
                          initial={false}
                          animate={{ pathLength: animation.progress }}
                          transition={{ duration: 0.08, ease: "linear" }}
                        />
                      </svg>
                      <small className="creator-mini-scale">
                        0–{Math.ceil(comparisonMax * 100)}% share · dashed lines mark resets
                      </small>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>

          <p className="creator-shadow-comparison__note">
            The intervention clears accumulated visibility scores, not prior views or modeled
            audience appeal. Each pair shares a vertical scale; scales differ between creators.
            These are policy counterfactuals. A shadow future keeps the rule unchanged and
            changes only the random draws: use “Run new recommendations” to see one.
          </p>
        </section>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            Creator {world.winner + 1} received{" "}
            <strong>{sharePercent(world.finalShare)}% of all recommendations</strong>.
            Without intervention, #2 and #3 received{" "}
            {sharePercent(runnerUpCounterfactuals[0].share)}% and{" "}
            {sharePercent(runnerUpCounterfactuals[1].share)}%. With ranking resets, their
            policy reruns reach {sharePercent(runnerUpCounterfactuals[0].resetShare)}% and{" "}
            {sharePercent(runnerUpCounterfactuals[1].resetShare)}%.
          </>
        ) : (
          <>
            Run the recommendations, then rerun them. Inputs and rules stay the same;
            only the random draws change. The model tracks exposure, not earnings or talent growth.
          </>
        )}
      </p>
    </div>
  );
}

export function ExperimentMonopolyGraph() {
  const animation = useGraphAnimation(2_700);
  const keptScores = useMemo(() => simulateCreatorWorld(31), []);
  const clearedScores = useMemo(() => simulateCreatorWorld(31, COMPARISON_RESET_INTERVAL), []);
  const keptScoreTotal = keptScores.comparison.at(-1) ?? 0;
  const clearedScoreTotal = clearedScores.comparison.at(-1) ?? 0;
  const chartWidth = 860;
  const chartHeight = 390;

  return (
    <div
      className="creator-graph"
      data-animation-state={animation.state}
      data-testid="experiment-monopoly-graph"
    >
      <div className="creator-graph__head">
        <div>
          <div className="panel__meta">Social media makes 1,600 recommendations</div>
          <strong>Transactions keep arriving. Does comparison keep growing?</strong>
        </div>
        <button className="button button--small" type="button" onClick={animation.play}>
          {animation.running ? "Comparing…" : "Compare both rules"}
        </button>
      </div>

      <div className="experiment-metric">
        <div>
          <span className="panel__meta">What the vertical axis measures</span>
          <strong>
            The comparison budget accumulated so far, rather than the number of transactions.
          </strong>
        </div>
        <p>
          A 70% chance for the current favorite adds 0.30 to the budget: everyone else has
          a combined 30% chance. A 99.9% chance adds only 0.001. Both rules make 1,600
          recommendations, but they accumulate different amounts of comparison.
        </p>
      </div>

      <div className="creator-graph__plot">
        <svg
          className="creator-line-chart"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          role="img"
          aria-label={`Cumulative comparison budget versus recommendation count. Continuous reinforcement accumulates ${Math.round(keptScoreTotal)} budget units. Clearing ranking scores ${COMPARISON_RESET_COUNT} times accumulates ${Math.round(clearedScoreTotal)}. Both allocate 1,600 recommendations.`}
        >
          <title>Activity and comparison need not grow together</title>
          <desc>
            A dashed line counts recommendations. The rust line adds the chance outside
            the favorite under continuous reinforcement. The blue line clears accumulated
            ranking scores nine times, retaining the same appeal scores and random draws.
          </desc>
          <text x="54" y="15" className="creator-chart-label comparison-axis-title">
            cumulative comparison budget
          </text>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const tickY = 24 + (1 - tick) * 314;
            return (
              <g key={tick}>
                <line
                  x1="54"
                  x2="832"
                  y1={tickY}
                  y2={tickY}
                  className="creator-chart-grid"
                />
                <text
                  x="42"
                  y={tickY + 5}
                  textAnchor="end"
                  className="creator-chart-label comparison-y-tick"
                >
                  {(tick * RECOMMENDATIONS).toLocaleString("en-US")}
                </text>
              </g>
            );
          })}
          <motion.path
            d={linePath(keptScores.comparison, chartWidth, chartHeight, RECOMMENDATIONS)}
            fill="none"
            stroke="var(--rust)"
            strokeWidth="5"
            strokeLinecap="round"
            initial={false}
            animate={{ pathLength: animation.progress }}
            transition={{ duration: 0.08, ease: "linear" }}
          />
          <motion.path
            d={linePath(clearedScores.comparison, chartWidth, chartHeight, RECOMMENDATIONS)}
            fill="none"
            stroke="var(--blue)"
            strokeWidth="5"
            strokeLinecap="round"
            initial={false}
            animate={{ pathLength: animation.progress }}
            transition={{ duration: 0.08, ease: "linear" }}
          />
          <path
            d={linePath([0, RECOMMENDATIONS], chartWidth, chartHeight, RECOMMENDATIONS)}
            fill="none"
            stroke="currentColor"
            strokeOpacity="0.5"
            strokeWidth="2"
            strokeDasharray="7 8"
          />
          <text x="54" y="370" className="creator-chart-label">
            0 recommendations
          </text>
          <text x="832" y="370" textAnchor="end" className="creator-chart-label">
            recommendation 1,600
          </text>
        </svg>
      </div>

      <div className="experiment-labels">
        <div>
          <span className="experiment-labels__line experiment-labels__line--rust" />
          <strong>Keep boosting the early leader</strong>
          <span>{Math.round(keptScoreTotal)} units of comparison budget</span>
        </div>
        <div>
          <span className="experiment-labels__line experiment-labels__line--blue" />
          <strong>Clear ranking scores {COMPARISON_RESET_COUNT} times</strong>
          <span>{Math.round(clearedScoreTotal)} units of comparison budget; appeal stays unchanged</span>
        </div>
      </div>
      <p className="chart-reading-note">
        The dashed line counts transactions, a different quantity shown for reference.
        Clearing scores creates ten ranking periods, not ten independent markets.
        A finite animation does not establish what happens over an infinite future.
      </p>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            Across the same 1,600 recommendations, clearing ranking scores {COMPARISON_RESET_COUNT}
            {" "}times accumulated <strong>{Math.round(clearedScoreTotal)} comparison-budget units</strong>,
            versus <strong>{Math.round(keptScoreTotal)}</strong> under continuous reinforcement.
            This run preserved{" "}
            <strong>
              {(clearedScoreTotal / keptScoreTotal).toFixed(1)} times as much comparison
            </strong>
            . A bigger budget leaves room for more evidence; it does not guarantee identification.
          </>
        ) : (
          <>
            If social media keeps boosting the early leader, everyone else gets fewer real
            chances to be seen.
          </>
        )}
      </p>
    </div>
  );
}

export function LorenzHistoryGraph() {
  const animation = useGraphAnimation(2_600);
  const baselineWorld = useMemo(() => simulateCreatorWorld(31), []);
  const interventionWorld = useMemo(
    () => simulateCreatorWorld(31, 0, 0.5),
    [],
  );
  const baselineShares = useMemo(
    () =>
      baselineWorld.series
        .map((values) => values.at(-1) ?? 0)
        .sort((left, right) => left - right),
    [baselineWorld],
  );
  const interventionShares = useMemo(
    () =>
      interventionWorld.series
        .map((values) => values.at(-1) ?? 0)
        .sort((left, right) => left - right),
    [interventionWorld],
  );
  const baselineCumulativeShares = useMemo(
    () => [
      0,
      ...baselineShares.map((_, index) =>
        baselineShares
          .slice(0, index + 1)
          .reduce((sum, share) => sum + share, 0),
      ),
    ],
    [baselineShares],
  );
  const interventionCumulativeShares = useMemo(
    () => [
      0,
      ...interventionShares.map((_, index) =>
        interventionShares
          .slice(0, index + 1)
          .reduce((sum, share) => sum + share, 0),
      ),
    ],
    [interventionShares],
  );
  const bottomThreeQuartersIndex = Math.floor(CREATOR_COUNT * 0.75);
  const baselineBottomShare =
    baselineCumulativeShares[bottomThreeQuartersIndex] ?? 0;
  const interventionBottomShare =
    interventionCumulativeShares[bottomThreeQuartersIndex] ?? 0;
  const baselineTopThreeShare = baselineShares
    .slice(-3)
    .reduce((sum, share) => sum + share, 0);
  const interventionTopThreeShare = interventionShares
    .slice(-3)
    .reduce((sum, share) => sum + share, 0);
  const baselineBudget = baselineWorld.comparison.at(-1) ?? 0;
  const interventionBudget = interventionWorld.comparison.at(-1) ?? 0;
  const baselineActiveCreators = baselineShares.filter(
    (share) => share >= 0.02,
  ).length;
  const interventionActiveCreators = interventionShares.filter(
    (share) => share >= 0.02,
  ).length;
  const chartWidth = 860;
  const chartHeight = 420;
  const annotationX = 54 + 0.75 * 778;
  const baselineAnnotationY = 24 + (1 - baselineBottomShare) * 344;
  const interventionAnnotationY =
    24 + (1 - interventionBottomShare) * 344;

  return (
    <div
      className="creator-graph"
      data-animation-state={animation.state}
      data-testid="lorenz-history-graph"
    >
      <div className="creator-graph__head">
        <div>
          <div className="panel__meta">A comparison-preserving intervention</div>
          <strong>In this run, preserving comparison spreads recommendations more widely.</strong>
        </div>
        <button className="button button--small" type="button" onClick={animation.play}>
          {animation.running
            ? "Protecting the comparison budget…"
            : animation.state === "complete"
              ? "Replay the intervention"
              : "Protect the comparison budget"}
        </button>
      </div>

      <div className="creator-graph__plot">
        <svg
          className="creator-line-chart"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          role="img"
          aria-label={`Two Lorenz curves of simulated recommendation shares. Under reinforcement, the top three receive ${sharePercent(baselineTopThreeShare)} percent of recommendations. With the comparison floor, they receive ${sharePercent(interventionTopThreeShare)} percent. Each curve sorts creators separately from least to most exposure.`}
        >
          <title>Recommendation concentration under two allocation rules</title>
          <desc>
            A rust curve shows the reinforcing baseline. A blue curve appears when a rule
            preserves at least half of each next-recommendation chance for creators other
            than the current favorite.
          </desc>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
            const tickX = 54 + tick * 778;
            const tickY = 24 + (1 - tick) * 344;
            return (
              <g key={tick}>
                <line
                  x1="54"
                  x2="832"
                  y1={tickY}
                  y2={tickY}
                  className="creator-chart-grid"
                />
                <text
                  x="42"
                  y={tickY + 5}
                  textAnchor="end"
                  className="creator-chart-label lorenz-y-tick"
                  data-extreme={tick === 0 || tick === 1}
                >
                  {Math.round(tick * 100)}%
                </text>
                <text
                  x={tickX}
                  y="402"
                  textAnchor="middle"
                  className="creator-chart-label lorenz-x-tick"
                  data-extreme={tick === 0 || tick === 1}
                >
                  {Math.round(tick * 100)}%
                </text>
              </g>
            );
          })}
          <path
            d={linePath([0, 1], chartWidth, chartHeight, 1)}
            fill="none"
            stroke="var(--line-strong)"
            strokeWidth="2"
            strokeDasharray="7 8"
          />
          <text x="610" y="112" className="creator-chart-label">
            equal distribution
          </text>
          <path
            d={linePath(baselineCumulativeShares, chartWidth, chartHeight, 1)}
            fill="none"
            stroke="var(--rust)"
            strokeWidth="6"
            strokeLinecap="round"
          />
          <motion.path
            d={linePath(interventionCumulativeShares, chartWidth, chartHeight, 1)}
            fill="none"
            stroke="var(--blue)"
            strokeWidth="6"
            strokeLinecap="round"
            initial={false}
            animate={{ pathLength: animation.progress }}
            transition={{ duration: 0.08, ease: "linear" }}
          />
          <circle
            cx={annotationX}
            cy={baselineAnnotationY}
            r="6"
            fill="var(--rust)"
          />
          <text
            x={annotationX - 12}
            y={Math.max(338, baselineAnnotationY - 18)}
            textAnchor="end"
            className="creator-chart-label"
          >
            baseline: bottom 75% receive {sharePercent(baselineBottomShare)}%
          </text>
          {animation.progress > 0.92 ? (
            <>
              <circle
                cx={annotationX}
                cy={interventionAnnotationY}
                r="6"
                fill="var(--blue)"
              />
              <text
                x={annotationX - 12}
                y={interventionAnnotationY - 18}
                textAnchor="end"
                className="creator-chart-label"
              >
                comparison rule: bottom 75% receive{" "}
                {sharePercent(interventionBottomShare)}%
              </text>
            </>
          ) : null}
          <text
            x="443"
            y="418"
            textAnchor="middle"
            className="creator-chart-label lorenz-axis-title"
          >
            creators, least to most exposure
          </text>
          <text
            x="-196"
            y="15"
            transform="rotate(-90)"
            textAnchor="middle"
            className="creator-chart-label lorenz-axis-title"
          >
            cumulative share of recommendations
          </text>
        </svg>
      </div>

      <div
        className="lorenz-comparison"
        aria-label="Comparison budget and recommendation concentration under two rules"
      >
        <div className="lorenz-comparison__legend" aria-label="Chart key">
          <div>
            <span
              className="lorenz-comparison__swatch lorenz-comparison__swatch--rust"
              aria-hidden="true"
            />
            <span>
              <strong>Reinforcing baseline</strong>
              The ranking can keep spending attention on its current favorite.
            </span>
          </div>
          <div>
            <span
              className="lorenz-comparison__swatch lorenz-comparison__swatch--blue"
              aria-hidden="true"
            />
            <span>
              <strong>Comparison-preserving rule</strong>
              At least half of the next-recommendation chance stays outside the favorite.
            </span>
          </div>
        </div>

        <div className="lorenz-comparison__metrics" aria-live="polite">
          <article>
            <span>Comparison budget</span>
            <strong>
              {Math.round(baselineBudget)}
              <span aria-hidden="true">→</span>
              {animation.state === "complete" ? Math.round(interventionBudget) : "?"}
            </strong>
            <small>Cumulative chance for someone else to receive the next recommendation</small>
          </article>
          <article>
            <span>Top three recommendation share</span>
            <strong>
              {sharePercent(baselineTopThreeShare)}%
              <span aria-hidden="true">→</span>
              {animation.state === "complete"
                ? `${sharePercent(interventionTopThreeShare)}%`
                : "?"}
            </strong>
            <small>Exposure share, not an estimate of earnings or contribution</small>
          </article>
          <article>
            <span>Creators reaching the 2% cutoff</span>
            <strong>
              {baselineActiveCreators}
              <span aria-hidden="true">→</span>
              {animation.state === "complete" ? interventionActiveCreators : "?"}
            </strong>
            <small>An illustrative cutoff, not a test of commercial viability</small>
          </article>
        </div>

        <section className="lorenz-interventions" aria-labelledby="lorenz-interventions-title">
          <div className="lorenz-interventions__intro">
            <span className="panel__meta">How shadow futures guide policy</span>
            <h3 id="lorenz-interventions-title">
              Treat the comparison budget as something institutions should protect.
            </h3>
          </div>
          <div className="lorenz-interventions__grid">
            <article>
              <span className="panel__meta">Modeled in the blue curve</span>
              <h4>Reserve discovery for alternatives</h4>
              <p>
                When the favorite’s lead starts closing the contest, redirect enough exposure
                to keep challengers genuinely testable.
              </p>
            </article>
            <article>
              <span className="panel__meta">Real-market counterpart</span>
              <h4>Make audiences and data portable</h4>
              <p>
                Interoperability and portability can make other routes to reputation,
                customers and distribution possible.
              </p>
            </article>
            <article>
              <span className="panel__meta">Real-market counterpart</span>
              <h4>Keep trials independent</h4>
              <p>
                Separate rankings, procurement trials and public options can create useful
                replications when their outcomes are genuinely independent and informative.
              </p>
            </article>
          </div>
        </section>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            In this stylized run, the intervention preserved{" "}
            <strong>
              {(interventionBudget / baselineBudget).toFixed(1)} times as much comparison
            </strong>
            , expanded the number receiving at least 2% of recommendations from{" "}
            <strong>
              {baselineActiveCreators} to {interventionActiveCreators}
            </strong>
            , and reduced the top three’s recommendation share from{" "}
            <strong>
              {sharePercent(baselineTopThreeShare)}% to{" "}
              {sharePercent(interventionTopThreeShare)}%
            </strong>
            .
          </>
        ) : (
          <>
            The rust curve is the reinforcing baseline. Apply the blue rule to keep alternatives
            in the experiment and compare the resulting competition and concentration.
          </>
        )}
      </p>
      <p className="chart-reading-note">
        Read 75% on the horizontal axis as the 18 least-exposed creators. The vertical
        value is their combined share of recommendations. Each curve sorts creators anew,
        so those 18 need not be the same people. These are two completed model runs;
        the animation reveals the comparison, not a transition through intermediate policies.
      </p>
    </div>
  );
}
