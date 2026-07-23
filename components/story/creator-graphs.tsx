"use client";

import { motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useMemo, useState } from "react";

import { mulberry32 } from "@/lib/model/prng";

const CREATOR_COUNT = 24;
const RECOMMENDATIONS = 1_600;
const SAMPLE_EVERY = 40;
const FEEDBACK_STRENGTH = 1.55;
const RESET_INTERVAL = 400;
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

function simulateCreatorWorld(seed: number, clearScoresEvery = 0): CreatorWorld {
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
    const probabilities = weights.map((weight) => weight / weightTotal);
    comparisonTotal += 1 - Math.max(...probabilities);

    let draw = random() * weightTotal;
    let recipient = 0;
    while (recipient < CREATOR_COUNT - 1 && (draw -= weights[recipient]) > 0) {
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
            Creators differ. The feed decides whose promise gets enough chances to grow.
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
          aria-label="Twenty-four creators with different modeled audience response compete for 1,600 recommendations. The ten leading observed paths are shown."
        >
          <title>One creator’s early exposure becomes a runaway platform lead</title>
          <desc>
            Twenty-four creators have different levels of modeled audience response. Small early
            differences in exposure are amplified until one creator receives much more of the
            platform’s attention. The ten leading observed paths are shown.
          </desc>
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
                <span>{Math.round(entry.share * 100)}%</span>
                {counterfactual ? <small>compared below</small> : null}
              </div>
            );
          })}
        </div>

        <section className="creator-shadow-comparison" aria-labelledby="shadow-paths-title">
          <div className="creator-shadow-comparison__intro">
            <div>
              <span className="panel__meta">Two shadow paths, separated from the crowd</span>
              <h3 id="shadow-paths-title">What changes when the platform reopens discovery?</h3>
            </div>
            <p>
              Same creator, same modeled audience response and same random sequence. Only the
              accumulated ranking score resets after recommendations 400, 800 and 1,200.
            </p>
          </div>

          <div className="creator-shadow-comparison__grid">
            {runnerUpCounterfactuals.map((entry, index) => {
              const comparisonMax = Math.max(
                0.12,
                Math.max(...world.series[entry.creator], ...entry.resetSeries) * 1.08,
              );
              const delta =
                Math.round(entry.resetShare * 100) - Math.round(entry.share * 100);
              return (
                <article className="creator-shadow-card" key={entry.creator}>
                  <header>
                    <div>
                      <span>Original #{index + 2}</span>
                      <strong>Creator {entry.creator + 1}</strong>
                    </div>
                    <span className="creator-shadow-card__delta">
                      {delta >= 0 ? "+" : ""}
                      {delta} points
                    </span>
                  </header>

                  <div className="creator-shadow-card__panels">
                    <div className="creator-shadow-card__panel">
                      <div className="creator-shadow-card__panel-head">
                        <span>Observed ranking</span>
                        <strong>{Math.round(entry.share * 100)}%</strong>
                      </div>
                      <svg
                        viewBox={`0 0 ${miniChartWidth} ${miniChartHeight}`}
                        role="img"
                        aria-label={`Creator ${entry.creator + 1} receives ${Math.round(entry.share * 100)} percent of recommendations under the observed ranking.`}
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
                    </div>

                    <div className="creator-shadow-card__panel creator-shadow-card__panel--reset">
                      <div className="creator-shadow-card__panel-head">
                        <span>Ranking reset</span>
                        <strong>{Math.round(entry.resetShare * 100)}%</strong>
                      </div>
                      <svg
                        viewBox={`0 0 ${miniChartWidth} ${miniChartHeight}`}
                        role="img"
                        aria-label={`Creator ${entry.creator + 1} receives ${Math.round(entry.resetShare * 100)} percent of recommendations when ranking scores reset every 400 recommendations.`}
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
                    </div>
                  </div>
                </article>
              );
            })}
          </div>

          <p className="creator-shadow-comparison__note">
            The intervention clears accumulated visibility scores, not prior views or modeled
            audience response. These are policy counterfactuals, not claims about a creator’s
            guaranteed potential.
          </p>
        </section>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            Creator {world.winner + 1} received{" "}
            <strong>{Math.round(world.finalShare * 100)}% of all recommendations</strong>.
            Without intervention, #2 and #3 received{" "}
            {Math.round(runnerUpCounterfactuals[0].share * 100)}% and{" "}
            {Math.round(runnerUpCounterfactuals[1].share * 100)}%. With ranking resets, their
            shadow paths reach {Math.round(runnerUpCounterfactuals[0].resetShare * 100)}% and{" "}
            {Math.round(runnerUpCounterfactuals[1].resetShare * 100)}%.
          </>
        ) : (
          <>
            Talent can improve the odds. It can’t be amplified if the platform stops showing the
            work.
          </>
        )}
      </p>
    </div>
  );
}

export function ExperimentMonopolyGraph() {
  const animation = useGraphAnimation(2_700);
  const keptScores = useMemo(() => simulateCreatorWorld(31), []);
  const clearedScores = useMemo(() => simulateCreatorWorld(31, 160), []);
  const keptScoreTotal = keptScores.comparison.at(-1) ?? 0;
  const clearedScoreTotal = clearedScores.comparison.at(-1) ?? 0;
  const startingOpenShare = (CREATOR_COUNT - 1) / CREATOR_COUNT;
  const averageOpenShare = (comparison: number[]) =>
    comparison.map((total, index) =>
      index === 0 ? startingOpenShare : total / (index * SAMPLE_EVERY),
    );
  const keptOpenShares = averageOpenShare(keptScores.comparison);
  const clearedOpenShares = averageOpenShare(clearedScores.comparison);
  const keptOpenShare = keptOpenShares.at(-1) ?? 0;
  const clearedOpenShare = clearedOpenShares.at(-1) ?? 0;
  const keptOpenPercent = Math.round(keptOpenShare * 100);
  const clearedOpenPercent = Math.round(clearedOpenShare * 100);
  const chartWidth = 860;
  const chartHeight = 390;
  const keptLabelY = Math.max(48, 24 + (1 - keptOpenShare) * 314 - 12);
  const clearedLabelY = Math.max(48, 24 + (1 - clearedOpenShare) * 314 - 12);

  return (
    <div
      className="creator-graph"
      data-animation-state={animation.state}
      data-testid="experiment-monopoly-graph"
    >
      <div className="creator-graph__head">
        <div>
          <div className="panel__meta">Social media makes 1,600 recommendations</div>
          <strong>How much opportunity remains for anyone besides the current leader?</strong>
        </div>
        <button className="button button--small" type="button" onClick={animation.play}>
          {animation.running ? "Comparing…" : "Compare both rules"}
        </button>
      </div>

      <div className="experiment-metric">
        <div>
          <span className="panel__meta">What the vertical axis measures</span>
          <strong>
            The average chance that the next recommendation goes to anyone except the current
            leader.
          </strong>
        </div>
        <p>
          If the leader has a 70% chance of receiving the next recommendation, the other 23
          creators together have 30%. A higher line means the recommendation system keeps more
          alternative paths open.
        </p>
      </div>

      <div className="creator-graph__plot">
        <svg
          className="creator-line-chart"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          role="img"
          aria-label={`The vertical axis measures the average chance that the next recommendation goes to anyone except the current leader. When the recommendation system keeps boosting the early leader, that chance averages ${keptOpenPercent} percent. Resetting everyone to equal visibility ten times raises it to ${clearedOpenPercent} percent.`}
        >
          <title>Average chance that anyone except the current leader is recommended next</title>
          <desc>
            One rule keeps boosting the current leader. The other resets every creator to
            equal visibility ten times. A higher line means someone else is more likely to be
            recommended.
          </desc>
          <text x="54" y="15" className="creator-chart-label comparison-axis-title">
            average chance anyone else is recommended
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
                  {Math.round(tick * 100)}%
                </text>
              </g>
            );
          })}
          <motion.path
            d={linePath(keptOpenShares, chartWidth, chartHeight, 1)}
            fill="none"
            stroke="var(--rust)"
            strokeWidth="5"
            strokeLinecap="round"
            initial={false}
            animate={{ pathLength: animation.progress }}
            transition={{ duration: 0.08, ease: "linear" }}
          />
          <motion.path
            d={linePath(clearedOpenShares, chartWidth, chartHeight, 1)}
            fill="none"
            stroke="var(--blue)"
            strokeWidth="5"
            strokeLinecap="round"
            initial={false}
            animate={{ pathLength: animation.progress }}
            transition={{ duration: 0.08, ease: "linear" }}
          />
          {animation.progress > 0.92 ? (
            <>
              <text
                x="820"
                y={clearedLabelY}
                textAnchor="end"
                className="creator-chart-label"
              >
                reset to equal visibility: {clearedOpenPercent}%
              </text>
              <text
                x="820"
                y={keptLabelY}
                textAnchor="end"
                className="creator-chart-label"
              >
                keep boosting the leader: {keptOpenPercent}%
              </text>
            </>
          ) : null}
          <text x="54" y="370" className="creator-chart-label">
            recommendation 1
          </text>
          <text x="832" y="370" textAnchor="end" className="creator-chart-label">
            recommendation 1,600
          </text>
        </svg>
      </div>

      <div className="experiment-labels" aria-hidden="true">
        <div>
          <span className="experiment-labels__line experiment-labels__line--rust" />
          <strong>Keep boosting the early leader</strong>
          <span>Anyone else: {keptOpenPercent}% average chance of the next recommendation</span>
        </div>
        <div>
          <span className="experiment-labels__line experiment-labels__line--blue" />
          <strong>Reset everyone to equal visibility 10 times</strong>
          <span>Anyone else: {clearedOpenPercent}% average chance of the next recommendation</span>
        </div>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            Across 1,600 recommendations, resetting visibility ten times gave someone other than
            the current leader a <strong>{clearedOpenPercent}% average chance</strong> of receiving
            the next recommendation. Continuous boosting cut that chance to{" "}
            <strong>{keptOpenPercent}%</strong>. The reset rule preserved{" "}
            <strong>
              about {Math.round(clearedScoreTotal / keptScoreTotal)} times as much opportunity
            </strong>
            {" "}for an alternative to break through.
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
  const world = useMemo(() => simulateCreatorWorld(31), []);
  const sortedShares = useMemo(
    () =>
      world.series
        .map((values) => values.at(-1) ?? 0)
        .sort((left, right) => left - right),
    [world],
  );
  const cumulativeShares = useMemo(
    () => [
      0,
      ...sortedShares.map((_, index) =>
        sortedShares
          .slice(0, index + 1)
          .reduce((sum, share) => sum + share, 0),
      ),
    ],
    [sortedShares],
  );
  const bottomThreeQuartersIndex = Math.floor(CREATOR_COUNT * 0.75);
  const bottomThreeQuartersShare =
    cumulativeShares[bottomThreeQuartersIndex] ?? 0;
  const topThreeShare = sortedShares
    .slice(-3)
    .reduce((sum, share) => sum + share, 0);
  const chartWidth = 860;
  const chartHeight = 420;
  const annotationX = 54 + 0.75 * 778;
  const annotationY =
    24 + (1 - bottomThreeQuartersShare) * 344;

  return (
    <div
      className="creator-graph"
      data-animation-state={animation.state}
      data-testid="lorenz-history-graph"
    >
      <div className="creator-graph__head">
        <div>
          <div className="panel__meta">The visible income distribution</div>
          <strong>The bend shows inequality. It doesn’t show what caused it.</strong>
        </div>
        <button className="button button--small" type="button" onClick={animation.play}>
          {animation.running ? "Income is concentrating…" : "Draw the income curve"}
        </button>
      </div>

      <div className="creator-graph__plot">
        <svg
          className="creator-line-chart"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          role="img"
          aria-label={`A stylized Lorenz curve. The bottom 75 percent of creators receive ${Math.round(bottomThreeQuartersShare * 100)} percent of income, while the top three creators receive ${Math.round(topThreeShare * 100)} percent.`}
        >
          <title>A bowed creator-income Lorenz curve</title>
          <desc>
            The curve reveals a highly unequal distribution. The same curve can be
            consistent with different mixtures of contribution and inherited position.
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
            equal income
          </text>
          <motion.path
            d={linePath(cumulativeShares, chartWidth, chartHeight, 1)}
            fill="none"
            stroke="var(--rust)"
            strokeWidth="6"
            strokeLinecap="round"
            initial={false}
            animate={{ pathLength: animation.progress }}
            transition={{ duration: 0.08, ease: "linear" }}
          />
          {animation.progress > 0.92 ? (
            <>
              <circle
                cx={annotationX}
                cy={annotationY}
                r="6"
                fill="var(--rust)"
              />
              <text
                x={annotationX - 12}
                y={Math.max(292, annotationY - 18)}
                textAnchor="end"
                className="creator-chart-label"
              >
                bottom 75% receive {Math.round(bottomThreeQuartersShare * 100)}%
              </text>
            </>
          ) : null}
          <text
            x="443"
            y="418"
            textAnchor="middle"
            className="creator-chart-label lorenz-axis-title"
          >
            creators, lowest to highest income
          </text>
          <text
            x="-196"
            y="15"
            transform="rotate(-90)"
            textAnchor="middle"
            className="creator-chart-label lorenz-axis-title"
          >
            share of all creator income
          </text>
        </svg>
      </div>

      <div
        className="lorenz-histories"
        aria-label="Two possible causal histories behind the same Lorenz curve"
      >
        <div className="lorenz-histories__shared">
          <span className="panel__meta">The same Lorenz curve above</span>
          <strong>One visible income gap</strong>
        </div>
        <div className="lorenz-histories__fork" aria-hidden="true" />
        <div className="lorenz-histories__cards">
          <article>
            <span className="panel__meta">Possible history A</span>
            <h3>Creators’ work explains more of the gap</h3>
            <div
              className="lorenz-cause-mix lorenz-cause-mix--work"
              role="img"
              aria-label="An illustrative mix in which work and contribution matter more than compounding visibility"
            >
              <span>Work and contribution</span>
              <span>Compounding visibility</span>
            </div>
            <p>Early visibility still amplifies the result, but it plays the smaller role.</p>
          </article>
          <article>
            <span className="panel__meta">Possible history B</span>
            <h3>Compounding visibility explains more of the gap</h3>
            <div
              className="lorenz-cause-mix lorenz-cause-mix--visibility"
              role="img"
              aria-label="An illustrative mix in which compounding visibility matters more than work and contribution"
            >
              <span>Work and contribution</span>
              <span>Compounding visibility</span>
            </div>
            <p>Differences in creators’ work still matter, but they play the smaller role.</p>
          </article>
        </div>
        <div className="lorenz-histories__conclusion">
          <strong>Same curve. Different causes.</strong>
          <span>The curve shows the income gap, not which causal mix produced it.</span>
        </div>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            The top three receive <strong>{Math.round(topThreeShare * 100)}% of income</strong>.
            The curve measures that gap, but not how much came from creators’ work versus
            compounding visibility.
          </>
        ) : (
          <>Draw the income curve, then compare two different histories that could produce it.</>
        )}
      </p>
    </div>
  );
}
