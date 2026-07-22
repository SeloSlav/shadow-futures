"use client";

import { motion, useReducedMotion } from "framer-motion";
import Image from "next/image";
import { useCallback, useEffect, useMemo, useState } from "react";

import { deriveSeed, mulberry32 } from "@/lib/model/prng";

const CREATOR_CAST = [
  { name: "Taylor Swift", image: "/creator-portraits/taylor-swift.webp" },
  { name: "Rihanna", image: "/creator-portraits/rihanna.webp" },
  { name: "Beyoncé", image: "/creator-portraits/beyonce.webp" },
  { name: "Selena Gomez", image: "/creator-portraits/selena-gomez.webp" },
  { name: "Lady Gaga", image: "/creator-portraits/lady-gaga.webp" },
  { name: "Adele", image: "/creator-portraits/adele.webp" },
  { name: "Billie Eilish", image: "/creator-portraits/billie-eilish.webp" },
  { name: "Ariana Grande", image: "/creator-portraits/ariana-grande.webp" },
  { name: "Dua Lipa", image: "/creator-portraits/dua-lipa.webp" },
  { name: "Ed Sheeran", image: "/creator-portraits/ed-sheeran.webp" },
  { name: "Bruno Mars", image: "/creator-portraits/bruno-mars.webp" },
  { name: "Justin Bieber", image: "/creator-portraits/justin-bieber.webp" },
  { name: "The Weeknd", image: "/creator-portraits/the-weeknd.webp" },
  { name: "Drake", image: "/creator-portraits/drake.webp" },
  { name: "Kendrick Lamar", image: "/creator-portraits/kendrick-lamar.webp" },
  { name: "Harry Styles", image: "/creator-portraits/harry-styles.webp" },
  { name: "Miley Cyrus", image: "/creator-portraits/miley-cyrus.webp" },
  { name: "Katy Perry", image: "/creator-portraits/katy-perry.webp" },
  { name: "Shakira", image: "/creator-portraits/shakira.webp" },
  { name: "Jennifer Lopez", image: "/creator-portraits/jennifer-lopez.webp" },
  { name: "Bad Bunny", image: "/creator-portraits/bad-bunny.webp" },
  { name: "SZA", image: "/creator-portraits/sza.webp" },
  { name: "Post Malone", image: "/creator-portraits/post-malone.webp" },
  { name: "Doja Cat", image: "/creator-portraits/doja-cat.webp" },
] as const;

const CREATOR_COUNT = CREATOR_CAST.length;
const RECOMMENDATIONS = 1_600;
const SAMPLE_EVERY = 40;
const FEEDBACK_STRENGTH = 1.55;
const WORLD_SEEDS = [31, 17, 42, 91, 2_026];
const WORLD_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] as const;

type CreatorWorld = {
  comparison: number[];
  finalShare: number;
  series: number[][];
  winner: number;
};

function simulateCreatorWorld(seed: number, clearScoresEvery = 0): CreatorWorld {
  const random = mulberry32(seed);
  const counts = Array.from({ length: CREATOR_COUNT }, () => 0);
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
      counts.fill(0);
    }

    const weights = counts.map((count) => (2 + count) ** FEEDBACK_STRENGTH);
    const weightTotal = weights.reduce((sum, weight) => sum + weight, 0);
    const probabilities = weights.map((weight) => weight / weightTotal);
    comparisonTotal += 1 - Math.max(...probabilities);

    let draw = random() * weightTotal;
    let recipient = 0;
    while (recipient < CREATOR_COUNT - 1 && (draw -= weights[recipient]) > 0) {
      recipient += 1;
    }
    counts[recipient] += 1;

    if ((recommendation + 1) % SAMPLE_EVERY === 0) {
      const total = counts.reduce((sum, count) => sum + count, 0) || 1;
      counts.forEach((count, index) => series[index].push(count / total));
      comparison.push(comparisonTotal);
    }
  }

  const total = counts.reduce((sum, count) => sum + count, 0) || 1;
  const winner = counts.indexOf(Math.max(...counts));
  return {
    comparison,
    finalShare: counts[winner] / total,
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
  const topThree = useMemo(
    () =>
      world.series
        .map((values, creator) => ({
          creator,
          share: values.at(-1) ?? 0,
        }))
        .sort((left, right) => right.share - left.share)
        .slice(0, 3),
    [world],
  );
  const topThreeRank = new Map(
    topThree.map((entry, index) => [entry.creator, index]),
  );
  const sampleIndex = Math.min(
    world.series[0].length - 1,
    Math.floor(animation.progress * (world.series[0].length - 1)),
  );
  const leaderShare = world.series[world.winner][sampleIndex] ?? 0;
  const chartWidth = 860;
  const chartHeight = 420;
  const x = 54 + (sampleIndex / Math.max(1, world.series[0].length - 1)) * 778;
  const y = 24 + (1 - leaderShare) * 344;
  const topThreeLabelY = topThree.map((entry, index) => {
    const desired = 24 + (1 - entry.share) * 344 - 12;
    if (index === 0) return Math.max(38, desired);
    const previous = 24 + (1 - topThree[index - 1].share) * 344 - 12;
    return Math.min(350, Math.max(desired, previous + 24));
  });

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
          <div className="panel__meta">One recommendation system in action</div>
          <strong>The model compares creators with deliberately similar skill.</strong>
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
          aria-label={`Twenty-four similarly skilled creators compete for 1,600 recommendations. The top three eventually receive ${topThree.map((entry) => Math.round(entry.share * 100)).join(", ")} percent of them.`}
        >
          <title>One similarly skilled creator breaks away</title>
          <desc>
            Twenty-four creators with similar skill start with equal visibility. Small early
            differences are amplified by recommendation systems until one creator receives much
            more exposure.
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
          {world.series.map((values, creator) => (
            <motion.path
              key={creator}
              d={linePath(values, chartWidth, chartHeight, 1)}
              fill="none"
              stroke={
                topThreeRank.get(creator) === 0
                  ? "var(--rust)"
                  : topThreeRank.get(creator) === 1
                    ? "var(--blue)"
                    : topThreeRank.get(creator) === 2
                      ? "var(--shadow)"
                      : "var(--line-strong)"
              }
              strokeWidth={
                topThreeRank.get(creator) === 0
                  ? 5
                  : topThreeRank.get(creator) === 1
                    ? 3.25
                    : topThreeRank.get(creator) === 2
                      ? 2.4
                      : 1.2
              }
              strokeLinecap="round"
              initial={false}
              animate={{ pathLength: animation.progress }}
              transition={{ duration: 0.08, ease: "linear" }}
              opacity={topThreeRank.has(creator) ? 1 : 0.52}
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
          {animation.progress > 0.92 ? (
            <>
              {topThree.map((entry, index) => {
                const endpointY = 24 + (1 - entry.share) * 344;
                const fill =
                  index === 0
                    ? "var(--rust)"
                    : index === 1
                      ? "var(--blue)"
                      : "var(--shadow)";
                return (
                  <g key={entry.creator}>
                    {index > 0 ? (
                      <circle cx="832" cy={endpointY} r="5" fill={fill} />
                    ) : null}
                    <text
                      x="820"
                      y={topThreeLabelY[index]}
                      textAnchor="end"
                      className="creator-chart-label"
                    >
                      {index + 1}
                      {index === 0 ? "st" : index === 1 ? "nd" : "rd"}:{" "}
                      {Math.round(entry.share * 100)}%
                    </text>
                  </g>
                );
              })}
            </>
          ) : null}
          <text x="54" y="402" className="creator-chart-label">
            first recommendation
          </text>
          <text x="832" y="402" textAnchor="end" className="creator-chart-label">
            recommendation 1,600
          </text>
        </svg>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            Creator {world.winner + 1} received{" "}
            <strong>{Math.round(world.finalShare * 100)}% of all recommendations</strong>.
            The runners-up received {Math.round(topThree[1].share * 100)}% and{" "}
            {Math.round(topThree[2].share * 100)}%, even though all 24 began with comparable
            skill and equal visibility.
          </>
        ) : (
          <>A tiny early lead changes which creator the recommendation system promotes next.</>
        )}
      </p>
    </div>
  );
}

export function ShadowFuturesGraph() {
  const animation = useGraphAnimation(2_400);
  const [replayBatch, setReplayBatch] = useState(0);
  const [hasReplayed, setHasReplayed] = useState(false);
  const worlds = useMemo(
    () =>
      Array.from({ length: 10 }, (_, index) =>
        simulateCreatorWorld(deriveSeed(31, replayBatch * 10 + index)),
      ),
    [replayBatch],
  );
  const uniqueWinners = new Set(worlds.map((world) => world.winner)).size;
  const winnerGroups = useMemo(() => {
    const groups = new Map<number, number[]>();
    worlds.forEach((world, worldIndex) => {
      groups.set(world.winner, [...(groups.get(world.winner) ?? []), worldIndex]);
    });
    return Array.from(groups, ([winner, worldIndexes]) => ({ winner, worldIndexes }));
  }, [worlds]);
  const worldReveals = worlds.map((_, index) =>
    Math.max(0, Math.min(1, animation.progress * worlds.length - index)),
  );
  const visibleWinnerGroups = winnerGroups
    .map((group) => ({
      ...group,
      visibleWorldIndexes: group.worldIndexes.filter((index) => worldReveals[index] > 0.08),
    }))
    .filter((group) => group.visibleWorldIndexes.length > 0);

  const playReplays = () => {
    if (hasReplayed) {
      setReplayBatch((current) => current + 1);
    } else {
      setHasReplayed(true);
    }
    animation.play();
  };

  return (
    <div
      className="creator-graph"
      data-animation-state={animation.state}
      data-testid="shadow-futures-graph"
    >
      <div className="creator-graph__head">
        <div>
          <div className="panel__meta">Same artists · same posts · same recommendation system</div>
          <strong>Only the opening views change from one world to the next.</strong>
        </div>
        <button
          className="button button--small"
          type="button"
          onClick={playReplays}
          disabled={animation.running}
        >
          {animation.running
            ? "Running ten worlds…"
            : hasReplayed
              ? "Run ten new worlds"
              : "Run ten worlds"}
        </button>
      </div>

      <div className="shadow-replay">
        <div className="shadow-replay__start">
          <span className="shadow-replay__label">24 music artists as visual stand-ins</span>
          <div
            className="shadow-replay__cast"
            aria-label="Twenty-four music-artist portraits act only as visual stand-ins for the possible winners in this hypothetical simulation."
          >
            {CREATOR_CAST.map((creator) => (
              <span
                className="shadow-replay__person"
                key={creator.name}
                title={creator.name}
                aria-label={creator.name}
              >
                <Image
                  src={creator.image}
                  alt=""
                  width={480}
                  height={480}
                  sizes="(max-width: 720px) 12vw, 7vw"
                />
              </span>
            ))}
          </div>
          <p className="shadow-replay__cast-note">
            These portraits are only visual stand-ins. The simulation doesn’t make any claim
            about these artists’ real talent, work or merit.{" "}
            <a href="/creator-portraits/credits.json" target="_blank" rel="noreferrer">
              Portrait credits
            </a>
          </p>
        </div>

        <div className="shadow-replay__turn" aria-hidden="true">
          <span>↻</span>
        </div>

        <div className="shadow-replay__world-map">
          <span className="shadow-replay__map-label">Ten separate starts</span>
          <div className="shadow-replay__seeds" aria-hidden="true">
            {worlds.map((_, index) => {
              const assigned = worldReveals[index] > 0.08;
              return (
                <motion.span
                  className={`shadow-replay__seed${assigned ? " is-assigned" : ""}`}
                  key={WORLD_LABELS[index]}
                  initial={false}
                  animate={{ opacity: assigned ? 0.22 : 1, scale: assigned ? 0.82 : 1 }}
                >
                  {WORLD_LABELS[index]}
                </motion.span>
              );
            })}
          </div>

          <div className="shadow-replay__branch" aria-hidden="true" />
          <span className="shadow-replay__map-label">Where each world lands</span>

          <div
            className={`shadow-replay__outcomes${
              visibleWinnerGroups.length === 0 ? " is-empty" : ""
            }`}
            role="list"
            aria-label={`Ten independent worlds produced ${uniqueWinners} different winners.`}
            aria-live="polite"
          >
            {visibleWinnerGroups.length === 0 ? (
              <span className="shadow-replay__pending" aria-hidden="true">
                ?
              </span>
            ) : (
              visibleWinnerGroups.map((group) => {
                const winner = CREATOR_CAST[group.winner];
                const worldNames = group.visibleWorldIndexes.map(
                  (index) => WORLD_LABELS[index],
                );
                return (
                  <motion.div
                    className="shadow-replay__winner"
                    key={winner.name}
                    role="listitem"
                    aria-label={`${winner.name} won ${
                      worldNames.length === 1
                        ? `world ${worldNames[0]}`
                        : `worlds ${worldNames.join(", ")}`
                    }.`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.18 }}
                  >
                    <span className="shadow-replay__winner-portrait" aria-hidden="true">
                      <Image
                        src={winner.image}
                        alt=""
                        width={480}
                        height={480}
                        sizes="84px"
                      />
                    </span>
                    <strong>{winner.name}</strong>
                    <span className="shadow-replay__won-worlds" aria-hidden="true">
                      {group.visibleWorldIndexes.map((index) => (
                        <motion.span
                          className="shadow-replay__won-world"
                          key={WORLD_LABELS[index]}
                          initial={{ opacity: 0, scale: 0.6 }}
                          animate={{ opacity: 1, scale: 1 }}
                        >
                          {WORLD_LABELS[index]}
                        </motion.span>
                      ))}
                    </span>
                  </motion.div>
                );
              })
            )}
          </div>
        </div>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            <strong>{uniqueWinners} different musicians</strong> won across the ten worlds.
          </>
        ) : (
          <>The same 24 artists begin every world.</>
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

      <div className="lorenz-histories">
        <article>
          <span className="panel__meta">History A</span>
          <strong>Differences in creators’ work explain more of the outcome.</strong>
          <p>Inherited visibility still matters, but less.</p>
        </article>
        <article>
          <span className="panel__meta">History B</span>
          <strong>Early visibility explains more of the outcome.</strong>
          <p>Direct contribution still matters, but less.</p>
        </article>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            The top three receive <strong>{Math.round(topThreeShare * 100)}% of income</strong>.
            The curve measures that inequality. The income record alone can’t tell us which
            history produced it.
          </>
        ) : (
          <>One income curve can hide very different contribution histories.</>
        )}
      </p>
    </div>
  );
}
