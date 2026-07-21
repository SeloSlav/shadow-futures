"use client";

import { motion, useReducedMotion } from "framer-motion";
import Image from "next/image";
import { useCallback, useEffect, useMemo, useState } from "react";

import { deriveSeed, mulberry32 } from "@/lib/model/prng";

const CREATOR_CAST = [
  { name: "MrBeast", image: "/creator-portraits/mrbeast.webp" },
  { name: "Taylor Swift", image: "/creator-portraits/taylor-swift.webp" },
  { name: "Rihanna", image: "/creator-portraits/rihanna.webp" },
  { name: "Beyoncé", image: "/creator-portraits/beyonce.webp" },
  { name: "Selena Gomez", image: "/creator-portraits/selena-gomez.webp" },
  { name: "Dwayne Johnson", image: "/creator-portraits/dwayne-johnson.webp" },
  { name: "Cristiano Ronaldo", image: "/creator-portraits/cristiano-ronaldo.webp" },
  { name: "Lionel Messi", image: "/creator-portraits/lionel-messi.webp" },
  { name: "Kylie Jenner", image: "/creator-portraits/kylie-jenner.webp" },
  { name: "Kim Kardashian", image: "/creator-portraits/kim-kardashian.webp" },
  { name: "Oprah Winfrey", image: "/creator-portraits/oprah-winfrey.webp" },
  { name: "Elon Musk", image: "/creator-portraits/elon-musk.webp" },
  { name: "Jeff Bezos", image: "/creator-portraits/jeff-bezos.webp" },
  { name: "Mark Zuckerberg", image: "/creator-portraits/mark-zuckerberg.webp" },
  { name: "Bill Gates", image: "/creator-portraits/bill-gates.webp" },
  { name: "Warren Buffett", image: "/creator-portraits/warren-buffett.webp" },
  { name: "Richard Branson", image: "/creator-portraits/richard-branson.webp" },
  { name: "Jensen Huang", image: "/creator-portraits/jensen-huang.webp" },
  { name: "Satya Nadella", image: "/creator-portraits/satya-nadella.webp" },
  { name: "Sundar Pichai", image: "/creator-portraits/sundar-pichai.webp" },
  { name: "Sara Blakely", image: "/creator-portraits/sara-blakely.webp" },
  { name: "Charli D'Amelio", image: "/creator-portraits/charli-damelio.webp" },
  { name: "Khaby Lame", image: "/creator-portraits/khaby-lame.webp" },
  { name: "Emma Chamberlain", image: "/creator-portraits/emma-chamberlain.webp" },
] as const;

const CREATOR_COUNT = CREATOR_CAST.length;
const RECOMMENDATIONS = 1_600;
const SAMPLE_EVERY = 40;
const FEEDBACK_STRENGTH = 1.55;
const WORLD_SEEDS = [31, 17, 42, 91, 2_026];

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
          <div className="panel__meta">One social media recommendation system</div>
          <strong>Every creator starts equally good.</strong>
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
          aria-label={`Twenty-four equally good creators compete for 1,600 recommendations. The top three eventually receive ${topThree.map((entry) => Math.round(entry.share * 100)).join(", ")} percent of them.`}
        >
          <title>One equally good creator breaks away</title>
          <desc>
            Twenty-four creators start equally. Small early differences are reinforced by
            the social media platform’s recommendation system until one creator receives
            much more exposure.
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
            Second and third place received {Math.round(topThree[1].share * 100)}% and{" "}
            {Math.round(topThree[2].share * 100)}%, even though all 24 started equal.
          </>
        ) : (
          <>A tiny early lead changes which creator the platform recommends next.</>
        )}
      </p>
    </div>
  );
}

export function ShadowFuturesGraph() {
  const animation = useGraphAnimation(2_400);
  const worlds = useMemo(
    () =>
      Array.from({ length: 10 }, (_, index) =>
        simulateCreatorWorld(deriveSeed(31, index)),
      ),
    [],
  );
  const uniqueWinners = new Set(worlds.map((world) => world.winner)).size;
  return (
    <div
      className="creator-graph"
      data-animation-state={animation.state}
      data-testid="shadow-futures-graph"
    >
      <div className="creator-graph__head">
        <div>
          <div className="panel__meta">Same people · same posts · same recommendation system</div>
          <strong>Only the first few views will change.</strong>
        </div>
        <button className="button button--small" type="button" onClick={animation.play}>
          {animation.running ? "Replaying…" : "Watch ten replays"}
        </button>
      </div>

      <div className="shadow-replay">
        <div className="shadow-replay__start">
          <span className="shadow-replay__label">The same 24 equally good creators</span>
          <div
            className="shadow-replay__cast"
            aria-label="An illustrative cast of 24 prominent creators and public figures begins together on the same starting line."
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
            Illustrative public-figure cast; no endorsement implied.{" "}
            <a href="/creator-portraits/credits.json" target="_blank" rel="noreferrer">
              Portrait credits
            </a>
          </p>
        </div>

        <div className="shadow-replay__turn" aria-hidden="true">
          <span>A few early views land differently</span>
          <span>↓</span>
        </div>

        <div
          className="shadow-replay__worlds"
          role="img"
          aria-label={`Ten replays of the same social media recommendation system produce ${uniqueWinners} different winners.`}
        >
          {worlds.map((world, index) => {
            const winner = CREATOR_CAST[world.winner];
            const reveal = Math.max(
              0,
              Math.min(1, animation.progress * worlds.length - index),
            );
            const revealed = reveal > 0.08;

            return (
              <motion.div
                className={`shadow-replay__world${revealed ? " is-revealed" : ""}`}
                key={index}
                initial={false}
                animate={{
                  opacity: 0.42 + reveal * 0.58,
                  y: (1 - reveal) * 8,
                }}
                transition={{ duration: 0.1, ease: "linear" }}
              >
                <span className="shadow-replay__take">Replay {index + 1}</span>
                <span className="shadow-replay__spotlight" aria-hidden="true">
                  <span>
                    {revealed ? (
                      <Image
                        src={winner.image}
                        alt=""
                        width={480}
                        height={480}
                        sizes="80px"
                      />
                    ) : (
                      "?"
                    )}
                  </span>
                </span>
                <strong>{revealed ? winner.name : "Who wins?"}</strong>
                <span>{revealed ? "wins this replay" : "Same starting line"}</span>
              </motion.div>
            );
          })}
        </div>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            Nothing about the creators changed. Changing only the opening produced{" "}
            <strong>{uniqueWinners} different winners</strong>. Real life shows us just one of
            these replays.
          </>
        ) : (
          <>Same people. Same posts. Same recommendation system. Change only the first few views.</>
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
  const keptOpenPercent = Math.round(
    (keptScoreTotal / RECOMMENDATIONS) * 100,
  );
  const clearedOpenPercent = Math.round(
    (clearedScoreTotal / RECOMMENDATIONS) * 100,
  );
  const maxValue = clearedScoreTotal * 1.08;
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
          <div className="panel__meta">The social media platform makes 1,600 recommendations</div>
          <strong>
            Does its recommendation system keep giving everyone a chance—or keep boosting
            the early leader?
          </strong>
        </div>
        <button className="button button--small" type="button" onClick={animation.play}>
          {animation.running ? "Comparing…" : "See both versions"}
        </button>
      </div>

      <div className="creator-graph__plot">
        <svg
          className="creator-line-chart"
          viewBox={`0 0 ${chartWidth} ${chartHeight}`}
          role="img"
          aria-label={`Both versions make 1,600 social media recommendations. When the platform keeps boosting the early leader, everyone else shares ${keptOpenPercent} percent of the chance to be shown, on average. Starting everyone equally ten times raises that share to ${clearedOpenPercent} percent.`}
        >
          <title>Boosting the early leader compared with starting everyone equally ten times</title>
          <desc>
            One version focuses more and more on the early leader. The other regularly
            starts everyone equally again.
          </desc>
          {[0, 0.5, 1].map((tick) => {
            const tickY = 24 + (1 - tick) * 314;
            return (
              <line
                key={tick}
                x1="54"
                x2="832"
                y1={tickY}
                y2={tickY}
                className="creator-chart-grid"
              />
            );
          })}
          <motion.path
            d={linePath(keptScores.comparison, chartWidth, chartHeight, maxValue)}
            fill="none"
            stroke="var(--rust)"
            strokeWidth="5"
            strokeLinecap="round"
            initial={false}
            animate={{ pathLength: animation.progress }}
            transition={{ duration: 0.08, ease: "linear" }}
          />
          <motion.path
            d={linePath(clearedScores.comparison, chartWidth, chartHeight, maxValue)}
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
              <text x="820" y="45" textAnchor="end" className="creator-chart-label">
                start everyone equally 10 times
              </text>
              <text x="820" y="253" textAnchor="end" className="creator-chart-label">
                keep boosting the early leader
              </text>
            </>
          ) : null}
          <text x="54" y="370" className="creator-chart-label">
            first choice
          </text>
          <text x="832" y="370" textAnchor="end" className="creator-chart-label">
            choice 1,600
          </text>
        </svg>
      </div>

      <div className="experiment-labels" aria-hidden="true">
        <div>
          <span className="experiment-labels__line experiment-labels__line--rust" />
          <strong>Keep boosting the early leader</strong>
          <span>Together, everyone else shares {keptOpenPercent}% of the chance to be shown</span>
        </div>
        <div>
          <span className="experiment-labels__line experiment-labels__line--blue" />
          <strong>Start everyone equally 10 times</strong>
          <span>Together, everyone else shares {clearedOpenPercent}% of the chance to be shown</span>
        </div>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            Both versions chose who to show 1,600 times. Starting over ten times gave other
            people{" "}
            <strong>
              almost {Math.round(clearedScoreTotal / keptScoreTotal)} times as much chance to
              break through
            </strong>
            .
          </>
        ) : (
          <>
            If the social media platform keeps boosting the early leader, everyone else gets
            fewer real chances to be seen.
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
          <strong>The bend shows inequality. It does not show what caused it.</strong>
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
          <strong>Differences in the work carry more of the outcome.</strong>
          <p>Inherited visibility still matters, but less.</p>
        </article>
        <article>
          <span className="panel__meta">History B</span>
          <strong>Early visibility carries more of the outcome.</strong>
          <p>Direct contribution still matters, but less.</p>
        </article>
      </div>

      <p className="creator-graph__result" aria-live="polite">
        {animation.state === "complete" ? (
          <>
            The top three receive <strong>{Math.round(topThreeShare * 100)}% of income</strong>.
            The curve measures that inequality. The income record alone cannot tell us which
            history produced it.
          </>
        ) : (
          <>One income curve can hide very different contribution histories.</>
        )}
      </p>
    </div>
  );
}
