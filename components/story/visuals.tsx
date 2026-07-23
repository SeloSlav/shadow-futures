"use client";

import { extent, max, quantileSorted } from "d3-array";
import { scaleLinear } from "d3-scale";
import { curveMonotoneX, line } from "d3-shape";
import { motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";

import { allocationProbabilities } from "@/lib/model/allocation";
import { gaugeTransform, herfindahl } from "@/lib/model/metrics";
import { deriveSeed } from "@/lib/model/prng";
import { simulateScenario, simulateWorlds } from "@/lib/model/simulation";
import type {
  AllocationStep,
  Scenario,
  SimulationResult,
  WorldSummary,
} from "@/lib/model/types";
import { useMarketStore } from "@/lib/store/market-store";
import { Math as EquationMath } from "@/components/ui/math";
import { RangeControl } from "@/components/ui/range-control";

const AGENT_COLORS = [
  "var(--blue)",
  "var(--rust)",
  "#6e7f58",
  "#8270a1",
  "#b4863f",
  "#56858a",
  "#a06470",
  "#70777a",
  "#3f7966",
  "#9c724d",
];

function useParallelWorlds(scenario: Scenario): {
  worlds: WorldSummary[];
  pending: boolean;
} {
  const [worlds, setWorlds] = useState<WorldSummary[]>([]);
  const [pending, setPending] = useState(true);
  const requestId = useRef(0);

  useEffect(() => {
    const id = requestId.current + 1;
    requestId.current = id;
    const timeout = window.setTimeout(() => {
      setPending(true);
      if (typeof Worker === "undefined") {
        setWorlds(simulateWorlds(scenario, scenario.worlds));
        setPending(false);
        return;
      }
      const worker = new Worker(
        new URL("../../lib/model/parallelWorlds.worker.ts", import.meta.url),
      );
      worker.onmessage = (event: MessageEvent<{ id: number; worlds: WorldSummary[] }>) => {
        if (event.data.id === requestId.current) {
          setWorlds(event.data.worlds);
          setPending(false);
        }
        worker.terminate();
      };
      worker.onerror = () => {
        setWorlds(simulateWorlds(scenario, Math.min(scenario.worlds, 128)));
        setPending(false);
        worker.terminate();
      };
      worker.postMessage({ id, scenario, count: scenario.worlds });
    }, 140);

    return () => window.clearTimeout(timeout);
  }, [scenario]);

  return { worlds, pending };
}

function pathForSeries(
  values: Array<[number, number]>,
  width: number,
  height: number,
  yDomain?: [number, number],
): string {
  const xDomain = extent(values, (value) => value[0]) as [number, number];
  const actualYDomain =
    yDomain ?? ([0, max(values, (value) => value[1]) ?? 1] as [number, number]);
  const x = scaleLinear().domain(xDomain).range([0, width]);
  const y = scaleLinear().domain(actualYDomain).range([height, 0]);
  return (
    line<[number, number]>()
      .x((value) => x(value[0]))
      .y((value) => y(value[1]))
      .curve(curveMonotoneX)(values) ?? ""
  );
}

function sampledSteps(steps: AllocationStep[], maximumPoints = 180): AllocationStep[] {
  if (steps.length <= maximumPoints) return steps;
  const stride = Math.ceil(steps.length / maximumPoints);
  return steps.filter((_, index) => index % stride === 0 || index === steps.length - 1);
}

export function HeroNetwork() {
  const branches = [
    "M 80 530 C 270 460, 280 250, 510 250 S 760 95, 1060 120",
    "M 80 530 C 270 460, 310 390, 520 410 S 810 310, 1120 360",
    "M 80 530 C 260 470, 350 555, 560 545 S 840 650, 1140 590",
    "M 80 530 C 260 470, 350 700, 610 690 S 890 760, 1160 725",
    "M 80 530 C 290 450, 350 90, 640 80 S 940 170, 1170 55",
  ];
  return (
    <svg
      className="hero-network"
      viewBox="0 0 1200 800"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="heroBranch" x1="0" x2="1">
          <stop offset="0" stopColor="var(--rust)" stopOpacity="0.62" />
          <stop offset="0.45" stopColor="var(--shadow)" stopOpacity="0.24" />
          <stop offset="1" stopColor="var(--shadow)" stopOpacity="0.03" />
        </linearGradient>
      </defs>
      {branches.map((branch, index) => (
        <motion.path
          key={branch}
          d={branch}
          fill="none"
          stroke="url(#heroBranch)"
          strokeDasharray={index === 1 ? undefined : "5 10"}
          strokeWidth={index === 1 ? 2.5 : 1.2}
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: 1.8, delay: index * 0.12 }}
        />
      ))}
      <circle cx="80" cy="530" r="7" fill="var(--rust)" />
      <circle cx="1120" cy="360" r="6" fill="var(--blue)" />
    </svg>
  );
}

export function MovingTrackRace() {
  const scenario = useMarketStore((state) => state.scenario);
  const rerun = useMarketStore((state) => state.rerun);
  const activeStep = useMarketStore((state) => state.activeStep);
  const race = useMemo(
    () =>
      simulateScenario({
        ...scenario,
        n: 2,
        periods: Math.max(120, scenario.periods),
      }),
    [scenario],
  );
  const visibleStep =
    race.steps[Math.max(0, Math.min(activeStep - 1, race.steps.length - 1))] ??
    race.steps[0];
  const total = Math.max(1, visibleStep.t);
  const shares = visibleStep.counts.map((count) => count / total);
  const firstWinner = race.steps[0]?.recipient ?? 0;
  const leader = shares.indexOf(Math.max(...shares));

  return (
    <div className="panel" data-testid="hero-simulation">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Thought experiment · the moving-track race</div>
          <strong>Runner A is a little faster. The first few steps are still uncertain.</strong>
        </div>
        <button className="button button--small" type="button" onClick={rerun}>
          Replay the start
        </button>
      </div>
      <div className="panel__body">
        <svg
          viewBox="0 0 760 330"
          role="img"
          aria-label={`A two-runner race after ${visibleStep.t} steps. Runner ${String.fromCharCode(
            65 + leader,
          )} leads, and the moving track now helps that runner.`}
        >
          <title>A race where the track helps whoever is ahead</title>
          <desc>
            Runner A begins slightly faster. An early lead activates a moving track beneath
            the leader, making future wins more likely.
          </desc>
          {[0, 1].map((runner) => {
            const y = 105 + runner * 125;
            const x = 95 + shares[runner] * 560;
            const isLeader = leader === runner;
            return (
              <g key={runner}>
                <text x="38" y={y + 5} className="chart-axis">
                  {runner === 0 ? "A · slightly faster" : "B · can still lead early"}
                </text>
                <line
                  x1="95"
                  x2="680"
                  y1={y}
                  y2={y}
                  stroke="var(--line-strong)"
                  strokeWidth="8"
                  strokeLinecap="round"
                />
                {isLeader ? (
                  <line
                    x1="95"
                    x2={Math.max(110, x)}
                    y1={y}
                    y2={y}
                    stroke="var(--rust-soft)"
                    strokeWidth="18"
                    strokeLinecap="round"
                  />
                ) : null}
                <motion.circle
                  cx={x}
                  cy={y}
                  r="17"
                  fill={AGENT_COLORS[runner]}
                  animate={{ cx: x }}
                  transition={{ type: "spring", stiffness: 120, damping: 22 }}
                />
                <text x={x} y={y - 30} textAnchor="middle" className="chart-axis">
                  {Math.round(shares[runner] * 100)}% of wins
                </text>
                {isLeader ? (
                  <text x="680" y={y + 32} textAnchor="end" className="chart-axis">
                    moving track is helping
                  </text>
                ) : null}
              </g>
            );
          })}
          <text x="95" y="294" className="chart-axis">
            start
          </text>
          <text x="680" y="294" className="chart-axis" textAnchor="end">
            later
          </text>
        </svg>
        <div className="callout">
          The first win went to <strong>Runner {String.fromCharCode(65 + firstWinner)}</strong>.
          From then on, winning changed the track. The final gap now mixes two things:
          running ability and help created by being ahead.
        </div>
      </div>
    </div>
  );
}

export function AIFlywheel() {
  const scenario = useMarketStore((state) => state.scenario);
  const patchScenario = useMarketStore((state) => state.patchScenario);
  const stages = [
    ["1", "Early contract", "A customer chooses one lab"],
    ["2", "More revenue", "The winner can spend sooner"],
    ["3", "More compute", "More chips, power, and data-center capacity"],
    ["4", "Better reach", "Faster service and wider distribution"],
    ["5", "Next contract", "The first win now helps cause the next"],
  ];
  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Translate the race into the AI economy</div>
          <strong>An early lead can buy the track that makes later leads easier.</strong>
        </div>
      </div>
      <div className="panel__body">
        <svg
          viewBox="0 0 780 370"
          role="img"
          aria-label="AI flywheel: early contract leads to revenue, compute, distribution, and a higher chance of winning the next contract."
        >
          <title>The AI compute flywheel</title>
          <desc>
            Five connected stages show how an early market result can become productive
            capacity and then influence the next market result.
          </desc>
          {stages.map(([number, title, detail], index) => {
            const angle = -Math.PI / 2 + (index / stages.length) * Math.PI * 2;
            const x = 390 + Math.cos(angle) * 245;
            const y = 185 + Math.sin(angle) * 125;
            const nextAngle =
              -Math.PI / 2 + (((index + 1) % stages.length) / stages.length) * Math.PI * 2;
            const nextX = 390 + Math.cos(nextAngle) * 245;
            const nextY = 185 + Math.sin(nextAngle) * 125;
            return (
              <g key={title}>
                <path
                  d={`M ${x} ${y} Q 390 185 ${nextX} ${nextY}`}
                  fill="none"
                  stroke="var(--line-strong)"
                  strokeWidth="2"
                  markerEnd="url(#flywheel-arrow)"
                />
                <circle
                  cx={x}
                  cy={y}
                  r="54"
                  fill={index < 3 ? "var(--blue-soft)" : "var(--rust-soft)"}
                  stroke="var(--line-strong)"
                />
                <text x={x} y={y - 12} textAnchor="middle" className="chart-axis">
                  {number}. {title}
                </text>
                <foreignObject x={x - 46} y={y + 2} width="92" height="42">
                  <div
                    style={{
                      color: "var(--ink)",
                      fontSize: "9px",
                      lineHeight: 1.25,
                      textAlign: "center",
                    }}
                  >
                    {detail}
                  </div>
                </foreignObject>
              </g>
            );
          })}
          <defs>
            <marker
              id="flywheel-arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--line-strong)" />
            </marker>
          </defs>
          <circle cx="390" cy="185" r="72" fill="var(--surface-solid)" stroke="var(--line)" />
          <text x="390" y="177" textAnchor="middle" className="chart-axis">
            real engineering
          </text>
          <text x="390" y="192" textAnchor="middle" className="chart-axis">
            + an inherited lead
          </text>
          <text x="390" y="207" textAnchor="middle" className="chart-axis">
            become tangled
          </text>
        </svg>
        <div style={{ maxWidth: "34rem", margin: "0 auto" }}>
          <RangeControl
            id="ai-snowball"
            label="How strongly yesterday’s lead affects tomorrow"
            min={0}
            max={2.5}
            step={0.05}
            value={scenario.rho}
            format={(value) =>
              value === 0
                ? "not at all"
                : value < 1
                  ? "a little"
                  : value === 1
                    ? "strong"
                    : "very strongly"
            }
            onChange={(rho) => patchScenario({ rho })}
          />
        </div>
        <p style={{ color: "var(--muted)", fontSize: "0.82rem", marginBottom: 0 }}>
          This is a thought experiment, not a claim that every AI market follows the same loop.
          The point is to see how genuine skill and a self-reinforcing position can coexist.
        </p>
      </div>
    </div>
  );
}

export function EvidenceFadeChart() {
  const scenario = useMarketStore((state) => state.scenario);
  const result = useMemo(() => simulateScenario(scenario), [scenario]);
  const steps = sampledSteps(result.steps, 180);
  const width = 720;
  const height = 320;
  const winner = result.winner;
  const record = steps.map(
    (step) => [step.t, step.counts[winner] / scenario.periods] as [number, number],
  );
  const initialInformation = Math.max(result.steps[0]?.information ?? 1, 0.0001);
  const evidence = steps.map(
    (step) => [step.t, Math.min(1, step.information / initialInformation)] as [
      number,
      number,
    ],
  );
  const last = result.steps.at(-1);
  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">The trophy shelf problem</div>
          <strong>The record gets bigger while each new trophy tells us less.</strong>
        </div>
      </div>
      <div className="panel__body">
        <svg
          viewBox={`0 0 ${width} ${height + 38}`}
          role="img"
          aria-label="The winner's record rises while the relative evidence in each next transaction can fall."
        >
          <title>More wins can mean less new evidence</title>
          <desc>
            A rust line shows the winner&apos;s growing record. A blue line shows how much the
            next allocation teaches relative to the start of the market.
          </desc>
          {[0.25, 0.5, 0.75].map((fraction) => (
            <line
              key={fraction}
              className="chart-grid"
              x1="0"
              x2={width}
              y1={height * fraction}
              y2={height * fraction}
            />
          ))}
          <path
            d={pathForSeries(record, width, height, [0, 1])}
            fill="none"
            stroke="var(--rust)"
            strokeWidth="4"
          />
          <path
            d={pathForSeries(evidence, width, height, [0, 1])}
            fill="none"
            stroke="var(--blue)"
            strokeWidth="3"
          />
          <text x={width - 8} y="38" textAnchor="end" className="chart-axis">
            winner&apos;s record ↑
          </text>
          <text x={width - 8} y={height - 18} textAnchor="end" className="chart-axis">
            what the next win teaches ↓
          </text>
          <text x="0" y={height + 26} className="chart-axis">
            market begins
          </text>
          <text x={width} y={height + 26} textAnchor="end" className="chart-axis">
            history hardens
          </text>
        </svg>
        <div className="callout">
          If the leader has a <strong>{((last?.probabilities[winner] ?? 0) * 100).toFixed(0)}%</strong>{" "}
          chance of getting the next customer, another win is impressive, but it isn’t a fresh
          head-to-head test.
        </div>
      </div>
    </div>
  );
}

export function ContributionSplit() {
  const scenario = useMarketStore((state) => state.scenario);
  const [displacement, setDisplacement] = useState(0.45);
  const input = scenario.inputs[0]?.[0] ?? 0.8;
  const baseContribution = Math.max(0.05, input * scenario.beta[0]);
  const basePosition = 0.55;
  const total = baseContribution + basePosition;
  const shiftedContribution = Math.max(0.02, baseContribution + input * displacement);
  const shiftedPosition = total - shiftedContribution;

  const renderBar = (label: string, contribution: number, position: number) => (
    <div className="scope-card">
      <div className="panel__meta">{label}</div>
      <div
        aria-label={`${label}: ${Math.round(
          (contribution / total) * 100,
        )} percent direct contribution and ${Math.round((position / total) * 100)} percent inherited position`}
        style={{
          display: "flex",
          height: "3.4rem",
          overflow: "hidden",
          marginTop: "0.8rem",
          borderRadius: "999px",
          background: "var(--line)",
        }}
      >
        <div
          style={{
            width: `${Math.max(0, (contribution / total) * 100)}%`,
            display: "grid",
            placeItems: "center",
            background: "var(--blue)",
            color: "var(--paper)",
            fontSize: "0.72rem",
          }}
        >
          skill
        </div>
        <div
          style={{
            width: `${Math.max(0, (position / total) * 100)}%`,
            display: "grid",
            placeItems: "center",
            background: "var(--rust)",
            color: "var(--paper)",
            fontSize: "0.72rem",
          }}
        >
          position
        </div>
      </div>
      <p>Observed winning odds: exactly the same</p>
    </div>
  );

  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">The attribution problem</div>
          <strong>Two explanations can fit the same visible market perfectly.</strong>
        </div>
      </div>
      <div className="panel__body">
        <RangeControl
          id="simple-gauge-d"
          label="Move part of the explanation from position to skill"
          min={-0.45}
          max={0.45}
          step={0.01}
          value={displacement}
          format={(value) => (value > 0 ? "more skill" : value < 0 ? "more position" : "even split")}
          onChange={setDisplacement}
        />
        <div className="experiment-grid" style={{ marginTop: "1.5rem" }}>
          {renderBar("Explanation A", shiftedContribution, shiftedPosition)}
          {renderBar("Explanation B", baseContribution, basePosition)}
        </div>
        <div className="callout" style={{ marginTop: "1rem" }}>
          The data sees the <strong>total</strong>. If inherited position is hidden, it may not
          reveal how that total should be split.
        </div>
      </div>
    </div>
  );
}

export function HonestPolicyDemo() {
  const [contribution, setContribution] = useState(64);
  const reward = 100;
  const alternative = 42;
  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">One income record · two plausible stories</div>
          <strong>A precise “earned versus inherited” split needs evidence the market may not have made.</strong>
        </div>
      </div>
      <div className="panel__body">
        <RangeControl
          id="simple-tax-contribution"
          label="How much of the $100 reward did direct contribution cause?"
          min={20}
          max={90}
          value={contribution}
          format={(value) => `$${value}`}
          onChange={setContribution}
        />
        <div className="experiment-grid" style={{ marginTop: "1.5rem" }}>
          <div className="scope-card">
            <div className="panel__meta">Story A</div>
            <h3>${contribution} contribution</h3>
            <p>${reward - contribution} left as position-driven reward.</p>
          </div>
          <div className="scope-card">
            <div className="panel__meta">Story B</div>
            <h3>${alternative} contribution</h3>
            <p>${reward - alternative} left as position-driven reward.</p>
          </div>
        </div>
        <div className="callout" style={{ marginTop: "1rem" }}>
          The same visible $100 can’t tell a tax rule which hidden story is true. This doesn’t
          <strong> not</strong> mean all high income is rent, risk is fake, or one tax rate is
          automatically correct.
        </div>
      </div>
    </div>
  );
}

type MarketPathProps = {
  result: SimulationResult;
  title: string;
  activeStep?: number;
  compact?: boolean;
};

export function MarketPath({
  result,
  title,
  activeStep = result.steps.length,
  compact = false,
}: MarketPathProps) {
  const steps = sampledSteps(result.steps.slice(0, activeStep));
  const width = 620;
  const height = compact ? 170 : 260;
  const total = Math.max(1, activeStep);

  return (
    <div className="chart-shell">
      <div className="panel__meta">{title}</div>
      <svg
        viewBox={`0 0 ${width} ${height + 34}`}
        role="img"
        aria-label={`${title}. Cumulative reward paths for ${result.scenario.n} agents over ${activeStep} allocations.`}
      >
        <title>{title}</title>
        <desc>
          Cumulative rewards by agent. Divergence can occur even with identical inputs and
          structural parameters because the seed changes early allocation shocks.
        </desc>
        {[0.25, 0.5, 0.75].map((fraction) => (
          <line
            key={fraction}
            className="chart-grid"
            x1="0"
            x2={width}
            y1={height * fraction}
            y2={height * fraction}
          />
        ))}
        {Array.from({ length: result.scenario.n }, (_, agent) => {
          const values: Array<[number, number]> = [
            [0, 0],
            ...steps.map(
              (step) => [step.t, step.counts[agent] / total] as [number, number],
            ),
          ];
          return (
            <path
              key={agent}
              d={pathForSeries(values, width, height, [0, 1])}
              fill="none"
              stroke={AGENT_COLORS[agent]}
              strokeWidth={agent < 2 ? 3 : 1.5}
              strokeOpacity={agent < 2 ? 1 : 0.62}
            />
          );
        })}
        <text className="chart-axis" x="0" y={height + 25}>
          0
        </text>
        <text className="chart-axis" x={width} y={height + 25} textAnchor="end">
          {activeStep} transactions
        </text>
      </svg>
    </div>
  );
}

export function SideBySideMarkets() {
  const scenario = useMarketStore((state) => state.scenario);
  const secondarySeed = useMarketStore((state) => state.secondarySeed);
  const activeStep = useMarketStore((state) => state.activeStep);
  const rerun = useMarketStore((state) => state.rerun);
  const primary = useMemo(() => simulateScenario({ ...scenario, n: 2 }), [scenario]);
  const secondary = useMemo(
    () => simulateScenario({ ...scenario, n: 2, seed: secondarySeed }),
    [scenario, secondarySeed],
  );

  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Identical inputs · different shocks</div>
          <strong>
            A quality {scenario.inputs[0]?.[0]?.toFixed(2)} / B quality{" "}
            {scenario.inputs[1]?.[0]?.toFixed(2)}
          </strong>
        </div>
        <button className="button button--small" type="button" onClick={rerun}>
          Rerun the same market
        </button>
      </div>
      <div className="panel__body experiment-grid">
        <MarketPath result={primary} title={`Replay A · start ${primary.scenario.seed}`} activeStep={activeStep} />
        <MarketPath
          result={secondary}
          title={`Replay B · start ${secondary.scenario.seed}`}
          activeStep={activeStep}
        />
      </div>
    </div>
  );
}

export function RewardRewrite() {
  const scenario = useMarketStore((state) => state.scenario);
  const activeStep = useMarketStore((state) => state.activeStep);
  const result = useMemo(() => simulateScenario(scenario), [scenario]);
  const step = result.steps[Math.max(0, Math.min(activeStep - 1, result.steps.length - 1))];
  const next = result.steps[Math.min(step.t, result.steps.length - 1)] ?? step;

  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Conditional allocation</div>
          <strong>Reward #{step.t} rewrites reward #{step.t + 1}</strong>
        </div>
      </div>
      <div className="panel__body">
        <svg
          viewBox="0 0 720 310"
          role="img"
          aria-label="Routes from agents to the next customer thicken as reward and position accumulate."
        >
          <title>Reward rewrites the odds</title>
          <desc>
            Each agent connects to the next customer. Route thickness encodes conditional
            allocation probability after accumulated position updates.
          </desc>
          {scenario.inputs.map((input, index) => {
            const y = 42 + index * (220 / Math.max(1, scenario.n - 1));
            const probability = next.probabilities[index];
            const isRecipient = step.recipient === index;
            return (
              <g key={index}>
                <path
                  d={`M 105 ${y} C 285 ${y}, 380 155, 615 155`}
                  fill="none"
                  stroke={AGENT_COLORS[index]}
                  strokeOpacity={0.35 + probability * 0.65}
                  strokeWidth={1 + probability * 28}
                />
                <circle
                  cx="78"
                  cy={y}
                  r={isRecipient ? 19 : 12}
                  fill={AGENT_COLORS[index]}
                  stroke={isRecipient ? "var(--ink)" : "none"}
                  strokeWidth="3"
                />
                <text className="chart-axis" x="45" y={y + 4} textAnchor="end">
                  {String.fromCharCode(65 + index)}
                </text>
                <text className="chart-axis" x="126" y={y - 8}>
                  {Math.round(probability * 100)}%
                </text>
              </g>
            );
          })}
          <circle cx="650" cy="155" r="30" fill="var(--surface-solid)" stroke="var(--line-strong)" />
          <text className="chart-axis" x="650" y="151" textAnchor="middle">
            next
          </text>
          <text className="chart-axis" x="650" y="164" textAnchor="middle">
            reward
          </text>
        </svg>
        <div className="stats-grid">
          <div className="stat">
            <span className="stat-label">Latest recipient</span>
            <strong>{String.fromCharCode(65 + step.recipient)}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Leader probability</span>
            <strong>{(Math.max(...next.probabilities) * 100).toFixed(1)}%</strong>
          </div>
          <div className="stat">
            <span className="stat-label">ρ</span>
            <strong>{scenario.rho.toFixed(2)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ShadowMap() {
  const scenario = useMarketStore((state) => state.scenario);
  const setScenario = useMarketStore((state) => state.setScenario);
  const { worlds, pending } = useParallelWorlds(scenario);
  const [selected, setSelected] = useState<number | null>(null);
  const width = 760;
  const height = 420;

  const distribution = useMemo(() => {
    const winnerCounts = Array.from({ length: scenario.n }, () => 0);
    worlds.forEach((world) => {
      winnerCounts[world.winner] += 1;
    });
    const topShares = worlds.map((world) => Math.max(...world.shares)).sort((a, b) => a - b);
    return {
      winnerCounts,
      median: quantileSorted(topShares, 0.5) ?? 0,
      q1: quantileSorted(topShares, 0.25) ?? 0,
      q3: quantileSorted(topShares, 0.75) ?? 0,
      uniqueWinners: winnerCounts.filter((count) => count > 0).length,
    };
  }, [scenario.n, worlds]);

  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Shadow-futures map</div>
          <strong>{scenario.worlds} structurally identical worlds</strong>
        </div>
        <span className="panel__meta">{pending ? "Simulating…" : "Ready"}</span>
      </div>
      <div className="panel__body">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={`Terminal results across ${worlds.length} independently seeded worlds. Select a branch to replay that world.`}
        >
          <title>Shadow futures</title>
          <desc>
            Each line is a possible history under the same inputs and parameters. Its
            terminal height is the leading agent&apos;s reward share.
          </desc>
          <line x1="70" x2="70" y1={height / 2} y2={height / 2} stroke="var(--rust)" strokeWidth="4" />
          {worlds.map((world, index) => {
            const angle = (index / Math.max(1, worlds.length - 1) - 0.5) * 2.35;
            const terminalShare = Math.max(...world.shares);
            const endX = width - 30;
            const endY = 28 + (1 - terminalShare) * (height - 56);
            const midY = height / 2 + Math.sin(angle) * height * 0.35;
            const isSelected = selected === index;
            const dimmed = selected !== null && !isSelected;
            return (
              <path
                key={world.seed}
                d={`M 70 ${height / 2} C 250 ${height / 2}, 365 ${midY}, ${endX} ${endY}`}
                fill="none"
                stroke={AGENT_COLORS[world.winner]}
                strokeWidth={isSelected ? 4 : 1.2}
                strokeOpacity={dimmed ? 0.12 : isSelected ? 1 : 0.42}
                tabIndex={0}
                role="button"
                aria-label={`World ${index + 1}, seed ${world.seed}, winner ${String.fromCharCode(
                  65 + world.winner,
                )}, terminal share ${Math.round(terminalShare * 100)} percent`}
                onMouseEnter={() => setSelected(index)}
                onFocus={() => setSelected(index)}
                onClick={() => {
                  setSelected(index);
                  setScenario({ ...scenario, seed: world.seed });
                }}
              />
            );
          })}
          <text className="chart-axis" x="70" y={height / 2 - 14} textAnchor="middle">
            same market
          </text>
          <text className="chart-axis" x={width - 30} y={height - 8} textAnchor="end">
            terminal outcomes
          </text>
        </svg>
        <div className="stats-grid">
          <div className="stat">
            <span className="stat-label">Typical winner’s share</span>
            <strong>{(distribution.median * 100).toFixed(0)}%</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Middle half of replays</span>
            <strong>
              {(distribution.q1 * 100).toFixed(0)}–{(distribution.q3 * 100).toFixed(0)}%
            </strong>
          </div>
          <div className="stat">
            <span className="stat-label">Different agents who won</span>
            <strong>
              {distribution.uniqueWinners} of {scenario.n}
            </strong>
          </div>
          <div className="stat">
            <span className="stat-label">Most frequent winner</span>
            <strong>
              {distribution.winnerCounts.length
                ? String.fromCharCode(
                    65 +
                      distribution.winnerCounts.indexOf(
                        Math.max(...distribution.winnerCounts),
                      ),
                  )
                : "Not available"}
            </strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export function BudgetChart() {
  const scenario = useMarketStore((state) => state.scenario);
  const result = useMemo(() => simulateScenario(scenario), [scenario]);
  const steps = sampledSteps(result.steps);
  const width = 720;
  const height = 340;
  const maxY = Math.max(scenario.periods, result.comparisonBudget);
  const transactionPath = pathForSeries(
    [[0, 0], ...steps.map((step) => [step.t, step.t] as [number, number])],
    width,
    height,
    [0, maxY],
  );
  const budgetPath = pathForSeries(
    [
      [0, 0],
      ...steps.map((step) => [step.t, step.comparisonBudget] as [number, number]),
    ],
    width,
    height,
    [0, maxY],
  );
  const finalStep = result.steps.at(-1);

  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">The conceptual center</div>
          <strong>Volume isn’t replication.</strong>
        </div>
      </div>
      <div className="panel__body">
        <div className="callout">
          Think of the comparison budget as <strong>room for surprise</strong>. A 50–50 choice
          teaches us a lot. A choice the leader was 99% sure to win teaches us very little.
        </div>
        <svg
          viewBox={`0 0 ${width} ${height + 36}`}
          role="img"
          aria-label={`Transactions rise to ${scenario.periods}; the comparison budget rises to ${result.comparisonBudget.toFixed(
            1,
          )}.`}
        >
          <title>Transactions versus comparison budget</title>
          <desc>
            The transaction line rises one-for-one with time. The comparison-budget line
            rises only by the probability mass outside the current leader.
          </desc>
          {[0.25, 0.5, 0.75].map((fraction) => (
            <line
              key={fraction}
              className="chart-grid"
              x1="0"
              x2={width}
              y1={height * fraction}
              y2={height * fraction}
            />
          ))}
          <path d={transactionPath} fill="none" stroke="var(--rust)" strokeWidth="3" />
          <path d={budgetPath} fill="none" stroke="var(--blue)" strokeWidth="4" />
          <text className="chart-axis" x={width - 8} y="18" textAnchor="end">
            transactions
          </text>
          <text
            className="chart-axis"
            x={width - 8}
            y={height - (result.comparisonBudget / maxY) * height - 10}
            textAnchor="end"
          >
            room for surprise
          </text>
          <text className="chart-axis" x="0" y={height + 26}>
            0
          </text>
          <text className="chart-axis" x={width} y={height + 26} textAnchor="end">
            {scenario.periods}
          </text>
        </svg>
        <div className="stats-grid">
          <div className="stat">
            <span className="stat-label">Deals we watched</span>
            <strong>{scenario.periods.toLocaleString()}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Room-for-surprise units</span>
            <strong>{result.comparisonBudget.toFixed(1)}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Chance the next deal goes elsewhere</span>
            <strong>{((finalStep?.residualContestability ?? 0) * 100).toFixed(1)}%</strong>
          </div>
        </div>
        <div style={{ marginTop: "1rem" }}>
          <div className="stat-label">How much room is left for a surprise?</div>
          <div
            aria-label={`${((finalStep?.residualContestability ?? 0) * 100).toFixed(
              1,
            )} percent residual contestability`}
            style={{
              height: "0.7rem",
              overflow: "hidden",
              marginTop: "0.5rem",
              borderRadius: "999px",
              background: "var(--line)",
            }}
          >
            <div
              style={{
                width: `${(finalStep?.residualContestability ?? 0) * 100}%`,
                height: "100%",
                background: "var(--blue)",
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export function InformationChart() {
  const scenario = useMarketStore((state) => state.scenario);
  const result = useMemo(() => simulateScenario(scenario), [scenario]);
  const steps = sampledSteps(result.steps, 220);
  const width = 720;
  const height = 330;
  const maximum = Math.max(
    0.001,
    max(steps, (step) => Math.max(step.information, step.informationBound)) ?? 1,
  );
  const actualPath = pathForSeries(
    steps.map((step) => [step.t, step.information]),
    width,
    height,
    [0, maximum],
  );
  const boundPath = pathForSeries(
    steps.map((step) => [step.t, step.informationBound]),
    width,
    height,
    [0, maximum],
  );
  const last = result.steps.at(-1);
  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Model illustration</div>
          <strong>The record lengthens; marginal evidence fades.</strong>
        </div>
      </div>
      <div className="panel__body">
        <EquationMath
          latex="\operatorname{tr} I_t(\beta)=\operatorname{tr}\operatorname{Var}_{p_t(\beta)}(x_{J,t})\le D_X^2\varepsilon_t(\beta)"
          label="Fisher information trace bounded by squared design diameter times residual contestability"
        />
        <svg
          viewBox={`0 0 ${width} ${height + 35}`}
          role="img"
          aria-label="One-period Fisher information and its comparison-budget upper bound over time."
        >
          <title>Information absorption</title>
          <desc>
            Actual conditional information is shown in blue. The upper bound based on
            residual contestability is shown as a rust dashed line.
          </desc>
          <path d={boundPath} fill="none" stroke="var(--rust)" strokeWidth="2" strokeDasharray="7 7" />
          <path d={actualPath} fill="none" stroke="var(--blue)" strokeWidth="3" />
          <text className="chart-axis" x="0" y={height + 25}>
            early comparison
          </text>
          <text className="chart-axis" x={width} y={height + 25} textAnchor="end">
            inherited position
          </text>
        </svg>
        <div className="stats-grid">
          <div className="stat">
            <span className="stat-label">Latest information</span>
            <strong>{(last?.information ?? 0).toFixed(4)}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Latest upper bound</span>
            <strong>{(last?.informationBound ?? 0).toFixed(4)}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Cumulative information</span>
            <strong>{result.cumulativeInformation.toFixed(1)}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Cumulative comparison</span>
            <strong>{result.comparisonBudget.toFixed(1)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export function LikelihoodIllustration() {
  const scenario = useMarketStore((state) => state.scenario);
  const result = useMemo(() => simulateScenario(scenario), [scenario]);
  const steps = sampledSteps(result.steps, 140);
  const width = 650;
  const height = 250;
  const evidence = steps.map((step) => Math.min(1, step.comparisonBudget / 18));
  const first = steps.map(
    (step, index) => [step.t, 0.5 + evidence[index] * 0.31] as [number, number],
  );
  const second = steps.map(
    (step, index) => [step.t, 0.5 - evidence[index] * 0.19] as [number, number],
  );
  return (
    <div className="panel" style={{ background: "color-mix(in srgb, var(--paper) 7%, transparent)" }}>
      <div className="panel__header">
        <div>
          <div className="panel__meta">Illustration, not proof</div>
          <strong>Candidate explanations stop separating</strong>
        </div>
      </div>
      <div className="panel__body">
        <svg
          viewBox={`0 0 ${width} ${height + 32}`}
          role="img"
          aria-label="Illustrative likelihood weights for two candidate parameters move early and flatten later."
        >
          <title>Likelihood illustration</title>
          <desc>
            This isn’t a calculation of the theorem. It illustrates early updating followed
            by flattening as residual comparison disappears.
          </desc>
          <line className="chart-grid" x1="0" x2={width} y1={height / 2} y2={height / 2} />
          <path d={pathForSeries(first, width, height, [0, 1])} fill="none" stroke="var(--blue)" strokeWidth="3" />
          <path d={pathForSeries(second, width, height, [0, 1])} fill="none" stroke="var(--rust)" strokeWidth="3" />
          <text className="chart-axis" x={width - 6} y="44" textAnchor="end">
            β
          </text>
          <text className="chart-axis" x={width - 6} y={height - 30} textAnchor="end">
            β′
          </text>
          <text className="chart-axis" x="0" y={height + 24}>
            evidence arrives
          </text>
          <text className="chart-axis" x={width} y={height + 24} textAnchor="end">
            comparison fades
          </text>
        </svg>
      </div>
    </div>
  );
}

export function StrongReinforcementChart() {
  const scenario = useMarketStore((state) => state.scenario);
  const noFeedback = useMemo(
    () => simulateScenario({ ...scenario, rho: 0 }),
    [scenario],
  );
  const linear = useMemo(
    () => simulateScenario({ ...scenario, rho: 1 }),
    [scenario],
  );
  const strong = useMemo(
    () => simulateScenario({ ...scenario, rho: Math.max(1.6, scenario.rho) }),
    [scenario],
  );
  const experiments = [
    { label: "no snowball", result: noFeedback, color: "var(--blue)" },
    { label: "boundary", result: linear, color: "var(--shadow)" },
    { label: "strong snowball", result: strong, color: "var(--rust)" },
  ];
  const width = 720;
  const height = 330;
  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Sharp primitive case</div>
          <strong>Feedback changes the comparison trajectory</strong>
        </div>
      </div>
      <div className="panel__body">
        <div className="callout">
          Blue is a race with no snowball. Gray is the boundary case. Rust is a stronger
          snowball, where being ahead can quickly become the main reason for staying ahead.
        </div>
        <svg
          viewBox={`0 0 ${width} ${height + 34}`}
          role="img"
          aria-label="Cumulative comparison budgets for no, linear, and superlinear reinforcement."
        >
          <title>Reinforcement regimes</title>
          <desc>
            Stronger feedback typically slows the growth of the comparison budget in this
            seeded model illustration. Linear reinforcement is shown as a boundary, not as
            proof of finite comparison.
          </desc>
          {experiments.map(({ label, result, color }) => {
            const series = sampledSteps(result.steps).map(
              (step) => [step.t, step.comparisonBudget] as [number, number],
            );
            return (
              <g key={label}>
                <path
                  d={pathForSeries(series, width, height, [0, noFeedback.comparisonBudget])}
                  fill="none"
                  stroke={color}
                  strokeWidth="3"
                />
                <text
                  className="chart-axis"
                  x={width - 6}
                  y={
                    height -
                    (result.comparisonBudget / noFeedback.comparisonBudget) * height -
                    6
                  }
                  textAnchor="end"
                >
                  {label}
                </text>
              </g>
            );
          })}
          <text className="chart-axis" x="0" y={height + 25}>
            0
          </text>
          <text className="chart-axis" x={width} y={height + 25} textAnchor="end">
            {scenario.periods} transactions
          </text>
        </svg>
        <div className="callout">
          <strong>Boundary note.</strong> Linear preferential attachment can generate power
          laws without satisfying the exact finite-comparison condition.
        </div>
      </div>
    </div>
  );
}

export function ReplicationExperiment() {
  const scenario = useMarketStore((state) => state.scenario);
  const patchScenario = useMarketStore((state) => state.patchScenario);
  const totalBudget = Math.min(5_000, Math.max(500, scenario.periods * scenario.channels));
  const channels = Math.max(1, scenario.channels);
  const perChannel = Math.max(10, Math.floor(totalBudget / channels));

  const oneMarket = useMemo(
    () => simulateScenario({ ...scenario, channels: 1, periods: totalBudget }),
    [scenario, totalBudget],
  );
  const replicated = useMemo(
    () =>
      Array.from({ length: channels }, (_, index) =>
        simulateScenario({
          ...scenario,
          periods: perChannel,
          seed: deriveSeed(scenario.seed, index),
        }),
      ),
    [channels, perChannel, scenario],
  );
  const replicatedInformation = replicated.reduce(
    (sum, result) => sum + result.cumulativeInformation,
    0,
  );
  const replicatedBudget = replicated.reduce(
    (sum, result) => sum + result.comparisonBudget,
    0,
  );

  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Same transaction budget</div>
          <strong>Time extends a path. Replication reopens the laboratory.</strong>
        </div>
      </div>
      <div className="panel__body">
        <div className="control-grid">
          <RangeControl
            id="channels"
            label="Independent channels"
            min={1}
            max={100}
            value={channels}
            onChange={(value) => patchScenario({ channels: value })}
          />
          <RangeControl
            id="channel-periods"
            label="Transactions per channel"
            min={10}
            max={500}
            step={10}
            value={Math.min(500, perChannel)}
            onChange={(value) =>
              patchScenario({ periods: value, channels: Math.max(1, channels) })
            }
          />
        </div>
        <div className="experiment-grid" style={{ marginTop: "2rem" }}>
          <div className="scope-card">
            <div className="panel__meta">One inherited path</div>
            <h3>{totalBudget.toLocaleString()} transactions</h3>
            <p>One sequence of early shocks becomes the common position inherited later.</p>
            <div className="stats-grid">
              <div className="stat">
                <span className="stat-label">Learning signal</span>
                <strong>{oneMarket.cumulativeInformation.toFixed(1)}</strong>
              </div>
              <div className="stat">
                <span className="stat-label">Fresh-comparison units</span>
                <strong>{oneMarket.comparisonBudget.toFixed(1)}</strong>
              </div>
            </div>
          </div>
          <div className="scope-card">
            <div className="panel__meta">{channels} independent paths</div>
            <h3>{perChannel} transactions each</h3>
            <p>Inputs and parameters are shared; early allocation shocks are independently redrawn.</p>
            <div className="stats-grid">
              <div className="stat">
                <span className="stat-label">Combined learning signal</span>
                <strong>{replicatedInformation.toFixed(1)}</strong>
              </div>
              <div className="stat">
                <span className="stat-label">Fresh-comparison units</span>
                <strong>{replicatedBudget.toFixed(1)}</strong>
              </div>
            </div>
          </div>
        </div>
        <div className="callout" style={{ marginTop: "1rem" }}>
          Fresh starts create new head-to-head moments. Watching the same inherited lead for
          longer doesn’t.
        </div>
      </div>
    </div>
  );
}

type PolicyState = {
  multihoming: boolean;
  randomizedExposure: boolean;
  portability: boolean;
  reset: boolean;
  separation: boolean;
};

export function EpistemicMonopoly() {
  const scenario = useMarketStore((state) => state.scenario);
  const patchScenario = useMarketStore((state) => state.patchScenario);
  const [policies, setPolicies] = useState<PolicyState>({
    multihoming: false,
    randomizedExposure: scenario.exploration > 0,
    portability: false,
    reset: scenario.resetCadence > 0,
    separation: scenario.channels > 1,
  });
  const result = useMemo(() => simulateScenario(scenario), [scenario]);
  const last = result.steps.at(-1);
  const updatePolicy = (key: keyof PolicyState, enabled: boolean) => {
    const next = { ...policies, [key]: enabled };
    setPolicies(next);
    if (key === "randomizedExposure") patchScenario({ exploration: enabled ? 0.08 : 0 });
    if (key === "reset") patchScenario({ resetCadence: enabled ? 100 : 0 });
    if (key === "separation") patchScenario({ channels: enabled ? 6 : 1 });
    if (key === "multihoming") patchScenario({ channels: enabled ? Math.max(3, scenario.channels) : 1 });
    if (key === "portability" && enabled) {
      patchScenario({ initialPositions: scenario.initialPositions.map(() => 0) });
    }
  };

  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Routes to market</div>
          <strong>Many sellers can still produce one correlated history.</strong>
        </div>
      </div>
      <div className="panel__body">
        <svg
          viewBox="0 0 760 300"
          role="img"
          aria-label={`${scenario.n} sellers feed into ${scenario.channels} independent allocation channel${
            scenario.channels === 1 ? "" : "s"
          }.`}
        >
          <title>Epistemic channel structure</title>
          <desc>
            Sellers on the left route through shared or independent recommendation channels
            before reaching audiences on the right.
          </desc>
          {Array.from({ length: scenario.n }, (_, index) => {
            const y = 35 + index * (230 / Math.max(1, scenario.n - 1));
            const channelY =
              scenario.channels === 1
                ? 150
                : 55 + (index % Math.min(6, scenario.channels)) * 38;
            return (
              <g key={index}>
                <circle cx="65" cy={y} r="11" fill={AGENT_COLORS[index]} />
                <path
                  d={`M 78 ${y} C 220 ${y}, 235 ${channelY}, 350 ${channelY}`}
                  fill="none"
                  stroke="var(--line-strong)"
                  strokeWidth="2"
                />
                <path
                  d={`M 410 ${channelY} C 520 ${channelY}, 555 150, 690 150`}
                  fill="none"
                  stroke="var(--line-strong)"
                  strokeWidth="2"
                />
              </g>
            );
          })}
          {Array.from({ length: Math.min(6, scenario.channels) }, (_, index) => {
            const y = scenario.channels === 1 ? 150 : 55 + index * 38;
            return (
              <rect
                key={index}
                x="350"
                y={y - 16}
                width="60"
                height="32"
                rx="16"
                fill={scenario.channels === 1 ? "var(--rust)" : "var(--blue)"}
              />
            );
          })}
          <circle cx="710" cy="150" r="30" fill="var(--surface-solid)" stroke="var(--line-strong)" />
          <text className="chart-axis" x="710" y="154" textAnchor="middle">
            audience
          </text>
        </svg>
        <div className="policy-grid">
          {(
            [
              ["multihoming", "Multihoming"],
              ["randomizedExposure", "Randomized exposure"],
              ["portability", "Portability"],
              ["reset", "Periodic state reset"],
              ["separation", "Structural separation"],
            ] as Array<[keyof PolicyState, string]>
          ).map(([key, label]) => (
            <label className="toggle policy-card" key={key}>
              <input
                type="checkbox"
                checked={policies[key]}
                onChange={(event) => updatePolicy(key, event.target.checked)}
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
        <div style={{ marginTop: "1rem" }}>
          <RangeControl
            id="exploration"
            label="Exploration floor, η"
            min={0}
            max={0.25}
            step={0.01}
            value={scenario.exploration}
            format={(value) => value.toFixed(2)}
            onChange={(value) => {
              patchScenario({ exploration: value });
              setPolicies((current) => ({ ...current, randomizedExposure: value > 0 }));
            }}
            help="Policy intervention: this changes the baseline allocation process."
          />
        </div>
        <div className="callout">
          Randomized exposure deliberately gives alternatives another chance to be seen. That
          changes the market rule; it’s a policy choice, not a neutral measurement trick.
        </div>
        <div className="stats-grid">
          <div className="stat">
            <span className="stat-label">Chance next deal goes elsewhere</span>
            <strong>{((last?.residualContestability ?? 0) * 100).toFixed(1)}%</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Room-for-surprise total</span>
            <strong>{result.comparisonBudget.toFixed(1)}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">How one-sided the market became</span>
            <strong>{result.concentration.toFixed(2)}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Learning signal</span>
            <strong>{result.cumulativeInformation.toFixed(1)}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export function GaugeDemo() {
  const scenario = useMarketStore((state) => state.scenario);
  const [displacement, setDisplacement] = useState(0.45);
  const inputs = scenario.inputs.slice(0, Math.min(5, scenario.n));
  const beta = scenario.beta;
  const latent = inputs.map((_, index) => 0.2 - index * 0.04);
  const transformed = gaugeTransform(beta, latent, inputs, [displacement]);
  const baseProbabilities = allocationProbabilities({
    inputs,
    beta,
    counts: inputs.map(() => 0),
    initialPositions: inputs.map(() => 0),
    baseline: 1,
    rho: 0,
    latentPositions: latent,
  });
  const transformedProbabilities = allocationProbabilities({
    inputs,
    beta: transformed.beta,
    counts: inputs.map(() => 0),
    initialPositions: inputs.map(() => 0),
    baseline: 1,
    rho: 0,
    latentPositions: transformed.positions,
  });
  const maximumDifference = Math.max(
    ...baseProbabilities.map((value, index) =>
      Math.abs(value - transformedProbabilities[index]),
    ),
  );

  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Exact point non-identification</div>
          <strong>The market observes the composite index.</strong>
        </div>
      </div>
      <div className="panel__body">
        <RangeControl
          id="gauge-d"
          label="Latent-position transformation, d"
          min={-1}
          max={1}
          step={0.01}
          value={displacement}
          format={(value) => value.toFixed(2)}
          onChange={setDisplacement}
        />
        <EquationMath
          latex="\beta^{(d)}=\beta+d,\qquad \lambda_i^{(d)}=\lambda_i-x_i^\top d"
          label="Contribution position gauge transformation"
        />
        <EquationMath
          latex="x_i^\top\beta^{(d)}+\lambda_i^{(d)}=x_i^\top\beta+\lambda_i"
          label="Composite index invariance identity"
        />
        <div className="experiment-grid">
          <div className="scope-card">
            <div className="panel__meta">Economy A</div>
            <h3>More direct contribution</h3>
            <p>
              β = {transformed.beta[0].toFixed(2)}; inherited position moves in the
              opposite direction.
            </p>
          </div>
          <div className="scope-card">
            <div className="panel__meta">Economy B</div>
            <h3>Less direct contribution</h3>
            <p>β = {beta[0].toFixed(2)}; inherited position carries more of the index.</p>
          </div>
        </div>
        <div className="stats-grid">
          <div className="stat">
            <span className="stat-label">Largest probability difference</span>
            <strong>{maximumDifference.toExponential(1)}</strong>
          </div>
          <div className="stat">
            <span className="stat-label">Observed history</span>
            <strong>identical</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export function TaxDemo() {
  const [contribution, setContribution] = useState(64);
  const reward = 100;
  const alternativeContribution = 42;
  const residualA = reward - contribution;
  const residualB = reward - alternativeContribution;
  return (
    <div className="panel">
      <div className="panel__header">
        <div>
          <div className="panel__meta">Merit-sensitive taxation</div>
          <strong>One observable record; two structural residuals.</strong>
        </div>
      </div>
      <div className="panel__body">
        <RangeControl
          id="tax-contribution"
          label="Contribution assigned in Economy A"
          min={20}
          max={90}
          value={contribution}
          format={(value) => `${value}`}
          onChange={setContribution}
        />
        <EquationMath
          latex="R_i(\theta,O)=Y_i(O)-C_i(\theta,O)\qquad \tau_i(O)=R_i(\theta,O)"
          label="Positional rent and merit separating tax condition"
        />
        <div className="experiment-grid">
          <div className="scope-card">
            <div className="panel__meta">Economy A</div>
            <h3>Y {reward} − C {contribution} = R {residualA}</h3>
            <p>Higher assigned contribution leaves a smaller positional residual.</p>
          </div>
          <div className="scope-card">
            <div className="panel__meta">Economy B</div>
            <h3>Y {reward} − C {alternativeContribution} = R {residualB}</h3>
            <p>The same observable reward supports a different structural decomposition.</p>
          </div>
        </div>
        <div className="callout" style={{ marginTop: "1rem" }}>
          <strong>One number can’t equal both residuals.</strong> The result doesn’t imply
          that all high income is rent, that risk-bearing is fictitious, or that an optimal
          tax rate follows from the theorem.
        </div>
      </div>
    </div>
  );
}

export function ClosingBranches() {
  return (
    <svg
      viewBox="0 0 900 360"
      role="img"
      aria-label="One realized history remains bright while surrounding shadow futures reappear."
      style={{ width: "100%", height: "auto" }}
    >
      <title>Realized history and shadow futures</title>
      <desc>
        A bright realized branch is surrounded by muted possible branches that weren’t
        observed but would have provided comparison.
      </desc>
      {Array.from({ length: 13 }, (_, index) => {
        const offset = (index - 6) * 26;
        const realized = index === 7;
        return (
          <motion.path
            key={index}
            d={`M 40 180 C 260 180, 390 ${180 + offset * 0.25}, 850 ${180 + offset}`}
            fill="none"
            stroke={realized ? "var(--rust)" : "var(--shadow)"}
            strokeWidth={realized ? 5 : 1.4}
            strokeOpacity={realized ? 1 : 0.36}
            strokeDasharray={realized ? undefined : "5 8"}
            initial={{ pathLength: realized ? 1 : 0, opacity: realized ? 1 : 0 }}
            whileInView={{ pathLength: 1, opacity: realized ? 1 : 0.36 }}
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 1.2, delay: index * 0.04 }}
          />
        );
      })}
    </svg>
  );
}

export function ConcentrationSummary({ result }: { result: SimulationResult }) {
  const shares = result.finalCounts.map(
    (count) => count / Math.max(1, result.scenario.periods),
  );
  return (
    <div className="stats-grid">
      <div className="stat">
        <span className="stat-label">Concentration</span>
        <strong>{herfindahl(shares).toFixed(2)}</strong>
      </div>
      <div className="stat">
        <span className="stat-label">Winner</span>
        <strong>{String.fromCharCode(65 + result.winner)}</strong>
      </div>
    </div>
  );
}
