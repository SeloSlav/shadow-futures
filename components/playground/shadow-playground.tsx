"use client";

import {
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { RangeControl } from "@/components/ui/range-control";
import { Math as EquationMath } from "@/components/ui/math";
import { DEFAULT_SCENARIO, simulateScenario } from "@/lib/model/simulation";
import type { AllocationStep, SimulationResult } from "@/lib/model/types";

const COMPETITORS = [
  { short: "A", name: "Aster", color: "#78f4e5" },
  { short: "B", name: "Bramble", color: "#84b8ff" },
  { short: "C", name: "Cinder", color: "#a997ff" },
  { short: "D", name: "Dune", color: "#f3a6dd" },
  { short: "E", name: "Elm", color: "#f2c879" },
];

const INPUTS = [[0.78], [0.7], [0.62], [0.56], [0.5]];
const PERIODS = 360;
const MAX_RESIDUAL = 1 - 1 / COMPETITORS.length;
const TERRAIN_HALF_WIDTH = 2.7;
const TERRAIN_HALF_DEPTH = 3.25;
const PAN_SCREEN_RATIO = 0.8;

const PRESETS = [
  {
    name: "Competitive Market",
    description: "Open entry; contribution decides the contest.",
    beta: 1,
    rho: 0,
    exploration: 0,
  },
  {
    name: "Big Tech",
    description: "Scale and an early lead compound into dominance.",
    beta: 1,
    rho: 1.65,
    exploration: 0,
  },
  {
    name: "Big Tech with Regulation",
    description: "Rules curb lock-in without erasing scale.",
    beta: 1,
    rho: 0.9,
    exploration: 0.08,
  },
] as const;

type ViewState = {
  yaw: number;
  tilt: number;
  zoom: number;
  targetX: number;
  targetZ: number;
};

type ProjectedPoint = {
  x: number;
  y: number;
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function wrapAngle(value: number) {
  const fullTurn = Math.PI * 2;
  return ((((value + Math.PI) % fullTurn) + fullTurn) % fullTurn) - Math.PI;
}

function heightAt(step: AllocationStep, worldX: number) {
  return step.probabilities.reduce((height, probability, index) => {
    const competitorX = index - (COMPETITORS.length - 1) / 2;
    const distance = worldX - competitorX;
    return height + probability * Math.exp(-(distance * distance) / 0.2) * 1.48;
  }, 0);
}

function sampleSteps(result: SimulationResult, cursor: number, maximum = 54) {
  const last = Math.min(cursor, result.steps.length - 1);
  const count = Math.min(maximum, last + 1);
  if (count <= 1) return [result.steps[0]];

  return Array.from({ length: count }, (_, index) => {
    const stepIndex = Math.round((index / (count - 1)) * last);
    return result.steps[stepIndex];
  });
}

function contestStatus(residual: number) {
  if (residual > 0.58) return "The contest is open";
  if (residual > 0.34) return "The field is narrowing";
  if (residual > 0.14) return "Shadow futures are fading";
  return "The market is nearly locked";
}

function giniCoefficient(values: number[]) {
  const total = values.reduce((sum, value) => sum + value, 0);
  if (total === 0) return 0;

  const pairwiseDifference = values.reduce(
    (outerSum, left) =>
      outerSum +
      values.reduce((innerSum, right) => innerSum + Math.abs(left - right), 0),
    0,
  );

  return pairwiseDifference / (2 * values.length * total);
}

function ShadowSurface({
  result,
  cursor,
  showShadows,
}: {
  result: SimulationResult;
  cursor: number;
  showShadows: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{
    pointerId: number;
    x: number;
    y: number;
    mode: "orbit" | "pan";
  } | null>(null);
  const [view, setView] = useState<ViewState>({
    yaw: -0.08,
    tilt: 0.52,
    zoom: 1,
    targetX: 0,
    targetZ: 0,
  });
  const renderStateRef = useRef({ result, cursor, showShadows, view });

  useEffect(() => {
    renderStateRef.current = { result, cursor, showShadows, view };
  }, [cursor, result, showShadows, view]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let width = 0;
    let height = 0;
    let frame = 0;

    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      width = Math.max(1, bounds.width);
      height = Math.max(1, bounds.height);
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    const draw = (now: number) => {
      const { result, cursor, showShadows, view } = renderStateRef.current;
      context.clearRect(0, 0, width, height);

      const background = context.createRadialGradient(
        width * 0.58,
        height * 0.4,
        20,
        width * 0.58,
        height * 0.4,
        Math.max(width, height) * 0.78,
      );
      background.addColorStop(0, "#142638");
      background.addColorStop(0.42, "#080f1d");
      background.addColorStop(1, "#04070e");
      context.fillStyle = background;
      context.fillRect(0, 0, width, height);

      for (let index = 0; index < 44; index += 1) {
        const x = ((Math.sin(index * 91.17) + 1) / 2) * width;
        const y = ((Math.cos(index * 47.31) + 1) / 2) * height * 0.8;
        const pulse = 0.2 + 0.15 * Math.sin(now / 1100 + index);
        context.fillStyle = `rgba(148, 211, 226, ${pulse})`;
        context.fillRect(x, y, index % 5 === 0 ? 1.5 : 1, index % 5 === 0 ? 1.5 : 1);
      }

      const baseScale = Math.min(width / 7.9, height / 6.7) * view.zoom;
      const centerX = width * (width < 700 ? 0.5 : 0.54);
      const centerY = height * 0.72;
      const cosine = Math.cos(view.yaw);
      const sine = Math.sin(view.yaw);

      const project = (worldX: number, worldY: number, worldZ: number): ProjectedPoint => {
        const relativeX = worldX - view.targetX;
        const relativeZ = worldZ - view.targetZ;
        const rotatedX = relativeX * cosine - relativeZ * sine;
        const rotatedZ = relativeX * sine + relativeZ * cosine;
        const perspective = clamp(1 - rotatedZ * 0.055, 0.68, 1.34);
        return {
          x: centerX + rotatedX * baseScale * perspective,
          y:
            centerY -
            rotatedZ * baseScale * Math.sin(view.tilt) * 0.82 -
            worldY * baseScale * Math.cos(view.tilt) * perspective,
        };
      };

      const planeCorners = [
        project(-2.7, 0, 3.25),
        project(2.7, 0, 3.25),
        project(2.7, 0, -3.25),
        project(-2.7, 0, -3.25),
      ];
      context.beginPath();
      context.moveTo(planeCorners[0].x, planeCorners[0].y);
      planeCorners.slice(1).forEach((point) => context.lineTo(point.x, point.y));
      context.closePath();
      const planeGradient = context.createLinearGradient(0, height * 0.1, 0, height);
      planeGradient.addColorStop(0, "rgba(43, 74, 101, 0.05)");
      planeGradient.addColorStop(1, "rgba(30, 76, 93, 0.24)");
      context.fillStyle = planeGradient;
      context.fill();
      context.strokeStyle = "rgba(130, 205, 221, 0.16)";
      context.lineWidth = 1;
      context.stroke();

      for (let grid = 0; grid <= 12; grid += 1) {
        const z = 3.2 - (grid / 12) * 6.4;
        const left = project(-2.7, 0, z);
        const right = project(2.7, 0, z);
        context.beginPath();
        context.moveTo(left.x, left.y);
        context.lineTo(right.x, right.y);
        context.strokeStyle =
          grid % 3 === 0 ? "rgba(113, 198, 217, 0.16)" : "rgba(113, 198, 217, 0.07)";
        context.stroke();
      }

      for (let grid = -2; grid <= 2; grid += 1) {
        const far = project(grid, 0, 3.2);
        const near = project(grid, 0, -3.2);
        context.beginPath();
        context.moveTo(far.x, far.y);
        context.lineTo(near.x, near.y);
        context.strokeStyle = "rgba(113, 198, 217, 0.09)";
        context.stroke();
      }

      const sampled = sampleSteps(result, cursor);
      const surfaceColumns = 42;
      const surfaceRows = sampled.map((step, rowIndex) => {
        const progress = sampled.length <= 1 ? 1 : rowIndex / (sampled.length - 1);
        const z = 3.15 - progress * 6.3;
        return Array.from({ length: surfaceColumns }, (_, columnIndex) => {
          const x = -2.55 + (columnIndex / (surfaceColumns - 1)) * 5.1;
          return {
            x,
            height: heightAt(step, x),
            point: project(x, heightAt(step, x), z),
          };
        });
      });

      for (let row = 0; row < surfaceRows.length - 1; row += 1) {
        const rowProgress = row / Math.max(1, surfaceRows.length - 1);
        for (let column = 0; column < surfaceColumns - 1; column += 1) {
          const a = surfaceRows[row][column];
          const b = surfaceRows[row][column + 1];
          const c = surfaceRows[row + 1][column + 1];
          const d = surfaceRows[row + 1][column];
          const averageHeight = (a.height + b.height + c.height + d.height) / 4;
          const hue = 188 + 94 * clamp(averageHeight / 0.62, 0, 1);
          context.beginPath();
          context.moveTo(a.point.x, a.point.y);
          context.lineTo(b.point.x, b.point.y);
          context.lineTo(c.point.x, c.point.y);
          context.lineTo(d.point.x, d.point.y);
          context.closePath();
          context.fillStyle = `hsla(${hue}, 78%, ${42 + averageHeight * 22}%, ${
            0.035 + rowProgress * 0.075 + averageHeight * 0.08
          })`;
          context.fill();
        }
      }

      surfaceRows.forEach((row, rowIndex) => {
        if (rowIndex % 2 !== 0 && rowIndex !== surfaceRows.length - 1) return;
        context.beginPath();
        row.forEach(({ point }, index) => {
          if (index === 0) context.moveTo(point.x, point.y);
          else context.lineTo(point.x, point.y);
        });
        const isLatest = rowIndex === surfaceRows.length - 1;
        context.strokeStyle = isLatest
          ? "rgba(119, 247, 231, 0.9)"
          : `rgba(105, 193, 224, ${0.08 + (rowIndex / surfaceRows.length) * 0.2})`;
        context.lineWidth = isLatest ? 2 : 0.8;
        context.shadowColor = isLatest ? "#6debdc" : "transparent";
        context.shadowBlur = isLatest ? 12 : 0;
        context.stroke();
        context.shadowBlur = 0;
      });

      const pathSteps = sampleSteps(result, cursor, 88);
      context.beginPath();
      pathSteps.forEach((step, index) => {
        const progress = pathSteps.length <= 1 ? 1 : index / (pathSteps.length - 1);
        const z = 3.15 - progress * 6.3;
        const x = step.recipient - (COMPETITORS.length - 1) / 2;
        const point = project(x, step.probabilities[step.recipient] * 1.48 + 0.08, z);
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.strokeStyle = "#b9fff4";
      context.lineWidth = 2.4;
      context.shadowColor = "#5ff6e1";
      context.shadowBlur = 16;
      context.stroke();
      context.shadowBlur = 0;

      if (showShadows && sampled.length > 2) {
        const branchEvery = Math.max(2, Math.floor(sampled.length / 13));
        sampled.forEach((step, rowIndex) => {
          if (rowIndex % branchEvery !== 0 || rowIndex === sampled.length - 1) return;
          const progress = rowIndex / Math.max(1, sampled.length - 1);
          const nextProgress = Math.min(1, progress + 0.08);
          const sourceX = step.recipient - (COMPETITORS.length - 1) / 2;
          const source = project(
            sourceX,
            step.probabilities[step.recipient] * 1.48 + 0.06,
            3.15 - progress * 6.3,
          );

          step.probabilities.forEach((probability, competitorIndex) => {
            if (competitorIndex === step.recipient) return;
            const targetX = competitorIndex - (COMPETITORS.length - 1) / 2;
            const target = project(
              targetX,
              probability * 1.48 + 0.05,
              3.15 - nextProgress * 6.3,
            );
            const alpha = clamp(probability * 1.7, 0.025, 0.34);
            const midpointX = (source.x + target.x) / 2;
            const midpointY = Math.min(source.y, target.y) - 12 - probability * 30;
            context.beginPath();
            context.moveTo(source.x, source.y);
            context.quadraticCurveTo(midpointX, midpointY, target.x, target.y);
            context.setLineDash([3, 6]);
            context.strokeStyle = `rgba(197, 176, 255, ${alpha})`;
            context.lineWidth = 1;
            context.stroke();
            context.setLineDash([]);

            const particleProgress = (now / 1900 + rowIndex * 0.17 + competitorIndex * 0.23) % 1;
            const inverse = 1 - particleProgress;
            const particleX =
              inverse * inverse * source.x +
              2 * inverse * particleProgress * midpointX +
              particleProgress * particleProgress * target.x;
            const particleY =
              inverse * inverse * source.y +
              2 * inverse * particleProgress * midpointY +
              particleProgress * particleProgress * target.y;
            context.beginPath();
            context.arc(particleX, particleY, 1.2 + probability * 2.2, 0, Math.PI * 2);
            context.fillStyle = `rgba(221, 208, 255, ${alpha * (1 - particleProgress)})`;
            context.fill();
          });
        });
      }

      const budgetPath = sampleSteps(result, cursor, 70);
      context.beginPath();
      budgetPath.forEach((step, index) => {
        const progress = budgetPath.length <= 1 ? 1 : index / (budgetPath.length - 1);
        const possible = Math.max(1, step.t * MAX_RESIDUAL);
        const normalizedBudget = clamp(step.comparisonBudget / possible, 0, 1);
        const point = project(2.76, 0.08 + normalizedBudget * 0.62, 3.15 - progress * 6.3);
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.strokeStyle = "rgba(245, 190, 105, 0.9)";
      context.lineWidth = 2;
      context.shadowColor = "#f1b85c";
      context.shadowBlur = 10;
      context.stroke();
      context.shadowBlur = 0;

      const latestStep = result.steps[Math.min(cursor, result.steps.length - 1)];
      COMPETITORS.forEach((competitor, index) => {
        const worldX = index - (COMPETITORS.length - 1) / 2;
        const point = project(worldX, latestStep.probabilities[index] * 1.48 + 0.02, -3.18);
        context.beginPath();
        context.arc(point.x, point.y, 8, 0, Math.PI * 2);
        context.fillStyle = `${competitor.color}22`;
        context.fill();
        context.strokeStyle = competitor.color;
        context.lineWidth = 1.2;
        context.stroke();
        context.fillStyle = "#eaf7f8";
        context.font = "600 10px Inter, Segoe UI, sans-serif";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(competitor.short, point.x, point.y + 0.5);
      });

      frame = window.requestAnimationFrame(draw);
    };

    frame = window.requestAnimationFrame(draw);
    return () => {
      observer.disconnect();
      window.cancelAnimationFrame(frame);
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      setView((current) => ({
        ...current,
        zoom: clamp(current.zoom - event.deltaY * 0.0009, 0.42, 3.6),
      }));
    };

    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, []);

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0 && event.button !== 2) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      mode: event.button === 2 ? "pan" : "orbit",
    };
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    dragRef.current = { ...drag, x: event.clientX, y: event.clientY };

    if (drag.mode === "pan") {
      const bounds = event.currentTarget.getBoundingClientRect();
      const viewportScale = Math.max(1, Math.min(bounds.width / 7.9, bounds.height / 6.7));
      setView((current) => {
        const panScale = PAN_SCREEN_RATIO / (viewportScale * current.zoom);
        const panRight = -deltaX * panScale;
        const panForward =
          (deltaY * panScale) / Math.max(0.2, Math.sin(current.tilt) * 0.82);
        const cosine = Math.cos(current.yaw);
        const sine = Math.sin(current.yaw);

        return {
          ...current,
          targetX: clamp(
            current.targetX + cosine * panRight + sine * panForward,
            -TERRAIN_HALF_WIDTH,
            TERRAIN_HALF_WIDTH,
          ),
          targetZ: clamp(
            current.targetZ - sine * panRight + cosine * panForward,
            -TERRAIN_HALF_DEPTH,
            TERRAIN_HALF_DEPTH,
          ),
        };
      });
      return;
    }

    setView((current) => ({
      ...current,
      yaw: wrapAngle(current.yaw + deltaX * 0.006),
      tilt: clamp(current.tilt + deltaY * 0.0045, 0.12, 1.36),
    }));
  };

  const releasePointer = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
    }
  };

  return (
    <canvas
      ref={canvasRef}
      className="playground-surface"
      data-testid="playground-surface"
      data-view={`${view.yaw.toFixed(2)}:${view.tilt.toFixed(2)}:${view.zoom.toFixed(2)}:${view.targetX.toFixed(2)}:${view.targetZ.toFixed(2)}`}
      aria-label="Interactive three-dimensional probability terrain. Left-drag to orbit fully around it, right-drag to pan, and use the wheel to zoom. The bright line is the one realized allocation history. Purple branches are unrealized shadow futures and fade as nonleader probability disappears. The amber edge tracks the comparison budget."
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={releasePointer}
      onPointerCancel={releasePointer}
      onContextMenu={(event) => event.preventDefault()}
    />
  );
}

export function ShadowPlayground() {
  const [beta, setBeta] = useState(1);
  const [rho, setRho] = useState(1.65);
  const [exploration, setExploration] = useState(0);
  const [seed, setSeed] = useState(42);
  const [speed, setSpeed] = useState(2);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [showShadows, setShowShadows] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);

  const result = useMemo(
    () =>
      simulateScenario({
        ...DEFAULT_SCENARIO,
        name: "Comparison playground",
        n: COMPETITORS.length,
        inputs: INPUTS,
        // Cinder starts with an installed-base advantage despite being only the
        // third-highest contributor. This lets the lab distinguish productive
        // scale from incumbent lock-in.
        initialPositions: [0, 0, 1, 0, 0],
        beta: [beta],
        rho,
        exploration,
        periods: PERIODS,
        seed,
      }),
    [beta, exploration, rho, seed],
  );

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => {
      setReducedMotion(media.matches);
      setCursor(media.matches ? PERIODS - 1 : 0);
      setPlaying(!media.matches);
    };
    const initialUpdate = window.setTimeout(update, 0);
    media.addEventListener("change", update);
    return () => {
      window.clearTimeout(initialUpdate);
      media.removeEventListener("change", update);
    };
  }, []);

  useEffect(() => {
    if (!playing || reducedMotion) return;
    let frame = 0;
    let previous = performance.now();
    let accumulatedRounds = 0;

    const advance = (now: number) => {
      accumulatedRounds += ((now - previous) / 1000) * speed * 22;
      previous = now;
      const wholeRounds = Math.floor(accumulatedRounds);
      if (wholeRounds > 0) {
        accumulatedRounds -= wholeRounds;
        setCursor((current) => {
          const next = Math.min(PERIODS - 1, current + wholeRounds);
          if (next === PERIODS - 1) setPlaying(false);
          return next;
        });
      }
      frame = window.requestAnimationFrame(advance);
    };

    frame = window.requestAnimationFrame(advance);
    return () => window.cancelAnimationFrame(frame);
  }, [playing, reducedMotion, speed]);

  const currentStep = result.steps[Math.min(cursor, result.steps.length - 1)];
  const comparisonYield =
    currentStep.comparisonBudget / Math.max(1, currentStep.t * MAX_RESIDUAL);
  const visibleSteps = result.steps.slice(0, Math.min(cursor + 1, result.steps.length));
  const bestContribution = Math.max(...INPUTS.map((input) => input[0] ?? 0));
  const productiveWealth =
    visibleSteps.reduce(
      (sum, step) => sum + (INPUTS[step.recipient]?.[0] ?? 0),
      0,
    ) / Math.max(Number.EPSILON, currentStep.t * bestContribution);
  const rewardInequality = giniCoefficient(currentStep.counts);

  const restartPlayback = () => {
    setCursor(reducedMotion ? PERIODS - 1 : 0);
    setPlaying(!reducedMotion);
  };

  const applyPreset = (preset: (typeof PRESETS)[number]) => {
    setBeta(preset.beta);
    setRho(preset.rho);
    setExploration(preset.exploration);
    restartPlayback();
  };

  const relaunch = () => {
    setSeed((current) => current + 1);
    restartPlayback();
  };

  return (
    <main className="playground-page" id="main-content">
      <header className="playground-intro">
        <div>
          <p className="playground-intro__eyebrow">
            Interactive equation lab <span>Live model</span>
          </p>
          <h1>The Comparison Playground</h1>
          <p className="playground-intro__dek">
            Compare a competitive-market benchmark with platform scale—and see when competition
            rules can raise productive wealth while reducing inequality.
          </p>
        </div>
        <div className="playground-equation" aria-label="The equation driving the playground">
          <EquationMath
            latex="s_{it}=e^{\beta x_i}(a+N_i(t))^\rho \;\to\; p_{it}=\frac{s_{it}}{\sum_j s_{jt}}"
            label="Allocation score becomes allocation probability"
          />
          <EquationMath
            latex="\varepsilon_t=1-\max_i p_{it} \;\to\; B_T=\sum_{t<T}\varepsilon_t"
            label="Residual contestability accumulates into the comparison budget"
          />
        </div>
      </header>

      <div className="playground-lab">
        <aside className="playground-controls" aria-label="Playground controls">
          <div className="playground-control-section">
            <div className="playground-control-heading">
              <span>Scenario</span>
              <small>Choose a starting point</small>
            </div>
            <div className="playground-presets">
              {PRESETS.map((preset) => {
                const selected =
                  beta === preset.beta &&
                  rho === preset.rho &&
                  exploration === preset.exploration;
                return (
                  <button
                    type="button"
                    key={preset.name}
                    data-selected={selected}
                    aria-pressed={selected}
                    onClick={() => applyPreset(preset)}
                  >
                    <span>{preset.name}</span>
                    <small>{preset.description}</small>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="playground-control-section playground-control-section--sliders">
            <div className="playground-control-heading">
              <span>Equation controls</span>
              <small>Change the market’s rules</small>
            </div>
            <RangeControl
              id="playground-beta"
              label="Contribution weight β"
              value={beta}
              min={0}
              max={2}
              step={0.05}
              format={(value) => value.toFixed(2)}
              onChange={(value) => {
                setBeta(value);
                restartPlayback();
              }}
              help="How strongly verified input changes the score."
            />
            <RangeControl
              id="playground-rho"
              label="Feedback strength ρ"
              value={rho}
              min={0}
              max={2.4}
              step={0.05}
              format={(value) => value.toFixed(2)}
              onChange={(value) => {
                setRho(value);
                restartPlayback();
              }}
              help="How strongly past wins produce future advantage."
            />
            <RangeControl
              id="playground-exploration"
              label="Reserved discovery η"
              value={exploration}
              min={0}
              max={0.3}
              step={0.01}
              format={(value) => `${Math.round(value * 100)}%`}
              onChange={(value) => {
                setExploration(value);
                restartPlayback();
              }}
              help="Exposure kept aside for alternatives."
            />
          </div>

          <div className="playground-control-section playground-actions">
            <div className="playground-speed" aria-label="Simulation speed">
              {[1, 2, 4].map((value) => (
                <button
                  type="button"
                  key={value}
                  data-selected={speed === value}
                  aria-pressed={speed === value}
                  onClick={() => setSpeed(value)}
                >
                  {value}×
                </button>
              ))}
            </div>
            <div className="playground-action-row">
              <button
                className="playground-action-button playground-action-button--primary"
                type="button"
                onClick={() => {
                  if (cursor === PERIODS - 1) setCursor(0);
                  setPlaying((current) => !current);
                }}
              >
                <span aria-hidden="true">{playing ? "Ⅱ" : "▶"}</span>
                {playing ? "Pause" : cursor === PERIODS - 1 ? "Replay" : "Continue"}
              </button>
              <button className="playground-action-button" type="button" onClick={relaunch}>
                <span aria-hidden="true">↻</span> New history
              </button>
            </div>
            <label className="playground-shadow-toggle">
              <input
                type="checkbox"
                checked={showShadows}
                onChange={(event) => setShowShadows(event.target.checked)}
              />
              <span>
                Show shadow futures
                <small>Unrealized paths that could’ve supplied comparison</small>
              </span>
            </label>
          </div>
        </aside>

        <section className="playground-stage" aria-labelledby="playground-stage-title">
          <div className="playground-stage__top">
            <div>
              <span className="playground-stage__live">
                <i aria-hidden="true" /> Round {currentStep.t} of {PERIODS}
              </span>
              <h2 id="playground-stage-title">{contestStatus(currentStep.residualContestability)}</h2>
            </div>
            <div className="playground-stage__help">
              <span>left drag</span> orbit · <span>right drag</span> pan ·{" "}
              <span>wheel</span> zoom
            </div>
          </div>

          <div className="playground-stage__canvas">
            <ShadowSurface result={result} cursor={cursor} showShadows={showShadows} />
            <div className="playground-stage__legend" aria-label="Visual key">
              <span>
                <i className="playground-legend-line playground-legend-line--observed" />
                Realized history
              </span>
              <span>
                <i className="playground-legend-line playground-legend-line--shadow" />
                Shadow futures
              </span>
              <span>
                <i className="playground-legend-line playground-legend-line--budget" />
                Comparison
              </span>
            </div>
          </div>

          <div className="playground-timeline">
            <label htmlFor="playground-round">
              <span>Early contest</span>
              <span>Reinforced history</span>
            </label>
            <input
              id="playground-round"
              type="range"
              min={0}
              max={PERIODS - 1}
              value={cursor}
              onChange={(event) => {
                setPlaying(false);
                setCursor(Number(event.target.value));
              }}
            />
          </div>

          <div className="playground-readout">
            <article className="playground-readout__outcome playground-readout__wealth">
              <span className="playground-readout__label">Productive wealth (proxy)</span>
              <div className="playground-readout__metric">
                <strong>{Math.round(productiveWealth * 100)}%</strong>
                <span>of best feasible output</span>
              </div>
              <div className="playground-outcome-bar" aria-hidden="true">
                <span style={{ width: `${clamp(productiveWealth * 100, 0, 100)}%` }} />
              </div>
              <p>Contribution-weighted value of the opportunities allocated so far.</p>
            </article>

            <article className="playground-readout__shadows">
              <span className="playground-readout__label">Competition preserved</span>
              <div className="playground-orbit-meter">
                <span style={{ "--meter": comparisonYield } as CSSProperties} />
                <strong>{Math.round(comparisonYield * 100)}%</strong>
              </div>
              <p>
                Of an ideal open benchmark’s cumulative comparison opportunities.
              </p>
            </article>

            <article className="playground-readout__outcome playground-readout__inequality">
              <span className="playground-readout__label">Reward inequality</span>
              <div className="playground-readout__metric">
                <strong>{rewardInequality.toFixed(2)}</strong>
                <span>Gini coefficient</span>
              </div>
              <div className="playground-outcome-bar" aria-hidden="true">
                <span
                  style={{
                    width: `${clamp((rewardInequality / MAX_RESIDUAL) * 100, 0, 100)}%`,
                  }}
                />
              </div>
              <p>0 is equal; 0.80 means one firm receives every reward.</p>
            </article>
          </div>
        </section>
      </div>

      <section className="playground-explainer" aria-labelledby="playground-explainer-title">
        <div>
          <span>How to read the world</span>
          <h2 id="playground-explainer-title">Less inequality need not mean less wealth.</h2>
        </div>
        <ol>
          <li>
            <span>01</span>
            <p>
              <strong>The colored terrain is the allocation rule.</strong> Higher peaks mean a
              competitor is more likely to receive the next opportunity.
            </p>
          </li>
          <li>
            <span>02</span>
            <p>
              <strong>The bright path is the one history we observe.</strong> Each win changes
              the next surface through accumulated position <EquationMath latex="N_i(t)" block={false} />.
            </p>
          </li>
          <li>
            <span>03</span>
            <p>
              <strong>The purple branches are shadow futures.</strong> As their probability
              fades, the market loses the comparisons that could separate contribution from
              reinforced position.
            </p>
          </li>
          <li>
            <span>04</span>
            <p>
              <strong>Competition policy is not only redistribution.</strong> When it keeps a
              merely early incumbent from blocking stronger alternatives, productive wealth can
              rise while reward inequality falls. Regulation approaches the open benchmark, but
              natural scale means it never fully recreates it.
            </p>
          </li>
        </ol>
      </section>
    </main>
  );
}
