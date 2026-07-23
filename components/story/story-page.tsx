"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import {
  BudgetChart,
  ClosingBranches,
  EpistemicMonopoly,
  GaugeDemo,
  HeroNetwork,
  InformationChart,
  LikelihoodIllustration,
  ReplicationExperiment,
  RewardRewrite,
  ShadowMap,
  SideBySideMarkets,
  StrongReinforcementChart,
  TaxDemo,
} from "@/components/story/visuals";
import { Math as EquationMath } from "@/components/ui/math";
import { RangeControl } from "@/components/ui/range-control";
import { PAPER } from "@/lib/paper/citation";
import { simulateScenario } from "@/lib/model/simulation";
import { SCENARIO_PRESETS } from "@/lib/scenarios/presets";
import {
  scenarioFromSearch,
  serializeScenario,
} from "@/lib/scenarios/serialization";
import { useMarketStore } from "@/lib/store/market-store";

function downloadText(filename: string, contents: string, type: string) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function ChapterIntro({
  number,
  title,
  children,
}: {
  number: string;
  title: string;
  children: React.ReactNode;
}) {
  const chapterMatch = number.match(/^\d+/u);
  const headingId =
    number === "Scope and limits"
      ? "scope-title"
      : chapterMatch
        ? `chapter-${Number(chapterMatch[0])}`
        : undefined;
  return (
    <div className="chapter__intro">
      <span className="chapter__number">{number}</span>
      <h2 id={headingId}>{title}</h2>
      <p>{children}</p>
    </div>
  );
}

export function MarketToolbar() {
  const scenario = useMarketStore((state) => state.scenario);
  const activeStep = useMarketStore((state) => state.activeStep);
  const playing = useMarketStore((state) => state.playing);
  const seedLocked = useMarketStore((state) => state.seedLocked);
  const setActiveStep = useMarketStore((state) => state.setActiveStep);
  const setPlaying = useMarketStore((state) => state.setPlaying);
  const toggleSeedLock = useMarketStore((state) => state.toggleSeedLock);
  const rerun = useMarketStore((state) => state.rerun);
  const reset = useMarketStore((state) => state.reset);
  const [notice, setNotice] = useState("");
  const result = useMemo(() => simulateScenario(scenario), [scenario]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      const current = useMarketStore.getState().activeStep;
      if (current >= scenario.periods) {
        setPlaying(false);
      } else {
        setActiveStep(current + 1);
      }
    }, 85);
    return () => window.clearInterval(timer);
  }, [playing, scenario.periods, setActiveStep, setPlaying]);

  const share = async () => {
    const url = new URL(window.location.href);
    url.searchParams.set("scenario", serializeScenario(scenario));
    await navigator.clipboard.writeText(url.toString());
    window.history.replaceState(null, "", url);
    setNotice("Share link copied");
    window.setTimeout(() => setNotice(""), 1800);
  };

  const exportCsv = () => {
    const header = [
      "t",
      "recipient",
      "residual_contestability",
      "comparison_budget",
      "information",
      "information_bound",
      ...Array.from({ length: scenario.n }, (_, index) => `count_${index + 1}`),
      ...Array.from({ length: scenario.n }, (_, index) => `probability_${index + 1}`),
    ];
    const rows = result.steps.map((step) =>
      [
        step.t,
        step.recipient + 1,
        step.residualContestability,
        step.comparisonBudget,
        step.information,
        step.informationBound,
        ...step.counts,
        ...step.probabilities,
      ].join(","),
    );
    downloadText(
      `shadow-futures-seed-${scenario.seed}.csv`,
      [header.join(","), ...rows].join("\n"),
      "text/csv;charset=utf-8",
    );
  };

  return (
    <div>
      <div className="button-row" style={{ justifyContent: "flex-start" }}>
        <button
          className="button button--primary"
          type="button"
          onClick={() => setPlaying(!playing)}
          data-testid="play-market"
        >
          {playing ? "Pause" : "Play"} market
        </button>
        <button
          className="button"
          type="button"
          onClick={() => setActiveStep(Math.min(scenario.periods, activeStep + 1))}
        >
          Step once
        </button>
        <button className="button" type="button" onClick={rerun}>
          Replay from a new beginning
        </button>
        <button className="button" type="button" onClick={() => setActiveStep(0)}>
          Back to the start
        </button>
      </div>
      <details style={{ marginTop: "0.85rem" }}>
        <summary style={{ cursor: "pointer", color: "var(--muted)", fontSize: "0.82rem" }}>
          Share, download, or lock this experiment
        </summary>
        <div className="button-row" style={{ justifyContent: "flex-start", marginTop: "0.75rem" }}>
          <button
            className="button button--small"
            type="button"
            aria-pressed={seedLocked}
            onClick={toggleSeedLock}
          >
            {seedLocked ? "Beginning locked" : "Lock beginning"}
          </button>
          <button className="button button--small" type="button" onClick={share}>
            Share this version
          </button>
          <button className="button button--small" type="button" onClick={exportCsv}>
            Download history
          </button>
          <button
            className="button button--small"
            type="button"
            onClick={() =>
              downloadText(
                `shadow-futures-scenario-${scenario.seed}.json`,
                JSON.stringify(scenario, null, 2),
                "application/json",
              )
            }
          >
            Export settings
          </button>
          <button className="button button--small" type="button" onClick={reset}>
            Reset everything
          </button>
        </div>
      </details>
      <p aria-live="polite" className="panel__meta">
        {notice || `Showing allocation ${activeStep} of ${scenario.periods}`}
      </p>
    </div>
  );
}

export function FullMarketControls() {
  const scenario = useMarketStore((state) => state.scenario);
  const patchScenario = useMarketStore((state) => state.patchScenario);
  const setScenario = useMarketStore((state) => state.setScenario);
  const updateInput = (
    agent: number,
    dimension: number,
    value: number,
  ) => {
    const inputs = scenario.inputs.map((input) => [...input]);
    inputs[agent][dimension] = value;
    patchScenario({ inputs });
  };
  const updateInitialPosition = (agent: number, value: number) => {
    const positions = [...scenario.initialPositions];
    positions[agent] = value;
    patchScenario({ initialPositions: positions });
  };
  const twoDimensional = scenario.beta.length === 2;

  return (
    <details className="panel" style={{ marginTop: "1.25rem" }}>
      <summary
        style={{
          cursor: "pointer",
          listStyle: "none",
          padding: "1rem 1.25rem",
          fontWeight: 720,
        }}
      >
        Open all market controls
      </summary>
      <div className="panel__body" style={{ borderTop: "1px solid var(--line)" }}>
        <div className="control-grid">
          <label className="range-control">
            <span className="range-control__top">
              <span>Scenario preset</span>
            </span>
            <select
              value={
                SCENARIO_PRESETS.some((preset) => preset.name === scenario.name)
                  ? scenario.name
                  : ""
              }
              onChange={(event) => {
                const selected = SCENARIO_PRESETS.find(
                  (preset) => preset.name === event.target.value,
                );
                if (selected) setScenario(selected);
              }}
              style={{
                minHeight: "2.7rem",
                border: "1px solid var(--line)",
                borderRadius: "0.7rem",
                background: "var(--surface-solid)",
                color: "var(--ink)",
                padding: "0.45rem 0.7rem",
              }}
            >
              <option value="" disabled>
                Custom scenario
              </option>
              {SCENARIO_PRESETS.map((preset) => (
                <option value={preset.name} key={preset.name}>
                  {preset.name}
                </option>
              ))}
            </select>
          </label>
          <RangeControl
            id="beta"
            label="Contribution coefficient, β"
            min={-2}
            max={3}
            step={0.05}
            value={scenario.beta[0]}
            format={(value) => value.toFixed(2)}
            onChange={(value) => patchScenario({ beta: [value, ...scenario.beta.slice(1)] })}
          />
          <RangeControl
            id="rho"
            label="Strength of feedback, ρ"
            min={0}
            max={2.5}
            step={0.05}
            value={scenario.rho}
            format={(value) => value.toFixed(2)}
            onChange={(value) => patchScenario({ rho: value })}
          />
          <RangeControl
            id="baseline"
            label="Baseline attachment, a"
            min={0.1}
            max={5}
            step={0.1}
            value={scenario.baseline}
            format={(value) => value.toFixed(1)}
            onChange={(value) => patchScenario({ baseline: value })}
          />
          <RangeControl
            id="agents"
            label="Number of agents, n"
            min={2}
            max={10}
            value={scenario.n}
            onChange={(value) => patchScenario({ n: value })}
          />
          <RangeControl
            id="periods"
            label="Transactions, T"
            min={10}
            max={10_000}
            step={10}
            value={scenario.periods}
            format={(value) => value.toLocaleString()}
            onChange={(value) => patchScenario({ periods: value })}
          />
          <RangeControl
            id="seed"
            label="Random seed"
            min={0}
            max={999}
            value={scenario.seed}
            onChange={(value) => patchScenario({ seed: value })}
          />
          <RangeControl
            id="worlds"
            label="Independent worlds"
            min={2}
            max={1_000}
            step={2}
            value={scenario.worlds}
            onChange={(value) => patchScenario({ worlds: value })}
          />
          <RangeControl
            id="global-channels"
            label="Independent channels"
            min={1}
            max={100}
            value={scenario.channels}
            onChange={(value) => patchScenario({ channels: value })}
          />
          <RangeControl
            id="global-exploration"
            label="Exploration rate, η"
            min={0}
            max={0.5}
            step={0.01}
            value={scenario.exploration}
            format={(value) => value.toFixed(2)}
            onChange={(value) => patchScenario({ exploration: value })}
          />
          <RangeControl
            id="reset-cadence"
            label="Reset cadence"
            min={0}
            max={1_000}
            step={10}
            value={scenario.resetCadence}
            format={(value) => (value === 0 ? "never" : `every ${value}`)}
            onChange={(value) => patchScenario({ resetCadence: value })}
          />
          <label className="toggle">
            <input
              type="checkbox"
              checked={twoDimensional}
              onChange={(event) => {
                if (event.target.checked) {
                  patchScenario({
                    beta: [scenario.beta[0], 0.6],
                    inputs: scenario.inputs.map((input, index) => [
                      input[0],
                      Math.max(0, 0.65 - index * 0.08),
                    ]),
                  });
                } else {
                  patchScenario({
                    beta: [scenario.beta[0]],
                    inputs: scenario.inputs.map((input) => [input[0]]),
                  });
                }
              }}
            />
            Two-dimensional verified inputs
          </label>
        </div>
        <div style={{ marginTop: "2rem" }}>
          <div className="panel__meta">Verified input profiles and inherited positions</div>
          <div className="control-grid" style={{ marginTop: "1rem" }}>
            {scenario.inputs.map((input, agent) => (
              <div className="scope-card" key={agent}>
                <strong>Agent {String.fromCharCode(65 + agent)}</strong>
                <RangeControl
                  id={`input-${agent}-0`}
                  label="Verified input x₁"
                  min={-1}
                  max={1.5}
                  step={0.01}
                  value={input[0]}
                  format={(value) => value.toFixed(2)}
                  onChange={(value) => updateInput(agent, 0, value)}
                />
                {twoDimensional ? (
                  <RangeControl
                    id={`input-${agent}-1`}
                    label="Verified input x₂"
                    min={-1}
                    max={1.5}
                    step={0.01}
                    value={input[1]}
                    format={(value) => value.toFixed(2)}
                    onChange={(value) => updateInput(agent, 1, value)}
                  />
                ) : null}
                <RangeControl
                  id={`position-${agent}`}
                  label="Initial position"
                  min={0}
                  max={20}
                  value={scenario.initialPositions[agent]}
                  onChange={(value) => updateInitialPosition(agent, value)}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </details>
  );
}

function Hero() {
  return (
    <section className="hero" aria-labelledby="hero-title">
      <HeroNetwork />
      <div className="hero__content">
        <p className="eyebrow">An interactive economics essay by Martin Erlic</p>
        <h1 id="hero-title">Shadow Futures</h1>
        <p className="hero__subtitle">Contribution Uncertainty and the Self-Reinforcing Market</p>
        <p className="hero__line">“A market can keep paying after it has stopped learning.”</p>
        <p className="hero__dek">
          Work can matter. Quality can matter. Risk can matter. But when yesterday’s reward
          determines tomorrow’s exposure, one winning history may not reveal how much any of
          them mattered.
        </p>
        <div className="button-row">
          <a className="button button--primary" href="#run-market">
            Run the market
          </a>
          <a className="button" href={PAPER.url} target="_blank" rel="noreferrer">
            Read the paper
          </a>
        </div>
      </div>
    </section>
  );
}

export function StoryPage() {
  const scenario = useMarketStore((state) => state.scenario);
  const patchScenario = useMarketStore((state) => state.patchScenario);
  const setScenario = useMarketStore((state) => state.setScenario);
  const rerun = useMarketStore((state) => state.rerun);
  const [citationCopied, setCitationCopied] = useState(false);

  useEffect(() => {
    const restored = scenarioFromSearch(window.location.search);
    if (restored) setScenario(restored);
  }, [setScenario]);

  const rhoLabel =
    scenario.rho === 0
      ? "no reinforcement"
      : scenario.rho < 1
        ? "sublinear"
        : scenario.rho === 1
          ? "linear boundary"
          : "superlinear";

  return (
    <>
      <Hero />
      <main id="main-content">
        <section className="chapter" id="run-market" aria-labelledby="chapter-1">
          <ChapterIntro number="01 / Realized histories" title="One market, many possible histories">
            The better input changes the odds. It doesn’t determine the realized history.
            These worlds share structural parameters and verified profiles; only their seeded
            allocation shocks differ.
          </ChapterIntro>
          <SideBySideMarkets />
          <MarketToolbar />
          <FullMarketControls />
        </section>

        <section className="chapter" id="reinforcement" aria-labelledby="chapter-2">
          <ChapterIntro number="02 / Reinforcement" title="Reward rewrites the odds">
            One reward raises accumulated position, thickens the route to the next customer,
            and changes every conditional allocation probability.
          </ChapterIntro>
          <div className="scrolly">
            <div className="scrolly__copy">
              <div className="scrolly__step">
                <h3>Productive inputs enter directly</h3>
                <p>
                  Verified work and quality change the allocation odds through β. The model
                  doesn’t assume they’re irrelevant.
                </p>
              </div>
              <div className="scrolly__step">
                <h3>Position compounds</h3>
                <p>
                  Past rewards enter through (a + Nᵢ(t))ᵨ. The result of one round changes the
                  opportunity set in the next.
                </p>
              </div>
              <div className="scrolly__step">
                <h3>Path dependence isn’t the theorem</h3>
                <p>
                  Exact single-history impossibility requires finite total comparison under
                  the theorem’s assumptions.
                </p>
              </div>
            </div>
            <div className="scrolly__sticky">
              <RewardRewrite />
              <div className="panel__body">
                <RangeControl
                  id="feedback-strength"
                  label="Strength of feedback, ρ"
                  min={0}
                  max={2.5}
                  step={0.05}
                  value={scenario.rho}
                  format={(value) => `${value.toFixed(2)} · ${rhoLabel}`}
                  onChange={(value) => patchScenario({ rho: value })}
                />
                <div className="qualitative-scale" aria-hidden="true">
                  <span>0 · none</span>
                  <span>sublinear</span>
                  <span>1 · boundary</span>
                  <span>&gt;1 · superlinear</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="chapter" id="shadow-map" aria-labelledby="chapter-3">
          <ChapterIntro number="03 / Missing observations" title="The shadow-futures map">
            Only one branch becomes history. The other branches are the missing observations
            needed to estimate contribution. Select any path to replay that world.
          </ChapterIntro>
          <ShadowMap />
          <div style={{ maxWidth: "34rem", marginTop: "1.25rem" }}>
            <RangeControl
              id="parallel-worlds"
              label="Independent random seeds"
              min={64}
              max={1_000}
              step={16}
              value={scenario.worlds}
              format={(value) => `${value} worlds`}
              onChange={(value) => patchScenario({ worlds: value })}
            />
          </div>
        </section>

        <section className="chapter" id="comparison-budget" aria-labelledby="chapter-4">
          <ChapterIntro number="04 / The comparison budget" title="Transactions count activity. Comparison counts alternatives.">
            Transactions count allocations. The comparison budget counts how much probability
            remained available for the allocation to go another way.
          </ChapterIntro>
          <BudgetChart />
          <div style={{ maxWidth: "36rem", marginTop: "1.5rem" }}>
            <RangeControl
              id="budget-periods"
              label="Transactions, T"
              min={50}
              max={10_000}
              step={50}
              value={scenario.periods}
              format={(value) => value.toLocaleString()}
              onChange={(value) => patchScenario({ periods: value })}
            />
          </div>
        </section>

        <section className="chapter" id="information" aria-labelledby="chapter-5">
          <ChapterIntro number="05 / Information absorption" title="The record grows as its evidentiary value shrinks">
            The record becomes more impressive at the same time that its evidentiary value
            becomes smaller. This is a model illustration of the theorem’s information
            mechanism.
          </ChapterIntro>
          <InformationChart />
        </section>

        <section className="chapter chapter--dark" id="theorem" aria-labelledby="chapter-6">
          <ChapterIntro number="06 / The theorem" title="Finite comparison makes one history insufficient">
            The formal result is about complete-history laws and universal learning from one
            endogenous path. It doesn’t say that different parameters generate identical
            distributions.
          </ChapterIntro>
          <div className="theorem">
            <div className="theorem__label">
              Theorem 1 · finite comparison-budget impossibility
            </div>
            <blockquote>
              “If a market generates only a finite total amount of comparison, distinct
              contribution parameters can remain statistically inseparable on the complete
              single history.”
            </blockquote>
            <div className="experiment-grid">
              <EquationMath
                latex="h_t^2(\beta,\beta')=\sum_i\left[\sqrt{p_{it}(\beta)}-\sqrt{p_{it}(\beta')}\right]^2"
                label="One period squared Hellinger distance"
              />
              <EquationMath
                latex="h_t^2(\beta,\beta')\le K_{\beta,\beta'}\varepsilon_t(\beta)"
                label="Comparison dominated separation"
              />
              <EquationMath
                latex="B_\infty(\beta)=\sum_t\varepsilon_t(\beta)<\infty"
                label="Finite total comparison budget"
              />
              <EquationMath
                latex="P_\beta\sim P_{\beta'}\quad\text{on }\mathcal F_\infty"
                label="Complete history laws are mutually absolutely continuous"
              />
            </div>
            <p>
              A history possible under one finite contribution parameter remains possible
              under the other. One path doesn’t accumulate decisive separating evidence.
            </p>
            <div className="callout">
              <strong>Boundary.</strong> This is a failure of universal learning from one
              endogenous history, not classical point non-identification of the entire model
              family.
            </div>
          </div>
          <div style={{ marginTop: "2rem" }}>
            <LikelihoodIllustration />
          </div>
        </section>

        <section className="chapter" id="strong-reinforcement" aria-labelledby="chapter-7">
          <ChapterIntro number="07 / A sharp case" title="Strong reinforcement can exhaust comparison">
            For polynomial feedback, the summability condition holds when ρ &gt; 1. Strong
            reinforcement is a sharp corollary, not the definition of contribution
            uncertainty.
          </ChapterIntro>
          <StrongReinforcementChart />
          <div style={{ maxWidth: "36rem", marginTop: "1.5rem" }}>
            <RangeControl
              id="strong-rho"
              label="Reinforcement exponent, ρ"
              min={0}
              max={2.5}
              step={0.05}
              value={scenario.rho}
              format={(value) => value.toFixed(2)}
              onChange={(value) => patchScenario({ rho: value })}
            />
          </div>
          <EquationMath
            latex="\sum_{m=0}^{\infty}\frac{1}{g(a+m)}<\infty"
            label="Strong reinforcement summability condition"
          />
        </section>

        <section className="chapter" id="replication" aria-labelledby="chapter-8">
          <ChapterIntro number="08 / Independent markets" title="One long market isn’t many markets">
            Statistical separation can accumulate across independent histories even when one
            path absorbs its comparison. The relevant distinction is independence, not nominal
            firm count.
          </ChapterIntro>
          <ReplicationExperiment />
        </section>

        <section className="chapter" id="epistemic-monopoly" aria-labelledby="chapter-9">
          <ChapterIntro number="09 / Market structure" title="Monopoly can be epistemic">
            Economic monopoly concerns durable control over allocation conditions. Epistemic
            monopoly concerns control over the production of the comparison paths needed to
            explain allocation.
          </ChapterIntro>
          <EpistemicMonopoly />
          <div className="callout" style={{ marginTop: "1.5rem" }}>
            <strong>Policy qualification.</strong> A merger may reduce contribution information
            when it eliminates genuinely independent future allocation paths. Welfare effects
            require additional assumptions about decision errors and the costs of preserving
            those paths.
          </div>
        </section>

        <section className="chapter" id="attribution-gauge" aria-labelledby="chapter-10">
          <ChapterIntro number="10 / The attribution gauge" title="Contribution and position can be exactly equivalent">
            This point non-identification result is distinct from the one-history learning
            impossibility. With latent position, two decompositions can preserve every
            observable allocation probability.
          </ChapterIntro>
          <GaugeDemo />
        </section>

        <section className="chapter" id="taxation" aria-labelledby="chapter-11">
          <ChapterIntro number="11 / Merit-sensitive taxation" title="One number can’t equal both residuals">
            The same observable reward can imply different positional rents in observationally
            equivalent structural economies. The theorem constrains an exact decomposition; it
            doesn’t supply a tax rate.
          </ChapterIntro>
          <TaxDemo />
          <div className="policy-grid" style={{ marginTop: "1.5rem" }}>
            {[
              ["Identified sets", "Report the range of decompositions consistent with the information set."],
              ["Mechanism-based taxation", "Target observable compounding rules rather than an unrecoverable moral decomposition."],
              ["Reinforcement-neutral policy", "Reduce superlinear compounding without asserting that every high reward is rent."],
              ["Attribution-invariant transfers", "Use transfers that don’t reconstruct the same unavailable merit ranking."],
              ["Unconditional social dividend", "Distribute without requiring an exact contribution-versus-position split."],
            ].map(([title, copy]) => (
              <article className="policy-card" key={title}>
                <h3>{title}</h3>
                <p>{copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="chapter" id="scope" aria-labelledby="scope-title">
          <ChapterIntro number="Scope and limits" title="What the argument does and doesn’t establish">
            Simulations expose the mechanism. They don’t prove the theorem, measure moral
            desert, or turn every concentration event into a policy conclusion.
          </ChapterIntro>
          <div className="scope-grid">
            {[
              ["Work and risk can matter", "A real causal effect can remain unrecoverable from the reinforced path that rewarded it."],
              ["Path dependence isn’t enough", "Finite comparison is sufficient for the exact theorem under its assumptions; it isn’t necessary for every identification failure."],
              ["Observe the full experiment", "Any additional parameter-dependent observations can carry information and must be included."],
              ["Contribution is narrowly defined", "The theorem concerns direct reward contribution inside the allocation mechanism, not total social value or moral desert."],
              ["Firm count isn’t replication", "Competition creates identifying variation only when routes produce genuinely independent histories."],
              ["Policy needs welfare assumptions", "Merger and tax implications are conditional. Decision errors and the cost of preserving alternate paths matter."],
            ].map(([title, copy]) => (
              <article className="scope-card" key={title}>
                <h3>{title}</h3>
                <p>{copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="chapter" id="closing" aria-labelledby="chapter-12">
          <ChapterIntro number="12 / Closing" title="History is bright. Its alternatives are missing.">
            Markets produce rewards. They also produce, or fail to produce, the evidence by
            which those rewards are later explained.
          </ChapterIntro>
          <ClosingBranches />
          <div style={{ maxWidth: "68rem", margin: "3rem auto 0", textAlign: "center" }}>
            <p className="hero__line" style={{ marginTop: 0 }}>
              “A self-reinforcing market can keep paying after it has stopped learning.”
            </p>
            <div className="button-row">
              <a className="button button--primary" href={PAPER.url} target="_blank" rel="noreferrer">
                Read the paper
              </a>
              <a className="button" href={PAPER.url} download>
                Download the paper
              </a>
              <button
                className="button"
                type="button"
                onClick={async () => {
                  await navigator.clipboard.writeText(PAPER.bibtex);
                  setCitationCopied(true);
                  window.setTimeout(() => setCitationCopied(false), 1800);
                }}
              >
                {citationCopied ? "Citation copied" : "Copy citation"}
              </button>
              <Link className="button" href="/math">
                Open the advanced mathematics
              </Link>
              <button className="button button--rust" type="button" onClick={rerun}>
                Rerun the market
              </button>
            </div>
          </div>
        </section>
      </main>
      <footer className="footer">
        <div className="footer__inner">
          <span>Shadow Futures · Martin Erlic · Revised July 2026</span>
          <span>
            Interactive model for explanation, not empirical estimation or policy advice.
          </span>
        </div>
      </footer>
    </>
  );
}
