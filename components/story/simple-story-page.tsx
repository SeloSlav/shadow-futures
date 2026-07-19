"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { FullMarketControls, MarketToolbar } from "@/components/story/story-page";
import {
  AIFlywheel,
  BudgetChart,
  ClosingBranches,
  ContributionSplit,
  EpistemicMonopoly,
  EvidenceFadeChart,
  HeroNetwork,
  HonestPolicyDemo,
  MovingTrackRace,
  ReplicationExperiment,
  ShadowMap,
  SideBySideMarkets,
  StrongReinforcementChart,
} from "@/components/story/visuals";
import { RangeControl } from "@/components/ui/range-control";
import { PAPER } from "@/lib/paper/citation";
import { scenarioFromSearch } from "@/lib/scenarios/serialization";
import { useMarketStore } from "@/lib/store/market-store";

function Chapter({
  id,
  number,
  eyebrow,
  title,
  intro,
  children,
  dark = false,
}: {
  id: string;
  number: number;
  eyebrow: string;
  title: string;
  intro: string;
  children: React.ReactNode;
  dark?: boolean;
}) {
  return (
    <section
      className={`chapter${dark ? " chapter--dark" : ""}`}
      id={id}
      aria-labelledby={`${id}-title`}
    >
      <div className="chapter__intro">
        <span className="chapter__number">
          {String(number).padStart(2, "0")} / {eyebrow}
        </span>
        <h2 id={`${id}-title`}>{title}</h2>
        <p>{intro}</p>
      </div>
      {children}
    </section>
  );
}

function SimpleHero() {
  return (
    <section className="hero" aria-labelledby="hero-title">
      <HeroNetwork />
      <div className="hero__content">
        <p className="eyebrow">A thought experiment for the AI age · no economics background needed</p>
        <h1 id="hero-title">Shadow Futures</h1>
        <p className="hero__subtitle">Contribution Uncertainty and the Self-Reinforcing Market</p>
        <p className="hero__line">
          Imagine a race where the track speeds up under whoever is ahead.
        </p>
        <p className="hero__dek">
          One runner may truly be better. But once the track starts helping the leader, the
          final gap no longer tells us how much better. That is the basic problem behind
          self-reinforcing markets—and it may matter enormously in the age of AI.
        </p>
        <div className="button-row">
          <a className="button button--primary" href="#run-market">
            Start the race
          </a>
          <a className="button" href={PAPER.url} target="_blank" rel="noreferrer">
            Read the paper
          </a>
        </div>
      </div>
    </section>
  );
}

export function SimpleStoryPage() {
  const scenario = useMarketStore((state) => state.scenario);
  const patchScenario = useMarketStore((state) => state.patchScenario);
  const setScenario = useMarketStore((state) => state.setScenario);
  const rerun = useMarketStore((state) => state.rerun);
  const [citationCopied, setCitationCopied] = useState(false);

  useEffect(() => {
    const restored = scenarioFromSearch(window.location.search);
    if (restored) setScenario(restored);
  }, [setScenario]);

  return (
    <>
      <SimpleHero />
      <main id="main-content">
        <Chapter
          id="run-market"
          number={1}
          eyebrow="The moving-track race"
          title="A fair race—until someone gets ahead"
          intro="Runner A is slightly faster. Runner B can still get the first break. After that break, the track begins moving under the leader."
        >
          <MovingTrackRace />
          <MarketToolbar />
          <FullMarketControls />
        </Chapter>

        <Chapter
          id="reinforcement"
          number={2}
          eyebrow="The AI translation"
          title="A first win can buy the next win"
          intro="Replace the runners with two AI labs. An early customer brings revenue, compute, data-center capacity, attention, and a record that helps win the next customer."
        >
          <AIFlywheel />
          <div className="scope-grid" style={{ marginTop: "1.25rem" }}>
            <article className="scope-card">
              <h3>Real skill still matters</h3>
              <p>
                Better engineers, models, judgment, capital, and risk-taking can genuinely
                improve the odds.
              </p>
            </article>
            <article className="scope-card">
              <h3>The lead also starts to matter</h3>
              <p>
                More compute, users, revenue, and visibility can make tomorrow’s contest less
                independent than today’s.
              </p>
            </article>
          </div>
        </Chapter>

        <Chapter
          id="shadow-map"
          number={3}
          eyebrow="Press rewind"
          title="Same labs. Same talent. Different future."
          intro="Run the same market twice. Change only the early random breaks. The stronger lab has better odds, but a better chance is not the same thing as a guaranteed history."
        >
          <SideBySideMarkets />
          <div className="callout" style={{ marginTop: "1rem" }}>
            If the “better” lab loses one replay, that does <strong>not</strong> prove quality
            was irrelevant. It proves that quality and history both affected the result.
          </div>
        </Chapter>

        <Chapter
          id="comparison-budget"
          number={4}
          eyebrow="Shadow futures"
          title="The races history did not record"
          intro="Now replay the market hundreds of times. Every pale branch is a future that could have happened with the same people, technology, and rules."
        >
          <ShadowMap />
          <div style={{ maxWidth: "34rem", marginTop: "1.25rem" }}>
            <RangeControl
              id="parallel-worlds"
              label="How many times should we rewind the market?"
              min={64}
              max={1_000}
              step={16}
              value={scenario.worlds}
              format={(value) => `${value} possible histories`}
              onChange={(worlds) => patchScenario({ worlds })}
            />
          </div>
          <div className="callout" style={{ marginTop: "1rem" }}>
            In real life, only one branch becomes history. The rest are <strong>shadow
            futures</strong>: the missing replays we would need to measure how much skill,
            effort, luck, and position each caused.
          </div>
        </Chapter>

        <Chapter
          id="information"
          number={5}
          eyebrow="Room for surprise"
          title="A thousand sales are not a thousand fair tests"
          intro="When the leader has a 99% chance of winning the next customer, another sale is a transaction—but barely a new comparison."
        >
          <BudgetChart />
          <div style={{ maxWidth: "36rem", marginTop: "1.5rem" }}>
            <RangeControl
              id="budget-periods"
              label="How long should we keep watching this one history?"
              min={50}
              max={10_000}
              step={50}
              value={scenario.periods}
              format={(value) => `${value.toLocaleString()} transactions`}
              onChange={(periods) => patchScenario({ periods })}
            />
          </div>
          <p className="hero__line" style={{ maxWidth: "16ch", marginTop: "3rem" }}>
            Volume is not replication.
          </p>
        </Chapter>

        <Chapter
          id="theorem"
          number={6}
          eyebrow="The evidence problem"
          title="The trophy shelf grows. The experiment disappears."
          intro="The winner’s record can become more impressive at the exact moment each new win becomes less informative."
          dark
        >
          <EvidenceFadeChart />
          <div className="theorem" style={{ marginTop: "1.5rem" }}>
            <div className="theorem__label">The paper’s theorem, in ordinary language</div>
            <blockquote>
              If genuine chances to compare eventually run out, one market history cannot tell
              us exactly how much of the outcome came from contribution.
            </blockquote>
            <div className="scope-grid">
              <article className="scope-card">
                <h3>The runner may really be faster</h3>
                <p>The theorem does not erase work, quality, judgment, or risk.</p>
              </article>
              <article className="scope-card">
                <h3>The moving track may also matter</h3>
                <p>The final distance mixes ability with help created by being ahead.</p>
              </article>
              <article className="scope-card">
                <h3>One race cannot untangle them perfectly</h3>
                <p>Watching the leader ride the moving track for longer is not a fresh test.</p>
              </article>
              <article className="scope-card">
                <h3>This is not “nothing can be known”</h3>
                <p>It is a precise limit on exact learning from one self-reinforcing history.</p>
              </article>
            </div>
            <div className="button-row" style={{ justifyContent: "flex-start" }}>
              <Link className="button" href="/math">
                See the exact assumptions and proof
              </Link>
            </div>
          </div>
        </Chapter>

        <Chapter
          id="strong-reinforcement"
          number={7}
          eyebrow="From snowball to avalanche"
          title="How quickly does the race close?"
          intro="A weak snowball leaves room for challengers. A strong snowball can make the leader nearly impossible to dislodge. The exact theorem needs more than ordinary path dependence."
        >
          <StrongReinforcementChart />
          <div style={{ maxWidth: "36rem", marginTop: "1.5rem" }}>
            <RangeControl
              id="strong-rho"
              label="Snowball strength"
              min={0}
              max={2.5}
              step={0.05}
              value={scenario.rho}
              format={(value) =>
                value === 0
                  ? "none"
                  : value < 1
                    ? "weak"
                    : value === 1
                      ? "boundary"
                      : "strong"
              }
              onChange={(rho) => patchScenario({ rho })}
            />
          </div>
          <div className="callout" style={{ marginTop: "1rem" }}>
            A normal feedback loop is not automatically the impossibility theorem. The sharp
            case occurs when the remaining chances to compare shrink fast enough that their
            total is finite.
          </div>
        </Chapter>

        <Chapter
          id="replication"
          number={8}
          eyebrow="A real laboratory"
          title="One long movie—or one hundred fresh starts?"
          intro="Five thousand customers in one inherited timeline are not the same evidence as one hundred independent markets with fifty customers each."
        >
          <ReplicationExperiment />
          <p className="hero__line" style={{ maxWidth: "20ch", marginTop: "3rem" }}>
            Time extends the movie. Replication changes the opening scene.
          </p>
        </Chapter>

        <Chapter
          id="epistemic-monopoly"
          number={9}
          eyebrow="AI infrastructure"
          title="Who owns the track?"
          intro="In an AI economy, many companies can exist while depending on the same compute bottleneck, app store, ranking system, cloud, or recommendation channel."
        >
          <div className="scope-grid" style={{ marginBottom: "1.25rem" }}>
            <article className="scope-card">
              <h3>Compute</h3>
              <p>
                Access to chips, power, and data-center capacity can turn early revenue into
                faster future growth.
              </p>
            </article>
            <article className="scope-card">
              <h3>Data and feedback</h3>
              <p>
                More use can create more opportunities to train, test, and improve a system.
              </p>
            </article>
            <article className="scope-card">
              <h3>Distribution</h3>
              <p>
                One default ranking or platform can make many firms part of a single correlated
                history.
              </p>
            </article>
            <article className="scope-card">
              <h3>Finance and trust</h3>
              <p>
                A visible lead can attract capital, workers, suppliers, and customers before
                the underlying contribution is fully known.
              </p>
            </article>
          </div>
          <EpistemicMonopoly />
          <div className="callout" style={{ marginTop: "1rem" }}>
            <strong>Economic monopoly</strong> is control over prices or access.{" "}
            <strong>Epistemic monopoly</strong> is control over the independent paths society
            would need to understand why one system won.
          </div>
        </Chapter>

        <Chapter
          id="attribution-gauge"
          number={10}
          eyebrow="Skill versus position"
          title="Was the winner better—or simply earlier?"
          intro="Sometimes the same visible winning odds can be explained by more direct contribution and less inherited position, or the other way around."
        >
          <ContributionSplit />
          <div className="callout" style={{ marginTop: "1rem" }}>
            This is a second, different problem. The first was too little comparison in one
            history. Here, hidden position and contribution can fit the same observations
            exactly.
          </div>
        </Chapter>

        <Chapter
          id="taxation"
          number={11}
          eyebrow="What policy can honestly know"
          title="Do not confuse a moral judgment with a measurement"
          intro="A society may still tax, regulate, invest, or redistribute. But an exact split between “earned contribution” and “position” may not be recoverable from the reward record alone."
        >
          <HonestPolicyDemo />
          <div className="policy-grid" style={{ marginTop: "1.5rem" }}>
            {[
              [
                "Preserve fresh starts",
                "Independent channels, randomized trials, public options, and new procurement rounds can create comparison.",
              ],
              [
                "Let people carry their work",
                "Portability and multihoming can keep one platform history from becoming the only history.",
              ],
              [
                "Target visible mechanisms",
                "Policy can address observable compounding rules without pretending to know a perfect moral split.",
              ],
              [
                "Use ranges, not false precision",
                "Identified sets can state which contribution stories remain possible.",
              ],
              [
                "Redistribute without reranking everyone",
                "A broad social dividend does not require the state to reconstruct an impossible merit table.",
              ],
            ].map(([title, copy]) => (
              <article className="policy-card" key={title}>
                <h3>{title}</h3>
                <p>{copy}</p>
              </article>
            ))}
          </div>
          <div className="scope-grid" style={{ marginTop: "1.5rem" }}>
            <article className="scope-card">
              <h3>This does not mean</h3>
              <p>
                all high income is rent, all mergers are harmful, risk is fake, or skill does
                not matter.
              </p>
            </article>
            <article className="scope-card">
              <h3>It does mean</h3>
              <p>
                the market’s final score may be a poor instrument for precisely measuring
                moral or causal contribution.
              </p>
            </article>
          </div>
        </Chapter>

        <Chapter
          id="closing"
          number={12}
          eyebrow="The idea to remember"
          title="Markets choose winners—and which evidence survives"
          intro="A market does not only distribute money, attention, compute, and power. Its rules also decide whether society gets enough alternate histories to understand those rewards."
        >
          <ClosingBranches />
          <div style={{ maxWidth: "68rem", margin: "3rem auto 0", textAlign: "center" }}>
            <p className="hero__line" style={{ marginTop: 0 }}>
              “A self-reinforcing market can keep paying after it has stopped learning.”
            </p>
            <p className="hero__dek">
              Shadow futures are the replays we never saw: the same people, the same work, and
              the same technology—meeting different early breaks.
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
                Replay the market
              </button>
            </div>
          </div>
        </Chapter>
      </main>
      <footer className="footer">
        <div className="footer__inner">
          <span>Shadow Futures · Martin Erlic · Revised July 2026</span>
          <span>
            The simulations explain a mechanism. They are not evidence about every AI market.
          </span>
        </div>
      </footer>
    </>
  );
}
