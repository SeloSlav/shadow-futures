"use client";

import Link from "next/link";
import { useState } from "react";

import {
  BreakoutGraph,
  ExperimentMonopolyGraph,
  LorenzHistoryGraph,
  ShadowFuturesGraph,
} from "@/components/story/creator-graphs";
import { HeroNetwork } from "@/components/story/visuals";
import { PAPER } from "@/lib/paper/citation";

function StorySection({
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

function CreatorHero() {
  return (
    <section className="hero creator-hero" aria-labelledby="hero-title">
      <HeroNetwork />
      <div className="hero__content">
        <p className="eyebrow">
          Early advantage compounds · other possible futures disappear
        </p>
        <h1 id="hero-title">Shadow Futures</h1>
        <p className="hero__subtitle">
          Contribution Uncertainty and the Self-Reinforcing Market
        </p>
        <p className="hero__line">
          A market can reward real contribution—and erase the evidence needed to measure it.
        </p>
        <p className="hero__dek">
          A creator’s early audience or a firm’s first customer can make the next win easier.
          As that advantage compounds, the market records one path in extraordinary detail
          while failing to produce the comparison paths that could explain it. Those missing
          experiments are shadow futures.
        </p>
        <div className="button-row">
          <a className="button button--primary" href="#breakout">
            See how the evidence disappears
          </a>
          <a className="button" href={PAPER.url} target="_blank" rel="noreferrer">
            Read the paper
          </a>
        </div>
      </div>
    </section>
  );
}

export function CreatorStoryPage() {
  const [citationCopied, setCitationCopied] = useState(false);

  return (
    <>
      <CreatorHero />
      <main id="main-content">
        <StorySection
          id="breakout"
          number={1}
          eyebrow="The familiar feedback loop"
          title="The feed can turn an early break into a lasting lead"
          intro="Imagine 24 equally good creators. One gets slightly more attention in the first few minutes. The platform treats that attention as a reason to show the same creator again."
        >
          <BreakoutGraph />
          <div className="concept-equation" aria-label="The creator-platform feedback loop">
            <span>an early break</span>
            <span aria-hidden="true">→</span>
            <span>shown to more people</span>
            <span aria-hidden="true">→</span>
            <span>more followers</span>
            <span aria-hidden="true">→</span>
            <span>shown even more</span>
          </div>
          <div className="scope-grid">
            <article className="scope-card">
              <h3>Skill matters. The ranking still overclaims.</h3>
              <p>
                Excellent work can help someone win. It does not turn the final follower count
                into a precise measure of what they contributed.
              </p>
            </article>
            <article className="scope-card">
              <h3>Follower count is not a merit score</h3>
              <p>
                A million followers can reflect both the creator’s work and all the extra
                chances that came from already having followers.
              </p>
            </article>
          </div>
        </StorySection>

        <StorySection
          id="shadow-futures"
          number={2}
          eyebrow="The missing histories"
          title="Press replay. A different creator wins."
          intro="The 24 creators, their work, and the feed are unchanged. Replay only the first few random views and the winner changes. Each replay is evidence the original market never produced."
        >
          <ShadowFuturesGraph />
          <div className="shadow-definition">
            <p>
              <strong>The history we saw:</strong> one creator’s rise, recorded in followers,
              views, sponsorships, and income.
            </p>
            <p>
              <strong>The shadow futures:</strong> all the equally possible histories in which
              another creator received the first break and kept getting shown more.
            </p>
          </div>
          <p className="hero__line creator-story-line">
            The platform does not merely discover a winner. It helps create the history that
            later looks like proof.
          </p>
        </StorySection>

        <StorySection
          id="experiment-monopoly"
          number={3}
          eyebrow="The familiar story—and the missing question"
          title="The problem is not simply that success compounds. It is what compounding erases."
          intro="Increasing returns and preferential attachment explain why an early lead can grow. Shadow Futures asks what happens to the evidence: after that lead shapes thousands of later decisions, can the one history we observed still tell us how much the winner contributed?"
          dark
        >
          <div className="platform-families">
            {[
              [
                "Attention",
                "Instagram, TikTok, YouTube and Twitch rank creators for enormous shared audiences.",
              ],
              [
                "Subscriptions",
                "OnlyFans, Fanvue, Patreon and Substack turn an audience lead into recurring income.",
              ],
              [
                "Work and sales",
                "Upwork, Fiverr, Etsy, Amazon and app stores carry reviews and ranking into the next sale.",
              ],
              [
                "Knowledge",
                "Popular search results and papers are easier to find and cite, so they can become even more popular.",
              ],
            ].map(([title, copy]) => (
              <article key={title}>
                <span className="panel__meta">{title}</span>
                <p>{copy}</p>
              </article>
            ))}
          </div>
          <div
            className="idea-distinction"
            aria-label="How Shadow Futures differs from familiar theories"
          >
            <article>
              <span className="panel__meta">The familiar question</span>
              <h3>Why does the winner keep winning?</h3>
              <p>
                Increasing returns, scaling laws, network effects and preferential attachment
                explain how early success can compound into concentration.
              </p>
            </article>
            <article>
              <span className="panel__meta">What Shadow Futures adds</span>
              <h3>What did the market stop letting us learn?</h3>
              <p>
                When one path absorbs the chances to try other paths, the market loses the
                experimental comparisons needed to measure contribution from its final score.
              </p>
            </article>
          </div>
          <div className="global-history-callout">
            A market can be extremely busy while producing almost no new evidence. Ten million
            views, sales or contracts can keep extending one inherited path instead of testing
            what the same inputs would have done along another.
          </div>
          <ExperimentMonopolyGraph />
          <div className="monopoly-definition">
            <article>
              <span className="panel__meta">The familiar kind of monopoly</span>
              <h3>One company controls prices or access</h3>
              <p>
                The company can charge more, set the rules, or keep competitors out.
              </p>
            </article>
            <article>
              <span className="panel__meta">The paper’s epistemic monopoly</span>
              <h3>One history controls the evidence</h3>
              <p>
                Thousands of creators or firms can remain in the market while one ranking,
                standard or route to customers decides which paths get recorded. The monopoly
                is over the comparisons society would need to explain the outcome.
              </p>
            </article>
          </div>
          <div className="theorem creator-theorem">
            <div className="theorem__label">The Shadow Futures result</div>
            <blockquote>
              Transactions are not the sample size. Real chances for the market to go another
              way are.
            </blockquote>
            <p>
              The paper calls the total of those chances the comparison budget. If that budget
              is finite, no method using the one market history can consistently recover a
              meaningful contribution measure—one that changes when contribution changes.
              More activity can lengthen the same path without adding the missing experiments.
            </p>
          </div>
        </StorySection>

        <StorySection
          id="firm-markets"
          number={4}
          eyebrow="From scale to evidence"
          title="A growing firm can improve—and make its own contribution harder to measure"
          intro="An early customer brings revenue, data, credibility and scale. Those are real productive gains. But as one firm captures customers, standards and distribution, the market can run out of independent paths for learning how much came from the firm’s inputs and how much came from the position built by earlier wins."
        >
          <div
            className="firm-flywheel"
            role="img"
            aria-label="An early contract brings revenue and data, which fund investment, which can lower costs and improve the product, which makes the next contract easier to win."
          >
            {[
              [
                "01",
                "Win early customers",
                "A first contract, retailer, standard or major buyer creates the opening lead.",
              ],
              [
                "02",
                "Gain money and information",
                "Sales provide cash, usage data, a track record and easier access to finance.",
              ],
              [
                "03",
                "Invest and improve",
                "The firm can hire, build capacity, lower unit costs and make the product more reliable.",
              ],
              [
                "04",
                "Win the next customer more easily",
                "Buyers see a proven supplier with scale, compatibility and distribution already in place.",
              ],
            ].map(([number, title, copy], index) => (
              <div className="firm-flywheel__item" key={title}>
                <span>{number}</span>
                <strong>{title}</strong>
                <p>{copy}</p>
                {index < 3 ? (
                  <span className="firm-flywheel__arrow" aria-hidden="true">
                    →
                  </span>
                ) : null}
              </div>
            ))}
            <div className="firm-flywheel__return" aria-hidden="true">
              The loop begins again
            </div>
          </div>

          <div className="firm-market-examples">
            {[
              [
                "AI and cloud computing",
                "Models, chips and data centers require enormous up-front investment. More customers can fund more capacity, lower average costs and—in some settings—provide data that improves the service.",
              ],
              [
                "Manufacturing and logistics",
                "A larger order book can pay for better machinery, cheaper purchasing and wider distribution. Those real efficiencies can make the largest supplier still cheaper.",
              ],
              [
                "Software and technical standards",
                "A large installed base attracts integrations, trained workers and compatible products. Switching becomes costly even when another firm has a strong alternative.",
              ],
              [
                "Finance and large contracts",
                "A proven sales record can unlock cheaper capital and make a firm look like the safe choice for the next major buyer or government contract.",
              ],
            ].map(([title, copy]) => (
              <article key={title}>
                <h3>{title}</h3>
                <p>{copy}</p>
              </article>
            ))}
          </div>

          <div className="firm-market-effects">
            <article>
              <span className="panel__meta">The public claim on scale</span>
              <h3>Scale should serve the public—not become proof of desert</h3>
              <p>
                Lower costs, better reliability, larger research budgets and useful standards
                are collective economic gains. They do not turn market power or profit into a
                precise measure of contribution.
              </p>
            </article>
            <article>
              <span className="panel__meta">A measurement problem</span>
              <h3>Market share is not an exact contribution score</h3>
              <p>
                Today’s profit can reflect better products and the advantages created by
                yesterday’s sales. One observed market path cannot always separate the two.
              </p>
            </article>
            <article>
              <span className="panel__meta">A competition problem</span>
              <h3>Many firms can still produce only one useful test</h3>
              <p>
                A market can contain many legal competitors while buyers, standards, finance
                and distribution all follow the same early leader.
              </p>
            </article>
            <article>
              <span className="panel__meta">A policy problem</span>
              <h3>Mergers can remove paths we would have learned from</h3>
              <p>
                Merger review should ask whether independent products, experiments and routes
                to customers will disappear—not only whether several company names remain.
              </p>
            </article>
          </div>

          <div className="global-history-callout firm-boundary">
            The goal is not to freeze every firm at equal size. It is to prevent today’s leader
            from closing tomorrow’s contest. Where feedback can exhaust real comparison, open
            standards, interoperability, independent procurement trials, new entry and
            structural separation are democratic infrastructure.
          </div>
        </StorySection>

        <StorySection
          id="lorenz-curve"
          number={5}
          eyebrow="What inequality cannot answer"
          title="The Lorenz curve is the symptom. Shadow futures are the missing evidence."
          intro="A Lorenz curve shows how unequal the final rewards became. It cannot tell us how much of that ranking came from contribution rather than a history that reinforced itself. Shadow Futures is not another description of the bend; it explains why the observed distribution may be unable to reveal its own cause."
        >
          <LorenzHistoryGraph />
          <div className="lorenz-takeaway">
            <strong>The Lorenz curve is the final scoreboard.</strong>
            <span>
              Shadow Futures asks whether the market kept enough alternate plays alive to
              explain that score.
            </span>
          </div>
          <div className="platform-lorenz-copy">
            <article>
              <h3>OnlyFans and Fanvue</h3>
              <p>
                A verified payout curve would show how subscription income is divided among
                creators. Even a perfect curve could not tell whether its bend came from
                better work, outside fame, early discovery, referrals, money, or simply being
                shown first and then shown again.
              </p>
            </article>
            <article>
              <h3>YouTube, TikTok, Twitch and Instagram</h3>
              <p>
                Views, followers, recommendations and sponsorships can reinforce one another.
                The visible earnings curve records the result of that history, not the missing
                histories in which different creators received the early audience.
              </p>
            </article>
            <article>
              <h3>Patreon, Substack, Spotify and marketplaces</h3>
              <p>
                Subscriptions, playlists, reviews and rankings can carry yesterday’s position
                into tomorrow’s income. Some rules give newcomers more real chances than
                others.
              </p>
            </article>
          </div>
          <p className="data-note">
            The curve above is an example, not OnlyFans or Fanvue payout data. To draw a real
            curve, we would need to know what individual creators earned. Company totals are
            not enough.
          </p>
          <div className="source-links" aria-label="Creator-platform sources">
            <a
              href="https://www.ucl.ac.uk/bartlett/sites/bartlett/files/2025-12/Rich_Get_Richer.pdf"
              target="_blank"
              rel="noreferrer"
            >
              Creator earnings research
            </a>
            <a
              href="https://doi.org/10.1038/s41598-022-26727-5"
              target="_blank"
              rel="noreferrer"
            >
              Twitch inequality study
            </a>
            <a
              href="https://landing.fanvue.com/report"
              target="_blank"
              rel="noreferrer"
            >
              Fanvue creator report
            </a>
            <a
              href="https://find-and-update.company-information.service.gov.uk/company/10354575/filing-history"
              target="_blank"
              rel="noreferrer"
            >
              OnlyFans parent filings
            </a>
          </div>
        </StorySection>

        <StorySection
          id="tax-and-ubi"
          number={6}
          eyebrow="Tax, UBI and social insurance"
          title="The market cannot tell tax policy who “deserved” each dollar"
          intro="A tax system can see personal income, company profits and wealth. It cannot look at one career or one firm’s market share and calculate the exact amount created by contribution rather than accumulated advantage."
        >
          <div className="tax-policy-split">
            <article>
              <span className="panel__meta">A false promise</span>
              <h3>“Tax only the part that was not earned.”</h3>
              <p>
                That rule would require us to know exactly how much success came from the
                person’s work and how much came from getting ahead early. Their earnings alone
                cannot answer that, so tax policy should not pretend that they can.
              </p>
            </article>
            <article>
              <span className="panel__meta">A democratic guarantee</span>
              <h3>“Give everyone a basic floor.”</h3>
              <p>
                A universal basic income or social dividend recognizes that everyone depends
                on shared institutions, infrastructure, knowledge and demand. It provides a
                floor without turning survival into a merit contest.
              </p>
            </article>
          </div>
          <div className="policy-grid">
            {[
              [
                "Progressive taxes are justified by power and capacity",
                "Higher incomes and wealth confer greater security, bargaining power and ability to pay. Progressive taxation shares the cost of the institutions and infrastructure that make private success possible.",
              ],
              [
                "Tax the gains we can actually measure",
                "Income, wealth, land values, company profits and platform fees are observable. Policy can tax them directly instead of inventing a false personal merit score.",
              ],
              [
                "UBI takes survival out of the merit contest",
                "A universal floor follows people through unstable work, automation and algorithmic exclusion. Nobody should lose the basics of life because a market stopped choosing them.",
              ],
              [
                "A social dividend recognizes shared production",
                "Technology, public research, infrastructure, institutions and accumulated knowledge are collective inheritances. Part of the income they generate should return to everyone.",
              ],
            ].map(([title, copy]) => (
              <article className="policy-card" key={title}>
                <h3>{title}</h3>
                <p>{copy}</p>
              </article>
            ))}
          </div>
        </StorySection>

        <StorySection
          id="conclusion"
          number={7}
          eyebrow="The distinct contribution"
          title="The market does not just choose a winner. It chooses what can still be known."
          intro="Increasing returns explain compounding. Scaling laws relate size to performance. Preferential attachment explains why success attracts more success. Lorenz curves describe inequality. Shadow Futures identifies the missing step: self-reinforcing allocation can destroy the comparison paths needed to measure contribution from the one history we observe."
        >
          <div className="creator-closing">
            <p className="hero__line">
              A market can keep paying the winner long after it has stopped producing evidence
              about why they won.
            </p>
            <p className="hero__dek">
              Shadow futures are not simply stories in which someone else got lucky. They are
              the missing experimental repetitions—the same inputs meeting different early
              audiences, customers, rankings or shocks—that would have let us estimate
              contribution.
            </p>
            <div className="button-row">
              <a className="button button--primary" href={PAPER.url} target="_blank" rel="noreferrer">
                Read the paper
              </a>
              <Link className="button" href="/methodology">
                Evidence and assumptions
              </Link>
              <Link className="button" href="/math">
                Open the mathematics
              </Link>
              <button
                className="button"
                type="button"
                onClick={async () => {
                  await navigator.clipboard.writeText(PAPER.bibtex);
                  setCitationCopied(true);
                  window.setTimeout(() => setCitationCopied(false), 1_800);
                }}
              >
                {citationCopied ? "Citation copied" : "Copy citation"}
              </button>
            </div>
          </div>
        </StorySection>
      </main>
      <footer className="footer">
        <div className="footer__inner">
          <span>Shadow Futures · Martin Erlic · Revised July 2026</span>
          <span>
            More views can repeat the same early result without telling us what would have
            happened to everyone else.
          </span>
        </div>
      </footer>
    </>
  );
}
