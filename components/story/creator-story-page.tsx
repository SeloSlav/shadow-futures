"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

import {
  BreakoutGraph,
  ExperimentMonopolyGraph,
  LorenzHistoryGraph,
} from "@/components/story/creator-graphs";
import { HeroNetwork } from "@/components/story/visuals";
import { PAPER } from "@/lib/paper/citation";

function StorySection({
  id,
  number,
  eyebrow,
  title,
  intro,
  illustration,
  children,
  dark = false,
}: {
  id: string;
  number: number;
  eyebrow: string;
  title: React.ReactNode;
  intro: string;
  illustration?: {
    src: string;
    alt: string;
  };
  children: React.ReactNode;
  dark?: boolean;
}) {
  return (
    <section
      className={`chapter${dark ? " chapter--dark" : ""}`}
      id={id}
      aria-labelledby={`${id}-title`}
    >
      <div className={`chapter__intro${illustration ? " chapter__intro--illustrated" : ""}`}>
        <span className="chapter__number">
          {String(number).padStart(2, "0")} / {eyebrow}
        </span>
        <h2 id={`${id}-title`}>{title}</h2>
        <p className="chapter__intro-copy">{intro}</p>
        {illustration ? (
          <figure className="chapter__illustration">
            <Image
              src={illustration.src}
              fill
              sizes="(max-width: 900px) 68vw, 26vw"
              alt={illustration.alt}
            />
          </figure>
        ) : null}
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
          A market can reward real contribution while erasing the evidence needed to measure it.
        </p>
        <p className="hero__dek">
          In some markets, each win makes the next win easier. Success brings more attention,
          customers, capital, data or distribution to the leader, while others lose the chances
          they need to show what they could have contributed. Over time, the market records the
          winner’s path in extraordinary detail but stops producing the comparisons needed to tell
          how much success came from contribution and how much came from already being ahead.
          Those missing experiments are shadow futures.
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

function NoveltyBridge() {
  return (
    <section className="novelty-bridge" id="novelty" aria-labelledby="novelty-title">
      <div className="novelty-bridge__intro">
        <p className="eyebrow">What’s new here</p>
        <h2 id="novelty-title">A million transactions can still add up to only one experiment.</h2>
        <div className="novelty-bridge__aside">
          <figure className="novelty-bridge__illustration">
            <Image
              src="/illustrations/chapters/one-experiment.png"
              width={1200}
              height={507}
              sizes="(max-width: 900px) calc(100vw - 2rem), 36vw"
              alt="Many transaction tiles circulate around one track and pass through a single observation window, while dashed alternative tracks remain unrealized."
            />
          </figure>
          <p>
            Economists already know that early success can compound. This paper asks a different
            question: when each win changes who gets the next chance, can one long market history
            still tell us how much of the final gap came from the winner&apos;s work?
          </p>
        </div>
      </div>

      <div className="novelty-bridge__body">
        <article className="novelty-analogy">
          <span className="panel__meta">A simple analogy</span>
          <h3>One product gets moved to the front shelf.</h3>
          <p>
            A shop moves one product to the front after its first sale. Front placement brings
            more sales, and each sale keeps it in front. After a year, the shop has thousands of
            receipts but only one shelf history.
          </p>
          <figure className="novelty-analogy__visual" id="shelf-analogy-visual">
            <Image
              src="/illustrations/shelf-shadow-future.webp"
              width={1400}
              height={684}
              sizes="(max-width: 900px) calc(100vw - 5rem), 40vw"
              alt="One product is repeatedly promoted by sales on the observed shelf, while a faint second shelf shows the unrealized rerun with a different product in front."
            />
            <figcaption>
              <span>Observed shelf history</span>
              <span>Missing rerun</span>
            </figcaption>
          </figure>
          <p className="novelty-analogy__answer">
            <strong>More receipts don’t mean more experiments.</strong> To separate product quality
            from placement, we’d need to rerun the shop with a different product in front.
            Those missing reruns are shadow futures.
          </p>
        </article>

        <div className="novelty-literature">
          <article>
            <span className="panel__meta">What’s already familiar</span>
            <h3>Why an early winner keeps winning</h3>
            <p>
              Increasing returns, lock-in, cumulative advantage and preferential attachment
              already explain how an early lead can grow.
            </p>
          </article>
          <article>
            <span className="panel__meta">What this paper adds</span>
            <h3>When that process destroys the evidence needed to explain the winner</h3>
            <p>
              At every step, Shadow Futures measures the chance that someone other than the current
              leader gets the next reward. It adds those chances to the comparison budget. If that
              total is finite, the paper proves that no method using only one market history can
              consistently recover how much of the reward came from contribution, even when work
              and quality are observed and new transactions keep arriving.
            </p>
          </article>
        </div>
      </div>

      <div className="novelty-boundary">
        <span className="panel__meta">The literature gap, stated carefully</span>
        <div className="novelty-boundary__copy">
          <p>
            The paper doesn’t claim that lock-in or cumulative advantage is new. Arthur&apos;s
            {" "}
            <cite>Competing Technologies, Increasing Returns, and Lock-In by Historical Events</cite>
            {" "}
            (1989), David&apos;s <cite>Clio and the Economics of QWERTY</cite>
            {" "}
            (1985), and Merton&apos;s <cite>The Matthew Effect in Science</cite>
            {" "}
            (1968) establish those foundations.
          </p>
          <p>
            Pemantle&apos;s <cite>A Survey of Random Processes with Reinforcement</cite>
            {" "}
            (2007)
            {" "}
            maps the reinforced-process literature, Oliveira&apos;s
            {" "}
            <cite>The Onset of Dominance in Balls-in-Bins Processes with Feedback</cite>
            {" "}
            (2009)
            {" "}
            proves a dominance result, and Bar-Yam&apos;s
            {" "}
            <cite>From Big Data to Important Information</cite>
            {" "}
            (2016)
            {" "}
            distinguishes abundant records from the information needed to evaluate interventions.
            Hayek&apos;s
            {" "}
            <cite>Competition as a Discovery Procedure</cite>
            {" "}
            (2002)
            {" "}
            gives competition its familiar discovery role.
          </p>
          <p>
            The closest statistical precedent is Le Goff and Soulier&apos;s
            {" "}
            <cite>Parameter Estimation of a Two-Colored Urn Model Class</cite>
            {" "}
            (2017),
            {" "}
            which proves an estimation failure in that narrower urn setting. Shadow Futures adds
            a market-level comparison budget tied to a single-history impossibility theorem for
            contribution attribution. It also gives competition an additional role: independent
            market paths are the replications needed to learn why outcomes diverged.
          </p>
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
        <NoveltyBridge />
        <StorySection
          id="breakout"
          number={1}
          eyebrow="How a platform manufactures the chart"
          title="A platform can bury talent before it has a chance to become visible"
          intro="Imagine 24 creators with a realistic spread of promise: some work will connect more strongly than others. But promise only becomes visible when people get to encounter the work. An early entrant who receives the first audience also gains followers, feedback, income and time to improve. The platform then reads those advantages as reasons to keep promoting them."
          illustration={{
            src: "/illustrations/chapters/platform-visibility.png",
            alt: "Creator cards remain screened at the bottom while one card rides a feedback staircase upward.",
          }}
        >
          <aside className="skill-objection">
            <div>
              <span className="panel__meta">Talent matters. So why model the platform?</span>
              <h3>
                Because talent can’t be discovered, rewarded or developed without opportunities
                to be seen.
              </h3>
            </div>
            <div className="skill-objection__answer">
              <p>
                The simulation gives creators different modeled audience appeal. Better work
                improves the odds of connecting with each person who sees it. It doesn’t
                guarantee that the ranking system will keep supplying those chances.
              </p>
              <p>
                Earlier entrants can build followers, reviews, data, capital and production
                capacity before a promising newcomer arrives. When platforms rank using those
                accumulated signals, past exposure buys future exposure. A creator can be
                talented and still be drowned out before enough people encounter the work.
              </p>
              <p>
                The question isn’t “Did the winner have talent?” It’s “Did the platform keep
                testing enough alternatives to know how much unrealized talent it buried?”
              </p>
            </div>
          </aside>
          <BreakoutGraph />
          <div
            className="concept-equation"
            aria-label="How a social media recommendation system reinforces an early lead"
          >
            <span>one creator gets an early break</span>
            <span aria-hidden="true">→</span>
            <span>social media shows them to more people</span>
            <span aria-hidden="true">→</span>
            <span>they gain more followers</span>
            <span aria-hidden="true">→</span>
            <span>recommendation systems show them even more</span>
          </div>
          <div className="scope-grid">
            <article className="scope-card">
              <h3>The ranking measures a shaped history</h3>
              <p>
                Better work can improve someone’s chances. But the final follower count combines
                audience response with every extra opportunity created by earlier visibility.
              </p>
            </article>
            <article className="scope-card">
              <h3>Unseen talent leaves almost no evidence</h3>
              <p>
                If the feed stops testing a creator, the absence of followers may tell us more
                about missing exposure than about the quality of what they could have built.
              </p>
            </article>
          </div>
          <aside
            className="musiclab-evidence"
            id="shadow-futures"
            aria-label="Experimental evidence"
          >
            <span className="panel__meta">Shown experimentally</span>
            <p>
              <strong>This pattern is more than a thought experiment.</strong> In 2006, Matthew
              J. Salganik, Peter Sheridan Dodds and Duncan J. Watts built an artificial music
              market with 14,341 participants. When listeners could see earlier download counts,
              success became more unequal and less predictable. Quality moved the odds, but it
              didn’t determine the ranking.
            </p>
            <p className="musiclab-evidence__citation">
              <a
                href="https://doi.org/10.1126/science.1121066"
                target="_blank"
                rel="noreferrer"
              >
                <cite>
                  Experimental Study of Inequality and Unpredictability in an Artificial Cultural
                  Market
                </cite>
              </a>
              , <span>Science 311 (2006), 854–856</span>
            </p>
          </aside>
          <div className="shadow-definition">
            <p>
              <strong>The chart we saw:</strong> one ranking after earlier rankings had already
              decided who received the chances to grow.
            </p>
            <p>
              <strong>The shadow charts:</strong> the other plausible rankings hidden by the one
              launch that actually happened.
            </p>
          </div>
        </StorySection>

        <StorySection
          id="experiment-monopoly"
          number={2}
          eyebrow="The familiar story and the missing question"
          title={
            <>
              <span className="chapter__title-lead">
                The problem isn’t simply that success compounds.
              </span>{" "}
              <span className="chapter__title-payoff">It’s what compounding erases.</span>
            </>
          }
          intro="Increasing returns and preferential attachment explain why an early lead can grow. Shadow Futures asks what happens to the evidence: once that lead has shaped thousands of later decisions, can the one history we observe still tell us how much the winner contributed?"
          illustration={{
            src: "/illustrations/chapters/erased-comparisons.png",
            alt: "Alternative branches are cut off as feedback loops feed one recorded path.",
          }}
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
                "Upwork, Fiverr, Etsy, Amazon and app stores carry reviews and rankings into each new sale.",
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
                explain how early success can grow into market concentration.
              </p>
            </article>
            <article>
              <span className="panel__meta">What Shadow Futures adds</span>
              <h3>What can the market no longer teach us?</h3>
              <p>
                When one path crowds out the chance to test others, the market loses the
                comparisons needed to separate contribution from position in the final score.
              </p>
            </article>
          </div>
          <div className="global-history-callout">
            A market can be extremely busy while producing almost no new evidence. Ten million
            views, sales or contracts can keep extending one inherited path instead of testing
            how the same inputs would’ve performed on another.
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
                standard or route to customers determines which paths get recorded. What it
                monopolizes is the evidence society needs to explain the outcome.
              </p>
            </article>
          </div>
          <div className="theorem creator-theorem">
            <div className="theorem__label">The Shadow Futures result</div>
            <blockquote>
              Transactions aren’t the sample size. Real chances for the market to go another way
              are.
            </blockquote>
            <p>
              The paper calls the total of those chances the comparison budget. If that budget
              is finite, no method using a single market history can consistently recover a
              meaningful measure of contribution that rises or falls when contribution does.
              More activity can lengthen the same path without adding the missing experiments.
            </p>
          </div>
        </StorySection>

        <StorySection
          id="firm-markets"
          number={3}
          eyebrow="From scale to evidence"
          title="A growing firm can improve while making its own contribution harder to measure"
          intro="An early customer brings revenue, data, credibility and scale. Those can produce real gains. But as one firm comes to dominate customers, standards and distribution, the market can run out of independent paths that would reveal how much success came from the firm’s inputs and how much from the position created by earlier wins."
          illustration={{
            src: "/illustrations/chapters/firm-flywheel.png",
            alt: "A contract, performance data, a factory and the next customer form a reinforcing loop.",
          }}
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
                "An early contract, retail placement, technical standard or major buyer creates an early lead.",
              ],
              [
                "02",
                "Gain money and information",
                "Sales bring cash, usage data, a track record and easier access to finance.",
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
                "Models, chips and data centers require enormous up-front investment. More customers can fund more capacity, lower average costs and sometimes provide data that improves the service.",
              ],
              [
                "Manufacturing and logistics",
                "A larger order book can pay for better machinery, cheaper purchasing and wider distribution. Those real efficiencies can make the largest supplier cheaper still.",
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
              <h3>Scale should serve the public, not prove what a firm deserves</h3>
              <p>
                Lower costs, better reliability, larger research budgets and useful standards
                are collective economic gains. They don’t turn market power or profit into a
                precise measure of contribution.
              </p>
            </article>
            <article>
              <span className="panel__meta">A measurement problem</span>
              <h3>Market share isn’t an exact contribution score</h3>
              <p>
                Today’s profit can reflect better products and the advantages created by
                yesterday’s sales. One observed market path can’t always separate the two.
              </p>
            </article>
            <article>
              <span className="panel__meta">A competition problem</span>
              <h3>Many firms can still offer only one useful test</h3>
              <p>
                A market can contain many legal competitors while buyers, standards, financing
                and distribution all converge on the same early leader.
              </p>
            </article>
            <article>
              <span className="panel__meta">A policy problem</span>
              <h3>Mergers can erase valuable comparisons</h3>
              <p>
                Merger review should ask whether independent products, experiments and routes
                to customers will disappear, not only whether several company names remain.
              </p>
            </article>
          </div>

          <div className="global-history-callout firm-boundary">
            The goal isn’t to freeze every firm at equal size. It’s to prevent today’s leader
            from closing tomorrow’s contest. When feedback loops can eliminate real comparisons,
            open standards, interoperability, independent procurement trials, support for new
            entrants and structural separation can serve as democratic infrastructure.
          </div>
        </StorySection>

        <StorySection
          id="lorenz-curve"
          number={4}
          eyebrow="What inequality can’t answer"
          title="The Lorenz curve is the symptom. Shadow futures are the missing evidence."
          intro="Debates about extreme inequality often split between two stories. One says the reward broadly reflects talent, work or risk. The other says a small early accident was amplified by cumulative advantage. Shadow Futures reframes the argument: the same visible curve can reflect many different mixes of contribution and reinforced position, and a single market history may not contain the comparisons needed to tell them apart."
          illustration={{
            src: "/illustrations/chapters/lorenz-causes.png",
            alt: "Contribution and reinforced position intertwine beneath the same visible Lorenz curve while other mixtures remain unobserved.",
          }}
        >
          <div
            className="inequality-frames"
            aria-label="Three ways to interpret extreme inequality"
          >
            <article>
              <span className="panel__meta">Common interpretation 1</span>
              <h3>The winner contributed proportionally more</h3>
              <p>
                A huge reward is taken as evidence of much greater talent, effort, judgment
                or risk-bearing. The final gap looks like a contribution score.
              </p>
            </article>
            <article>
              <span className="panel__meta">Common interpretation 2</span>
              <h3>A small early accident became a giant lead</h3>
              <p>
                Cumulative advantage and preferential attachment show how an early break can
                attract more attention, customers and rewards until inequality becomes extreme.
              </p>
            </article>
            <article className="inequality-frames__shadow">
              <span className="panel__meta">What Shadow Futures changes</span>
              <h3>The curve can’t tell us how much of either story is true</h3>
              <p>
                Talent, effort and risk can matter while early position compounds. Because the
                market records only one path, the same final inequality can fit very different
                mixtures of contribution and reinforced advantage.
              </p>
            </article>
          </div>
          <p className="inequality-question">
            Once comparison becomes a design target, the question becomes practical: can a
            platform keep more alternatives testable while reducing the concentration created
            by an inherited lead?
          </p>
          <LorenzHistoryGraph />
          <div className="lorenz-takeaway">
            <strong>The comparison budget is also a design target.</strong>
            <span>
              Platforms and policymakers can preserve independent chances to learn, compete and
              grow before one path becomes the only path the market records.
            </span>
          </div>
          <div className="platform-lorenz-copy">
            <article>
              <h3>OnlyFans and Fanvue</h3>
              <p>
                A curve built from verified payouts would show how subscription income is divided
                among creators. Even a perfect curve couldn’t reveal whether the inequality came
                from better work, an existing following, early discovery, referrals, investment
                or simply being shown first and then shown again.
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
                into tomorrow’s income. Some platform rules leave newcomers more room to break
                through than others.
              </p>
            </article>
          </div>
          <p className="data-note">
            Both curves are model illustrations, not forecasts or OnlyFans or Fanvue payout data.
            The blue curve models one narrow rule: once alternatives’ combined chance would fall
            below 50%, the ranking reserves enough discovery to preserve that comparison floor.
            Portability and independent trials are related institutional examples, not additional
            inputs to the plotted simulation. A real Lorenz curve would require individual creator
            earnings; company totals aren’t enough.
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
          number={5}
          eyebrow="Tax, UBI and social insurance"
          title="The income record can’t isolate contribution from position"
          intro="Existing tax systems use observable measures such as income, profits and wealth; they don’t try to calculate how much of each dollar came from the recipient’s contribution. The paper asks whether one market history could ever isolate the share created by position. Under the theorem’s conditions, it can’t. That means extreme rewards shouldn’t be treated as proof that recipients deserve every dollar. Progressive taxation, antitrust, UBI and social dividends each address a different part of the problem."
          illustration={{
            src: "/illustrations/chapters/social-floor.png",
            alt: "A narrow tower of rewards is partly distributed through channels into a broad floor supporting many people.",
          }}
        >
          <div className="tax-policy-split">
            <article>
              <span className="panel__meta">A theoretical benchmark</span>
              <h3>Could a tax isolate only positional rent?</h3>
              <p>
                The paper tests this demanding ideal to find the limits of what a market record
                can reveal. Current tax systems generally don’t attempt this calculation. The
                result limits claims about exactly what someone deserves; it isn’t a description
                of ordinary tax administration.
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
                "Tax extreme rewards progressively",
                "The largest creator incomes, founder gains and monopoly profits combine real contribution with advantages that scale and history magnify. Higher rates are justified by ability to pay, concentrated power and the public systems that made those gains possible.",
              ],
              [
                "Use antitrust to keep alternative paths open",
                "Merger enforcement, interoperability, structural separation and public options can prevent one platform, standard or distribution channel from becoming the only experiment society gets to observe.",
              ],
              [
                "UBI takes survival out of the merit contest",
                "A universal floor follows people through unstable work, automation and algorithmic exclusion. Nobody should lose the basics of life because a market stops choosing them.",
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
          number={6}
          eyebrow="The distinct contribution"
          title="The market doesn’t just choose a winner. It chooses what can still be known."
          intro="Increasing returns explain compounding. Scaling laws relate size to performance. Preferential attachment explains why success attracts more success. Lorenz curves describe inequality. Shadow Futures identifies the missing step: self-reinforcing markets can destroy the comparison paths needed to measure contribution from the one history we observe."
          illustration={{
            src: "/illustrations/chapters/recorded-path.png",
            alt: "Several possible paths approach a selector, but only one continues into the market record.",
          }}
          dark
        >
          <div className="creator-closing">
            <div className="creator-closing__brand">
              <p className="creator-closing__eyebrow">One observed market. Many missing experiments.</p>
              <p className="creator-closing__title">Shadow Futures</p>
              <p className="creator-closing__subtitle">
                Contribution Uncertainty and the Self-Reinforcing Market
              </p>
            </div>
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
              <Link className="button" href="/faq">
                Questions and answers
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
          <Link href="/faq">Read the Shadow Futures FAQ</Link>
        </div>
      </footer>
    </>
  );
}
