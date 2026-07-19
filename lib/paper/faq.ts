import { PAPER } from "@/lib/paper/citation";

export type FaqEntry = {
  id: string;
  question: string;
  answer: string[];
  inlineLink?: {
    paragraphIndex: number;
    text: string;
    href: string;
  };
};

export type FaqGroup = {
  id: string;
  label: string;
  title: string;
  intro: string;
  entries: FaqEntry[];
};

export const FAQ_GROUPS: FaqGroup[] = [
  {
    id: "core-idea",
    label: "01 / Core idea",
    title: "What Shadow Futures means",
    intro:
      "The central claim is about missing evidence—not merely unequal rewards or the fact that success compounds.",
    entries: [
      {
        id: "what-are-shadow-futures",
        question: "What are shadow futures?",
        answer: [
          "Shadow futures are the unrealized market histories that could have happened with the same people, firms, productive inputs, and rules, but different early customers, rankings, audiences, or random shocks.",
          "They matter because those alternate histories are the missing experimental repetitions needed to estimate how much work, quality, effort, judgment, capital, or risk contributed to the reward we actually observed.",
        ],
      },
      {
        id: "what-is-contribution-uncertainty",
        question: "What is contribution uncertainty?",
        answer: [
          "Contribution uncertainty is uncertainty about how much an observed productive input caused the final reward inside a self-reinforcing market.",
          "It is not uncertainty about whether the work happened. Hours, code, investment, quality, and risk can be perfectly verified while the market still lacks the comparison histories needed to measure their causal contribution.",
        ],
      },
      {
        id: "different-from-preferential-attachment",
        question:
          "How is Shadow Futures different from preferential attachment, increasing returns, or network effects?",
        answer: [
          "Preferential attachment, increasing returns, network effects, and scaling laws explain why an early lead can grow. Shadow Futures asks a different question: what happens to our ability to measure contribution while that lead grows?",
          "The distinctive claim is that self-reinforcing allocation can destroy the independent comparisons needed to explain its own rewards. The familiar theories explain concentration; Shadow Futures identifies the resulting attribution limit.",
        ],
      },
      {
        id: "different-from-inequality",
        question: "Is Shadow Futures simply an argument about inequality?",
        answer: [
          "No. Inequality is a distributional outcome. Shadow Futures is an information problem: one realized market history may be unable to reveal how much of its reward ranking came from contribution rather than accumulated position.",
          "The problem can exist whether society considers the resulting inequality fair or unfair. It concerns what the market record can actually prove.",
        ],
      },
      {
        id: "transaction-count-versus-evidence",
        question: "Why are more transactions not necessarily more evidence?",
        answer: [
          "A transaction adds useful attribution evidence only when the allocation still has a meaningful chance to go another way. If the leader has a 99.9 percent chance of receiving the next customer, another thousand sales mostly extend the inherited path.",
          "The market can therefore be commercially busy while its experiment is nearly exhausted. Volume is not the same thing as independent comparison.",
        ],
      },
      {
        id: "what-is-comparison-budget",
        question: "What is the comparison budget?",
        answer: [
          "The comparison budget is the total probability, accumulated over time, that the next reward could go to someone other than the current favorite.",
          "It measures how much real comparison the market still produces. When total comparison is finite under the paper’s assumptions, no method using that single history can consistently recover every meaningful measure of contribution.",
        ],
      },
    ],
  },
  {
    id: "markets-ai",
    label: "02 / Markets and AI",
    title: "Creators, firms, and data centers",
    intro:
      "The mechanism applies wherever an early reward changes access to the next opportunity.",
    entries: [
      {
        id: "creator-platforms",
        question: "How does Shadow Futures apply to social media and creator platforms?",
        answer: [
          "On TikTok, YouTube, Instagram, Twitch, OnlyFans, Fanvue, Patreon, and similar platforms, an early audience can raise the chance of receiving the next recommendation, subscriber, sponsor, or sale.",
          "The final follower or income ranking then records both the creator’s work and the extra opportunities created by earlier visibility. Without alternate exposure histories, the platform cannot recover the exact split from the winning path alone.",
        ],
      },
      {
        id: "competitive-firms",
        question: "How does Shadow Futures apply to firms in a competitive market?",
        answer: [
          "An early customer can give a firm revenue, data, credibility, financing, distribution, and lower unit costs. Those gains can improve the product while also making the next customer easier to win.",
          "Many firms may remain legally present even as customers, standards, finance, and distribution follow one inherited path. Market share can therefore reflect real productive gains and accumulated position without revealing the exact contribution of either.",
        ],
      },
      {
        id: "ai-data-centers",
        question: "Why does Shadow Futures matter for AI, chips, cloud computing, and data centers?",
        answer: [
          "AI and cloud markets combine enormous fixed costs with feedback through customers, compute, data, engineering talent, financing, and ecosystem compatibility. An early lead can fund more capacity and better service, which attracts the next customer and finances the next expansion.",
          "The concern is not merely that scale creates concentration. It is that a small number of reinforced development paths may become the only histories society gets to observe, making it harder to know which firms, models, or technical choices would have succeeded under different early allocations of compute, capital, and demand.",
        ],
      },
      {
        id: "ai-agents-x402-agentic-economy",
        question:
          "How do AI agents, x402, and the agentic economy relate to Shadow Futures and UBI?",
        answer: [
          "AI agents are software systems that can choose and act with less human input. x402 is an open internet payment standard built on HTTP 402 that lets software pay programmatically for APIs, data, compute, and other digital services. In an agentic economy, agents could buy inputs, hire services, sell outputs, and make payments around the clock.",
          "x402 does not itself cause monopoly. The Shadow Futures problem appears if early agent purchases improve a seller’s ranking, reputation, data, revenue, or compatibility, which then makes later agents more likely to choose the same seller. Millions of machine payments may look like millions of tests while mostly extending one inherited path. Competing agents and firms may never receive enough business to show what they could have contributed.",
          "That prospect strengthens the case for UBI or a social dividend. If automation shifts income toward the owners of models, compute, data, platforms, and agent networks, everyone should share in the productivity gains without having to prove an exact personal contribution to each machine transaction. Progressive taxes on the largest incomes, fortunes, profits, and economic rents can fund that floor; antitrust, interoperability, open standards, and public options are still needed to keep alternative economic paths open.",
        ],
        inlineLink: {
          paragraphIndex: 0,
          text: "x402 is an open internet payment standard",
          href: "https://docs.x402.org/introduction",
        },
      },
      {
        id: "lorenz-curve",
        question: "What can a Lorenz curve tell us—and what can it not tell us?",
        answer: [
          "A Lorenz curve shows how concentrated income or rewards are. It can accurately describe the final distribution.",
          "It cannot reveal how much of that distribution came from contribution, early visibility, inherited position, or feedback. The Lorenz curve is the scoreboard; shadow futures are the missing repetitions needed to explain the score.",
        ],
      },
    ],
  },
  {
    id: "competition-evidence",
    label: "03 / Competition and evidence",
    title: "When a market stops learning",
    intro:
      "Competition matters not only for price and choice, but also for producing independent evidence.",
    entries: [
      {
        id: "epistemic-monopoly",
        question: "What is an epistemic monopoly?",
        answer: [
          "An epistemic monopoly exists when one market history controls the production of the comparisons needed to explain an outcome.",
          "It does not require one legal seller. Thousands of creators or firms can remain active while one ranking system, technical standard, procurement channel, or distribution route determines which paths receive enough opportunities to generate evidence.",
        ],
      },
      {
        id: "competition-as-discovery",
        question: "Why does independent competition produce information?",
        answer: [
          "Independent marketplaces, distributors, funders, journals, procurement channels, and evaluators let similar inputs meet different audiences, rankings, and early shocks.",
          "Those separate paths act like replications. Time only lengthens an inherited history; independent competition can reopen the experiment.",
        ],
      },
      {
        id: "mergers-and-antitrust",
        question: "What does Shadow Futures imply for mergers and antitrust?",
        answer: [
          "Shadow Futures supports strong antitrust because concentrated control does more than raise prices or reduce choice. It can eliminate the independent market paths society needs to discover which firms, products, and technologies could succeed.",
          "Merger review should ask how many genuinely independent routes to customers, capital, distribution, and experimentation will remain—not only how many company names survive. Where one platform, standard, or buyer controls the experiment, interoperability, structural separation, public options, and merger enforcement can keep alternate futures open.",
        ],
      },
      {
        id: "preserve-comparisons",
        question: "What institutions can preserve shadow futures and useful comparison?",
        answer: [
          "Randomized exposure, independent trials, multihoming, data and audience portability, interoperability, open standards, public options, structural separation, and independent procurement can keep alternate paths alive.",
          "These policies do more than increase fairness or entry. They create variation that helps society learn what people, firms, and technologies actually contribute.",
        ],
      },
    ],
  },
  {
    id: "tax-redistribution",
    label: "04 / Tax and redistribution",
    title: "What market income cannot certify",
    intro:
      "The paper does not say success is fake. It shows why a self-reinforcing market may be unable to certify that an outsized reward was created by the winner alone.",
    entries: [
      {
        id: "tax-policy",
        question: "What does Shadow Futures imply for tax policy?",
        answer: [
          "Shadow Futures strengthens the case for taxing extreme incomes, wealth, founder gains, creator fortunes, and monopoly profits at higher rates. A person or firm can do genuinely valuable work and still receive a reward far larger than the realized market history can attribute to that work.",
          "Once an early win brings the next customer, recommendation, dataset, contract, investor, or technical advantage, the winning path grows while other people and firms lose chances to build a record, improve, attract capital, or even enter. Some excluded rivals may have been less capable and some may have been equally capable; the point is that the market closes the comparisons that could have told us. Under the paper’s conditions, the exact split between contribution and accumulated position cannot be recovered from the winner’s history.",
          "Policy should not resolve that missing evidence by awarding the entire surplus to the winner and calling it merit. Progressive taxation can return part of outsized rewards through UBI, social dividends, public services, and shared investment. Antitrust, interoperability, open standards, and public options should lower barriers to entry and keep rival paths alive. The theorem does not select an exact tax rate, but it rejects the idea that the market payout itself proves exact personal desert.",
        ],
      },
      {
        id: "progressive-taxation",
        question: "Does the argument support progressive taxation?",
        answer: [
          "Yes. Shadow Futures strengthens the case for progressive taxation because very high market incomes are not clean measurements of individual contribution. At the top, real work can be combined with early visibility, inherited position, global scale, and feedback that turns one break into years of additional opportunity.",
          "The case for taxing those rewards at higher rates rests on ability to pay, concentrated economic power, social insurance, and the shared institutions and infrastructure behind private success. The theorem does not choose an exact rate or prove that every dollar is unearned; it shows why pretax income should not be treated as a precise certificate of desert.",
        ],
      },
      {
        id: "tax-successful-creators",
        question: "Should highly successful platform creators be taxed more?",
        answer: [
          "Yes, the project supports applying strongly progressive income and wealth taxation to the largest creator fortunes and platform windfalls. A top creator may be talented and hardworking, while the size of the final reward also reflects a global ranking system that repeatedly amplified an early lead.",
          "That makes the outcome partly lottery-like without making the work fake: many similarly capable creators cannot rerun the same market with a different first audience. We cannot assign an exact luck percentage to one person, but that uncertainty is not a reason to treat an extreme payout as pure merit or exempt it from progressive taxation.",
        ],
      },
      {
        id: "ubi-social-dividend",
        question: "Why are UBI and social dividends relevant to Shadow Futures?",
        answer: [
          "UBI and social dividends provide security without asking a market ranking to decide who deserves the basics of life. That matters when global platforms and automated markets can direct enormous rewards toward a few winners while equally capable people lose visibility, customers, or bargaining power.",
          "These policies should complement rather than replace progressive taxation and antitrust. UBI provides a floor, a social dividend shares gains built on public knowledge and infrastructure, and antitrust keeps independent economic paths open.",
        ],
      },
    ],
  },
  {
    id: "theorem-scope",
    label: "05 / The theorem",
    title: "Conditions, boundaries, and authorship",
    intro:
      "The formal result is stronger than a simulation, but it applies under stated information and comparison conditions.",
    entries: [
      {
        id: "does-work-matter",
        question: "Does Shadow Futures claim that work, talent, quality, or risk do not matter?",
        answer: [
          "No. Productive inputs can be real, perfectly observed, and directly affect every reward probability. A better product or greater effort can genuinely improve the chance of winning.",
          "The problem is that one winning history may not contain enough genuine comparison to tell us how much those strengths mattered. Policy does not have to choose between “all merit” and “all luck”: society can reward creation while taxing extreme incomes progressively and using antitrust to preserve rival paths.",
        ],
      },
      {
        id: "theorem-result",
        question: "What does the Shadow Futures theorem prove?",
        answer: [
          "Imagine two worlds. In one, better work has a very large effect on who wins. In the other, it has a smaller effect. If an early winner eventually receives almost every new customer, view, or contract, both worlds can leave behind records that look compatible with the same winning story.",
          "The theorem proves that, under its conditions, no statistical method can always look at that one history and work out which world produced it—even if the market continues forever. Once genuine chances to compare different people or firms run out, more activity keeps extending the story but cannot recreate the missing experiment.",
        ],
      },
      {
        id: "superlinear-reinforcement",
        question: "Does the result require superlinear preferential attachment?",
        answer: [
          "No. The general theorem is organized around finite comparison, not a particular urn model or power law. Strong or superlinear reinforcement is one sharp case because it can exhaust the comparison budget and produce eventual allocation monopoly.",
          "Linear preferential attachment can generate heavy tails or power laws without satisfying the paper’s exact impossibility condition. Concentration alone is not the theorem.",
        ],
      },
      {
        id: "hidden-quality",
        question: "How is this different from hidden quality or Akerlof’s market for lemons?",
        answer: [
          "A lemons problem begins with relevant quality hidden from one side of a trade. The Shadow Futures problem can remain even when productive inputs are public, measured without error, and explicitly used by the allocation rule.",
          "The missing information is historical rather than private: the market never generated the alternate allocation paths needed to estimate what those inputs caused.",
        ],
      },
      {
        id: "author-and-paper",
        question: "Who developed the Shadow Futures argument, and where can I read the paper?",
        answer: [
          "Shadow Futures: Contribution Uncertainty and the Self-Reinforcing Market is by Martin Erlic. The paper was first posted in December 2025 and revised in July 2026.",
          "The complete paper and technical appendix are available on SSRN at abstract ID 6003994.",
        ],
        inlineLink: {
          paragraphIndex: 1,
          text: "complete paper and technical appendix",
          href: PAPER.ssrnUrl,
        },
      },
    ],
  },
];
