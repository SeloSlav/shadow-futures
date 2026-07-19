export type FaqEntry = {
  id: string;
  question: string;
  answer: string[];
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
          "Merger review should ask how many genuinely independent routes to customers, capital, distribution, and experimentation will remain—not only how many company names survive.",
          "Two channels that look duplicative in a static cost calculation may be informational complements. Combining them can save fixed costs while eliminating a comparison path that society would otherwise learn from.",
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
      "The paper does not describe how taxes are currently calculated. It studies a theoretical benchmark used in debates about desert: separating earned contribution from positional rent.",
    entries: [
      {
        id: "tax-policy",
        question: "What does Shadow Futures imply for tax policy?",
        answer: [
          "Shadow Futures does not claim that current tax systems calculate each person’s causal contribution. They already rely on observable bases such as income, profits, property, consumption, and wealth.",
          "The paper’s narrower implication is that realized income, profit, or market share cannot identify an exact contribution-versus-position split. Tax choices should be defended through observable effects and public purposes rather than presented as a forensic measurement of that split.",
        ],
      },
      {
        id: "progressive-taxation",
        question: "Does the argument support progressive taxation?",
        answer: [
          "Progressive taxation can be defended on familiar grounds such as ability to pay, public revenue, social insurance, economic power, and the shared institutions and infrastructure behind private success.",
          "Shadow Futures contributes a narrower point: pretax market income does not settle the separate moral question of exact desert. The theorem does not mechanically select a tax rate.",
        ],
      },
      {
        id: "ubi-social-dividend",
        question: "Why are UBI and social dividends relevant to Shadow Futures?",
        answer: [
          "A universal basic income or social dividend is attribution-invariant: eligibility and payment do not depend on estimating each recipient’s exact causal contribution to market rewards.",
          "That makes these policies conceptually compatible with contribution uncertainty. They can provide a floor or share common gains without requiring the missing earned-versus-positional calculation.",
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
          "The claim is about recoverability: one reinforced history may not contain enough independent comparison to estimate how large those causal effects were.",
        ],
      },
      {
        id: "theorem-result",
        question: "What does the Shadow Futures theorem prove?",
        answer: [
          "Under a common predictable design, local equivalence, comparison-dominated separation, and finite total comparison, distinct contribution parameters generate mutually absolutely continuous complete-history laws.",
          "In plain language, no estimator based on one realized history can consistently recover every nonconstant contribution quantity, and no test can perfectly separate two contribution parameters as the history grows.",
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
      },
    ],
  },
];
