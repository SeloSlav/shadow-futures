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
      "The central claim is about missing evidence, not merely unequal rewards or the fact that success compounds.",
    entries: [
      {
        id: "what-are-shadow-futures",
        question: "What are shadow futures?",
        answer: [
          "Shadow futures are alternative market histories that had a positive chance of happening under the same inputs, rules, and starting conditions. Different allocation shocks—such as who gets the first customer—can lead to different final rewards.",
          "Imagine two equally equipped bakeries. An early order gives one reviews and visibility, helping it win later orders. A history in which the other bakery got that first order is a shadow future. This illustrates a possible feedback mechanism; it doesn’t establish how any actual bakery market works.",
          "Those alternative histories are the missing repetitions needed to estimate what productive inputs caused. They do not show that work, quality, effort, capital, or risk had no effect.",
        ],
      },
      {
        id: "what-is-contribution-uncertainty",
        question: "What’s contribution uncertainty?",
        answer: [
          "Contribution uncertainty is uncertainty about the causal difference a productive input made to a reward inside a self-reinforcing market. The question is: how would rewards change if quality, effort, or another input changed, including the feedback that followed?",
          "It isn’t uncertainty about whether the work happened. Hours, code, investment, quality, and risk can be perfectly verified while the market still lacks the comparison histories needed to measure their causal contribution.",
        ],
      },
      {
        id: "different-from-preferential-attachment",
        question:
          "How’s Shadow Futures different from preferential attachment, increasing returns, or network effects?",
        answer: [
          "Preferential attachment, increasing returns, network effects, and scaling laws explain why an early lead can grow. Shadow Futures asks a different question: what happens to our ability to measure contribution while that lead grows?",
          "The paper shows that, under stated conditions, self-reinforcing allocation can exhaust the comparisons needed to estimate the effects of productive inputs from one market history. Increasing returns or network effects alone do not establish that those conditions hold.",
        ],
      },
      {
        id: "different-from-inequality",
        question: "Is Shadow Futures simply an argument about inequality?",
        answer: [
          "No. Inequality describes how rewards are distributed. Shadow Futures studies whether one realized market history contains enough evidence to estimate what productive inputs caused. A concentrated outcome alone does not prove the paper’s learning impossibility.",
          "The problem can exist whether society considers the resulting inequality fair or unfair. It concerns what the market record can actually prove.",
        ],
      },
      {
        id: "transaction-count-versus-evidence",
        question: "Why aren’t more transactions necessarily more evidence?",
        answer: [
          "When a leader is almost certain to win, the model limits how much information the next allocation can add about the effects of productive inputs. If the leader’s chance stays at 99.9 percent for 1,000 rounds, those rounds add just 1 to the comparison budget: 1,000 × 0.001. That is a sum of probabilities, not a claim that exactly one rival wins.",
          "The market can therefore be commercially busy while adding little attribution evidence. The 99.9 percent example shows weak comparison; by itself it does not prove a finite lifetime budget, because a fixed small chance added forever still sums to infinity.",
        ],
      },
      {
        id: "what-is-comparison-budget",
        question: "What’s the comparison budget?",
        answer: [
          "At each round, take the probability that the next reward goes to someone other than the current favorite. Add those probabilities across rounds. That sum is the comparison budget; the favorite may change along the way. It is a sum of model probabilities, not a count of actual rival wins.",
          "In the paper’s model, it places an upper bound on information about the effects of productive inputs. Under the theorem’s other assumptions, a finite lifetime budget prevents one method from learning a contribution measure reliably across all candidate explanations as that single history grows. A large or infinite budget alone does not guarantee learning: the comparisons must also distinguish the relevant input effects.",
        ],
      },
    ],
  },
  {
    id: "markets-ai",
    label: "02 / Markets and AI",
    title: "Creators, firms, and data centers",
    intro:
      "These examples illustrate feedback. Applying the theorem to a real market requires checking its conditions and available evidence.",
    entries: [
      {
        id: "creator-platforms",
        question: "How does Shadow Futures apply to social media and creator platforms?",
        answer: [
          "A creator’s early audience can increase the chance of receiving the next recommendation, subscriber, sponsor, or sale. Making a better video can still improve the chance of success while earlier visibility also feeds later exposure.",
          "This illustrates the paper’s mechanism. If a platform’s comparison budget is finite and the theorem’s other conditions hold, no method can learn a contribution measure reliably across all candidate input effects from its single history. The paper does not test individual platforms or calculate a creator’s percentage of merit or luck.",
        ],
      },
      {
        id: "competitive-firms",
        question: "How does Shadow Futures apply to firms in a competitive market?",
        answer: [
          "An early customer can give a firm revenue, data, credibility, financing, distribution, and lower unit costs. Those gains can improve the product while also making the next customer easier to win.",
          "Many firms may remain legally present even as customers, finance, and distribution follow one inherited path. Market share then records an outcome, but it does not by itself measure the causal contribution of the firm’s inputs. Whether the theorem applies depends on how opportunities and evidence are generated.",
        ],
      },
      {
        id: "ai-data-centers",
        question: "Why does Shadow Futures matter for AI, chips, cloud computing, and data centers?",
        answer: [
          "As an application beyond the paper’s formal model, consider a firm whose first customers finance more compute and better service. Those improvements may attract the next customers, who fund the next expansion. That is a possible self-reinforcing development path.",
          "The question is whether competing firms or technical approaches get enough independent opportunities to produce informative comparisons. Concentration and high fixed costs alone do not prove the theorem applies to AI or cloud markets; the paper provides no sector-specific estimate of how much success comes from contribution or position.",
        ],
      },
      {
        id: "ai-agents-x402-agentic-economy",
        question:
          "How do AI agents, x402, and the agentic economy relate to Shadow Futures and UBI?",
        answer: [
          "This is an application beyond the paper. AI agents can make software-mediated purchasing decisions. x402 is an open internet payment standard that lets software pay for APIs and content over HTTP using the 402 Payment Required status code.",
          "Imagine agents repeatedly buying from a seller because earlier purchases boosted its ranking or reputation. A million automated purchases may mostly continue that lead rather than provide a million independent tests. Whether this reaches the theorem’s impossibility depends on the comparison budget and its other assumptions, not on the payment protocol itself.",
          "A UBI or social dividend could share gains without reconstructing an exact merit ranking. Choosing it also requires judgments about income security, ownership, financing, and incentives. The theorem neither predicts where automation income will go nor selects UBI as the uniquely correct response.",
        ],
        inlineLink: {
          paragraphIndex: 0,
          text: "x402 is an open internet payment standard",
          href: "https://docs.x402.org/introduction",
        },
      },
      {
        id: "lorenz-curve",
        question: "What can a Lorenz curve tell us, and what can’t it tell us?",
        answer: [
          "A Lorenz curve shows how concentrated income or rewards are. It can accurately describe the final distribution.",
          "By itself, it can’t reveal how much of that distribution came from contribution, early visibility, inherited position, or feedback. Think of it as a scoreboard: it tells us how rewards ended up divided, but does not explain what caused the scores. Other comparisons or evidence are needed for that question.",
        ],
      },
    ],
  },
  {
    id: "competition-evidence",
    label: "03 / Competition and evidence",
    title: "When a market stops learning",
    intro:
      "Competition can preserve paths that produce evidence, provided those paths are sufficiently independent and informative.",
    entries: [
      {
        id: "epistemic-monopoly",
        question: "What’s an epistemic monopoly?",
        answer: [
          "In this paper, an epistemic monopoly means control over the production of comparison paths needed to answer a particular contribution question, using a stated set of evidence.",
          "It doesn’t require one legal seller. Many firms can share one ranking and distribution history. Conversely, a regulated monopoly could preserve learning through randomized pilots and independent evaluation. Economic monopoly and epistemic monopoly can coincide, but neither automatically implies the other.",
        ],
      },
      {
        id: "competition-as-discovery",
        question: "Why does independent competition produce information?",
        answer: [
          "Separate marketplaces, distributors, funders, journals, procurement channels, and evaluators can let similar inputs meet different audiences and early shocks. Think of giving the same songs to several fresh listening groups instead of treating repeat plays on one popularity chart as independent tests.",
          "The appendix shows that many independent markets can identify a shared contribution parameter under additional conditions, including that different parameter values predict different observable outcomes. More platform names or a reset alone do not ensure this: paths may share the same ranking, inherited advantage, or uninformative inputs.",
        ],
      },
      {
        id: "mergers-and-antitrust",
        question: "What does Shadow Futures imply for mergers and antitrust?",
        answer: [
          "The paper adds an evidence question to competition policy: could a merger remove an independent route through which rivals receive exposure and build a record? Two channels may look duplicative in cost terms while providing different comparison histories.",
          "This supports examining the number and independence of routes to customers, capital, distribution, and experimentation. It does not prove every merger is harmful or select a remedy automatically. The paper says welfare depends on decision errors and the cost of preserving independent channels.",
        ],
      },
      {
        id: "preserve-comparisons",
        question: "What institutions can preserve shadow futures and useful comparison?",
        answer: [
          "The paper discusses randomized exposure, multihoming, portability, interoperability, public options, structural separation, and independent procurement as ways to preserve alternative paths. Independent trials can also supply new comparisons.",
          "Their value depends on design: do they actually create informative variation rather than repeat the same inherited advantage? They can improve the evidence available to learn contribution, but none is a universal guarantee of identification.",
        ],
      },
    ],
  },
  {
    id: "tax-redistribution",
    label: "04 / Tax and redistribution",
    title: "What market income can’t certify",
    intro:
      "The formal result limits exact contribution-based taxation. Choosing redistribution or tax rates also requires explicit social goals and evidence about costs and incentives.",
    entries: [
      {
        id: "tax-policy",
        question: "What does Shadow Futures imply for tax policy?",
        answer: [
          "The paper’s tax theorem asks whether a rule using the observed record can tax exactly the part of a reward left after assigned contribution. If two structural economies produce the same observable law but assign different contributions to the same reward, that rule must choose the same tax even though the target taxes differ. It cannot be exact in both.",
          "This is a separate result from the finite-comparison theorem. It requires observationally identical economies with different contribution assignments; it does not follow merely from a high income or a concentrated market. Nor does it establish that all high income is rent, that risk is fictitious, or that taxation is impossible.",
          "The paper suggests targeting observable compounding, using ranges of contribution compatible with the evidence, and considering an unconditional social dividend. Progressive taxation, public services, or UBI can also be argued for using goals such as ability to pay, security, and sharing common gains. Those are additional policy judgments; the theorem supplies no exact tax rate or deserved share.",
        ],
      },
      {
        id: "progressive-taxation",
        question: "Does the argument support progressive taxation?",
        answer: [
          "It can strengthen a case for progressive taxation by challenging the claim that a large market payout is a precise certificate of individual contribution. Real work may combine with early visibility and feedback, while the observed record cannot reliably measure the input’s causal effect under the paper’s conditions.",
          "Progressivity still requires an argument about goals such as ability to pay, social insurance, or limiting concentrated power, alongside evidence about incentives and costs. The paper does not derive a progressive schedule, prove that every dollar is unearned, or establish how much any individual should pay.",
        ],
      },
      {
        id: "tax-successful-creators",
        question: "Should highly successful platform creators be taxed more?",
        answer: [
          "The paper does not establish a special tax rate for creators. A creator can be talented and hardworking, and early exposure may also amplify later opportunities. A large payout alone does not tell us how much either mechanism contributed.",
          "Applying progressive taxation to large creator incomes is consistent with policy goals such as ability to pay and income security. That choice needs those goals and evidence about tax effects; it cannot be justified by an invented luck percentage or a claim that the theorem measured one creator’s merit.",
        ],
      },
      {
        id: "ubi-social-dividend",
        question: "Why are UBI and social dividends relevant to Shadow Futures?",
        answer: [
          "The paper notes that an unconditional social dividend does not have to reconstruct the contribution ranking that the evidence may fail to identify. An equal unconditional payment can be made without deciding whose market success was deserved. UBI has a similar feature.",
          "This explains why such transfers are relevant, but it does not prove they are optimal or specify their size or funding. Their case also rests on goals such as security and sharing common gains. Transfers and competition policies address different tasks: providing income and preserving paths that may produce new evidence.",
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
        question: "Does Shadow Futures claim that work, talent, quality, or risk don’t matter?",
        answer: [
          "No. Productive inputs can be real, perfectly observed, and directly affect every reward probability. A better product or greater effort can genuinely improve the chance of winning.",
          "The problem is that one market history may not contain enough comparison to estimate how much those inputs mattered. Large prizes may still encourage useful experimentation, and risk-bearing may justify a premium before the outcome is known. Incentives for taking a risk and explaining a realized jackpot are different questions.",
        ],
      },
      {
        id: "theorem-result",
        question: "What does the Shadow Futures theorem prove?",
        answer: [
          "Imagine two versions of the same allocation system: better work has a larger effect on winning in one than in the other. Under the theorem’s conditions, their complete history laws remain mutually absolutely continuous: neither assigns positive probability to an event the other calls impossible. They need not assign the same probabilities.",
          "Consequently, no one method using a single history can learn a contribution measure reliably across all candidate explanations when that measure differs between them, even as more rounds are added. A test between two input effects cannot make both kinds of mistake vanish. Some evidence and partial learning are still possible; the theorem rules out universally reliable recovery from that evidence.",
          "The conditions matter: every candidate effect uses the same input and position rules applied to the recorded past; the same next recipients remain possible; the statistical difference between competing predictions is bounded by a fixed multiple of the chance of choosing outside the leader; and the lifetime comparison budget is finite under every candidate effect. The paper gives a bounded conditional-logit model with these properties. Any extra observations whose probabilities also depend on the effect must have their information counted separately.",
        ],
      },
      {
        id: "superlinear-reinforcement",
        question: "Does the result require superlinear preferential attachment?",
        answer: [
          "No. The general theorem is organized around a finite lifetime comparison budget, not a particular power law. In the paper’s reinforced model with finitely many alternatives and fixed input profiles, sufficiently strong feedback makes one alternative receive every reward after a finite random time and exhausts that budget. Polynomial feedback with exponent above 1 is one such case.",
          "Linear preferential attachment can generate heavy tails or power laws without satisfying the paper’s exact impossibility condition. Concentration alone isn’t the theorem.",
        ],
      },
      {
        id: "latent-position",
        question: "What changes when starting position is hidden?",
        answer: [
          "The paper gives a separate example in which a stronger input effect and a compensating reduction in hidden starting advantage produce exactly the same observable allocation probabilities. The data then cannot distinguish those explanations at all.",
          "That is different from the finite-budget result, which can hold even when position is observed and different parameter values produce different distributions. Hidden-position ambiguity is not automatically cured by running more copies of the same unidentified design; new evidence or restrictions would be needed.",
        ],
      },
      {
        id: "hidden-quality",
        question: "How’s this different from hidden quality or Akerlof’s market for lemons?",
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
