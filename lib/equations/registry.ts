import type {
  ControlDefinition,
  EquationDefinition,
  EquationKind,
  VariableDefinition,
} from "./types";

const v = (symbol: string, name: string, definition: string): VariableDefinition => ({
  symbol,
  name,
  definition,
});

const betaControl: ControlDefinition = {
  symbol: "\\beta",
  label: "Contribution coefficient β",
  min: -2,
  max: 3,
  step: 0.05,
  defaultValue: 1,
};

const rhoControl: ControlDefinition = {
  symbol: "\\rho",
  label: "Reinforcement exponent ρ",
  min: 0,
  max: 2.5,
  step: 0.05,
  defaultValue: 1.35,
};

const commonAssumptions = [
  "The verified-input design and state indices are common predictable functions of the canonical history.",
  "Any additional parameter-dependent observations belong in the statistical experiment.",
];

type Entry = Omit<EquationDefinition, "assumptions" | "visualization" | "derivationSteps"> & {
  assumptions?: string[];
  visualization?: string;
  derivationSteps?: string[];
};

const eq = (entry: Entry): EquationDefinition => ({
  ...entry,
  assumptions: entry.assumptions ?? commonAssumptions,
  visualization:
    entry.visualization ??
    "The numerical trace below changes one meaningful scalar while holding the displayed design fixed.",
  derivationSteps: entry.derivationSteps ?? [entry.plainLanguage],
});

export const EQUATIONS: EquationDefinition[] = [
  eq({
    id: "conditional-allocation",
    title: "Conditional allocation probability",
    section: "Main article §2",
    equationNumber: "1",
    kind: "definition",
    latex:
      "\\Pr_\\beta(J_{t+1}=i\\mid\\mathcal F_t)=p_{it}(\\beta)=\\frac{\\exp(x_{it}^{\\top}\\beta+s_{it})}{\\sum_{j=1}^{n}\\exp(x_{jt}^{\\top}\\beta+s_{jt})}",
    plainLanguage:
      "Verified inputs load directly into reward odds through β, while the predictable state index records accumulated position.",
    variables: [
      v("J_{t+1}", "recipient", "The alternative receiving the next reward."),
      v("x_{it}", "verified input", "Observed productive profile for alternative i at date t."),
      v("\\beta", "contribution parameter", "Direct loading of verified inputs on log reward odds."),
      v("s_{it}", "state index", "Predictable position inherited from the observed history."),
    ],
    controls: [betaControl],
    visualization: "Probability bars show how β shifts the next allocation while the state is fixed.",
    derivationSteps: [
      "Form each composite index xᵢₜᵀβ + sᵢₜ.",
      "Exponentiate each index to obtain a positive weight.",
      "Divide by the sum of all weights; the probabilities are positive and sum to one.",
    ],
  }),
  eq({
    id: "log-odds-decomposition",
    title: "Log-odds decomposition",
    section: "Main article §2",
    kind: "identity",
    source: "inline",
    latex:
      "\\log\\frac{p_{it}(\\beta)}{p_{kt}(\\beta)}=(x_{it}-x_{kt})^\\top\\beta+(s_{it}-s_{kt})",
    plainLanguage:
      "Pairwise log odds separate the direct verified-input loading from the predictable difference in accumulated position.",
    variables: [
      v("p_{it}/p_{kt}", "pairwise odds", "Conditional odds of i relative to k."),
      v("(x_{it}-x_{kt})^\\top\\beta", "contribution contrast", "Direct input difference loaded by β."),
      v("s_{it}-s_{kt}", "position contrast", "Difference in predictable state indices."),
    ],
    controls: [betaControl],
    visualization: "Contribution and position contrasts stack to form pairwise log odds.",
    derivationSteps: [
      "Take the ratio of two conditional-logit probabilities.",
      "The common softmax denominator cancels.",
      "Taking logs separates the input and state contrasts additively.",
    ],
  }),
  eq({
    id: "residual-contestability",
    title: "Residual contestability",
    section: "Main article §2",
    equationNumber: "2",
    kind: "definition",
    latex:
      "\\varepsilon_t(\\beta)=1-p_{w_t t}(\\beta)=1-\\max_i p_{it}(\\beta)",
    plainLanguage:
      "Residual contestability is the probability mass available for the next reward to go somewhere other than the current leader.",
    variables: [
      v("\\varepsilon_t", "residual contestability", "Probability mass outside the current leader."),
      v("w_t", "current leader", "Any maximizer of the conditional allocation probability."),
      v("p_{it}", "allocation probability", "Conditional chance that i receives the next reward."),
    ],
    controls: [rhoControl],
    visualization: "The gauge reports probability mass outside the largest probability bar.",
  }),
  eq({
    id: "comparison-budget",
    title: "Comparison budget",
    section: "Main article §2",
    equationNumber: "3",
    kind: "definition",
    latex:
      "B_T(\\beta)=\\sum_{t=0}^{T-1}\\varepsilon_t(\\beta),\\qquad B_\\infty(\\beta)=\\sum_{t=0}^{\\infty}\\varepsilon_t(\\beta)",
    plainLanguage:
      "The budget accumulates remaining alternatives, not transactions. One nearly certain allocation adds almost no comparison.",
    variables: [
      v("B_T", "finite comparison budget", "Cumulative residual contestability through horizon T."),
      v("B_\\infty", "total comparison budget", "The infinite-horizon sum when it exists."),
      v("T", "transaction horizon", "Number of observed allocations."),
    ],
    controls: [rhoControl],
    visualization: "A linear transaction counter is contrasted with the cumulative ε path.",
    derivationSteps: [
      "Compute the leader probability at each date.",
      "Subtract it from one to obtain εₜ.",
      "Add εₜ over dates; transaction count and comparison budget needn’t grow together.",
    ],
  }),
  eq({
    id: "information-upper-bound",
    title: "Fisher information and its upper bound",
    section: "Main article §3",
    equationNumber: "4",
    kind: "bound",
    latex: "\\operatorname{tr} I_t(\\beta)\\le D_X^2\\varepsilon_t(\\beta)",
    plainLanguage:
      "One-period contribution information can’t exceed squared design diameter times probability mass outside the leader.",
    variables: [
      v("I_t", "conditional Fisher information", "Covariance matrix of the chosen input profile."),
      v("D_X", "design diameter", "Uniform bound on distance between any two input profiles."),
      v("\\varepsilon_t", "residual contestability", "Probability mass outside the current leader."),
    ],
    controls: [rhoControl],
    visualization: "Actual conditional information is plotted below the Dₓ²εₜ ceiling.",
    derivationSteps: [
      "Write the information trace as expected squared distance from the conditional mean.",
      "The conditional mean minimizes expected squared distance.",
      "Compare with the current leader’s profile and bound all other distances by Dₓ².",
    ],
  }),
  eq({
    id: "hellinger-distance",
    title: "One-period Hellinger distance",
    section: "Main article §3",
    kind: "definition",
    latex:
      "h_t^2(\\beta,\\beta')=\\sum_{i=1}^{n}\\left[\\sqrt{p_{it}(\\beta)}-\\sqrt{p_{it}(\\beta')}\\right]^2",
    plainLanguage:
      "Hellinger distance measures how much the next-recipient law changes between two candidate contribution parameters.",
    variables: [
      v("h_t^2", "squared Hellinger distance", "Separation between the two one-period recipient laws."),
      v("\\beta,\\beta'", "candidate parameters", "Two finite contribution parameter values."),
      v("p_{it}", "recipient law", "Conditional allocation probability after the same history."),
    ],
    controls: [betaControl],
    visualization: "Two candidate probability vectors are compared component by component.",
  }),
  eq({
    id: "comparison-dominated-separation",
    title: "Comparison-dominated separation",
    section: "Main article §3",
    kind: "condition",
    latex:
      "h_t^2(\\beta,\\beta')\\le K_{\\beta,\\beta'}\\varepsilon_t(\\beta)",
    plainLanguage:
      "Parameter separation must be controlled by the market’s remaining comparison, with the corresponding condition also imposed after reversing the parameters.",
    variables: [
      v("K_{\\beta,\\beta'}", "domination constant", "Finite constant for the candidate parameter pair."),
      v("h_t^2", "Hellinger increment", "One-period statistical separation."),
      v("\\varepsilon_t", "residual contestability", "Remaining probability mass outside the leader."),
    ],
    assumptions: [
      "Local equivalence: the one-period laws have the same support after every history.",
      "The same domination condition holds with β and β′ reversed.",
    ],
    visualization: "The separation increment is shown against its Kε envelope.",
  }),
  eq({
    id: "finite-comparison-condition",
    title: "Finite comparison-budget condition",
    section: "Main article §3",
    equationNumber: "5",
    kind: "condition",
    latex:
      "B_\\infty(\\beta)<\\infty\\quad P_\\beta\\text{-almost surely for every }\\beta\\in\\Theta",
    plainLanguage:
      "Every parameter value generates only a finite total amount of residual comparison almost surely.",
    variables: [
      v("B_\\infty", "total comparison budget", "Infinite sum of residual contestability."),
      v("P_\\beta", "complete-history law", "Probability law induced by contribution parameter β."),
      v("\\Theta", "parameter space", "Admissible finite contribution parameters."),
    ],
    assumptions: [
      "The common predictable design, local equivalence, and comparison-dominated separation hold.",
      "The condition is sufficient for the theorem; it isn’t asserted to be necessary for every identification failure.",
    ],
    visualization: "A cumulative comparison line approaches a finite plateau in the sharp case.",
  }),
  eq({
    id: "finite-horizon-kl-bound",
    title: "Finite-horizon KL bound",
    section: "Main article §3",
    equationNumber: "6",
    kind: "bound",
    latex:
      "D_{\\mathrm{KL}}\\!\\left(P_\\beta^T\\,\\|\\,P_{\\beta'}^T\\right)\\le K_{X,\\Theta}\\|\\beta-\\beta'\\|^2\\,\\mathbb E_\\beta B_T(\\beta)",
    plainLanguage:
      "Statistical separation through horizon T is limited by expected comparison budget, not by transaction count alone.",
    variables: [
      v("D_{\\mathrm{KL}}", "KL divergence", "Finite-history separation between two parameter laws."),
      v("P_\\beta^T", "finite-history law", "Restriction of the complete law to observations through T."),
      v("K_{X,\\Theta}", "uniform constant", "Constant determined by bounded design and parameter set."),
    ],
    controls: [betaControl],
    visualization: "A KL envelope grows with expected Bₜ and squared parameter distance.",
  }),
  eq({
    id: "strong-reinforcement-main",
    title: "Strong-reinforcement summability",
    section: "Main article §3",
    equationNumber: "7",
    kind: "condition",
    latex: "\\sum_{m=0}^{\\infty}\\frac{1}{g(a+m)}<\\infty",
    plainLanguage:
      "The reciprocal feedback weights must be summable. For polynomial feedback g(u)=uᵨ, this holds when ρ>1.",
    variables: [
      v("g", "feedback function", "Positive nondecreasing position multiplier."),
      v("a", "baseline attachment", "Positive initial mass."),
      v("m", "occupancy level", "Accumulated rewards in the reinforcement clock."),
    ],
    controls: [rhoControl],
    assumptions: [
      "The set of agents is finite and verified-input profiles are fixed in the strong-reinforcement corollary.",
      "g is positive and nondecreasing.",
    ],
    visualization: "Partial sums of 1/(a+m)ᵨ reveal the boundary at ρ=1.",
  }),
  eq({
    id: "latent-position-allocation",
    title: "Allocation with latent position",
    section: "Main article §3",
    equationNumber: "8",
    kind: "definition",
    latex:
      "p_{it}(\\beta,\\lambda)=\\frac{\\exp(x_i^\\top\\beta+\\lambda_i+s_{it})}{\\sum_{j=1}^{n}\\exp(x_j^\\top\\beta+\\lambda_j+s_{jt})}",
    plainLanguage:
      "Inherited visibility or access enters the same composite index as direct contribution.",
    variables: [
      v("\\lambda_i", "latent position", "Inherited visibility, sponsorship, distribution access, or reputation."),
      v("x_i^\\top\\beta", "direct contribution index", "Verified input profile loaded by β."),
      v("s_{it}", "observed state", "Predictable state component from the allocation history."),
    ],
    controls: [betaControl],
    visualization: "Two decompositions produce the same composite index bars.",
  }),
  eq({
    id: "gauge-transformation",
    title: "Contribution-position gauge transformation",
    section: "Main article §3",
    equationNumber: "9",
    kind: "identity",
    latex:
      "\\beta^{(d)}=\\beta+d,\\qquad \\lambda_i^{(d)}=\\lambda_i-x_i^\\top d",
    plainLanguage:
      "Move loading from latent position into contribution, or back again, without changing the composite index.",
    variables: [
      v("d", "gauge displacement", "Any admissible vector shift in the contribution parameter."),
      v("\\beta^{(d)}", "shifted contribution", "Direct contribution coefficient after displacement."),
      v("\\lambda_i^{(d)}", "shifted position", "Compensating latent-position term."),
    ],
    controls: [
      {
        symbol: "d",
        label: "Gauge displacement d",
        min: -1,
        max: 1,
        step: 0.01,
        defaultValue: 0.4,
      },
    ],
    visualization: "Contribution and position bars move in opposite directions while probability bars remain fixed.",
    derivationSteps: [
      "Substitute β+d for β.",
      "Subtract xᵢᵀd from latent position.",
      "The added and subtracted terms cancel exactly in xᵢᵀβ+λᵢ.",
    ],
  }),
  eq({
    id: "positional-rent",
    title: "Positional rent",
    section: "Main article §5",
    equationNumber: "10",
    kind: "definition",
    latex: "R_i(\\theta,O)=Y_i(O)-C_i(\\theta,O)",
    plainLanguage:
      "Positional rent is defined residually as observed reward minus contribution assigned by structural economy θ.",
    variables: [
      v("R_i", "positional rent", "Residual component of reward."),
      v("Y_i(O)", "observed reward", "Reward measurable from observable record O."),
      v("C_i(\\theta,O)", "assigned contribution", "Contribution under structural economy θ."),
    ],
    visualization: "A fixed reward bar is decomposed into contribution and residual rent.",
  }),
  eq({
    id: "merit-separating-tax",
    title: "Merit-separating tax condition",
    section: "Main article §5",
    equationNumber: "11",
    kind: "policy implication",
    latex: "\\tau_i(O)=R_i(\\theta,O)",
    plainLanguage:
      "Exact merit separation would require the record-measurable tax to equal the structural residual rent.",
    variables: [
      v("\\tau_i(O)", "tax liability", "A rule measurable from the observable record."),
      v("R_i(\\theta,O)", "positional residual", "Reward minus assigned contribution in economy θ."),
    ],
    assumptions: [
      "Two structural economies can generate the same observable law yet assign different contribution.",
      "The result doesn’t determine an optimal tax rate or imply that all high income is rent.",
    ],
    visualization: "One observable tax marker is compared with two incompatible residuals.",
  }),
  eq({
    id: "conditional-mean",
    title: "Conditional mean input",
    section: "Appendix A.1",
    kind: "definition",
    latex: "\\mu_t(\\beta)=\\sum_i p_{it}(\\beta)x_{it}",
    plainLanguage:
      "The conditional mean is the probability-weighted average verified-input profile.",
    variables: [
      v("\\mu_t", "conditional mean", "Expected chosen input profile at date t."),
      v("p_{it}", "allocation weight", "Conditional recipient probability."),
      v("x_{it}", "input profile", "Verified vector for alternative i."),
    ],
    controls: [betaControl],
    visualization: "The mean marker shifts inside the convex hull of the profiles.",
  }),
  eq({
    id: "one-period-score",
    title: "One-period score",
    section: "Appendix A.1",
    kind: "identity",
    source: "inline",
    latex:
      "S_t(\\beta)=x_{J_{t+1},t}-\\mu_t(\\beta),\\qquad \\mu_t(\\beta)=\\sum_i p_{it}(\\beta)x_{it}",
    plainLanguage:
      "The score is the chosen verified-input profile minus its conditional probability-weighted mean.",
    variables: [
      v("S_t(\\beta)", "one-period score", "Gradient of the conditional log likelihood."),
      v("x_{J_{t+1},t}", "chosen profile", "Verified inputs of the next recipient."),
      v("\\mu_t(\\beta)", "conditional mean", "Probability-weighted average profile."),
    ],
    controls: [betaControl],
    visualization: "A chosen profile’s deviation from the conditional mean is shown as the score vector.",
    derivationSteps: [
      "Differentiate the chosen alternative’s linear index with respect to β.",
      "Differentiate the log softmax normalizer to obtain μₜ(β).",
      "Subtract the conditional mean from the chosen profile.",
    ],
  }),
  eq({
    id: "mean-minimizes-distance",
    title: "Mean minimizes squared distance",
    section: "Appendix A.1",
    kind: "bound",
    latex:
      "\\operatorname{tr}\\operatorname{Var}_{p_t(\\beta)}(x_{J,t})=\\mathbb E_{p_t(\\beta)}\\|x_{J,t}-\\mu_t(\\beta)\\|^2\\le\\mathbb E_{p_t(\\beta)}\\|x_{J,t}-a\\|^2",
    plainLanguage:
      "The probability-weighted mean minimizes expected squared distance over fixed comparison points a.",
    variables: [
      v("x_{J,t}", "chosen profile", "Random verified-input profile of the next recipient."),
      v("\\mu_t", "conditional mean", "Squared-error minimizing comparison point."),
      v("a", "fixed vector", "Any alternative comparison point."),
    ],
    visualization: "Expected squared distance is minimized at the probability-weighted center.",
    derivationSteps: [
      "Expand x−a as (x−μ)+(μ−a).",
      "The cross term vanishes because the weighted deviations from μ sum to zero.",
      "The remaining ||μ−a||² term is nonnegative.",
    ],
  }),
  eq({
    id: "leader-distance-bound",
    title: "Leader-distance information bound",
    section: "Appendix A.1",
    kind: "bound",
    latex:
      "\\operatorname{tr}I_t(\\beta)\\le\\sum_{j\\ne w_t}p_{jt}(\\beta)\\|x_{jt}-x_{w_t t}\\|^2\\le D_X^2\\varepsilon_t(\\beta)",
    plainLanguage:
      "Using the leader’s profile as comparison point leaves only nonleader probability mass, each distance bounded by Dₓ.",
    variables: [
      v("w_t", "probability leader", "Index maximizing pᵢₜ(β)."),
      v("D_X", "profile diameter", "Maximum pairwise profile distance."),
      v("\\varepsilon_t", "nonleader mass", "Sum of probabilities away from the leader."),
    ],
    visualization: "Distance rays from the leader are weighted by nonleader probabilities.",
  }),
  eq({
    id: "log-partition",
    title: "Log-partition function",
    section: "Appendix A.2",
    kind: "definition",
    latex:
      "\\delta=\\beta'-\\beta,\\qquad A_t(u)=\\log\\sum_i\\exp(x_{it}^\\top u+s_{it})",
    plainLanguage:
      "Aₜ is the conditional-logit log normalizer; δ is the displacement between candidate parameters.",
    variables: [
      v("\\delta", "parameter displacement", "Difference β′−β."),
      v("A_t(u)", "log partition", "Log of the total exponential weight at parameter u."),
      v("u", "intermediate parameter", "Point along the segment between candidates."),
    ],
    controls: [betaControl],
    visualization: "The convex log-partition curve is traced between β and β′.",
  }),
  eq({
    id: "kl-bregman",
    title: "KL divergence as Bregman divergence",
    section: "Appendix A.2",
    kind: "identity",
    latex:
      "D_{\\mathrm{KL}}\\!\\left(p_t(\\beta)\\,\\|\\,p_t(\\beta')\\right)=A_t(\\beta')-A_t(\\beta)-\\nabla A_t(\\beta)^\\top\\delta",
    plainLanguage:
      "Conditional KL divergence is the gap between the convex log-partition function and its tangent at β.",
    variables: [
      v("D_{\\mathrm{KL}}", "conditional KL divergence", "One-period separation of recipient laws."),
      v("\\nabla A_t", "log-partition gradient", "Conditional mean verified-input profile."),
      v("\\delta", "parameter displacement", "β′−β."),
    ],
    visualization: "A tangent-line gap illustrates the Bregman divergence.",
  }),
  eq({
    id: "kl-taylor",
    title: "Taylor representation of KL",
    section: "Appendix A.2",
    equationNumber: "A1",
    kind: "identity",
    latex:
      "D_{\\mathrm{KL}}\\!\\left(p_t(\\beta)\\,\\|\\,p_t(\\beta')\\right)=\\int_0^1(1-r)\\,\\delta^\\top\\operatorname{Var}_{p_t(\\beta+r\\delta)}(x_{J,t})\\delta\\,dr",
    plainLanguage:
      "KL divergence integrates curvature, the conditional covariance, along the parameter segment.",
    variables: [
      v("r", "path coordinate", "Position from β to β′."),
      v("\\operatorname{Var}", "log-partition Hessian", "Conditional covariance at the intermediate parameter."),
      v("\\delta", "direction", "Parameter displacement entering the quadratic form."),
    ],
    visualization: "Curvature contributions are accumulated along the parameter path.",
  }),
  eq({
    id: "tilted-leader-odds",
    title: "Leader odds under exponential tilt",
    section: "Appendix A.2",
    kind: "bound",
    latex:
      "\\frac{1-p_{w_t t}(\\beta+r\\delta)}{p_{w_t t}(\\beta+r\\delta)}\\le e^{D_X\\|\\delta\\|}\\frac{\\varepsilon_t(\\beta)}{1-\\varepsilon_t(\\beta)}",
    plainLanguage:
      "Moving along the parameter segment can’t inflate the leader’s outside odds by more than a bounded exponential factor.",
    variables: [
      v("r", "tilt amount", "Intermediate point in [0,1]."),
      v("D_X\\|\\delta\\|", "maximum log-odds shift", "Bound from profile diameter and parameter displacement."),
      v("\\varepsilon_t/(1-\\varepsilon_t)", "outside odds", "Nonleader probability relative to leader probability."),
    ],
    visualization: "Outside odds before and after the exponential tilt are compared.",
  }),
  eq({
    id: "tilted-residual-bound",
    title: "Residual contestability under tilt",
    section: "Appendix A.2",
    kind: "bound",
    latex:
      "1-\\max_i p_{it}(\\beta+r\\delta)\\le n\\exp\\!\\left(D_X\\operatorname{diam}\\Theta\\right)\\varepsilon_t(\\beta)",
    plainLanguage:
      "On a bounded parameter set, intermediate residual contestability is uniformly controlled by residual contestability at β.",
    variables: [
      v("n", "number of alternatives", "Finite choice-set size."),
      v("\\operatorname{diam}\\Theta", "parameter diameter", "Maximum distance within the bounded parameter set."),
      v("D_X", "design diameter", "Maximum profile distance."),
    ],
    assumptions: ["The parameter set Θ and input-profile diameter are bounded.", "The largest probability is at least 1/n."],
    visualization: "The intermediate ε curve remains below a constant multiple of the base curve.",
  }),
  eq({
    id: "log-likelihood-increment",
    title: "Log-likelihood increment",
    section: "Appendix A.2",
    kind: "identity",
    latex:
      "\\log\\frac{p_{J,t}(\\beta')}{p_{J,t}(\\beta)}=\\delta^\\top x_{J,t}-\\left[A_t(\\beta')-A_t(\\beta)\\right]",
    plainLanguage:
      "The one-period log-likelihood ratio is a chosen-profile term minus the change in log normalizer.",
    variables: [
      v("J", "observed recipient", "Alternative selected in the current period."),
      v("\\delta^\\top x_{J,t}", "profile score shift", "Parameter displacement applied to the chosen input."),
      v("A_t(\\beta')-A_t(\\beta)", "normalizer shift", "Conditional constant given the past."),
    ],
    visualization: "Observed score and normalizer contributions form each likelihood increment.",
  }),
  eq({
    id: "likelihood-variance-bound",
    title: "Likelihood-ratio variance bound",
    section: "Appendix A.2",
    kind: "bound",
    latex:
      "\\delta^\\top I_t(\\beta)\\delta\\le\\|\\delta\\|^2D_X^2\\varepsilon_t(\\beta)",
    plainLanguage:
      "Conditional variance of the log-likelihood increment is bounded by parameter distance, design diameter, and residual contestability.",
    variables: [
      v("\\delta^\\top I_t\\delta", "directional information", "Fisher information along the candidate displacement."),
      v("\\|\\delta\\|^2", "squared parameter distance", "Magnitude of candidate separation."),
      v("D_X^2\\varepsilon_t", "comparison envelope", "Information bound from Proposition 1."),
    ],
    visualization: "Directional information shrinks with ε even while transactions continue.",
  }),
  eq({
    id: "finite-likelihood-ratio",
    title: "Finite-horizon likelihood ratio",
    section: "Appendix B",
    equationNumber: "B1",
    kind: "definition",
    latex:
      "Z_T=\\frac{dQ_T}{dP_T}=\\prod_{t=0}^{T-1}\\frac{p_{J_{t+1},t}(\\beta)}{p_{J_{t+1},t}(\\beta')}",
    plainLanguage:
      "The finite-history likelihood ratio multiplies conditional recipient-probability ratios along the realized path.",
    variables: [
      v("Z_T", "likelihood ratio", "Density of Qₜ relative to Pₜ."),
      v("Q_T,P_T", "finite-history laws", "Restrictions of Pβ and Pβ′ through horizon T."),
      v("J_{t+1}", "realized recipient", "Observed allocation at date t+1."),
    ],
    visualization: "Multiplicative updates accumulate along one realized recipient sequence.",
  }),
  eq({
    id: "likelihood-ratio-increment",
    title: "Likelihood-ratio increment",
    section: "Appendix B",
    kind: "definition",
    latex: "R_{t+1}=\\frac{Z_{t+1}}{Z_t}",
    plainLanguage:
      "Rₜ₊₁ is the one-step multiplicative update to the finite-history likelihood ratio.",
    variables: [
      v("R_{t+1}", "density increment", "One-period likelihood-ratio factor."),
      v("Z_t", "running density", "Likelihood ratio through time t."),
    ],
    visualization: "Each new allocation scales the running likelihood ratio.",
  }),
  eq({
    id: "sqrt-density-expectation",
    title: "Square-root density expectation",
    section: "Appendix B",
    equationNumber: "B2",
    kind: "identity",
    latex:
      "\\mathbb E_P\\!\\left[\\sqrt{R_{t+1}}\\mid\\mathcal F_t\\right]=\\sum_{i=1}^{n}\\sqrt{p_{it}(\\beta)p_{it}(\\beta')}",
    plainLanguage:
      "The conditional expected square root of the density increment is the Hellinger affinity of the two recipient laws.",
    variables: [
      v("\\mathbb E_P", "reference-law expectation", "Conditional expectation under P=Pβ′."),
      v("\\sqrt{R_{t+1}}", "square-root increment", "Order-one-half density increment."),
      v("\\sum_i\\sqrt{p_iq_i}", "Hellinger affinity", "Overlap between the two one-period laws."),
    ],
    visualization: "Overlap between two probability vectors determines the expected square-root update.",
  }),
  eq({
    id: "hellinger-density-identity",
    title: "Hellinger-density identity",
    section: "Appendix B",
    equationNumber: "B3",
    kind: "identity",
    latex:
      "2\\left(1-\\mathbb E_P[\\sqrt{R_{t+1}}\\mid\\mathcal F_t]\\right)=\\sum_{i=1}^{n}\\left(\\sqrt{p_{it}(\\beta)}-\\sqrt{p_{it}(\\beta')}\\right)^2=h_t^2(\\beta,\\beta')",
    plainLanguage:
      "One minus Hellinger affinity, multiplied by two, equals the squared Hellinger increment.",
    variables: [
      v("R_{t+1}", "density increment", "Likelihood update between the two laws."),
      v("h_t^2", "Hellinger increment", "One-period separation."),
    ],
    visualization: "Affinity loss and squared root-probability distance move together exactly.",
  }),
  eq({
    id: "hellinger-criterion",
    title: "Filtered-space Hellinger criterion",
    section: "Appendix B",
    equationNumber: "B4",
    kind: "theorem",
    latex:
      "Q\\ll P\\quad\\Longleftrightarrow\\quad\\sum_{t=0}^{\\infty}h_t^2(\\beta,\\beta')<\\infty\\quad Q\\text{-almost surely}",
    plainLanguage:
      "In the locally equivalent discrete-time filtered experiment, absolute continuity is equivalent to finite cumulative order-one-half Hellinger process.",
    variables: [
      v("Q\\ll P", "absolute continuity", "Every P-null complete-history event is also Q-null."),
      v("\\sum_t h_t^2", "Hellinger process", "Cumulative one-period statistical separation."),
    ],
    assumptions: ["The discrete-time filtered experiment is locally equivalent at every finite horizon."],
    visualization: "The cumulative Hellinger process is compared with a finite ceiling.",
  }),
  eq({
    id: "cumulative-hellinger-bound",
    title: "Cumulative separation bound",
    section: "Appendix B",
    kind: "bound",
    latex:
      "\\sum_{t=0}^{\\infty}h_t^2(\\beta,\\beta')\\le K_{\\beta,\\beta'}B_\\infty(\\beta)<\\infty\\quad P_\\beta\\text{-almost surely}",
    plainLanguage:
      "Comparison-dominated separation plus finite total comparison makes the Hellinger process finite.",
    variables: [
      v("\\sum_t h_t^2", "total separation", "Cumulative Hellinger increments."),
      v("K_{\\beta,\\beta'}", "domination constant", "Finite pair-specific multiplier."),
      v("B_\\infty", "total comparison", "Cumulative residual contestability."),
    ],
    visualization: "Cumulative statistical separation is trapped below K times the comparison budget.",
  }),
  eq({
    id: "complete-law-equivalence",
    title: "Complete-history equivalence",
    section: "Appendix B",
    equationNumber: "B5",
    kind: "theorem",
    latex: "P_\\beta\\sim P_{\\beta'}\\quad\\text{on }\\mathcal F_\\infty",
    plainLanguage:
      "The complete single-history laws are mutually absolutely continuous; they aren’t claimed to be identical.",
    variables: [
      v("\\sim", "mutual absolute continuity", "Each complete-history law is absolutely continuous with respect to the other."),
      v("\\mathcal F_\\infty", "terminal sigma-field", "Events observable from the entire infinite history."),
    ],
    assumptions: [
      "Finite comparison holds under each candidate parameter.",
      "Local equivalence and comparison-dominated separation hold in both directions.",
    ],
    visualization: "Two law-support regions overlap completely while retaining different densities.",
  }),
  eq({
    id: "estimator-event",
    title: "Estimator error event",
    section: "Appendix B",
    kind: "definition",
    latex: "A_T(\\epsilon)=\\{d(\\psi_T,\\psi(\\beta))>\\epsilon\\}",
    plainLanguage:
      "Aₜ(ε) is the event that the estimator remains farther than ε from its target at β.",
    variables: [
      v("\\psi_T", "estimator", "Any statistic measurable from the history through T."),
      v("\\psi(\\beta)", "target functional", "Contribution quantity at parameter β."),
      v("d", "target-space metric", "Distance used to define consistency."),
    ],
    visualization: "An ε-neighborhood around the target classifies estimator outcomes.",
  }),
  eq({
    id: "target-separation",
    title: "Target separation",
    section: "Appendix B",
    kind: "definition",
    latex: "\\Delta=d(\\psi(\\beta),\\psi(\\beta'))>0",
    plainLanguage:
      "A nonconstant contribution functional takes separated values at two parameters.",
    variables: [
      v("\\Delta", "functional separation", "Positive distance between two target values."),
      v("\\psi", "contribution functional", "Quantity the estimator aims to recover."),
    ],
    visualization: "Two target points are shown Δ apart in the functional space.",
  }),
  eq({
    id: "confidence-event",
    title: "Shrinking confidence event",
    section: "Appendix B",
    kind: "definition",
    latex:
      "E_T=\\{\\psi(\\beta)\\in C_T,\\ \\operatorname{diam}_d(C_T)<\\Delta/2\\}",
    plainLanguage:
      "The event combines coverage of ψ(β) with a set too small to also contain the separated value ψ(β′).",
    variables: [
      v("C_T", "confidence set", "History-measurable random subset of the target space."),
      v("\\operatorname{diam}_d", "set diameter", "Largest distance between two members of Cₜ."),
      v("\\Delta/2", "shrinkage threshold", "Less than half the two-target separation."),
    ],
    visualization: "A shrinking interval covers one separated target and can’t cover the other.",
  }),
  eq({
    id: "kl-chain-rule",
    title: "Conditional KL chain rule",
    section: "Appendix C",
    kind: "identity",
    latex:
      "D_{\\mathrm{KL}}\\!\\left(P_\\beta^T\\,\\|\\,P_{\\beta'}^T\\right)=\\mathbb E_\\beta\\sum_{t=0}^{T-1}D_{\\mathrm{KL}}\\!\\left(p_t(\\beta)\\,\\|\\,p_t(\\beta')\\right)",
    plainLanguage:
      "Finite-history KL divergence is the expected sum of conditional one-period divergences.",
    variables: [
      v("P_\\beta^T", "joint finite-history law", "Law of allocations through T."),
      v("p_t(\\beta)", "conditional kernel", "Next-recipient law given the history."),
      v("\\mathbb E_\\beta", "truth-law expectation", "Expectation under contribution parameter β."),
    ],
    visualization: "Per-period KL increments stack into total finite-history separation.",
  }),
  eq({
    id: "le-cam-bound",
    title: "Le Cam two-point bound",
    section: "Appendix C",
    kind: "bound",
    latex:
      "\\max_{\\vartheta\\in\\{\\beta,\\beta'\\}}P_\\vartheta\\!\\left(d(\\widehat\\psi_T,\\psi(\\vartheta))\\ge\\Delta_\\psi/2\\right)\\ge\\frac{1-\\|P_\\beta^T-P_{\\beta'}^T\\|_{\\mathrm{TV}}}{2}",
    plainLanguage:
      "If two finite-history laws remain close in total variation, every estimator has substantial error at one of the two parameter points.",
    variables: [
      v("\\widehat\\psi_T", "estimator", "Any finite-history contribution estimate."),
      v("\\Delta_\\psi", "target gap", "Distance between the two contribution values."),
      v("\\|\\cdot\\|_{\\mathrm{TV}}", "total variation", "Maximum event-probability gap between the laws."),
    ],
    visualization: "Lower-bound risk remains high while total variation remains below one.",
  }),
  eq({
    id: "pinsker",
    title: "Pinsker inequality",
    section: "Appendix C",
    kind: "bound",
    latex:
      "\\|P_\\beta^T-P_{\\beta'}^T\\|_{\\mathrm{TV}}\\le\\sqrt{\\tfrac12 D_{\\mathrm{KL}}\\!\\left(P_\\beta^T\\,\\|\\,P_{\\beta'}^T\\right)}",
    plainLanguage:
      "Total variation is controlled by the square root of KL divergence.",
    variables: [
      v("\\|P-Q\\|_{\\mathrm{TV}}", "total variation", "Maximum probability difference over measurable events."),
      v("D_{\\mathrm{KL}}", "KL divergence", "Relative-entropy separation."),
    ],
    visualization: "The square-root KL envelope bounds total variation.",
  }),
  eq({
    id: "local-alternative",
    title: "Local alternative",
    section: "Appendix C",
    kind: "definition",
    latex: "\\beta_T=\\beta_0+\\frac{h v}{\\sqrt{b_T}}",
    plainLanguage:
      "The hard alternative approaches β₀ at the inverse square root of the available comparison scale bₜ.",
    variables: [
      v("\\beta_0", "base parameter", "Reference contribution parameter."),
      v("v", "local direction", "Fixed direction in parameter space."),
      v("b_T", "comparison scale", "Upper scale for expected Bₜ."),
    ],
    visualization: "The candidate distance contracts as bₜ grows.",
  }),
  eq({
    id: "functional-local-expansion",
    title: "Local functional expansion",
    section: "Appendix C",
    kind: "identity",
    latex:
      "\\psi(\\beta_T)-\\psi(\\beta_0)=\\frac{h\\nabla\\psi(\\beta_0)^\\top v}{\\sqrt{b_T}}+o(b_T^{-1/2})",
    plainLanguage:
      "Differentiability converts the local parameter shift into a target difference of order bₜ⁻¹ᐟ².",
    variables: [
      v("\\nabla\\psi(\\beta_0)", "functional gradient", "Local sensitivity of the contribution target."),
      v("o(b_T^{-1/2})", "remainder", "Term negligible relative to the comparison-limited rate."),
    ],
    visualization: "The leading linear term dominates the smaller remainder.",
  }),
  eq({
    id: "reward-count-process",
    title: "Reward indicators and occupancy counts",
    section: "Appendix D",
    equationNumber: "D1",
    kind: "definition",
    latex:
      "R_{i,t+1}=\\mathbf 1\\{J_{t+1}=i\\},\\qquad N_i(t)=\\sum_{r=1}^{t}R_{i,r},\\qquad N_i(0)=0",
    plainLanguage:
      "Each reward indicator increments the recipient’s accumulated occupancy count.",
    variables: [
      v("R_{i,t+1}", "reward indicator", "One when i receives the next reward, zero otherwise."),
      v("N_i(t)", "occupancy count", "Cumulative rewards received by i through t."),
    ],
    visualization: "A step in the recipient indicator increments one count path.",
  }),
  eq({
    id: "reinforced-allocation",
    title: "Reinforced allocation rule",
    section: "Appendix D",
    equationNumber: "D2",
    kind: "definition",
    latex:
      "p_{it}(\\beta)=\\frac{\\exp(x_i^\\top\\beta)g(a+N_i(t))}{\\sum_{j=1}^{n}\\exp(x_j^\\top\\beta)g(a+N_j(t))}",
    plainLanguage:
      "Direct contribution and accumulated position multiply to determine each conditional reward weight.",
    variables: [
      v("\\exp(x_i^\\top\\beta)", "contribution weight", "Positive direct-input component."),
      v("g(a+N_i(t))", "feedback weight", "Positive reinforcement from accumulated rewards."),
      v("a", "baseline", "Positive starting attachment."),
    ],
    controls: [betaControl, rhoControl],
    visualization: "Contribution and feedback weights combine before normalization.",
  }),
  eq({
    id: "strong-reinforcement-appendix",
    title: "Strong-reinforcement condition",
    section: "Appendix D",
    equationNumber: "D3",
    kind: "condition",
    latex: "\\sum_{m=0}^{\\infty}\\frac{1}{g(a+m)}<\\infty",
    plainLanguage:
      "Finite reciprocal total clock time is the primitive condition used in the absorption construction.",
    variables: [
      v("g(a+m)", "level-m feedback", "Rate multiplier after m rewards."),
      v("1/g(a+m)", "mean waiting scale", "Reciprocal feedback weight."),
    ],
    controls: [rhoControl],
    visualization: "Partial reciprocal sums converge for polynomial ρ>1 and diverge at the linear boundary.",
  }),
  eq({
    id: "polynomial-feedback",
    title: "Polynomial feedback",
    section: "Appendix D",
    kind: "definition",
    source: "inline",
    latex:
      "g(u)=u^\\rho,\\qquad \\sum_{m=0}^{\\infty}(a+m)^{-\\rho}<\\infty\\ \\Longleftrightarrow\\ \\rho>1",
    plainLanguage:
      "For polynomial reinforcement, the reciprocal feedback series converges precisely in the superlinear regime.",
    variables: [
      v("g(u)", "feedback function", "Position multiplier at level u."),
      v("\\rho", "reinforcement exponent", "Curvature of the feedback rule."),
      v("a", "baseline attachment", "Positive starting mass."),
    ],
    controls: [rhoControl],
    assumptions: ["The baseline a is strictly positive."],
    visualization: "Partial reciprocal sums flatten for ρ>1 and keep growing at or below the linear boundary.",
  }),
  eq({
    id: "quality-weight",
    title: "Fixed contribution weight",
    section: "Appendix D",
    kind: "definition",
    latex: "q_i=\\exp(x_i^\\top\\beta)>0",
    plainLanguage:
      "The verified input profile creates a strictly positive fixed contribution multiplier for each agent.",
    variables: [
      v("q_i", "contribution weight", "Exponentiated direct contribution index."),
      v("x_i^\\top\\beta", "direct index", "Input profile loaded by the contribution parameter."),
    ],
    controls: [betaControl],
    visualization: "Exponential weights remain positive and change smoothly with β.",
  }),
  eq({
    id: "explosion-time",
    title: "Exponential-clock total time",
    section: "Appendix D",
    kind: "definition",
    latex: "T_i=\\sum_{m=0}^{\\infty}E_{i,m}",
    plainLanguage:
      "Tᵢ is the total exponential-clock time for agent i to accumulate infinitely many rings.",
    variables: [
      v("E_{i,m}", "clock waiting time", "Independent exponential variable with rate qᵢg(a+m)."),
      v("T_i", "explosion time", "Infinite sum of occupancy-level waiting times."),
    ],
    assumptions: ["The Eᵢ,ₘ variables are independent across agents and occupancy levels."],
    visualization: "Waiting-time segments accumulate toward a finite or infinite endpoint.",
  }),
  eq({
    id: "expected-explosion-time",
    title: "Expected clock time",
    section: "Appendix D",
    kind: "bound",
    latex:
      "\\mathbb E T_i=\\sum_{m=0}^{\\infty}\\frac{1}{q_i g(a+m)}<\\infty",
    plainLanguage:
      "Strong reinforcement makes expected explosion time finite, hence Tᵢ is finite almost surely.",
    variables: [
      v("\\mathbb E T_i", "expected explosion time", "Expected sum of exponential waits."),
      v("q_i g(a+m)", "clock rate", "Contribution weight times feedback weight."),
    ],
    assumptions: ["qᵢ is positive and the strong-reinforcement reciprocal sum is finite."],
    visualization: "Expected waiting-time tails contract fast enough to have finite total length.",
  }),
  eq({
    id: "post-absorption-residual",
    title: "Post-absorption residual bound",
    section: "Appendix D",
    kind: "bound",
    latex:
      "\\varepsilon_t(\\beta)\\le\\sum_{j\\ne I^\\star}\\frac{q_j g(a+n_j)}{q_{I^\\star}g(a+m)}",
    plainLanguage:
      "After absorption, loser weights are fixed while the winner’s feedback weight grows with post-absorption count m.",
    variables: [
      v("I^\\star", "absorbing winner", "Unique agent with minimal exponential-clock explosion time."),
      v("n_j", "final loser count", "Fixed terminal occupancy of loser j."),
      v("m", "winner count", "Post-absorption occupancy level."),
    ],
    visualization: "The fixed loser numerator is divided by a growing winner denominator.",
  }),
  eq({
    id: "tail-path-probability",
    title: "Probability of an eventual constant tail",
    section: "Appendix D",
    kind: "definition",
    latex:
      "\\prod_{m=0}^{\\infty}\\frac{q_i g(a+n_i+m)}{q_i g(a+n_i+m)+\\sum_{j\\ne i}q_j g(a+n_j)}",
    plainLanguage:
      "Conditional on a finite prefix, this infinite product is the probability that i receives every later reward.",
    variables: [
      v("n_i,n_j", "prefix counts", "Occupancies at the end of the fixed finite prefix."),
      v("q_i,q_j", "contribution weights", "Positive exponentiated direct-input indices."),
      v("m", "tail index", "Winner’s additional reward count."),
    ],
    visualization: "Tail success probabilities multiply across all later allocations.",
  }),
  eq({
    id: "loser-mass-constant",
    title: "Fixed loser mass",
    section: "Appendix D",
    kind: "definition",
    latex: "C_{\\beta,\\omega}=\\sum_{j\\ne i}q_j g(a+n_j)",
    plainLanguage:
      "For an eventually constant history ω, all losing alternatives contribute a fixed positive total tail weight.",
    variables: [
      v("C_{\\beta,\\omega}", "loser mass", "Sum of fixed loser weights after the prefix."),
      v("\\omega", "eventually constant history", "A recipient sequence with one permanent tail winner."),
    ],
    visualization: "Loser weights freeze into one constant against the growing winner.",
  }),
  eq({
    id: "tail-product",
    title: "Tail probability product",
    section: "Appendix D",
    kind: "identity",
    latex:
      "\\prod_{m=0}^{\\infty}\\left(1+\\frac{C_{\\beta,\\omega}}{q_i g(a+n_i+m)}\\right)^{-1}",
    plainLanguage:
      "Factoring the winner weight rewrites the eventual-tail probability in infinite-product form.",
    variables: [
      v("C_{\\beta,\\omega}", "fixed loser mass", "Total nonwinner weight."),
      v("q_i g(a+n_i+m)", "growing winner weight", "Tail winner’s weight at step m."),
    ],
    visualization: "Each factor approaches one as the winner’s feedback weight grows.",
    derivationSteps: [
      "Divide numerator and denominator of every tail probability by the winner weight.",
      "The denominator becomes 1 plus loser mass over winner mass.",
      "Multiply the reciprocal factors over the infinite tail.",
    ],
  }),
  eq({
    id: "infinite-product-sum",
    title: "Infinite-product positivity condition",
    section: "Appendix D",
    kind: "condition",
    latex:
      "\\sum_{m=0}^{\\infty}\\frac{C_{\\beta,\\omega}}{q_i g(a+n_i+m)}<\\infty",
    plainLanguage:
      "Strong reinforcement makes the deviations of the tail-product factors summable, so the infinite product stays strictly positive.",
    variables: [
      v("C_{\\beta,\\omega}/q_i", "fixed scale", "Positive constant for the parameter and history."),
      v("g(a+n_i+m)", "winner feedback", "Growing tail denominator."),
    ],
    assumptions: ["The strong-reinforcement reciprocal series is finite."],
    visualization: "Summable factor deviations leave a nonzero infinite-product limit.",
  }),
];

export const EQUATION_BY_ID = new Map(EQUATIONS.map((equation) => [equation.id, equation]));

export const EQUATION_SECTIONS = Array.from(
  new Set(EQUATIONS.map((equation) => equation.section)),
);

export function equationCountByKind(kind: EquationKind): number {
  return EQUATIONS.filter((equation) => equation.kind === kind).length;
}
