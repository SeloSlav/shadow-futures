# Chapter 8

## A billion answers

Inference turns a trained model into a stream of outputs. Each interaction can also create evidence, but what it reveals depends on which requests arrive, how the model is selected, and whether the answer is checked.

The records can include the request, selected model, answer, time taken, and a user's acceptance. A check against an independently known outcome adds another observation. These records answer different questions because they arise from different measurement choices.

The trained model need not change its numerical settings during this exchange. A later development process might use collected examples to update it. The service could also retrieve a newer document or apply a different instruction without retraining the underlying model. Improvement can occur in the model, the surrounding software, or the people evaluating the result.

Now repeat the exchange a billion times across many workplaces and tasks. The records can become immensely useful. They can expose recurring mistakes, reveal delays, and help developers improve a service. But the number of completed requests does not, by itself, tell us which questions the resulting evidence can answer.

One question concerns this particular order: did the service find the correct deadline? Another concerns the alternative model: would it have done better on comparable work? A third concerns the development process: which investment produced the capability? A fourth concerns an economic arrangement: what would have happened if another supplier had received the early opportunities and resources to develop?

We can learn a great deal about the first question without settling all the others. We can also design the service to collect evidence about more than one. The choice matters before the billionth answer arrives. It matters when we decide what happens to the first request.

## Two answers to one question

Chatbot Arena offers a concrete departure from an ordinary exchange. In the platform described by Wei-Lin Chiang and colleagues in their 2024 paper, a user enters a prompt and receives responses from two anonymously selected models. The user compares them and votes; the identities are revealed afterward. The interaction creates a paired preference observation, rather than simply another conversation with a named favorite.[^c8-1]

The two columns on the screen do useful work. They bring alternatives into the same encounter, expose them to the same prompt, and delay the information that a famous brand might provide. We can argue about the design while recognizing what it adds. A record of which service people already use supplies a different kind of evidence from a vote after they have seen two responses.

Preference and correctness remain different. A polished wrong answer can be more convincing than a correct but awkwardly expressed one. Assessing accuracy may require evidence beyond the preference judgment.

Arena's paper itself distinguishes benchmarks based on human preference from those based on a known correct answer. That distinction helps us resist treating one ranking as a certificate for every use. A buyer interested in pleasant conversation and a buyer extracting deadlines may need different measurements. Neither has to dismiss the other's question to insist on measuring its own.

Prompts and users arriving at a public platform can differ from those in a specialized application. Long documents, damaged scans, or technical vocabulary can change performance. A general benchmark is evidence to assess for transportability, rather than an entire decision basis.

Unequal task assignments complicate performance comparisons. A service handling clear documents can have a higher acceptance rate than one handling damaged scans even if its underlying capability is no better. Evaluation must examine the assignment process as well as the score.

A paired test can help with that particular problem. We could send suitable examples from each category to both services and assess their outputs using a consistent procedure. The cost includes additional computation and evaluation. The benefit is evidence directed toward a decision that a single unbalanced stream would have made harder.

The Arena design also reminds us that comparison has to be produced. Someone builds the interface, selects model pairs, collects responses, and maintains the analysis. Those activities are part of the institution. The platform's methodological choices affect what a vote means, which differences it can reveal, and how confidently a ranking can be interpreted.

Suppose a purchasing department then uses a ranking to select its default supplier. The ranking can influence subsequent exposure. That possibility does not invalidate the original tests. It creates a new question about how the tests and the resulting deployment interact. A good measured result can become an input into a process that later generates a different kind of evidence.

Service and evaluation can be designed together. Delivering an accurate answer at an acceptable cost remains the productive aim; a suitable comparison can reveal when another system better serves it.

## When the judge likes the packaging

Yann Dubois and colleagues tested how an automatic judge responds to presentation in their length-controlled AlpacaEval study. They prompted models to give concise or verbose answers. For one GPT-4 version, the unadjusted win rate ranged from about twenty-three to sixty-four percent as the instruction changed. The underlying model had not received a new training run; the prompt and resulting answers differed. Their statistical length adjustment reduced the swing. The experiment exposes sensitivity to presentation, while leaving us to investigate which changes also improve the answer.[^c8-12]

Each response faced a reference answer to the same instruction. The judge's preference supplied the score. It was a measure of which answer the evaluator favored, rather than an independent check against a separately established correct solution.[^c8-12]

Arena records users' comparisons of two answers. AlpacaEval uses the judgment of a particular automatic evaluator. An adjustment to that evaluator's scores tells us about its measurements; human voting needs evidence of its own.

Think of the deadline in the purchase order. One service returns the date and identifies the relevant passage. Another returns the same date inside a long explanation of the document. A buyer might prefer either presentation for a legitimate reason. An evaluation intended to measure the extraction task should nevertheless understand how much of its ranking depends on the surrounding prose.

Now imagine that the longer response contains an error elsewhere. The extra detail has changed more than the length of the answer; it has created another opportunity to mislead the worker. Alternatively, the longer explanation might expose an ambiguity that the short answer concealed. Length by itself cannot tell us which happened. The measurement must remain connected to the purpose.

The length-controlled study also makes a counterfactual question explicit: how would the estimated preference compare if the responses had the same length? Its statistical adjustment supplies an estimate under a model. The authors are not claiming to have observed every possible rewritten answer or eliminated every source of evaluation bias. A defined counterfactual can be useful without becoming an omniscient reconstruction.

For a developer, the practical stakes concern which improvements an evaluation rewards. If a score rises after a formatting change, the developer should investigate whether users actually complete their work better. If better reasoning raises the score, that is worth distinguishing too. The same final ranking can emerge through changes with different consequences for the customer.

Who defines the test helps determine who receives the next opportunity. An accessible benchmark can support discovery, but its convenience is insufficient: its design must expose differences relevant to the intended use.

## Choosing before paying

The next design decision concerns which models to call. Sending every request to the most expensive available system may waste resources on tasks a cheaper system handles adequately. Calling every model for every request can produce comparisons, but also an unnecessary bill. The problem invites a selector: software that routes the request according to the work it appears to require.

Isaac Ong and colleagues' RouteLLM research investigates such a selector. Their framework learns from preference data and routes between a stronger, more expensive model and a cheaper alternative. They evaluate the resulting tradeoffs on specified benchmarks and compare their methods with random routing. Their reported improvements concern those models, data, and evaluations, rather than a guarantee for every future workplace.[^c8-2]

The purchase order makes the motivation easy to understand. A short document with a clearly labeled date may require less capability than an ambiguous revision spread across several pages. A well-designed router could reserve the more costly service for the cases where it adds enough value. That is a productive improvement even if the user never sees the selection happening.

Notice what the router must learn. It needs evidence about relative performance on kinds of requests. A general reputation for being stronger is not the same as an advantage worth paying for on every task. A useful selection rule takes variation seriously instead of reading an overall ranking as an instruction to make the same choice forever.

The RouteLLM paper also reports limits to transferring a router across tasks. Data resembling the eventual evaluation can improve performance. Correspondence among training examples, evaluation cases, and actual use therefore matters to the interpretation of a score.

Lingjiao Chen, Matei Zaharia, and James Zou explore a related route in FrugalGPT. Their system can call a model, assess the response, and decide whether to stop or call another. The answer already obtained becomes an input into the next decision. The study compares cost and performance on defined tasks, including classification and reading comprehension.[^c8-3]

In the workplace illustration, the system might first ask a cheaper model to extract the deadline. If its assessment of that response is unfavorable, it could ask another model. This differs from selecting once before seeing an answer. It also differs from obtaining an independent human check of correctness. The assessment mechanism itself needs evaluation.

FrugalGPT's paper includes cases where additional calls are made even though the models' answers agree. Its discussion identifies the selector's uncertainty as a source of avoidable expense. Delegating a decision to a model leaves another judgment to evaluate: when does that model decide to stop?

An economical cascade might lower the cost of trying alternative services. If the interface reports only one answer and one brand, the buyer may be unable to inspect those comparisons. Several logos can conceal dependence on one underlying service; a simple interface can conceal useful tests of several.

Routing and cascades can make it cheaper to compare services as well as to use them. A buyer who can afford only one expensive answer may be able to check several cheaper ones. Whether that happens depends on what the selection system tests and records.

One distinction is essential before we try to measure a comparison budget. A router's estimate that one model will produce the preferred answer differs from the chance that the request will actually be assigned to that model. RouteLLM uses a predicted preference and a threshold to make its selection. The prediction and the assignment rule do different jobs.

A predicted seventy percent chance of producing the preferred answer is not a seventy percent allocation probability. A selector could send every request to the predicted winner. Adding the remaining thirty percent figures would then mismeasure the process that actually assigns work.

A different rule could deliberately randomize some assignments. Its allocation probabilities would describe that design, while its predicted performance scores would describe the selector's expectations. Both can be recorded. Their relationship matters to the analysis, but they are not interchangeable numbers simply because both can be expressed as percentages.

Checking answers reveals performance on the tasks a system encounters; independent evaluation can supply information about alternatives. Those measurements belong alongside the assignment record when we assess what can be learned.

The work of checking also has a production process. Someone may need to open an amendment, resolve an ambiguous field, or determine which document version governs the task. AI could assist with that work, but an automated score remains something to validate against the relevant standard. A service can lower the cost of producing an answer while leaving costly uncertainty about whether the answer is useful.

The relevant chain runs from the observations training a selector through assignment rules, checked responses, and decisions using those checks. Following it distinguishes efficient production from useful comparison and unsupported precision.

The objectives still need to be named. A router optimized for the best current cost/performance tradeoff need not be optimized to discover a new supplier's potential. It may already have enough evidence for its operational decision, or it may need occasional new observations. We cannot determine which merely by noting that the routing is automated.

## A record worth reusing

Long before today's language models, a news page posed a related problem. The Yahoo Today Module could highlight an article for a visitor and observe whether the visitor clicked. It could not observe that same visit's response to every article left unshown. Lihong Li, Wei Chu, John Langford, and Xuanhui Wang investigated how previously collected data could help evaluate another recommendation rule.[^c8-4]

Their work used records gathered through randomized article selection. Under their stated conditions, an evaluator can retain suitable events where a proposed rule would have selected the article that was actually shown. The retained record then includes a real response to that choice. The researchers compared offline evaluations with online results in the Yahoo application.

A historical record can contain deliberately created variation. It can then support questions that simply repeating the current favorite would leave unanswered. Collecting the right record expands what later analysis can discover.

Randomized request logs with checked outcomes can support evaluation of another selection rule under the relevant coverage and identification assumptions. A literal parallel economy is not required for every counterfactual question.

That last qualification matters. If a service was never available for a particular kind of order, the record may contain no relevant responses from it. Knowing that the probability was zero does not supply the missing performance. An estimator needs suitable observations or additional defensible assumptions, rather than an increasingly elaborate way of describing their absence.

The Yahoo authors call their method replay. Their use of the word concerns evaluating a recommendation policy with suitable logged events. My shadow futures concern alternative histories generated under specified economic conditions and rules. These are distinct uses. A procedure that reuses article records is not a time machine for recreating the entire industry with a different first winner.

Evaluation of current systems, variation across alternative development histories, and the effects of changing a development policy require different comparisons. Randomized request logs can help with the first without establishing how capabilities would have evolved after different early opportunities.

The same record can nevertheless be valuable. If it supports a better procurement decision, it need not explain every earlier fortune to justify its collection. The point of the comparison budget is to clarify a particular information constraint, not to demand a complete causal explanation of the economy before allowing any useful inference.

This gives us a practical reason to preserve details that a success count omits. Which alternatives were available? Why was this one selected? What was measured afterward? A usable assignment record and meaningful outcome can make a small pilot worth more, for a specific decision, than a much larger archive lacking those features.

The pilot also has costs. Someone receives the less suitable answer when the comparison goes badly. Someone checks the outcome. Some data may be inappropriate to retain or share. Useful evaluation requires an observation design that fits the task and evidence worth those costs.

## The same ancestor

Our hypothetical buyer opens three services and receives three answers to the order question. Three storefronts are visible. What has been compared?

Perhaps each service has independently trained a different system. Perhaps each has adapted a shared model. Perhaps all three pass the request to the same supplier, with different surrounding instructions. The answers may still differ. The number of interfaces does not identify which part of the production process varied.

Meta's Llama 3 report supplies a concrete example of a relationship between models. It describes using its flagship model to improve smaller models during post-training. Different members of a model family can therefore be connected through the development process. Treating their outputs as evidence from wholly unrelated research histories would miss that connection.[^c8-5]

Shared ancestry does not make their performance identical. A smaller model can behave differently, require different resources, and fit a particular application better. Comparing those actual systems can teach us something useful. The mistake would be to suppose the comparison automatically tests every assumption inherited from their common origin.

Imagine that our three services extract the same wrong date because the underlying document representation loses the same footnote. Asking all three may produce reassuring agreement while leaving that error untouched. Now imagine that one service uses a better document parser and finds the footnote. A common language model has not prevented a meaningful improvement elsewhere in the service.

These are illustrative cases, but they identify an empirical task. Follow the chain far enough to understand which components are common, which vary, and where the outcome is measured. Independence is valuable relative to a question. It should not be a ceremonial word attached to a company name.

A thousand carefully checked orders can expose failure patterns that one order could not reveal. New requests and new checks can add information about a system even when the same model handles them all.

But generating a thousand responses without a new check does not create a thousand independent confirmations of the underlying claim. If the uncertainty concerns the actual delivery date, the order and its amendments remain relevant evidence. A longer conversation among programs might help interpret them. It cannot make an unread amendment irrelevant by agreeing without it.

Yaneer Bar-Yam's “From Big Data to Important Information” distinguishes detailed records from information useful for understanding an intervention.[^c8-6] For the delivery order, the decisive detail may be an amendment that thousands of generated answers have never checked.

The workplace can act on that warning. It can measure whether the extracted date is correct, whether the answer helped a worker complete the task, and whether a different service changes the result. Each measurement has a purpose. We gain clarity by naming those purposes before admiring the size of the resulting database.

## Predicting the next choice

Forecasts of the next allocation can converge while explanations of productive contribution remain different. Chapter 3 states the conditions under which that distinction persists.

David Blackwell and Lester Dubins showed in 1962 how conditional forecasts can merge under an appropriate relationship between probability laws. Hao Zhang later established an example in spatial statistics where some underlying parameters cannot be consistently recovered even though important prediction can remain effective.[^c8-7][^c8-8]

Forecasting which supplier wins an allocation is a different task from checking whether an AI answer is true. The answer needs evidence of its own.

Acceptance, correctness, and causal contribution are separate targets. Predicting that a user accepts an answer does not establish that it is accurate, or identify which investment caused the capability. A common interface does not make the underlying inference identical.

For the buyer, the next action could be straightforward. If errors matter, check a suitable sample against the original orders. If supplier choice matters, preserve a meaningful comparison. If a historical attribution claim matters, identify the intervention, horizon, and evidence that could support it. The inquiry becomes more useful as it becomes more specific.

## Sixteen agents and one compiler

A swarm sounds like a crowd. On a computer, it can mean several copies of a model pursuing assigned work, using tools, keeping records, and returning results. The economic question begins with what those activities accomplish. Sixteen running programs can perform sixteen useful investigations, or spend the afternoon rediscovering one mistake.

In February 2026, Nicholas Carlini at Anthropic reported setting sixteen Claude instances to build a C compiler, which translates code into instructions a computer can run. Compiling the Linux operating system, they stalled on the same bug and interfered with one another's fixes. He redesigned the tests: GCC, an established compiler, handled most files while the new compiler handled varying subsets. Failures became easier to isolate. The reported achievement still had limits, including inefficient generated code and dependence on other tools. This was an attributed capability experiment, not a measured replacement for a human workforce.[^c8-agents]

The revised tests helped separate failures. The checking arrangement belonged to the productive machinery as surely as the model did. A candidate program needed reliable tests that could expose its mistakes.

There are different ways to organize that machinery. Anthropic's June 2025 research system used a lead agent to distribute searches among others. Carlini's compiler account explicitly described a setup without a lead agent assigning tasks. In either arrangement, ask who directs the work and supplies the feedback.[^c8-agents]

Agent systems can organize candidate generation, execution, and checking, as the compiler account demonstrates. Their productive value depends on the full arrangement: human-defined objectives, tools, tests, and judgments about errors, as well as the cost of model calls.

Retained code, test cases, and usable project records can support later comparison when a model changes. The rights to those assets therefore affect the cost of developing and evaluating another approach.

Failed attempts can matter as much as final outputs. Preserving their records may prevent repeated mistakes, while requiring storage, documentation, and protection of confidential material.

Control of tools and records can make switching expensive even when software generation becomes cheap. This is an institutional question about portability and ownership, distinct from the capability demonstrated by an agent experiment.

Carlini's account describes checks within one firm using related models. A legally independent project could still use a defective test. Validity depends on feedback that can expose an error, wherever the test is organized.

A thousand agents can reproduce one mistaken interpretation; a single check against a relevant document can reveal it. Abundant software labor can multiply attempts faster than it produces trustworthy judgments of their results.

## A town throws a party

In a 2023 study, Joon Sung Park and colleagues populated Smallville with twenty-five software agents. One received the intention to organize a Valentine's Day party. The authors report invitations spreading and five invited agents arriving at the cafe. Another resident mistakenly identified her neighbor, Adam Smith, as the author of The Wealth of Nations. Knowledge inherited from a language model had intruded into the neighborhood.[^c8-10]

These characters supply a tempting contrast with a compiler. A simulated town can be rerun, with rules changed and histories recorded. But what would establish that its response to a tax change predicts the response of people? Believable conversation and a program's successful execution do not supply that validation. The study investigated believability and agent architecture, not the accuracy of economic policy forecasts.

Simulated demand can reveal implications of assumptions and help design an empirical test. Its preferences may also reflect the instructions and training inherited from a common model. Multiplying simulated participants reduces some sampling noise without correcting a mistaken demand mechanism.

A shadow replay keeps underlying parameters, starting conditions, and response rules fixed while the random history changes. Altering procurement or training policy is a separate intervention. For either operation, the claim about an actual economy requires evidence linking the model to that economy. The party makes generated interaction visible. The compiler's checks point toward the additional work of connecting a proposed answer to a result that can prove it wrong.

## When a generated answer can be tested

There is a more encouraging route from generation to evidence when a proposed answer can be checked directly. Google DeepMind's 2025 AlphaEvolve report describes a system that proposes computer programs, runs evaluations, and uses the results to guide further proposals. The language model helps generate candidates; executable tests supply feedback about what those candidates do.[^c8-13]

One application concerned scheduling jobs in Google's data centers. A machine can have spare computing capacity but be unable to take another job because another resource, such as memory, is already exhausted. The report describes searching for a better rule for assigning work to available machines. This is the factory's coordination problem turned into an object of automated investigation.

The researchers used a simulator based on historical workloads, tested on unseen recent data, and then measured operation after deployment. They reported recovering an average of 0.7 percent of fleet-wide computing resources that would otherwise have been stranded. That measures available computing capacity, rather than an equivalent reduction in electricity use.

The sequence matters as much as the headline. A proposal entered a test with a defined outcome. A promising simulated result faced another check. Deployment supplied further observations. The program was not considered better simply because another language model produced a fluent explanation of why it ought to work.

Smallville's party unfolds under programmed rules. A scheduling simulation can also misrepresent a real system. In the AlphaEvolve account, measurements from subsequent operation provided a way to check the simulated gains against actual performance.

For software generation, useful evidence comes from executed programs and checked outcomes. Candidate code can help organize a search, but its correctness needs testing beyond the model's ability to produce it.

The test set still needs attention. A program might exploit a quirk of the examples, perform well on clear documents, and fail on revisions. A good development process would investigate such limits. Automating the proposals makes the quality of evaluation more consequential because many candidates can now be optimized against whatever test we provide.

This possibility strengthens the constructive part of the comparison argument. An AI system can help us propose and assess alternatives at a cost that changes which experiments are worth attempting. It can do so inside a large firm or in a smaller undertaking. The organizational label does not tell us whether the tests are informative.

Evidence that a scheduling rule uses machines better can justify adopting that rule. The experiment compares ways to schedule the available machines. It leaves the contributions of earlier investors and the effects of alternative ownership arrangements to other evidence.

Generated proposals can help us test ambitious ideas when we can define an outcome that establishes improvement. The question is how to connect each proposal to evidence that can distinguish success from a plausible story. That connection lets us use AI's capabilities while examining whether the resulting system accomplishes its purpose.

## What the billion requests leave behind

Record design determines what inference can follow. A final answer and acceptance signal supply different evidence from a record also containing task characteristics, assignment rules, checked outcomes, and appropriate alternative-system tests. Equal request volumes need not produce equally informative records.

Acceptance can reflect verification, difficulty checking, or time pressure. Comparing outcomes with independently checked evidence helps distinguish those explanations and identify specific failures. The feedback signal must be interpreted before it is treated as correctness.

Production can be concentrated while a limited experiment preserves a relevant comparison. Assignments justified as learning must expose informative differences and be capable of affecting a decision, rather than merely adding a second name to a record.

The favorite can vary by task. Aggregate request totals can conceal a lack of within-task comparison, while a dominant service can host useful internal experiments. Logged assignment rules and checked outcomes matter together.

The comparison budget accumulates modeled allocation chances outside the current favorite, conditional on the relevant history and inputs. It counts neither actual calls nor agents. Small persistent chances can accumulate indefinitely; shrinking chances need not have a finite sum. Additional informative outcomes belong in the observed experiment. A busy agent team supplies no shortcut around these distinctions.

Stanford's 2026 AI Index distinguishes training from inference in discussing energy use. Repeated service remains a physical activity after a model has been developed. That reminder belongs beside the compiler and the order: extra attempts use resources, and their value depends on what they accomplish.[^c8-11]

AI can lower the cost of generating and testing alternatives. It can also operate within institutions that concentrate control of records, distribution, and evaluation. Its useful output and its effect on independent discovery both belong in the economic assessment.

The constructive choice is to identify a valuable uncertainty and the observation that could reduce it. A paired vote, checked document, randomized assignment, reusable log, or independent deployment can help under suitable conditions. Testing today's extractor also differs from testing whether its use teaches workers to recognize tomorrow's difficult cases. The next question follows the person who has to do that work.

## The person at the other end

A date in a report depends on more than the model that supplied it. The computing, service selection, evaluation, and person checking the answer all help determine whether the date can be trusted and used.

Someone supplied examples used in development. Someone expressed a preference in an evaluation, prepared a document, checked an answer, or corrected an error. Arena's paired votes make one such contribution visible. In a workplace, the corresponding judgments can be less conspicuous, folded into the ordinary activity of getting a task done.

If an AI service improves the worker's performance, that is a real possibility worth measuring. If it changes which tasks the worker gets to attempt, how expertise develops, or what an employer can observe about ability, those are further questions. An answer that helps finish today's task can also change the conditions under which tomorrow's worker learns.

Behind the answers are people doing work, people whose work is changing, and beginners still waiting for a first assignment. The system that handles today's task also helps determine who gets to learn tomorrow's job.

## Notes

[^c8-1]: Wei-Lin Chiang and colleagues, “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference,” *ICML*, PMLR 235 (2024): 8359–8388, [published paper](https://raw.githubusercontent.com/mlresearch/v235/main/assets/chiang24b/chiang24b.pdf), PDF pp. 1–3, Figure 1 and Section 3.1.

[^c8-2]: Isaac Ong and colleagues, “RouteLLM: Learning to Route LLMs with Preference Data,” *ICLR* (2025), sections 3–5 and limitations. [Version 4, February 23, 2025](https://arxiv.org/pdf/2406.18665v4), 3–7 and 10.

[^c8-3]: Lingjiao Chen, Matei Zaharia, and James Zou, [“FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance”](https://lingjiaochen.com/papers/2024_FrugalGPT_TMLR.pdf), *TMLR*, December 2024, Section 4, pp. 4–5, and experiments and Figure 3, pp. 6–10.

[^c8-4]: Lihong Li, Wei Chu, John Langford, and Xuanhui Wang, “Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms,” *WSDM* 2011, 297–306, DOI 10.1145/1935826.1935878; [author version 2](https://arxiv.org/pdf/1003.5956v2), March 1, 2012, Sections 3–4, especially Theorem 1 and the Yahoo study.

[^c8-5]: Llama Team, AI at Meta, *The Llama 3 Herd of Models*, [version 3](https://arxiv.org/pdf/2407.21783v3), November 23, 2024, p. 2, paragraph on the flagship Llama 3.1 model improving smaller models during post-training.

[^c8-6]: Yaneer Bar-Yam, “From Big Data to Important Information,” *Complexity* 21(S2) (2016): 73–98, DOI 10.1002/cplx.21785; [April 2016 author preprint](https://arxiv.org/pdf/1604.00976), abstract and Sections I–II, pp. 1–3.

[^c8-7]: David Blackwell and Lester Dubins, “Merging of Opinions with Increasing Information,” *Annals of Mathematical Statistics* 33 (1962): 882–886, [doi:10.1214/aoms/1177704456](https://doi.org/10.1214/aoms/1177704456), pp. 882–883 and main theorem.

[^c8-8]: Hao Zhang, “Inconsistent Estimation and Asymptotically Equal Interpolations in Model-Based Geostatistics,” *JASA* 99 (2004), 250–261, [full paper](https://www.stat.purdue.edu/~zhanghao/Paper/JASA2004.pdf), introduction and Section 3, pp. 250–253.

[^c8-10]: Joon Sung Park and colleagues, “Generative Agents: Interactive Simulacra of Human Behavior,” *UIST* 2023, [author version 2](https://arxiv.org/pdf/2304.03442v2), August 6, 2023, Sections 3, 6.5.2, 7 and 8.2, PDF pp. 5 and 15–17.

[^c8-11]: Stanford HAI, [*AI Index Report 2026*](https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf), Section 1.4, inference subsection, printed p. 36.

[^c8-12]: Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto, “Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators,” first released 2024, [author version 2](https://arxiv.org/pdf/2404.04475v2), March 10, 2025, pp. 1–4, Section 4.1 and Figure 3, p. 6, and limitations, p. 9.

[^c8-13]: Alexander Novikov and colleagues, *AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery*, [2025 white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf), pp. 1–3 and data-center scheduling discussion pp. 13–14; Google DeepMind, [May 14, 2025 account](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/).

[^c8-agents]: Nicholas Carlini, [“Building a C Compiler with a Team of Parallel Claudes,”](https://www.anthropic.com/engineering/building-c-compiler) February 5, 2026, sections “Make Parallelism Easy” and “Evaluation.” Anthropic, [“How We Built Our Multi-Agent Research System,”](https://www.anthropic.com/engineering/multi-agent-research-system) June 13, 2025, section “Architecture Overview.”
