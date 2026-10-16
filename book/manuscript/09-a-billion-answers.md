# Chapter 8

## A billion answers

A purchase order arrives with a delivery deadline somewhere in its pages. In our hypothetical workplace, a worker asks an AI service to find the date. Behind the familiar interface, software chooses a model, sends the document, receives an answer, and returns it to the worker.

The exchange can leave several records. There is the request itself, the model selected, the answer, the time taken, and perhaps a signal that the worker accepted it. If the workplace checks the answer against the order, there can also be a record of whether it was right. These are different observations, produced by different decisions about what to measure.

The trained model need not change its numerical settings during this exchange. A later development process might use collected examples to update it. The service could also retrieve a newer document or apply a different instruction without retraining the underlying model. We should keep those changes distinct when asking what another successful answer teaches us: the learning can occur in the model, in the surrounding software, or in the people evaluating the result.

Now repeat the exchange a billion times across many workplaces and tasks. The records can become immensely useful. They can expose recurring mistakes, reveal delays, and help developers improve a service. But the number of completed requests does not, by itself, tell us which questions the resulting evidence can answer.

One question concerns this particular order: did the service find the correct deadline? Another concerns the alternative model: would it have done better on comparable work? A third concerns the development process: which investment produced the capability? A fourth concerns an economic arrangement: what would have happened if another supplier had received the early opportunities and resources to develop?

We can learn a great deal about the first question without settling all the others. We can also design the service to collect evidence about more than one. The choice matters before the billionth answer arrives. It matters when we decide what happens to the first request.

## Two answers to one question

Chatbot Arena offers a concrete departure from an ordinary exchange. In the platform described by Wei-Lin Chiang and colleagues in their 2024 paper, a user enters a prompt and receives responses from two anonymously selected models. The user compares them and votes; the identities are revealed afterward. The interaction creates a paired preference observation, rather than simply another conversation with a named favorite.[^c8-1]

The two columns on the screen do useful work. They bring alternatives into the same encounter, expose them to the same prompt, and delay the information that a famous brand might provide. We can argue about the design while recognizing what it adds. A record of which service people already use supplies a different kind of evidence from a vote after they have seen two responses.

In the hypothetical workplace, we could adopt part of that design. A worker might compare two proposed readings of the same purchase order before seeing the suppliers' names. If one answer identifies the date and the other confuses it with the date of issue, the document can supply a further check. The comparison has a defined object and an observable mistake.

But preference and correctness remain different. A neatly presented wrong date might look more convincing than a correct answer expressed awkwardly. A worker who knows the underlying document can evaluate differently from one who does not. Deciding which answer is preferred is an intelligible task; deciding which answer is accurate may require additional evidence.

Arena's paper itself distinguishes benchmarks based on human preference from those based on a known correct answer. That distinction helps us resist treating one ranking as a certificate for every use. A buyer interested in pleasant conversation and a buyer extracting deadlines may need different measurements. Neither has to dismiss the other's question to insist on measuring its own.

There is also a difference between the people and prompts that arrive at a public platform and those in a particular workplace. If our buyer uses unusually long orders, scanned documents, or a specialized vocabulary, it should investigate performance on that work. A useful general comparison can be a starting point for a decision without being its entire evidential basis.

Imagine Aster and Birch, our fictional suppliers, receiving different mixtures of requests. Aster handles mostly short, clear documents. Birch receives damaged scans and complicated revisions. Aster's higher acceptance rate might reflect a better service, easier work, or both. Comparing the rates without investigating the assignments would give a tidy number to an untidy question.

A paired test can help with that particular problem. We could send suitable examples from each category to both services and assess their outputs using a consistent procedure. The cost includes additional computation and evaluation. The benefit is evidence directed toward a decision that a single unbalanced stream would have made harder.

The Arena design also reminds us that comparison has to be produced. Someone builds the interface, selects model pairs, collects responses, and maintains the analysis. Those activities are part of the institution. The platform's methodological choices affect what a vote means, which differences it can reveal, and how confidently a ranking can be interpreted.

Suppose a purchasing department then uses a ranking to select its default supplier. The ranking can influence subsequent exposure. That possibility does not invalidate the original tests. It creates a new question about how the tests and the resulting deployment interact. A good measured result can become an input into a process that later generates a different kind of evidence.

For the order on our worker's screen, the immediate achievement remains concrete: find the deadline accurately and at an acceptable cost. We can pursue that aim while arranging an occasional comparison that could show when another system performs better. The service and the experiment can be designed together.

## When the judge likes the packaging

Yann Dubois and colleagues tested how an automatic judge responds to presentation in their length-controlled AlpacaEval study. They prompted models to give concise or verbose answers. For one GPT-4 version, the unadjusted win rate ranged from about twenty-three to sixty-four percent as the instruction changed. The underlying model had not received a new training run; the prompt and resulting answers differed. Their statistical length adjustment reduced the swing. The experiment exposes sensitivity to presentation, while leaving us to investigate which changes also improve the answer.[^c8-12]

Each response faced a reference answer to the same instruction. The judge's preference supplied the score. It was a measure of which answer the evaluator favored, rather than an independent check against a separately established correct solution.[^c8-12]

This is a different evaluation procedure from Arena's direct user votes. The distinction prevents a lazy conclusion that every human judgment is merely a preference for long answers. The study examines a particular automatic judge and an adjustment to its measurements. It gives us a concrete example of how the apparatus of evaluation can influence the result.

Think of the deadline in the purchase order. One service returns the date and identifies the relevant passage. Another returns the same date inside a long explanation of the document. A buyer might prefer either presentation for a legitimate reason. An evaluation intended to measure the extraction task should nevertheless understand how much of its ranking depends on the surrounding prose.

Now imagine that the longer response contains an error elsewhere. The extra detail has changed more than the length of the answer; it has created another opportunity to mislead the worker. Alternatively, the longer explanation might expose an ambiguity that the short answer concealed. Length by itself cannot tell us which happened. The measurement must remain connected to the purpose.

The length-controlled study also makes a counterfactual question explicit: how would the estimated preference compare if the responses had the same length? Its statistical adjustment supplies an estimate under a model. The authors are not claiming to have observed every possible rewritten answer or eliminated every source of evaluation bias. A defined counterfactual can be useful without becoming an omniscient reconstruction.

For a developer, the practical stakes concern which improvements an evaluation rewards. If a score rises after a formatting change, the developer should investigate whether users actually complete their work better. If better reasoning raises the score, that is worth distinguishing too. The same final ranking can emerge through changes with different consequences for the customer.

For the buyer choosing Aster or Birch, this adds an institutional question. Who defines the test that helps decide who receives the next opportunity? An accessible evaluation can support discovery, but its convenience is insufficient by itself. Its design should expose important differences rather than repeatedly reward a feature that the buyer does not actually need.

## Choosing before paying

The next design decision concerns which models to call. Sending every request to the most expensive available system may waste resources on tasks a cheaper system handles adequately. Calling every model for every request can produce comparisons, but also an unnecessary bill. The problem invites a selector: software that routes the request according to the work it appears to require.

Isaac Ong and colleagues' RouteLLM research investigates such a selector. Their framework learns from preference data and routes between a stronger, more expensive model and a cheaper alternative. They evaluate the resulting tradeoffs on specified benchmarks and compare their methods with random routing. Their reported improvements concern those models, data, and evaluations, rather than a guarantee for every future workplace.[^c8-2]

The purchase order makes the motivation easy to understand. A short document with a clearly labeled date may require less capability than an ambiguous revision spread across several pages. A well-designed router could reserve the more costly service for the cases where it adds enough value. That is a productive improvement even if the user never sees the selection happening.

Notice what the router must learn. It needs evidence about relative performance on kinds of requests. A general reputation for being stronger is not the same as an advantage worth paying for on every task. A useful selection rule takes variation seriously instead of reading an overall ranking as an instruction to make the same choice forever.

The RouteLLM paper also reports limits to transferring a router across tasks. Data resembling the eventual evaluation can improve performance. This is an ordinary but consequential lesson: evidence collected for one use does not automatically retain the same meaning in another. Our buyer should care about the correspondence between training examples, evaluation examples, and actual orders.

Lingjiao Chen, Matei Zaharia, and James Zou explore a related route in FrugalGPT. Their system can call a model, assess the response, and decide whether to stop or call another. The answer already obtained becomes an input into the next decision. The study compares cost and performance on defined tasks, including classification and reading comprehension.[^c8-3]

In the workplace illustration, the system might first ask a cheaper model to extract the deadline. If its assessment of that response is unfavorable, it could ask another model. This differs from selecting once before seeing an answer. It also differs from obtaining an independent human check of correctness. The assessment mechanism itself needs evaluation.

FrugalGPT's paper includes cases where additional calls are made even though the models' answers agree. Its discussion treats this as room for improvement: the selector's uncertainty can cause extra expense. The example makes an important point about automation. Delegating the decision to a model does not remove the decision problem; it creates a system whose judgment about when to continue has consequences.

For a buyer, an economical cascade might lower the cost of trying alternative services. It might also make comparisons less visible if the final interface reports only one answer and one brand. We should inspect what the system actually records. A useful comparison can occur behind a simple interface, while an interface displaying several logos can conceal dependence on the same underlying service.

The two research programs give AI a positive role in this book's argument. Software can help direct scarce resources toward useful tests and useful service. More expensive machinery need not mean every opportunity becomes more expensive. Better selection and cheaper evaluation can change what is feasible.

One distinction is essential before we try to measure a comparison budget. A router's estimate that one model will produce the preferred answer differs from the chance that the request will actually be assigned to that model. RouteLLM uses a predicted preference and a threshold to make its selection. The prediction and the assignment rule do different jobs.

Suppose, in an illustrative setting, a selector judges that Aster has a seventy percent chance of producing the preferred response on this kind of order. It could then send every such order to Aster. The seventy percent figure describes the predicted comparison outcome; it does not imply that Birch receives thirty percent of the requests. Adding those thirty percent figures would not measure the allocation process we have actually specified.

A different rule could deliberately randomize some assignments. Its allocation probabilities would describe that design, while its predicted performance scores would describe the selector's expectations. Both can be recorded. Their relationship matters to the analysis, but they are not interchangeable numbers simply because both can be expressed as percentages.

Nor does deterministic selection make all subsequent observations uninformative. If our workplace checks the dates extracted by Aster, those checks can reveal performance on the orders it receives. If independent tests also assess Birch, they can add evidence about alternatives. Applying the allocation-history bound while ignoring these outcomes would misdescribe the experiment and exaggerate the book's conclusion.

The work of checking also has a production process. Someone may need to open an amendment, resolve an ambiguous field, or determine which document version governs the task. AI could assist with that work, but an automated score remains something to validate against the relevant standard. A service can lower the cost of producing an answer while leaving costly uncertainty about whether the answer is useful.

The buyer should therefore investigate the entire chain. What observation trains the selector? Which rule makes the assignment? What check evaluates the response? Which later decision can use that check? Our order does not need to become a giant economic experiment to make those questions worthwhile. They help us distinguish an efficient service, a useful comparison, and a record whose apparent precision exceeds what was measured.

The objectives still need to be named. A router optimized for the best current cost/performance tradeoff need not be optimized to discover a new supplier's potential. It may already have enough evidence for its operational decision, or it may need occasional new observations. We cannot determine which merely by noting that the routing is automated.

## A record worth reusing

Long before today's language models, a news page posed a related problem. The Yahoo Today Module could highlight an article for a visitor and observe whether the visitor clicked. It could not observe that same visit's response to every article left unshown. Lihong Li, Wei Chu, John Langford, and Xuanhui Wang investigated how previously collected data could help evaluate another recommendation rule.[^c8-4]

Their work used records gathered through randomized article selection. Under their stated conditions, an evaluator can retain suitable events where a proposed rule would have selected the article that was actually shown. The retained record then includes a real response to that choice. The researchers compared offline evaluations with online results in the Yahoo application.

This is an instructive answer to the claim that one history must leave every alternative forever beyond reach. A historical record can contain deliberately created variation. The variation can support questions that the institution operating the system did not answer by simply repeating its current favorite. Collecting the right record changes the possibilities for later analysis.

Return to the order. Suppose our buyer sometimes assigns requests randomly between suitable services and records the assignment mechanism and checked outcomes. A later analyst can ask questions about another selection rule using those observations, if the relevant assumptions and coverage hold. We need not build a literal parallel workplace for every policy we wish to assess.

That last qualification matters. If a service was never available for a particular kind of order, the record may contain no relevant responses from it. Knowing that the probability was zero does not supply the missing performance. An estimator needs suitable observations or additional defensible assumptions, rather than an increasingly elaborate way of describing their absence.

The Yahoo authors call their method replay. Their use of the word concerns evaluating a recommendation policy with suitable logged events. My shadow futures concern alternative histories generated under specified economic conditions and rules. These are distinct uses. A procedure that reuses article records is not a time machine for recreating the entire industry with a different first winner.

Our hypothetical buyer might use randomized logs to compare current Aster and current Birch on a defined task population. That result would not, on its own, establish how Birch would have developed after a different early allocation sequence, building another team as the same response rules allowed. Deliberately changing its training strategy would be a separate intervention. Current-system evaluation, historical variation under fixed rules, and a changed development policy require different comparisons.

The same record can nevertheless be valuable. If it supports a better procurement decision, it need not explain every earlier fortune to justify its collection. The point of the comparison budget is to clarify a particular information constraint, not to demand a complete causal explanation of the economy before allowing any useful inference.

This gives us a practical reason to preserve details that a success count omits. Which alternatives were available? Why was this one selected? What was measured afterward? A usable assignment record and meaningful outcome can make a small pilot worth more, for a specific decision, than a much larger archive lacking those features.

The pilot also has costs. Someone receives the less suitable answer when the comparison goes badly. Someone checks the outcome. Some data may be inappropriate to retain or share. The Yahoo research illustrates an opportunity created by deliberate observation; it does not imply that the opportunity is free or that every problem meets its assumptions.

## The same ancestor

Our hypothetical buyer opens three services and receives three answers to the order question. Three storefronts are visible. What has been compared?

Perhaps each service has independently trained a different system. Perhaps each has adapted a shared model. Perhaps all three pass the request to the same supplier, with different surrounding instructions. The answers may still differ. The number of interfaces does not identify which part of the production process varied.

Meta's Llama 3 report supplies a concrete example of a relationship between models. It describes using its flagship model to improve smaller models during post-training. Different members of a model family can therefore be connected through the development process. Treating their outputs as evidence from wholly unrelated research histories would miss that connection.[^c8-5]

Shared ancestry does not make their performance identical. A smaller model can behave differently, require different resources, and fit a particular application better. Comparing those actual systems can teach us something useful. The mistake would be to suppose the comparison automatically tests every assumption inherited from their common origin.

Imagine that our three services extract the same wrong date because the underlying document representation loses the same footnote. Asking all three may produce reassuring agreement while leaving that error untouched. Now imagine that one service uses a better document parser and finds the footnote. A common language model has not prevented a meaningful improvement elsewhere in the service.

These are illustrative cases, but they identify an empirical task. Follow the chain far enough to understand which components are common, which vary, and where the outcome is measured. Independence is valuable relative to a question. It should not be a ceremonial word attached to a company name.

The same caution applies to repeated answers from one system. New requests and new checks can provide fresh information about that system's performance. Repeating a model call is not automatically repeating the same observation. A thousand carefully checked orders can expose failure patterns that one order could not reveal.

But generating a thousand responses without a new check does not create a thousand independent confirmations of the underlying claim. If the uncertainty concerns the actual delivery date, the order and its amendments remain relevant evidence. A longer conversation among programs might help interpret them. It cannot make an unread amendment irrelevant by agreeing without it.

Yaneer Bar-Yam's discussion of moving from big data to important information provides a predecessor to this way of asking the question. Detailed records and information useful for understanding an intervention are different. His multiscale framework is not my allocation theorem, but the warning against treating the quantity of detail as an answer to every consequential question belongs in the lineage.[^c8-6]

The workplace can act on that warning. It can measure whether the extracted date is correct, whether the answer helped a worker complete the task, and whether a different service changes the result. Each measurement has a purpose. We gain clarity by naming those purposes before admiring the size of the resulting database.

## Predicting the next choice

Suppose two analysts watch the buyer's use of Aster. One credits Aster's engineering; the other assigns more importance to its early access and accumulated references. Over time, both become increasingly good at anticipating which supplier the buyer will choose. Their forecasts can become similar while their accounts of the earlier achievement remain different.

The statistical distinction is older than this book. David Blackwell and Lester Dubins showed in 1962 how conditional forecasts can merge under an appropriate relationship between probability laws. Hao Zhang later established a different example in spatial statistics where some underlying parameters cannot be consistently recovered even though important prediction can remain effective. Both deserve credit.[^c8-7][^c8-8]

Under the conditions in Chapter 3, forecasts of the next award can converge while the effect of work on those awards remains unresolved. Both observers might confidently predict Aster's next contract without knowing what another engineer would change. This is a result about forecasting the modeled allocation process; checking an AI's answers requires evidence about the answers themselves.

Keep the purchase order in view. A system could predict accurately that the worker will accept Aster's answer. That prediction concerns acceptance. Whether Aster extracted the correct date is another measurable question. Whether an early investment caused the difference between Aster and Birch requires a further causal comparison. These questions should remain distinct even when the software handles them behind one interface.

We should also resist reversing the theorem into a general celebration of ignorance. An accurately checked answer teaches us something. A well-designed comparison can identify a performance difference. Some input effects may be learned from additional measurements. The finite-budget result concerns a particular obstacle under particular conditions, rather than a rule that economic evidence must always fail.

For the buyer, the next action could be straightforward. If errors matter, check a suitable sample against the original orders. If supplier choice matters, preserve a meaningful comparison. If a historical attribution claim matters, identify the intervention, horizon, and evidence that could support it. The inquiry becomes more useful as it becomes more specific.

## Sixteen agents and one compiler

A swarm sounds like a crowd. On a computer, it can mean several copies of a model pursuing assigned work, using tools, keeping records, and returning results. The economic question begins with what those activities accomplish. Sixteen running programs can perform sixteen useful investigations, or spend the afternoon rediscovering one mistake.

In February 2026, Nicholas Carlini at Anthropic reported setting sixteen Claude instances to build a C compiler, which translates code into instructions a computer can run. Compiling the Linux operating system, they stalled on the same bug and interfered with one another's fixes. He redesigned the tests: GCC, an established compiler, handled most files while the new compiler handled varying subsets. Failures became easier to isolate. The reported achievement still had limits, including inefficient generated code and dependence on other tools. This was an attributed capability experiment, not a measured replacement for a human workforce.[^c8-agents]

The revised tests helped separate failures. The checking arrangement belonged to the productive machinery as surely as the model did. A candidate program needed reliable tests that could expose its mistakes.

There are different ways to organize that machinery. Anthropic's June 2025 research system used a lead agent to distribute searches among others. Carlini's compiler account explicitly described a setup without a lead agent assigning tasks. In either arrangement, ask who directs the work and supplies the feedback.[^c8-agents]

Now give fictional Birch an agent team. One program reads purchase orders, another checks amendments, and a third runs tests on proposed extraction code. The team might help a small business investigate more designs than its staff could previously afford. Employees still define a useful deadline, resolve disputed documents, and decide which errors matter. The firm buys a capacity to attempt work, with a bill for the attempts and a further cost of checking them.

Suppose Birch retains the resulting code, test cases, and usable project records. If one model becomes unsuitable, the firm can try another against those cases. Its accumulated work helps make the next comparison possible. A supplier of model access and the owner of a developing business then have different roles, even when the business depends heavily on that supplier today.

A usable record could also include failed attempts. If Birch transfers only the final code, the next team may repeat an abandoned approach. Keeping that history costs storage and care over the customer's confidential material.

Change the stipulated contract. The useful records now remain inside an interface Birch cannot transfer, and its customer route depends on the same provider. Switching could require rebuilding the tests and regaining access to buyers. These are hypothetical terms, not an allegation about an existing service. They show why cheaper software work and easier independent entry are separate propositions. An agent team can build valuable capabilities while the rights governing those capabilities make departure difficult.

Ownership alone cannot settle the information question. The compiler experiment illustrates useful checking inside one company using related models. Conversely, Birch could own every generated file and still rely on a defective test. The relevant inquiry follows the attempted design to an outcome capable of exposing its failure. Different letterheads do not provide that connection automatically.

For the order on our worker's desk, a thousand agents agreeing on a date could still miss the same amendment. One program that finds the amendment could improve the answer. A buyer should investigate which happened before counting the votes. The prospect of abundant software workers makes this discipline more consequential: we may be able to multiply attempts faster than we can produce trustworthy judgments of their results.

## A town throws a party

In a 2023 study, Joon Sung Park and colleagues populated Smallville with twenty-five software agents. One received the intention to organize a Valentine's Day party. The authors report invitations spreading and five invited agents arriving at the cafe. Another resident mistakenly identified her neighbor, Adam Smith, as the author of The Wealth of Nations. Knowledge inherited from a language model had intruded into the neighborhood.[^c8-10]

These characters supply a tempting contrast with a compiler. A simulated town can be rerun, with rules changed and histories recorded. But what would establish that its response to a tax change predicts the response of people? Believable conversation and a program's successful execution do not supply that validation. The study investigated believability and agent architecture, not the accuracy of economic policy forecasts.

We could simulate buyers choosing between Aster and Birch. The exercise might reveal an overlooked implication or help design an empirical test. The buyers' preferences could also reflect instructions or assumptions inherited from their model. Generating more residents would reduce some simulation noise without correcting a mistaken account of demand.

A shadow replay keeps underlying parameters, starting conditions, and response rules fixed while the random history changes. Altering procurement or training policy is a separate intervention. For either operation, the claim about an actual economy requires evidence linking the model to that economy. The party makes generated interaction visible. The compiler's checks point toward the additional work of connecting a proposed answer to a result that can prove it wrong.

## When a generated answer can be tested

There is a more encouraging route from generation to evidence when a proposed answer can be checked directly. Google DeepMind's 2025 AlphaEvolve report describes a system that proposes computer programs, runs evaluations, and uses the results to guide further proposals. The language model helps generate candidates; executable tests supply feedback about what those candidates do.[^c8-13]

One application concerned scheduling jobs in Google's data centers. A machine can have spare computing capacity but be unable to take another job because another resource, such as memory, is already exhausted. The report describes searching for a better rule for assigning work to available machines. This is the factory's coordination problem turned into an object of automated investigation.

The researchers used a simulator based on historical workloads, tested on unseen recent data, and then measured operation after deployment. They reported recovering an average of 0.7 percent of fleet-wide computing resources that would otherwise have been stranded. This is a developer-reported result about their operation, rather than an independently audited economy-wide benefit.

The sequence matters as much as the headline. A proposal entered a test with a defined outcome. A promising simulated result faced another check. Deployment supplied further observations. The program was not considered better simply because another language model produced a fluent explanation of why it ought to work.

Compare that with the simulated party. Smallville can show what its agents do under its rules. A scheduling simulation can also be wrong about a real system. In the AlphaEvolve account, subsequent measurements provided an additional connection to actual operation. We should examine the quality of that connection while recognizing why it adds something that another unvalidated simulation would not.

For fictional Birch, AI could help generate alternative document-processing programs. The company could run them on a test set with known deadlines, inspect failures, and assess the promising candidates on fresh orders. Useful evidence would then come from executed programs and checked documents. The generated code would help organize the search; it would not supply correctness merely by existing.

The test set still needs attention. A program might exploit a quirk of the examples, perform well on clear documents, and fail on revisions. A good development process would investigate such limits. Automating the proposals makes the quality of evaluation more consequential because many candidates can now be optimized against whatever test we provide.

This possibility strengthens the constructive part of the comparison argument. An AI system can help us propose and assess alternatives at a cost that changes which experiments are worth attempting. It can do so inside a large firm or in a smaller undertaking. The organizational label does not tell us whether the tests are informative.

The information also has a boundary. Evidence that a scheduling rule uses machines better can justify adopting that rule. It does not, by itself, measure the contribution of every earlier investor or determine whether another ownership arrangement would have produced a superior computing industry. A successful local experiment deserves its success without being asked to answer every historical question.

Generated proposals can help us test ambitious ideas when we can define an outcome that establishes improvement. The question is how to connect each proposal to evidence that can distinguish success from a plausible story. That connection lets us use AI's capabilities while examining whether the resulting system accomplishes its purpose.

## What the billion requests leave behind

Return to the order and let our hypothetical workplace choose its records. One design retains only the final answer, supplier, and a signal that the worker moved on. Another also retains task characteristics, the selection procedure, checked correctness, and a limited alternative-service trial. Both could process a billion requests. The records would support different questions.

Acceptance itself needs interpretation. The worker might have checked the date, found checking difficult, or hurried toward another deadline. These are possibilities in our illustration. Comparing a sample against the governing orders could reveal a recurring confusion between the issue date and delivery date, giving the company a specific error to investigate.

Aster's strong performance could justify giving it most production work while preserving a limited test of alternatives. Birch should not receive assignments merely to decorate a spreadsheet with another supplier. Its inputs, tasks, evaluation, and opportunity to be selected after a good result must make the trial relevant to a decision.

The favorite can also change with the request. Aster may be favored for one document class and Birch for another. Aggregate request totals can conceal little within-class comparison, just as a dominant supplier can host informative internal tests. The assignment process and checked outcomes need to be examined together.

The comparison budget accumulates modeled allocation chances outside the current favorite, conditional on the relevant history and inputs. It counts neither actual calls nor agents. Small persistent chances can accumulate indefinitely; shrinking chances need not have a finite sum. Additional informative outcomes belong in the observed experiment. A busy agent team supplies no shortcut around these distinctions.

Stanford's 2026 AI Index distinguishes training from inference in discussing energy use. Repeated service remains a physical activity after a model has been developed. That reminder belongs beside the compiler and the order: extra attempts use resources, and their value depends on what they accomplish.[^c8-11]

Some arrangements could make comparison cheaper. Birch could generate candidates, retain tests, and purchase independent checks on unfamiliar orders. Others could extend a favored provider's control over records and distribution while improving its current service. The service's usefulness would remain real; the route by which another approach could develop would remain a separate institutional question.

The constructive choice is to identify a valuable uncertainty and the observation that could reduce it. A paired vote, checked document, randomized assignment, reusable log, or independent deployment can help under suitable conditions. Testing today's extractor also differs from testing whether its use teaches workers to recognize tomorrow's difficult cases. The next question follows the person who has to do that work.

## The person at the other end

At the beginning of this chapter, the worker asked for a date. By the end, that small request has connected a computing facility to a model, a selection rule, an evaluation procedure, and a future procurement decision. The person remains part of the chain.

Someone supplied examples used in development. Someone expressed a preference in an evaluation, prepared a document, checked an answer, or corrected an error. Arena's paired votes make one such contribution visible. In a workplace, the corresponding judgments can be less conspicuous, folded into the ordinary activity of getting a task done.

If an AI service improves the worker's performance, that is a real possibility worth measuring. If it changes which tasks the worker gets to attempt, how expertise develops, or what an employer can observe about ability, those are further questions. An answer that helps finish today's order can also change the conditions under which tomorrow's worker learns.

The next chapter asks what happens to the worker. Its subject is the work behind the answers, the work changed by them, and the first opportunities through which people develop a record of their own.

## Notes

[^c8-1]: Wei-Lin Chiang and colleagues, “Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference,” *ICML*, PMLR 235 (2024), 8359–8388, [published paper](https://raw.githubusercontent.com/mlresearch/v235/main/assets/chiang24b/chiang24b.pdf), PDF pp. 1–3, Figure 1 and Section 3.1. Model-pair sampling can be deliberately nonuniform; randomized does not mean every pair has equal probability. The chapter describes the published design, not a verified current platform implementation, universal truth metric, or representative census of workplace demand.

[^c8-2]: Isaac Ong and colleagues, “RouteLLM: Learning to Route LLMs with Preference Data,” *ICLR* 2025, [full version 4](https://arxiv.org/pdf/2406.18665v4), February 23, 2025, Sections 3–5, pp. 3–7, and limitations, p. 10. Original report June 2024. Training uses human preferences and augmentation; evaluation spans distinct benchmarks and scoring procedures. No particular cost-saving percentage, universal router, unbiased factual judgment, or guarantee on our fictional orders is claimed.

[^c8-3]: Lingjiao Chen, Matei Zaharia, and James Zou, “FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance,” *TMLR*, December 2024, [published paper](https://lingjiaochen.com/papers/2024_FrugalGPT_TMLR.pdf), Section 4, pp. 4–5; experiments and Figure 3, pp. 6–10. Its main historical experiments use March 2023 models/prices, with a November 2024 extension. The chapter uses the cascade/scorer/stopping design and recorded extra-call limitation, not current prices or a universal savings estimate.

[^c8-4]: Lihong Li, Wei Chu, John Langford, and Xuanhui Wang, “Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms,” *WSDM* 2011, 297–306, DOI 10.1145/1935826.1935878; [author version 2](https://arxiv.org/pdf/1003.5956v2), March 1, 2012, Sections 3–4, especially Theorem 1 and the Yahoo study. The theorem assumes independent identically distributed events and uniformly randomized logging; modifications require corresponding conditions. A finite-stream ratio and concentration for arbitrary adaptive algorithms need additional care. Logged policy evaluation differs from a structural replay of economic development.

[^c8-5]: Llama Team, AI at Meta, *The Llama 3 Herd of Models*, [version 3](https://arxiv.org/pdf/2407.21783v3), November 23, 2024, p. 2, paragraph on the flagship model improving smaller models during post-training. The paper's language results concern Llama 3.1. Shared development does not establish identical errors, useless downstream tests, or statistical independence of outputs. Our three storefronts and document-parser examples are explicit illustrations.

[^c8-6]: Yaneer Bar-Yam, “From Big Data to Important Information,” *Complexity* 21(S2) (2016), 73–98, DOI 10.1002/cplx.21785; [April 2016 author preprint](https://arxiv.org/pdf/1604.00976), abstract and Sections I–II, pp. 1–3. Credit concerns detail, scale and intervention-relevant information. His framework is an antecedent, not the comparison-budget bound or a general rejection of large datasets.

[^c8-7]: David Blackwell and Lester Dubins, “Merging of Opinions with Increasing Information,” *Annals of Mathematical Statistics* 33 (1962), 882–886, [DOI](https://doi.org/10.1214/aoms/1177704456), pp. 882–883 and main theorem. Under the specified absolute-continuity relationship, conditional future laws merge almost surely under the relevant law. This does not establish accurate chatbot statements, agreement of arbitrary observers, or nonlearning of parameters by itself. Chapter 3 explains why predictive merging and contribution identification are different questions.

[^c8-8]: Hao Zhang, “Inconsistent Estimation and Asymptotically Equal Interpolations in Model-Based Geostatistics,” *JASA* 99 (2004), 250–261, [full paper](https://www.stat.purdue.edu/~zhanghao/Paper/JASA2004.pdf), introduction and Section 3, pp. 250–253. Fixed-domain infill can leave individual covariance parameters inconsistent while an identifiable combination supports prediction. This is a prior statistical result, not evidence that any particular AI supplier's contribution is unknowable or that accurate prediction lacks value.

[^c8-10]: Joon Sung Park and colleagues, “Generative Agents: Interactive Simulacra of Human Behavior,” *UIST* 2023, [author version 2](https://arxiv.org/pdf/2304.03442v2), August 6, 2023, Sections 3, 6.5.2, 7 and 8.2, PDF pp. 5 and 15–17. Party coordination and the mistaken Adam Smith biography occur among software characters. The study evaluates believability and architecture, not predictive validity for real economic policy. No author interview, observed real-world town, consciousness claim or validated tax simulation is implied.

[^c8-11]: Stanford HAI, *AI Index Report 2026*, [full report](https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf), Section 1.4, inference subsection, printed p. 36. The chapter uses the distinction between developing a model and repeatedly serving it. It adopts no model-specific energy estimate, current usage count, or inference from energy consumption to the comparison budget.

[^c8-12]: Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto, “Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators,” first released 2024, [author version 2](https://arxiv.org/pdf/2404.04475v2), March 10, 2025, pp. 1–4, Section 4.1 and Figure 3, p. 6, limitations p. 9. Section 4.1 reports gpt4_1106_preview win rates of 22.9%–64.3% under verbosity prompts, reduced to 41.9%–51.6% with length control. Scores aggregate the automatic judge's preference probabilities against reference outputs, not independently checked correct-answer rates. Prompting also changes the actual answers; the study does not hold all answer quality fixed. The adjustment assumes a same-length comparison is desired and does not remove every judge bias. No universal correction or current ranking is adopted.

[^c8-13]: Alexander Novikov and colleagues, *AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery*, [2025 white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf), pp. 1–3 and data-center scheduling discussion pp. 13–14; Google DeepMind, [May 14, 2025 account](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/). Reported resource recovery is not an equivalent percentage electricity saving or proof of overall welfare. Simulation, held-out testing and post-deployment measurement are distinct stages; developer report, not independent causal audit.

[^c8-agents]: Nicholas Carlini, [“Building a C Compiler with a Team of Parallel Claudes”](https://www.anthropic.com/engineering/building-c-compiler), February 5, 2026, “Make parallelism easy” and “Evaluation”; Anthropic, [“How We Built Our Multi-Agent Research System”](https://www.anthropic.com/engineering/multi-agent-research-system), June 13, 2025, “Architecture overview.” Distinct architectures; attributed developer experiments, not labor-market estimates or independently replicated benchmarks. Human test design and external tools count. Birch’s contracts and agent tasks are hypothetical.
