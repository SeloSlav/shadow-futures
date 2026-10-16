# Chapter 7

## The intelligence factory

In the International Energy Agency's 2026 account of artificial intelligence, one of the consequential technologies is a connection to the electricity grid.

The report describes delays that have little to do with a chatbot's ability to write a sentence. Transformers can take years to arrive. In many jurisdictions, a grid connection can involve a wait of five to ten years. A company may have money, a plan, and customers eager to try its service while still lacking a route by which enough electricity can reach the machines.[^c7-1]

This is an unusual setting for a discussion of something described as intelligence. We expect intelligence to concern ideas. Here it also concerns a delivery schedule for electrical equipment. A better algorithm cannot, by itself, carry power down a transmission line that has not been built.

Nor can the phrase data center tell us what a facility will accomplish. A building can house many kinds of computing. The useful economic questions concern the equipment inside it, the work assigned to that equipment, and the arrangements through which its output reaches people. The same physical capacity could support different services and different opportunities to test them.

In its April report, the IEA estimated that all data centers together consumed about 485 terawatt-hours of electricity in 2025. A terawatt-hour is a billion kilowatt-hours, the unit familiar from electricity bills. Its central projection put consumption at roughly 950 terawatt-hours in 2030. That is an outlook, not an observation from the future, and both totals cover more than AI alone. The seemingly weightless answer occupies a material economy.

We have reached that economy from an argument about rewards. The previous chapter asked how an inventor, a financier, and a worker might have different claims arising from the same achievement. AI gives the question a new scale. Someone must obtain chips, install them, keep them working, and finance the interval before a useful product earns anything.

There are strong reasons for some of that work to occur in large organizations. There are also reasons to ask whether the resulting arrangements let other approaches demonstrate what they can do. Understanding both requires following the power cable into the building.

## The machine needs a building

At Oak Ridge National Laboratory, preparations for the Frontier supercomputer involved a cooling plant, elaborate piping, and foundations capable of carrying cabinets weighing four tons apiece. The laboratory's published construction account records a failed joint in a large pipe during a pressure test. Water poured out; the team identified the failed weld and repaired it. Before the computer could do its work, the building had to survive the plumbing.[^c7-2]

The failed weld belongs in the account of the machine. Buying processors had not completed the job; the supporting systems had to pass their own tests. The pressurized pipe offered a plain measure: a joint had to hold before the planned operation could proceed. Acknowledging that work does not require calculating the crew's share of every future scientific discovery the computer will support.

Frontier is a scientific supercomputer with that proper name. When people call an AI model a frontier model, they mean something different: a system near the leading edge of relevant capabilities at a particular time. The distinction is worth making before the language conceals the machinery. A leading model is produced and operated on physical systems; it is not the building itself.

Begin with the chips. An accelerator is a processor designed to carry out suitable computations efficiently, often by performing many operations in parallel. Graphics processing units, or GPUs, became central tools for this work. A collection of them does not automatically behave like one enormous, dependable computer. They need data, memory, connections, and software that assigns and coordinates the work.

Imagine a team whose members must repeatedly exchange intermediate results. Adding people may help, but only if they can communicate quickly enough to keep the work moving. A training cluster presents an engineering version of that problem. A useful processor waiting for data or another processor is an expensive object accomplishing less than its specifications suggest.

Cooling creates another coordination problem. The electricity used in computation becomes heat that must leave the equipment. At Oak Ridge, an account of the cooling system describes circulating water warm enough to avoid the refrigeration required by older arrangements. The design uses the fact that water can carry heat away even when it would feel warm to us.[^c7-3]

That small reversal of intuition matters economically. Cooling is a design problem with alternatives, rather than a fixed surcharge attached to every computation. Improvements can change how much useful work a facility gets from a given expenditure. They can require knowledge that is embodied in installation, controls, and operating practice as well as in the chip.

At LUMI, the European supercomputer in Kajaani, Finland, the operator describes another destination for the heat: the city's district-heating network. Its account says excess heat warms hundreds of households annually. The physical by-product of computing becomes an input into another service.[^c7-13]

The important feature is the connection between activities. A facility in one place may be near a usable heat network; another may have no comparable outlet. A household's demand for heating can vary with the season while the computer's workload follows a different schedule. Evaluating the arrangement requires following those actual uses and alternatives.

The LUMI example therefore makes location more than a choice about cheap land. There must be a practical way to move useful heat to people who need it. We should ask what heating it replaces and what equipment and operating costs the transfer requires. The operator's description establishes a particular use; the wider economic judgment needs a wider account.

That is relevant to a comparison of AI factories even when they offer similar services. Their private accounts can omit different benefits and costs falling outside their walls. A meter reading alone does not identify those effects. Neither does a claim that all computation is virtual. The heat has to go somewhere, and in Kajaani some of it goes into homes.

Counting processors can miss those differences. Counting electricity can miss what the equipment accomplishes. Revenue records what a company receives. Each measure answers a question; none alone explains the whole productive system.

The Oak Ridge construction record makes the missing work visible. Someone had to discover the defect during testing and correct it. That is a contribution we can describe concretely. We do not need an exact division of the supercomputer's lifetime scientific value to acknowledge that preventing a failed pipe from interrupting operation is useful.

The same lesson applies to AI facilities. Engineering competence can be real and consequential even when we cannot infer its entire economic contribution from the winning company's market position. Respect for physical accomplishment and caution about retrospective attribution belong together.

## Keeping the chips working

Meta's technical report on the Llama 3 family provides a view inside large-scale training. For its largest language model, the researchers describe a fifty-four-day period containing 419 unexpected interruptions, in addition to planned ones. Their account also reports more than 90 percent effective training time. A long list of things went wrong, yet the operation continued doing useful work for most of the elapsed time.[^c7-4]

The two facts belong together. Reporting the interruptions alone would make the exercise sound like a collapse. Reporting the useful training time alone would hide the organizational and technical work needed to achieve it. Automation, diagnosis, and recovery turned a collection of fallible components into a functioning undertaking.

A single failed GPU could require the entire training job to restart. A processor that still worked but ran slowly could hold back thousands of others. Meta's report describes tools for finding those stragglers. The coordination problem was physical: many expensive processors depended on the pace and reliability of the components around them. Their number alone did not establish the useful work they could perform.[^c7-4]

The Llama team reported significant manual intervention was needed only three times during that period; automation handled the other issues. That is an operational claim with a specified boundary. An entrant renting well-run computing might purchase the benefit of such procedures without developing them afresh. The institutional question is whether the service lets the entrant investigate a different product, rather than whether every entrant owns its own maintenance operation.[^c7-4]

Scale can help in several different ways. The work of developing a useful system can be spread across many uses, provided the resulting service remains useful to those users. An operating team can support more than one project. A costly improvement in diagnosis may pay off repeatedly, rather than needing to be rediscovered for every customer.

A second possibility concerns idle capacity. Imagine a shared facility serving customers whose compatible workloads arrive at different times. Combining their demand could keep equipment occupied more consistently than separate facilities would. This is a conditional illustration of an advantage from pooling, not evidence that every large operator uses its machines efficiently or that every workload can be scheduled interchangeably.

Centralizing production need not make downstream services identical. Shared infrastructure can support different research methods and applications. The comparison question follows what can vary above that common layer.

Renting access to well-run computing can reduce the need to duplicate physical infrastructure before attempting a product. The provider's expertise can make more projects feasible. Access terms determine which of those projects can actually proceed.

Now imagine requiring each supplier to reproduce the entire cooling plant, network, maintenance system, and staff before it may offer a service. That would create organizational independence at a considerable resource cost. We should not assume the resulting duplication produces enough useful information to justify itself. A desire for comparison is not a reason to ignore engineering economics.

An exclusive capacity arrangement can restrict downstream experiments while preserving useful operating expertise. The productive value of a facility and the informational consequences of its allocation rules therefore require separate assessment.

We would want to examine the reason for the reservation. A long commitment might finance construction that would otherwise be impossible. A priority rule might protect work that cannot easily tolerate interruption. Or a restriction might exclude an alternative despite feasible capacity being available. These possibilities require evidence about the particular arrangement; the existence of a large cluster does not choose among them.

Large shared computing is more than a hypothetical possibility. For 2026, the Department of Energy's INCITE program allocated access to seventy-five computational science projects at Argonne and Oak Ridge. Its published account describes scientific peer review alongside assessments of whether proposed work is technically ready for the machines. The organization operating the facility and the teams using it need not be identical.[^c7-14]

The listed projects include William Collins's work on large ensembles of weather extremes using machine learning and research on foundation models for materials. These are allocations for proposed work, not evidence that every promised result was achieved. They make the institutional possibility concrete: a costly shared machine can support distinct questions chosen by different teams.

INCITE's policy document distinguishes technical readiness from scientific impact. Readiness alone is insufficient for an award. A proposal that needs preparation can be considered for preliminary access through another program. The distinction recognizes a practical problem: a team may need experience with the resource before it can demonstrate readiness to use a large allocation well.

There are choices to evaluate here, rather than a guarantee supplied by public ownership. Reviewers can make mistakes; preparing applications takes effort; access remains scarce. The program's existence does not establish the best selection rule or a measured comparison budget. It shows how sharing one physical operation and preserving several undertakings can be parts of the same institutional design.

Financing an entire facility and qualifying to use an existing one are different barriers to entry. Each selects projects differently and distributes the cost of infrastructure differently. Their effects must be assessed at the relevant layer.

Llama's interruption record also counters an easy version of the information argument. The company plainly learned about its own operation. A failing component, a useful recovery method, or a bottleneck in communication can generate actionable evidence inside one organization. Calling the organization large cannot make that learning disappear.

The more demanding question is which question the learning answers. An improved recovery procedure can establish how to keep this cluster operating. It does not, without further work, establish what a differently financed rival would have built or which product a buyer would prefer after a fair trial. A factory can become better understood while its place in the wider economy remains harder to explain.

## Three different jobs

The equipment performs different jobs at different stages. In pre-training, a language model's numerical settings are adjusted using large collections of examples. One common task is predicting text from preceding text. The point is to acquire patterns that support later performance, rather than to write a useful answer for a particular customer at that moment.

Post-training then changes the system further. It can use demonstrations and preferences to improve instruction following, usefulness, or particular capabilities. Meta's Llama report distinguishes these stages explicitly. A system that has learned patterns in text is not automatically the assistant its developer wants to offer.[^c7-5]

Inference is the use of a trained model to produce outputs. A person asks a question; computation produces a response. The work may include additional steps, such as retrieving information or using tools. That use has a cost each time it occurs. Training a model does not eliminate the equipment, electricity, and operations required to serve it.

For an ordinary analogy, designing a machine differs from operating it to make the next item. The analogy has limits because a trained model is a collection of numerical settings that can be copied. But it helps prevent a common accounting confusion: the expense of developing a capability and the expense of using it are different parts of the undertaking.

DeepSeek's report on its V3 model separates pre-training, extending the context the model can handle, and post-training in its cost table. That division tells us that even a reported training figure can combine several activities. It does not represent a timeless price for producing intelligence in every setting.[^c7-6]

Entry can mean training a model, adapting one, or building a service using an existing model. These undertakings require different resources and permissions. A claim about the cost of one cannot stand for the cost of all three.

A supplier that cannot finance a large general training run could still develop a valuable application. A supplier with a powerful model could still build a poor application. The difference might involve reliability, the customer's information, the surrounding software, or the way a worker uses the output. Counting large training runs does not count every opportunity for useful innovation.

Access to inference does not grant control of model development. A product requiring changes to the underlying model may need other permissions, technical capabilities, or a different design. The ability to submit a prompt does not supply every input needed to compete.

The word frontier also needs a task attached to it. A system can lead on one evaluation and be unsuitable for another use. Its position can change as methods and competitors improve. Treating frontier as a permanent title bestowed on a few companies obscures the very discovery process we want to understand.

In its 2026 report, Stanford's AI Index attributes more than 90 percent of the notable models in its 2025 dataset to industry. The dataset is curated for notability; it is not a census of every model or application. It gives evidence of industry's prominent role in this part of development, while leaving open the reasons for that pattern and the consequences of particular access arrangements.[^c7-7]

An assessment of entry must identify the layer being entered and the experiments possible there. A training strategy can be blocked while another useful application remains feasible, or application-level access can coexist with dependence on a single development frontier.

## Two budgets

In 2022, Jordan Hoffmann and colleagues at DeepMind investigated how to spend a fixed amount of training computation. Their Chinchilla model used fewer parameters and more training data than the larger Gopher model, with the same total training compute. The researchers reported stronger performance across a range of evaluations. The study challenged the idea that making the model larger was the best use of the available resource.[^c7-8]

The concrete choice makes the word budget useful. A research team has access to computing for a period. It can spend that resource on different combinations of model size, training duration, and data. Experimenting with those choices can reveal a better recipe. The factory supplies a constraint; it does not dictate a single design.

The Chinchilla work also involved multiple smaller training experiments. This is a case of comparison produced inside a large organization, directed at a particular technical problem. It is evidence against an argument that scale must always suppress experimentation. Resources can make demanding comparisons possible.

Now distinguish the computing budget from the comparison budget. The first concerns resources used to perform computation. The second, in my allocation model, adds the chances that successive opportunities go to someone outside the current favorite. They are different objects. A very large computing budget could support useful rival tests, or mainly serve a product already selected for deployment.

Spending on competing training recipes produces different evidence from spending on repeated deployment of a fixed recipe. Both can be productive. Only observations relevant to a defined comparison belong in an account of what the spending establishes about that comparison.

Neither description can be inferred from the electricity meter. The meter tells us about energy use. To understand the evidence, we need the designs, measurements, and decisions attached to it. A low electricity bill can accompany an informative test; an enormous bill can accompany repetition. The economic value of either depends on what is being accomplished.

Chinchilla also changed how its settings were adjusted, how text was divided into units, and the proportions drawn from different parts of the training dataset. The paper records those differences. Its result therefore compares two training packages; it does not isolate model size as the sole cause. The productive lesson is the researchers' ability to investigate a prevailing recipe. Future data, hardware, and uses could make another recipe preferable.[^c7-8]

Measured performance across competing services belongs in the evidence. The allocation-only limit cannot be applied by retaining the award record while discarding informative trials conducted before it.

Where deployment mainly reproduces the record of an established service, the questions concern how alternatives obtain use, which outcomes are checked, and whether tasks are comparable. A market share cannot answer those questions.

A million generated answers can reveal performance on encountered tasks without supplying the missing comparison. Applying Chapter 3's result requires specifying the observation process and counting every additional informative outcome and independent test.

Putting the two budgets beside one another helps us ask a constructive question. Can we use some of the computing resource to produce a more informative comparison at tolerable cost? Chinchilla shows the importance of technical experiments. A buyer's pilot might address a different uncertainty about practical performance. Neither requires treating useful scale as a mistake.

## The five-million-dollar number

DeepSeek's V3 report contains a number likely to attract anyone interested in the economics of the intelligence factory: approximately 5.6 million dollars. It is the report's calculated cost for specified training work, using an assumed rental price of two dollars per H800 GPU hour. The authors explicitly exclude earlier research and experiments on architectures, algorithms, and data from that total.[^c7-6]

The qualification is part of the number's meaning. It does not turn an achievement into a failure. Producing an effective training run with fewer resources can be important. But a reader asking what it costs to create a capable AI organization needs a different account from a reader asking about the computing attributed to one completed training pipeline.

The distinction resembles the difference between paying for a successful reactor run and financing the attempts needed to develop the reactor. The earlier LED chapter showed how repeated modifications could make later production possible. An efficient final operation can rest on costly discovery. Measuring the operation accurately does not make those earlier costs vanish.

Published methods can reduce the cost of reproducing a successful approach. That is one benefit of knowledge transmission, and one reason reproduction cost differs from discovery cost.

Incomplete methods, unavailable data, and different operating conditions can still require fresh experiments. An expenditure figure in a research report does not, by itself, establish the cost at which another organization can reproduce the capability.

The same caution applies in the opposite direction. A company may spend heavily on research that does not improve its final product. We should not assume that every dollar in a large research program was necessary merely because the company later succeeded. To assess necessity, we need comparisons or other evidence about alternatives, rather than either accepting or dismissing the entire bill.

Kerr, Nanda, and Rhodes-Kropf's account of entrepreneurship supplies the established economic starting point. Early tests can inform later commitments, and the option to stop can matter. Bringing that account into AI asks us to examine the sequence of experiments and financing decisions. It does not give the final winner exclusive ownership of the concept of experimentation.[^c7-10]

DeepSeek's table is useful because it defines its boundary. Our job is to preserve it. A quoted training cost should tell us which work is included, how resources were valued, and which comparison the figure permits. Without that discipline, a number intended to explain production becomes a rhetorical device for whatever conclusion we already favored.

## What can be copied

There is an asymmetry inside the factory. A trained model's numerical settings can be distributed without building an additional power station for each copy. Running the copied model still requires suitable equipment. Reproducing all the research that led to it is a further undertaking. Sharing the artifact, sharing access to computing, and sharing the means of discovery are different actions.

Kenneth Arrow's 1962 analysis of invention explains an older version of this difficulty. Knowledge can be costly and uncertain to produce while being relatively cheap to use or transmit once available. The social benefits of using it can exceed the rewards the producer can capture. Arrangements that create incentives to invent can also restrict its subsequent use.[^c7-11]

This is an established economic problem, not a new property discovered when AI appeared. What changes are the particular artifacts, costs, and control points. A model's settings might be available while its training data are not. A technical paper might describe a method while the equipment needed for a competitive implementation remains expensive. Different forms of access answer different constraints.

A common released model can support different services, customer problems, and error-reduction methods where its terms permit. Shared inputs can lower the cost of useful experiments without making their results redundant.

Suppose instead that our question concerns the training process itself: would another dataset or architecture have produced a better model? Two applications using unchanged copies of one model do not perform that comparison merely by having different company names. They may answer important application questions while leaving the training question unresolved.

That is why independence needs an object. Independence of ownership, computing provider, model ancestry, task assignment, or evaluation can each matter differently. Demanding independence in every respect could be prohibitively expensive. Assuming it exists because a storefront has a new name could be equally misleading.

Arrow's appropriation problem also gives an incumbent a serious argument. If immediate unrestricted access prevents it from recovering the costs of valuable research, some research might not occur. We need to examine that possibility. An argument for preserving comparison becomes incomplete when it treats the creation of the thing being compared as somebody else's concern.

The incentive argument still requires a comparison of feasible arrangements. Terms permitting returns and downstream experiments can differ from terms closing a customer route without enabling additional research. Both can be described as protecting investment while having different consequences.

We should compare feasible arrangements rather than compare an actual factory with an imaginary world in which discovery, equipment, and coordination cost nothing. Public support, contractual access, released artifacts, and competing private facilities can have different strengths and costs. Which is useful depends on the problem we want to solve.

The point is neither to award all knowledge to its first producer nor to require every producer to give everything away. It is to investigate how the terms of producing and using knowledge affect later useful attempts. The factory creates possibilities. Its institutional arrangements determine who can act on them.

## From computing access to a useful comparison

Computing access is one stage in producing evidence. A system must also reach relevant tasks, return interpretable outcomes, and permit those outcomes to affect a decision. These requirements connect infrastructure to discovery.

Task selection, comparable difficulty, valid outcome measures, privacy, and operational risk determine whether a service comparison is useful. The evaluation design has to justify its cost.

Distribution can close the route even when computing is available. A default that receives nearly every request can accumulate a record while alternatives obtain little ordinary-use evidence. Availability of an input does not guarantee access to an informative use.

This resembles the distinction between Bell's patent access and customer access, and between search technology and the distribution arrangements examined in Chapter 5. Possessing a useful input is a stage in the process. A practical route to being tried is another. We should not announce that the second problem has been solved merely because the first input is for sale.

Performance can nevertheless justify a strong default. An investigation must include measured quality, actual switching, available comparisons, and the costs of alternatives. Access-dependent evidence is something to interpret carefully, rather than to dismiss.

Limited, well-measured comparisons can sometimes retain the convenience of a default while supplying useful evidence. Their value depends on the uncertainty resolved, implementation costs, and the decisions they can change.

The comparison budget prescribes no universal spending ratio. Small persistent chances can accumulate indefinitely, and large nominal opportunities can fail to identify a relevant effect. Evaluation concerns the process and its information.

Discovery also needs a route to action. A demonstration has less practical value if its result cannot influence subsequent adoption. Switching terms and interoperability help determine whether learning can change an economic arrangement.

At the factory level, this inquiry asks about available capacity and access terms. At the service level, it asks about routing and usable performance evidence. At the level of research, it asks whether another training strategy can be attempted. A single statement that AI is concentrated cannot answer all three.

Nor does having many firms guarantee that all three are open. Several services could depend on the same model and the same route to users. They could still innovate meaningfully in their own domains. The useful question is what uncertainty their differences can resolve, and what important uncertainty remains outside their experiments.

Taking counterfactual economies seriously requires specifying feasible alternatives, what their tests could establish, and the full cost of producing that evidence. The resulting comparison can justify preserving a route without treating every proposed competitor as entitled to infrastructure.

## Who pays for the connection

The grid now returns to the story. On October 14, 2026, the IEA published another report on integrating data centers into electricity systems. It described connection queues containing duplicate or speculative proposals. A large queue can therefore present planners with both an urgent problem and an uncertain measure of how much demand will materialize.[^c7-12]

Imagine, solely as an illustration, one developer exploring sites on several grids while intending to build only one facility. Adding every request as if it represented a separate, certain future load would exaggerate the developer's eventual use. Ignoring the requests entirely could leave the winning location unprepared. The planner needs evidence about commitments and timing, rather than faith in a single headline total.

Here is another economic record that does not explain itself. A proposed project, financed construction, installed computing equipment, and sustained customer demand are different stages. Moving from one to the next can change the risks facing everyone involved. Treating a proposal as completed capacity confuses a possible future with an observed one.

The IEA's October report also describes the importance of allocating connection costs and risks. If new infrastructure is built for a large customer that consumes less than expected or leaves early, somebody must absorb the cost. Contractual commitments and tariffs can determine how much falls on other users of the electricity system. These are institutional choices alongside the engineering.

The IEA's comparison gives AEP Ohio's large-data-center arrangements a concrete form: a ramp-up period, minimum capacity payments, a long commitment, and exit fees. The customer reserving infrastructure cannot assume its obligation ends whenever its appetite for computing falls. That shifts some risk of unused capacity toward the customer requesting it. The terms are a way to allocate risk, not proof that the resulting investment is optimal or that households bear no remaining costs.[^c7-12]

Operating commitments matter as well as construction commitments. In another illustrative arrangement, a customer could accept occasional interruptions in exchange for earlier or cheaper access. A computing job able to pause safely might fit that bargain; a service promising immediate responses might find it costly. The relevant unit is the particular workload and obligation, rather than an undifferentiated category called AI demand.

That choice connects the contract back to the engineers. Recovery procedures, stored progress, and scheduling can change which interruptions are tolerable. A physical improvement might make a more flexible arrangement feasible; an inflexible promise to customers might rule it out. The institution and the machine have to be understood together.

We need evidence to distinguish those cases. A large electricity demand is not inherently a social loss. Nor does calling the demand innovative establish that every associated cost ought to be paid by others. The public question concerns useful output, displaced uses, environmental effects, and the allocation of risk as well as the private business plan.

The information question can also enter, but it must be specified separately. If a valuable pilot uses scarce capacity, it competes with other uses. The value of learning is part of the comparison, alongside electricity, equipment, labor, and the service forgone. Preserving evidence should not become a phrase that makes physical resources free.

Long commitments can support investment while restricting access to scarce capacity. The institutional question is whether a feasible arrangement retains the financing benefit and preserves useful independent experiments. Their costs can exceed their benefits; that possibility belongs in the appraisal.

This is the tension that a price or concentration statistic alone cannot resolve. A low price today can coexist with a poor route to testing tomorrow's alternative. A large producer can supply the very capacity that makes new experiments affordable. The economic task is to trace the mechanisms rather than assign one moral meaning to size.

## Leaving the factory

The physical story has carried us from a grid connection to a model ready for use. Along the way, there were pipes to test, interruptions to overcome, training choices to investigate, and costs to define. These are productive activities. The argument about missing comparisons gains nothing by pretending otherwise.

But the completed training run is a beginning for the economic story. A model can be served repeatedly, incorporated into different products, evaluated on changing tasks, and routed toward some users rather than others. Those arrangements affect which observations accumulate and which alternatives develop a usable record.

Follow the next request. Someone chooses a system, the machines produce an answer, and a person or another program does something with it. Does that sequence teach us what we need to know? Does it improve the next decision? Can it reveal when a different system would work better?

The factory has made an answer possible. The next chapter asks what we learn when there are a billion of them.

## Notes

[^c7-1]: IEA, *Key Questions on Energy and AI* (2026), [full report](https://iea.blob.core.windows.net/assets/3179f7f8-01f6-4dd6-bffa-c9f7b73f1dc9/KeyQuestionsonEnergyandAI.pdf), published April 16, 2026; executive summary, pp. 9–11; physical bottlenecks, p. 21; demand outlook and accelerator definitions, pp. 24–26. The summary rounds 2030 consumption to 950 TWh. These are total data-center estimates/projections, not AI-only consumption or observed future demand. Wait times vary across jurisdictions.

[^c7-2]: Matt Lakin, ORNL, [“Super Construction at Super Scale: How ORNL Built a New Home for Frontier”](https://www.ornl.gov/news/super-construction-super-scale-how-ornl-built-new-home-frontier/), November 14, 2023, construction account from cooling plant and piping through foundations and failed pressure test. The laboratory reports the event and identifies its staff. No visit, conversation, or private reaction is invented here. Frontier is a DOE scientific supercomputer, not the name of a language model.

[^c7-3]: ORNL, [“Computer Engineers at ORNL Pioneer Approaches to Energy Efficient Supercomputing”](https://www.ornl.gov/news/computer-engineers-ornl-pioneer-approaches-energy-efficient-supercomputing/), section describing David Grant's cooling-system work and the transition from Titan's cooler supply to Frontier's warm-water design. The chapter uses the qualitative engineering account, not a claimed causal estimate for AI companies or an assertion that every data center uses this design.

[^c7-4]: Llama Team, AI at Meta, *The Llama 3 Herd of Models*, [arXiv version 3](https://arxiv.org/pdf/2407.21783v3), November 23, 2024, Section 3.3.4, pp. 13–14 and Table 5. The reported fifty-four-day snapshot contains 466 interruptions: 47 planned and 419 unexpected. More than 90 percent effective training time concerns the reported operation; it is not a measurement of economic efficiency or a claim that the whole training run lasted fifty-four days. The same section reports single-GPU restart risk, slow stragglers affecting thousands of GPUs, and significant manual intervention only three times in the snapshot. Developer self-report, not an independent operational audit.

[^c7-5]: Llama Team, same report, introduction p. 1 and Section 4, p. 15, for pre-training and post-training; Section 3 for training infrastructure. The paper says its language-model results concern Llama 3.1, shortened to Llama 3 in its text. This chapter's general explanation does not imply that serving each request updates model weights, or that use requires all stages to be repeated.

[^c7-6]: DeepSeek-AI, *DeepSeek-V3 Technical Report*, [arXiv version 2](https://arxiv.org/pdf/2412.19437v2), February 18, 2025, p. 5, Table 1 and accompanying paragraphs. The calculation values 2.788 million H800 GPU hours at an assumed USD2 per hour, yielding USD5.576 million across the specified official training stages. The report explicitly excludes prior research and ablations. It is an attributed developer calculation, not a total company spending figure, audited invoice, or cost-equivalent comparison with another model or firm. The original report appeared in December 2024.

[^c7-7]: Stanford HAI, *Artificial Intelligence Index Report 2026*, [full report](https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf), Section 1.1, printed pp. 16–18, particularly p. 18 and Figure 1.1.4. Industry's reported share is 91.2 percent. The Epoch AI dataset is manually curated for notability; the report records an April 22, 2026 snapshot. This is neither all-model market share nor a causal estimate of exclusion, and a model's sector differs from an application's ownership or its users' options.

[^c7-8]: Jordan Hoffmann and colleagues, “Training Compute-Optimal Large Language Models,” *NeurIPS* 2022, [published paper](https://papers.nips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf), abstract, pp. 1–2, Section 4 and Section 4.1, p. 7. The same-compute comparison is Chinchilla versus Gopher, not a universal constant across architectures, dates, or objectives. Section 4.1 records additional optimizer, tokenizer and data-mix changes; no isolated randomized model-size effect is claimed. The [authors' April 12, 2022 account](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/) corroborates the qualitative result.

[^c7-10]: William R. Kerr, Ramana Nanda, and Matthew Rhodes-Kropf, “Entrepreneurship as Experimentation,” *JEP* 28(3) (2014), 25–48, [published paper](https://www.hbs.edu/faculty/Shared%20Documents/conferences/2014-strategy-research/Entrepreneruship%20as%20Experimentation.pdf), pp. 27–30 and 43–44. Chapter 6 develops its role and boundaries. A staged experiment's value and stopping option are prior ideas; the comparison-budget bound is a different proposed contribution.

[^c7-11]: Kenneth J. Arrow, “Economic Welfare and the Allocation of Resources for Invention,” in *The Rate and Direction of Inventive Activity* (1962), 609–626, [NBER chapter](https://www.nber.org/system/files/chapters/c2144/c2144.pdf), pp. 609 and 614–617, especially pp. 615 and 617. Arrow supplies the established uncertainty, use and appropriation problem. No particular AI access or intellectual-property rule follows automatically from it.

[^c7-12]: IEA, *Powering Artificial Intelligence: Strategies for Integrating Data Centres into Electricity Systems*, October 14, 2026, [report](https://www.iea.org/reports/powering-artificial-intelligence) and [full PDF](https://iea.blob.core.windows.net/assets/cd360a30-aaef-43cd-84e6-a125c3a0bc67/PoweringArtificialIntelligence.pdf), pp. 7–10 for queue visibility and cost allocation, and pp. 52–57, especially the AEP Ohio row on p. 57. The tariff example is the IEA's dated survey description, not a fresh inspection of every operative contract or a legal/welfare recommendation. The multi-grid developer is a labeled illustration, not an identified company. Queue entries, completed capacity, tariffs and realized welfare remain distinct. The chapter uses the report available by its October 16, 2026 source-check date.

[^c7-13]: LUMI consortium, [“Sustainable Future”](https://lumi-supercomputer.eu/sustainable-future/), opening paragraphs on waste-heat use, accessed October 16, 2026. This is an operator account of heat supplied to Kajaani households. The chapter does not adopt the site's broader claims of optimality or its strongest environmental descriptions, or substitute older projected heating shares for measured current ones.

[^c7-14]: Argonne Leadership Computing Facility, [“INCITE Program Awards Supercomputing Time to 75 High-Impact Projects”](https://www.alcf.anl.gov/news/incite-program-awards-supercomputing-time-75-high-impact-projects-0), May 5, 2026, opening, selection paragraphs and listed awards; *INCITE Overview and Policies*, [version 13, December 2024, distributed for the 2026 cycle](https://doeleadershipcomputing.org/wp-content/uploads/sites/123/2025/04/INCITE-Overview-and-Policies_2026.pdf), Section 1.2.1, printed pp. 1–2. Awards describe allocated opportunity, not completed research or a causal evaluation of public computing. The source's dated policy version is preserved.
