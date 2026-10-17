# Chapter 2

## The comparison budget

### A movie behind several doors

The film is Good Will Hunting. One image emphasizes Matt Damon and Minnie Driver; another emphasizes Robin Williams. In a 2017 account of artwork personalization, Netflix engineers used these possibilities to illustrate a choice their system had to make. A member drawn to romance might respond to one invitation, a member drawn to comedy to another. The same film could be introduced through different parts of its appeal.[^c2-1]

There is a small economic experiment hidden in that image. Showing an invitation supplies a chance for somebody to respond. Showing a different invitation supplies a chance to learn something else. An artwork that remains in a designer's folder can be beautiful, but the folder does not contain the audience's response to seeing it. An additional play following the established image does not automatically answer the question about the image still in the folder.

Netflix's engineers described a practical response: deliberately vary some selections, record how they were selected, and use responses to improve later choices. They explained that learning could carry an immediate cost when a member received an image other than the one currently predicted to work best. Their account concerned how to combine that experimentation with personalization, rather than simply displaying the same apparent winner indefinitely.[^c2-1]

A company with a large audience can create comparisons inside its own service. The same scale that helps an established choice dominate can spread the cost of testing another. What matters is how the service assigns those opportunities.

The image also keeps us honest about the question. Comparing two invitations to watch Good Will Hunting would not measure the contribution of every actor, the director, the screenwriter, or the company carrying the film. A useful experiment can answer a narrow question very well. Its strength depends partly on resisting the urge to attach every neighboring question to the same result.

The engineers also asked what happens when an image changes between visits. A member might remember a film by its picture and then fail to recognize the new invitation. Or the replacement might finally persuade the member to try it. If several pictures preceded a play, deciding which deserved credit became harder. The team treated the frequency of image changes as part of the learning problem.[^c2-1]

Getting somebody to press play was not enough, either. A striking picture could attract an audience that soon abandoned the film. The engineers described using the quality of engagement when labeling their observations, to avoid rewarding that kind of clickbait. Now the record needed to connect the member, title, image, and subsequent response. A million presentations would be less revealing if the file lost those connections.[^c2-1]

The comparison budget names one feature of the process producing such records: accumulated allocation chance outside the current favorite. Brief probability examples will make that accounting concrete before the documented systems show what an application additionally requires.

### What we add up

Start with two alternatives that each have a one-half chance of receiving an allocation. Either can be designated the current favorite when they tie. The chance outside it is one half, which becomes the first addition to the comparison budget. This is a probability before the result is known, rather than a prediction of the realized winner.

At the next allocation, suppose the leading alternative has a three-in-four chance. The chance outside it contributes one quarter. If the other alternative becomes the favorite at a third allocation with the same margin, the addition is again one quarter. Across these three decisions, the additions total one.

That total does not promise exactly one outsider victory. The awards could repeatedly go to the favorite or sometimes go elsewhere. A quarter added before an allocation stays a quarter in the accounting, whichever alternative wins.

The favorite is redefined at every step. The quantity concerns the concentration of the next opportunity, conditional on the history, rather than loyalty to a permanent incumbent.

The favorite is defined by the chance of receiving the allocation, not by an independent verdict that it is the best product. A buyer can favor a familiar supplier because switching is costly. An experimenter can deliberately give a plausible challenger substantial attention. The reason for the choice and the quality of the option both matter economically, but neither should be smuggled into the word "favorite." The budget first describes how opportunities are distributed.

Calling this a budget does not mean that somebody deposited a fixed fund of experimental tokens at the beginning. The quantity accumulates as opportunities arrive. We can ask how much has accumulated so far, or what the remaining sequence could add under a specified process. The economic issue is whether continuing the market continues to supply meaningful possibilities outside its current favorite.

With three alternatives, an eight-in-ten chance for the favorite leaves a combined two-in-ten chance outside it. Adding another name changes nothing if it receives no chance and the existing probabilities remain the same. A longer directory is not itself a larger comparison budget.

A company can preserve a variety of informative trials while remaining a single company, as Netflix's artwork experiments show. For image selection, the relevant alternatives are the pictures users could see. Counting corporate registrations would count something else.

The comparison budget describes opportunities through which evidence can emerge. It leaves service quality, price, workers' incomes, and consumers' enjoyment to be assessed separately. Its economic value depends on what those opportunities reveal and which decisions the evidence improves.

The allocation needs a consistent unit. For a study of image selection, candidate images could be the alternatives and a presentation the allocated opportunity. Whether somebody subsequently plays the film is an additional outcome. For a procurement study, suppliers and contracts could take those roles instead. We cannot count presentations for one question, switch to completed plays for another, and call the resulting total the same budget. The definition becomes useful by fixing the economic question and the process before counting.

When verified productive inputs enter the allocation rule in my model, and their differences remain within the stated bounds, the comparison budget places an upper limit on the information the allocation history provides about their effects. Scarce outside chances can therefore limit how precisely contribution can be recovered from the record. A large budget can still leave attribution unresolved if the trials never distinguish the relevant inputs.

### The news story that was not shown

Yahoo's front page gives us a documented example of an organization designing a record for later comparison. In research presented in 2010, Lihong Li, Wei Chu, John Langford, and Robert Schapire studied personalized news recommendation. Part of the underlying traffic came from a randomized bucket: candidate articles were selected at random for the highlighted position, with user and article information and the resulting click or nonclick recorded.[^c2-3]

The unshown article creates the difficulty. A person encounters one highlighted story, and the system sees whether that person clicks it. The same encounter does not reveal whether the person would have clicked every other candidate. The researchers' procedure started from that limited observation, rather than treating a visitor's arrival as a completed test of every story in the pool.

Random selection made a different use of the record possible. A proposed recommendation rule could be evaluated against logged encounters in which its choice matched the article actually selected. Under the paper's assumptions, this replay procedure supplied a reliable way to evaluate the rule using existing experimental traffic. The record was useful because of how the articles had been chosen and which responses had been retained.[^c2-3]

It is a beautiful answer to one version of the data problem. The observer does not invent a visitor's reaction to an unseen story. The evaluation uses reactions to stories that were genuinely shown. Its credibility comes from the logging design and the assumptions connecting those encounters to the proposed rule. The large database helps, but the relevant form of variation was built into the process before the later analyst opened it.

Now imagine a different file in which every visitor had been shown the same predicted favorite. The analyst might still learn much about that story and those visitors. What would be absent is the same coverage of choices under the randomized design. Replaying another rule would require further assumptions or further data. A more powerful computer could search the file faster without putting the missing article in front of a visitor who never saw it.

Yahoo recorded what visitors did after an article appeared, including nonclicks, as well as information about each visitor and article. A study of contribution has to use that evidence alongside the selection record. Counting only which story won the position would discard observations the experiment was designed to collect.

A replay based on yesterday's visits cannot, by itself, tell us how a new recommendation rule would change tomorrow's audience. Readers might return more often; publishers might offer different stories. Studying those changes requires a design or model that accounts for those responses, rather than assuming that the same encounters would keep arriving.

Imagine four groups of viewers, each always assigned its own particular image. Across an equally sized population, the images would have equal shares of presentations. Yet within a group, the rule leaves no chance for another image. Aggregate diversity can hide a complete lack of alternatives within each group.

The converse is possible too. One image could dominate overall presentations while a carefully selected trial supplies useful evidence about a question involving another audience. The minority share alone would not tell us whether that evidence is weak or strong. We would need the selection rule, the groups involved, and the responses observed. Equal totals can conceal different processes; unequal totals can coexist with purposeful learning. Neither pattern in a chart substitutes for inspecting how each opportunity arose.

An allocation record can contain more than the winner. Task characteristics, eligibility terms, selection chances, and measured outcomes may supply additional information. Those observations should be included explicitly, even when they answer questions outside the simple allocation model.

### A connection someone gets to make

A recommendation can also change a person's economic network. Karthik Rajkumar and colleagues studied randomized variations in LinkedIn's People You May Know recommendations. Different variants changed the kinds of connections suggested. The research, published in 2022, linked these experiments to subsequent job mobility. The effects of weak connections depended on how tie strength was measured; simply seeking ever weaker relationships was not a universal prescription.[^c2-8]

The connection makes the sequence particularly tangible. A recommended person can become a contact. That contact can help information about an opportunity reach a job seeker. A subsequent job change is then part of a history influenced by an earlier selection. The study's design allowed the researchers to investigate aspects of this process that an ordinary list of members and their later jobs would have left entangled.

Imagine studying a member's career after the connection has been made. The profile could show qualifications, previous employment, and a real new role. The question about the recommendation is whether changing who was suggested affected the path into that role. The person's abilities need not be fictitious for the route to matter. A useful match can involve both a qualified worker and information that reached that worker through a particular contact.

By varying connection recommendations and following later employment outcomes, LinkedIn's researchers could investigate how an introduction changed someone's opportunities. The useful evidence came from deliberately changing which connections people encountered.

What made a tie weak? The researchers used both the number of mutual connections and the intensity of bilateral messaging. These measures describe different relationships: knowing few of the same people and rarely exchanging messages need not coincide. Moderately weak ties did best by the mutual-connections measure; the weakest ties did best by interaction intensity. Even the practical meaning of trying a different connection depended on the measurement.[^c2-8]

The authors also examined the risk that treating one member could affect another member's outcomes. Connected people can affect one another even when their accounts appear separately in the data. Suppliers sharing a distribution channel can likewise share influences that a comparison has to account for.[^c2-8]

A network can improve access to opportunities while complicating attribution of their causes. References, connections, and observed performance need to be distinguished in an account of how the next allocation occurs.

### Two long futures

Suppose the chance outside the favorite stays at one in a thousand at every allocation. Each opportunity adds a thousandth to the budget. A thousand opportunities add one; another thousand add another one. The process is heavily concentrated, but its accumulated outside chance keeps growing.

Concentration and a finite lifetime comparison budget are therefore different properties. A tiny chance that persists differs from a chance that vanishes quickly enough for its total to remain bounded. Nor does either calculation promise that awards arrive on schedule or reveal an informative input contrast.

Now consider a different, explicitly hypothetical allocation schedule. Start with a one-in-four chance outside the favorite. At every subsequent opportunity, halve that chance. The first few opportunities contribute most of what the entire future can add. Continuing forever adds successively smaller pieces, and all those pieces together amount to one half. There can be indefinitely many decisions while the accumulated outside chance has a finite ceiling.

The contrast concerns the whole sequence. A single photograph of either process could show a dominant favorite and a neglected rival. The later history depends on whether meaningful chances continue arriving or become scarce fast enough. To distinguish those possibilities empirically, we need evidence about the rules and their evolution. A concentration chart is a record of realized shares, not a measurement of that future sequence.

A third schedule makes the need for care clearer still. Let outside chances decline, but so slowly that the additions continue accumulating without limit. We do not need the arithmetic of that schedule to see the conceptual point. "Shrinking" is not a synonym for "finite in total." Before a claim about disappearing comparison does serious work, the rate of disappearance must be established in the setting being discussed.

The halving schedule is an accounting illustration, not a fitted model of a named company. An actual supplier's prospects could rise after a useful demonstration or fall after an incompatible product change. Its future chance could depend on who won earlier contracts. The budget follows the relevant conditional chances through those changes. Replacing that evolving process with a single fixed market-share number would remove precisely the history the object is meant to describe.

The schedules expose a distinction ordinary language can blur. A very small chance may feel equivalent to none when deciding whether to make one costly attempt. For what an indefinitely long record can reveal, persistence matters: a long sequence can accumulate what each opportunity adds only in a tiny amount.

A business, worker, or policymaker may need an answer long before an indefinitely accumulating process becomes helpful. An infinite lifetime budget could coexist with painfully little relevant opportunity over five years. A trial available after the purchasing decision cannot inform that purchase.

Nor can the opposite inference be made automatically. A finite amount of opportunity need not leave every practical question unresolved. A buyer might already have enough evidence to choose a service sensibly. The stronger issue is a promise of reliable, increasingly precise attribution throughout the admissible explanations of the model. That is a more demanding promise than choosing an option that works well with the information currently available.

Netflix might need to choose an image for a member; a public buyer might need to decide whether another supplier merits a pilot. Each can act with some uncertainty remaining. The useful comparison is one that could improve the decision enough to justify its cost.

### How a lead can close a contest

There is a long mathematical history behind the difference between feedback that preserves comparison and feedback that eventually overwhelms it. One classic model begins with an urn holding colored balls. Draw a ball, return it, and add another of the same color. A color selected early becomes more likely to be selected again. Robin Pemantle's survey describes this family of reinforced processes and the many different behaviors produced by their rules.[^c2-4]

The urn isolates a feedback rule: an earlier draw changes the chances of later draws. Changing that rule can change how the sequence develops, even when every version can be described casually as success breeding success.

In the familiar urn, adding a ball of the drawn color can change long-run proportions while other colors continue to appear. Unequal histories can still keep producing comparisons. The rule governing each addition determines whether the outside chances have a finite lifetime total.

A stronger version gives accumulated success a much more powerful effect on the next selection. For a simple example, imagine weights that grow with the square of the accumulated score: doubling a score quadruples its weight. Research on appropriate strong-feedback models establishes a striking outcome. After some finite, random point, a single alternative receives every subsequent allocation. The result depends on the specified reinforcement conditions, not merely on a large lead.[^c2-5]

Roberto Oliveira studied the onset of such dominance, building on earlier monopoly results. Thomas Gottfried and Stefan Grosskinsky later distinguished several feedback regimes, including strong and weaker forms of monopoly. Their results explain how different reinforcement rules can produce different forms of dominance.[^c2-5]

What the comparison budget adds is an information question about the resulting record. In the strong-feedback model developed here, the remaining chances outside the eventual winner are summable. The budget stays finite. Under the other conditions of the attribution result, indefinitely extending that single allocation record cannot supply a universally reliable recovery of the contribution effect. The prior dominance result and the information interpretation do different jobs.

There is a feature here worth holding onto. An alternative can retain a positive chance at every finite stage, yet receive no further awards along the realized history. A positive chance is not a promise of an eventual event. When those chances get small quickly enough, their continuing existence need not create an endless supply of informative deviations. Nominal eligibility and sustained practical contestability can therefore diverge within a precisely specified model.

Entry, new products, new tasks, or independent tests can reopen comparisons. Whether positive feedback closes them permanently depends on the allocation rule and which changes remain possible. Gottfried and Grosskinsky distinguish reinforcement processes with different long-run behavior; the rule determines which result applies.

The urn and Netflix's artwork system both use selection rules that change with earlier results. The urn makes the feedback easy to inspect. Netflix shows how a business can design those rules to keep learning.

### The calls made to choose a policy

An experiment with rice farmers in India gives the selection question a different purpose. Maximilian Kasy and Anja Sautmann worked with Precision Agriculture for Development on ways to enroll farmers in a phone-based extension service. Six strategies combined morning or evening automated calls with different advance text-message arrangements. The recorded success was completion of five questions during the call. It was not a measure of a farmer's eventual harvest.[^c2-6]

The experiment proceeded in waves. Results from completed calls informed assignment to later strategies. The aim was to help select an approach for use after the experiment. That purpose matters: maximizing the number of completed calls during the trial is a different objective from collecting the evidence needed to make a better choice for the much larger implementation that may follow it.[^c2-6]

The distinction is familiar once we put a calendar beside the experiment. This week's calls can deliver useful service. They can also teach the organization which method to use next month. Giving every remaining call to the current leader may improve the immediate recorded success rate while leaving a close challenger insufficiently tested. A reasonable allocation depends partly on whose outcomes, at which dates, the organization is trying to improve.

Kasy and Sautmann's exploration-sampling method retained substantial testing of plausible alternatives instead of simply concentrating on the strategy most likely to be best. Results from earlier calls helped determine which strategies received later ones.[^c2-6]

The case supplies an important counterweight to the idea that experimentation must mean equal shares forever. Once an approach looks clearly unpromising, continuing to assign it the same share may be a poor use of opportunities. A close challenger may deserve more investigation. Relevant comparison depends on which uncertainty remains, how much resolving it would matter, and the consequences of sending another call along each route.

Adaptive design ties opportunities to the uncertainty being investigated. A limited experiment on a relevant task can distinguish plausible explanations; a nominally balanced allocation on incomparable tasks may add little. The assignment process must be related to what the observer wants to learn.

The measure of success should remain visible as well. Completing five questions is something an organization can record promptly. It is useful for choosing among recruitment methods on that outcome. If the later decision concerns agricultural income or the quality of advice, additional evidence is needed. The outcome recorded for fast adaptation should not acquire authority over every consequence simply because it arrives in time to guide the next wave.

This is especially important when comparisons have different costs. An ineffective image may waste a small amount of a viewer's attention. An unsuitable software deployment can demand weeks of staff time. A policy pilot may affect people whose welfare matters during the experiment as well as afterward. The agricultural study's stated objective helps identify the decision being optimized; other settings may require a different balance.

An institution can therefore preserve comparison while making increasingly selective choices. It need not suspend judgment until every candidate has been exhausted. The discipline is to state why the next allocation is useful, which possible responses would change the decision, and what costs are justified by that learning. Those questions turn a general preference for experimentation into something a buyer or program manager can actually implement.

### What a large budget cannot buy

Equal chances at every allocation would make the budget grow, but still need not identify an input effect. If the measured input is identical across the alternatives, reallocating awards changes the recipient without creating the contrast needed to learn that input's effect.

For a less abstract picture, imagine testing two artwork files that display exactly the same image. More exposures can help estimate how that image performs with particular audiences. They cannot reveal the difference between the two visual treatments because the treatments do not differ. A system can create a great deal of nominal variation while leaving the intended comparison untouched. Counting chances is necessary for one bound; it is insufficient for a guarantee.

A change of name need not create an independent trial either. Products may share an underlying service, a bottleneck, or a faulty outcome measure. The budget does not certify independence or meaningful variation merely because a different alternative receives an award.

Yaneer Bar-Yam's distinction between large quantities of detailed data and information relevant to an intervention is useful here. His framework concerns how to identify what matters for a system-level question, rather than treating detail as automatically sufficient. The comparison budget addresses a narrower allocation setting, but the shared discipline is to ask what the record can distinguish for the question under study.[^c2-7]

Yahoo's article evaluation depended on randomized logging and observed responses. The farmer calls depended on the measured completion outcome and adaptive assignment. In each case, the record's usefulness followed from how it was generated, rather than its size alone.

A buyer might obtain an independent performance test without awarding another full contract. If that test distinguishes effects the award history leaves uncertain, it adds information to the record. The allocation-only limit then describes only part of what the buyer can observe.

Likewise, known selection probabilities can be useful information, not merely quantities an analyst may assume into existence. A designed experiment may log them. An ordinary market may require estimates based on observed rules and behavior. If obtaining those estimates supplies additional information about contribution, that information has to be counted. An empirically defensible budget needs a defensible observation model alongside it.

Defining the object and measuring it are therefore separate achievements. The rice-farmer experiment could implement assignment probabilities deliberately; an analyst of an ordinary market cannot assume comparable access to its mechanism. Sometimes only a range of plausible budgets will be justified. That can still organize an investigation, provided the range and its consequences remain visible. Precision written into a definition should never be mistaken for precision supplied by observations.

The comparison budget cannot be read from a market-share table. A rival's sales tell us what happened, not its chance on the next job. Even a well-measured budget needs evidence about differences in what suppliers do and how well they perform. Frequent trials would tell us little about engineering if they never compared meaningful differences in engineering.

### A large company that keeps testing

Microsoft's Bing supplies a particularly direct challenge to the idea that the number of firms tells us how much learning remains. In a 2013 paper, Ron Kohavi and colleagues described more than two hundred experiments running concurrently inside the service. Some users were held out to assess the experimentation system's overall effect. The organization was deliberately maintaining differences in experience rather than relying solely on the record of the version everyone already used.[^c2-9]

One example involved adding links within advertisements so people could navigate directly to parts of an advertiser's site. The initial version increased revenue but worsened user metrics and page-loading time. Adjusting the presentation, including how many main advertisements appeared, changed that balance. The investigators were tracking more than a single increase in a commercial total. A result that looked attractive on one outcome still needed interpretation alongside others.[^c2-9]

Keeping only the revenue increase would lose part of what Bing's experiment revealed. Keeping only the company's size would lose the comparisons it arranged. The useful evidence combined an intervention, assignment, recorded responses, and a decision about which outcomes mattered.

The extra links offered a plausible convenience. Testing revealed consequences the designers had not settled in advance. Their skill helped create the proposal; comparison helped decide how to improve it.

A business can test many changes to its own product while leaving independent suppliers little opportunity to demonstrate theirs. Bing's internal trials can improve Bing without answering how another provider would perform with a route to users.

Sometimes the missing comparison calls for another supplier. Sometimes a company can learn more cheaply by changing its own trial or using records it already keeps. The question is which route will reveal something useful about the decision at hand, and what it will cost.

### The price of keeping a door open

Keeping a comparison available has a price. Testing can require moving data, training staff, checking outcomes, and accepting disruption before its result is known. The hoped-for information must justify those costs. Some alternatives will sensibly remain unevaluated because testing them would cost more than the decision could gain.

Netflix's engineers described their artwork setting as favorable partly because exploration costs could be spread across a large membership. The cost still depends on what a person experiences during the test. A setting that makes outside scrutiny difficult can sometimes make internal experimentation economical.[^c2-1]

The practical question is which uncertainty could change a consequential decision. A small reliability difference may justify investigation when it affects years of expensive operation. A difference incapable of outweighing known switching costs may not. There is no economic instruction to maximize the comparison budget everywhere.

Existing records, a shared test environment, or a limited task can sometimes answer a question more cheaply than a full deployment. Whether they do so depends on their design and on the inference required. A lost route has no automatically adequate substitute.

The order matters. Start with the question, identify a comparison capable of answering it, and assess the cost of producing that comparison. Only then decide whether an access arrangement or purchasing rule helps. Beginning with a preferred number of suppliers and assuming that the resulting record will be informative skips the hardest part of the task. The directory can be long while the relevant test remains absent.

Learning can strengthen the case for an incumbent. An artwork experiment can confirm that the established image works better for the relevant audience. An informative test must allow a result to change our view, including by increasing justified confidence in an existing choice.

The comparison budget measures the accumulated allocation chance outside the current favorite. What those chances reveal depends on the inputs, observed outcomes, and response rules. Preserving more comparison is worthwhile when the resulting evidence improves decisions enough to justify its cost.

A different early award could produce another history under the same rules. Changing those rules could produce a different economy. Explaining a reward requires knowing which comparison is being made.

## Notes

[^c2-1]: Ashok Chandrashekar, Fernando Amat, Justin Basilico, and Tony Jebara, [“Artwork Personalization at Netflix”](https://netflixtechblog.com/artwork-personalization-c589f074ad76), Netflix TechBlog, December 7, 2017, sections “Challenges,” “Contextual bandits approach,” and “Performance evaluation.”

[^c2-3]: Lihong Li, Wei Chu, John Langford, and Robert E. Schapire, “A Contextual-Bandit Approach to Personalized News Article Recommendation,” *Proceedings of WWW 2010*: 661–670, [doi:10.1145/1772690.1772758](https://doi.org/10.1145/1772690.1772758). See arXiv version 2 (2012), Sections 2.1, 4, and 5.2.1, [full paper](https://arxiv.org/pdf/1003.0146).

[^c2-4]: Robin Pemantle, “A Survey of Random Processes with Reinforcement,” *Probability Surveys* 4 (2007): 1–79, [doi:10.1214/07-PS094](https://doi.org/10.1214/07-PS094). See Section 2.1, especially pp. 3–5, for the classical urn and its variants, [full survey](https://arxiv.org/pdf/math/0610076).

[^c2-5]: Roberto Imbuzeiro Oliveira, “The Onset of Dominance in Balls-in-Bins Processes with Feedback,” *Random Structures & Algorithms* 34 (2009): 454–477, [doi:10.1002/rsa.20261](https://doi.org/10.1002/rsa.20261); [author preprint](https://arxiv.org/pdf/math/0510415), introduction and Theorem 1, pp. 1–2. Thomas Gottfried and Stefan Grosskinsky, “Asymptotics of Generalized Pólya Urns with Non-Linear Feedback,” *Electronic Journal of Probability* 29 (2024): 1–56, [doi:10.1214/24-EJP1157](https://doi.org/10.1214/24-EJP1157); [arXiv version 3](https://arxiv.org/pdf/2303.01210), Sections 2.1–2.3, especially Theorem 2.2.

[^c2-6]: Maximilian Kasy and Anja Sautmann, “Adaptive Treatment Assignment in Experiments for Policy Choice,” *Econometrica* 89 (2021): 113–132, [doi:10.3982/ECTA17527](https://doi.org/10.3982/ECTA17527), pp. 113–114 and Section 6, pp. 122–123, [full published paper](https://maxkasy.github.io/home/files/papers/adaptiveexperimentspolicy.pdf); authors' [November 10, 2021 correction](https://maxkasy.github.io/home/files/papers/correction_adaptiveexperimentspolicy.pdf), especially corrected Theorem 1.

[^c2-7]: Yaneer Bar-Yam, “From Big Data to Important Information,” *Complexity* 21, supplement 2 (2016): 73–98, [doi:10.1002/cplx.21785](https://doi.org/10.1002/cplx.21785), abstract and Sections I–II; [author preprint](https://arxiv.org/pdf/1604.00976).

[^c2-8]: Karthik Rajkumar, Guillaume Saint-Jacques, Iavor Bojinov, Erik Brynjolfsson, and Sinan Aral, “A Causal Test of the Strength of Weak Ties,” *Science* 377 (2022): 1304–1310, [doi:10.1126/science.abl4476](https://doi.org/10.1126/science.abl4476). See design and main results, pp. 1304–1309; tie-strength definitions, pp. 1305–1306; and interference discussion, pp. 1309–1310, [full paper hosted by MIT](https://ide.mit.edu/wp-content/uploads/2022/09/abl4476.pdf).

[^c2-9]: Ron Kohavi, Alex Deng, Brian Frasca, Toby Walker, Ya Xu, and Nils Pohlmann, “Online Controlled Experiments at Large Scale,” *Proceedings of KDD 2013*: 1168–1176, [doi:10.1145/2487575.2488217](https://doi.org/10.1145/2487575.2488217); [author version](https://www.exp-platform.com/Documents/2013%20controlledExperimentsAtScale.pdf), introduction and Section 1.1, pp. 1–2.
