import edition from "@/public/book/edition.json";

export const BOOK = {
  ...edition,
  amazon: "https://www.amazon.com/dp/B0HN91X19C",
};
export const BOOK_PDF_SIZE = `${(BOOK.pdf_bytes / 1024 / 1024).toFixed(1)} MB`;

export const BOOK_FULL_TITLE = `${BOOK.title}: ${BOOK.subtitle}`;
export const BOOK_DESCRIPTION =
  "Read Martin Erlic’s Shadow Futures free: how markets choose winners, lose sight of alternatives, and reshape AI, work, competition, and economic power.";

export const BOOK_SYNOPSIS = [
  "Shadow Futures is a book about what success can tell us—and what it leaves unseen. It begins with an online music experiment in which the same song finished first in one parallel market and fortieth in another. The recording stayed the same. The history of who listened, and what they saw other people choosing, changed.",
  "Martin Erlic follows that problem from music markets to telephone networks, search platforms, and artificial intelligence. An early contract can bring references; a popular platform can attract more users; a leading firm can gain resources for its next investment. Each success can change who receives a meaningful chance next. The alternative market histories that could have unfolded under the same rules and starting conditions are shadow futures.",
  "At the center of the argument is the comparison budget: the accumulated probability that opportunities go to someone other than the current favorite. A growing record of transactions need not settle how much work, quality, or investment contributed to success. Productive inputs can matter even when one market history cannot reliably reveal their effects.",
  "Written without equations, the book connects this information problem to AI infrastructure, agent workforces, the future of work, ownership, competition policy, and taxation. It asks how institutions can preserve opportunities to test alternatives, while taking the benefits of scale, the costs of experimentation, and the evidence needed for policy seriously.",
];

const chapterDescriptions: Record<string, string> = {
  Prologue:
    "The different fortunes of Lockdown in parallel music markets introduce the gap between the winner we observe and the other histories that were possible.",
  Introduction:
    "Sets out the book’s central question: how much can a record of success establish about productive contribution when success itself reshapes opportunity?",
  "Chapter 1":
    "Separates achievement from its explanation and follows the feedback between early rewards, later access, and the development of productive ability.",
  "Chapter 2":
    "Introduces the comparison budget in ordinary language and explains why transaction counts, rival wins, and informative comparisons measure different things.",
  "Chapter 3":
    "Explains alternative market histories and the conditions under which forecasts can converge while the effects of productive inputs remain unresolved.",
  "Chapter 4":
    "Follows the history of telephone networks, licensing, and access to show how earlier choices can become infrastructure governing later opportunities.",
  "Chapter 5":
    "Examines who controls the routes through which challengers develop and receive a meaningful test, including alternatives that cannot yet reach viable scale.",
  "Chapter 6":
    "Asks what market rewards establish about effort, quality, luck, and position, and how judgments about deserved rewards depend on additional values.",
  "Chapter 7":
    "Examines the computing, energy, capital, and coordination behind AI, keeping the productive benefits of scale alongside the barriers facing independent entrants.",
  "Chapter 8":
    "Follows AI deployment, evaluation, and agent workforces to distinguish repeated performance from evidence about alternative development paths and institutions.",
  "Chapter 9":
    "Explores how AI changes work, learning, and access to a first assignment, and why better assisted output need not establish independently acquired skill.",
  "Chapter 10":
    "Considers competition policy and institutional choices in Europe, the United States, and China through the opportunities they preserve for future discovery.",
  "Chapter 11":
    "Separates distributional reasons for taxation from the additional evidence needed to connect a loss of comparison to a cost that corrective policy could address.",
  "Chapter 12":
    "Develops an economics of possible worlds through experiments, simulation, and institutions that keep independent discovery feasible and open to evaluation.",
  Epilogue:
    "Returns to the practical work of keeping a comparison alive and to the next opportunity a worker, researcher, or rival system needs to be tested.",
};

export const BOOK_CHAPTERS = BOOK.contents.map((chapter) => ({
  ...chapter,
  description: chapterDescriptions[chapter.section],
}));
