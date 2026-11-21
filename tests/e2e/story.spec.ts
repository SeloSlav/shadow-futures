import { expect, test } from "@playwright/test";

test.describe("interactive essay", () => {
  test("plays the creator graphs and reaches every chapter", async ({ page, isMobile }) => {
    test.skip(isMobile, "covered by the desktop narrative journey");
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Shadow Futures", exact: true }),
    ).toBeVisible();
    await expect(page.getByRole("navigation", { name: "Primary navigation" }).getByRole("link", { name: /Read the paper/ })).toHaveAttribute(
      "href",
      "/paper",
    );
    const firstSectionAfterHero = page.locator("#main-content > section").first();
    await expect(firstSectionAfterHero).toHaveAttribute("id", "novelty");
    await expect(
      firstSectionAfterHero.getByRole("heading", {
        name: "A million transactions can still add up to only one experiment.",
      }),
    ).toBeVisible();
    await expect(firstSectionAfterHero.getByText(/The closest statistical precedent/)).toBeVisible();
    await page.locator("#book").getByRole("link", { name: /Explore the idea/ }).click();
    await expect(page).toHaveURL(/#novelty$/);
    await page.getByTestId("breakout-graph").scrollIntoViewIfNeeded();
    await expect(page.getByTestId("breakout-graph")).toBeVisible();

    await page.getByRole("button", { name: "Run the recommendations" }).click();
    await expect(page.getByTestId("breakout-graph")).toHaveAttribute(
      "data-animation-state",
      "complete",
    );

    await expect(
      page.getByTestId("breakout-graph").locator(".creator-chart-key__item"),
    ).toHaveCount(10);
    await expect(
      page.getByTestId("breakout-graph").locator(".creator-shadow-card"),
    ).toHaveCount(2);
    await expect(
      page.getByTestId("breakout-graph").locator(".creator-shadow-card svg"),
    ).toHaveCount(4);
    await expect(page.getByRole("link", { name: /Experimental Study of Inequality/ })).toHaveAttribute(
      "href",
      "https://doi.org/10.1126/science.1121066",
    );

    await page.getByTestId("experiment-monopoly-graph").scrollIntoViewIfNeeded();
    await page.getByRole("button", { name: "Compare both rules" }).click();
    await expect(page.getByTestId("experiment-monopoly-graph")).toHaveAttribute(
      "data-animation-state",
      "complete",
    );

    await page.getByTestId("lorenz-history-graph").scrollIntoViewIfNeeded();
    await expect(
      page.getByRole("heading", {
        name: "The curve can’t tell us how much of either story is true",
      }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Protect the comparison budget" }).click();
    await expect(page.getByTestId("lorenz-history-graph")).toHaveAttribute(
      "data-animation-state",
      "complete",
    );
    await expect(
      page.getByRole("heading", { name: "Reserve discovery for alternatives" }),
    ).toBeVisible();
    await expect(
      page.getByTestId("lorenz-history-graph").getByText(/3\.0 times as much comparison/),
    ).toBeVisible();

    const chapterIds = [
      "novelty",
      "breakout",
      "experiment-monopoly",
      "firm-markets",
      "lorenz-curve",
      "tax-and-ubi",
      "conclusion",
    ];
    for (const id of chapterIds) {
      await page.locator(`#${id}`).scrollIntoViewIfNeeded();
      await expect(page.locator(`#${id}`)).toBeVisible();
    }
  });

  test("reruns the same creator feed with a different early history", async ({ page, isMobile }) => {
    test.skip(isMobile, "covered by the desktop interaction journey");
    await page.goto("/#breakout");
    const graph = page.getByTestId("breakout-graph");
    const result = graph.locator(".creator-graph__result");
    await page.getByRole("button", { name: "Run the recommendations" }).click();
    await expect(graph).toHaveAttribute("data-animation-state", "complete");
    const firstWorld = await result.textContent();
    await page.getByRole("button", { name: "Run new recommendations" }).click();
    await expect(graph).toHaveAttribute("data-animation-state", "complete");
    await expect(result).not.toHaveText(firstWorld ?? "");
  });

  test("explains the central equation", async ({ page, isMobile }) => {
    test.skip(isMobile, "covered by the desktop mathematics journey");
    await page.goto("/math");
    await expect(
      page.getByRole("heading", { name: "From winning odds to the learning limit." }),
    ).toBeVisible();
    await expect(page.getByText("Choose who receives the next opportunity")).toBeVisible();
    await expect(page.getByText("Apply the finite-budget condition")).toBeVisible();
  });

  test("explores the comparison budget and rotates the shelf", async ({ page, isMobile }) => {
    const hydrationErrors: string[] = [];
    page.on("console", message => {
      if (/hydrat/i.test(message.text()) && message.type() === "error") hydrationErrors.push(message.text());
    });
    await page.goto("/");
    if (isMobile) await page.getByRole("button", { name: "Explore", exact: true }).click();
    await page.getByRole("link", { name: /^Playground/ }).click();
    await expect(page).toHaveURL(/\/playground$/);
    await page.reload();
    await expect(page.getByRole("heading", { name: "Comparison budget playground", exact: true })).toBeVisible();
    const chart = page.getByRole("img", { name: /^After .* sales?,/ });
    const slider = page.getByRole("slider", { name: "Sales completed" });
    await expect(chart).toBeVisible();
    await expect(slider).toHaveValue("4000");
    const baseline = "After 4,000 sales, Big Tech has 423.8 points and Commodities has 1,892.8 points.";
    const first = page.getByRole("combobox", { name: "First market" });
    const second = page.getByRole("combobox", { name: "Compare with" });
    await expect(chart).toHaveAttribute("aria-label", baseline);
    await expect(first).toHaveValue("Big Tech");
    await expect(second).toHaveValue("Commodities");
    const techCurve = await chart.locator('path[data-series="Big Tech"]').getAttribute("d");
    const wheatCurve = await chart.locator('path[data-series="Commodities"]').getAttribute("d");
    expect(techCurve).not.toBe(wheatCurve);
    await second.selectOption("AI");
    await expect(chart).toHaveAttribute("aria-label", /Big Tech has 423.8 points and AI has 752.0 points/);
    await expect(chart.locator('path[data-series="Big Tech"]')).toHaveAttribute("d", techCurve!);
    await first.selectOption("AI");
    await expect(second).toHaveValue("Big Tech");
    await expect(chart).toHaveAttribute("aria-label", /AI has 752.0 points and Big Tech has 423.8 points/);
    await first.selectOption("Big Tech");
    await second.selectOption("");
    await expect(chart).toHaveAttribute("aria-label", /Equal shelf space has 1,702.2 points/);
    await second.selectOption("Commodities");
    await expect(chart).toHaveAttribute("aria-label", baseline);
    await page.getByRole("button", { name: "Rotate the shelf" }).click();
    await expect(chart).toHaveAttribute("aria-label", /Big Tech has 553.3 points/);
    await expect(chart.locator('path[data-series="Commodities"]')).not.toHaveAttribute("d", wheatCurve!);
    await expect(page.getByText(/In the shaded interval/)).toBeVisible();
    await page.getByRole("button", { name: "Rank by sales" }).click();
    await expect(chart).toHaveAttribute("aria-label", baseline);
    await slider.press("Home");
    await expect(chart).toHaveAttribute("aria-label", "After 0 sales, Big Tech has 0.0 points and Commodities has 0.0 points.");
    await slider.press("ArrowRight");
    await expect(chart).toHaveAttribute("aria-label", "After 1 sale, Big Tech has 0.4 points and Commodities has 0.5 points.");
    await second.selectOption("Restaurants");
    await expect(chart).toHaveAttribute("aria-label", "After 1 sale, Big Tech has 0.4 points and Restaurants has 0.4 points.");
    await second.selectOption("Commodities");
    await page.getByRole("button", { name: "Play sales" }).click();
    await expect.poll(async () => Number(await slider.inputValue())).toBeGreaterThan(1);
    await page.getByRole("button", { name: "Pause" }).click();
    await slider.press("End");
    await expect(chart).toHaveAttribute("aria-label", baseline);
    const markets = page.getByRole("img", { name: /^Twenty-four simulated outcomes/ });
    const firstBatch = await markets.innerHTML();
    await page.getByRole("button", { name: /Rerun the 24 markets/ }).click();
    await expect.poll(() => markets.innerHTML()).not.toBe(firstBatch);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    expect(hydrationErrors).toEqual([]);
    await page.goto("/comparison-budget");
    await expect(page).toHaveURL(/\/playground$/);
  });

  test("the operating system reduced-motion setting is respected", async ({ page, isMobile }) => {
    test.skip(isMobile, "covered by the desktop accessibility journey");
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");
    await page.getByRole("button", { name: "Run the recommendations" }).click();
    await expect(page.getByTestId("breakout-graph")).toHaveAttribute(
      "data-animation-state",
      "complete",
    );
    await expect(page.getByRole("button", { name: /reduced motion/i })).toHaveCount(0);
    await expect(page.getByRole("button", { name: /mode/i })).toHaveCount(0);
  });

  test("mobile navigation reaches the methodology", async ({ page, isMobile }) => {
    test.skip(!isMobile, "mobile-only coverage");
    await page.goto("/");
    await expect(page.locator("body")).not.toHaveCSS("overflow-x", "scroll");
    await page.getByRole("button", { name: "Explore", exact: true }).click();
    const navigation = page.getByRole("navigation", { name: "Explore Shadow Futures" });
    await expect(navigation.getByRole("link")).toHaveCount(10);
    await navigation.getByRole("link", { name: /Methodology/ }).click();
    await expect(navigation).toHaveCount(0);
    await expect(page.getByRole("heading", { name: "How the argument works" })).toBeVisible();
  });

  test("FAQ exposes visible and machine-readable answers", async ({ page, isMobile }) => {
    test.skip(isMobile, "covered by desktop semantic rendering");
    await page.goto("/faq");
    await expect(
      page.getByRole("heading", { name: "Shadow Futures, explained" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "What are shadow futures?" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", {
        name: "How’s Shadow Futures different from preferential attachment, increasing returns, or network effects?",
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", {
        name: "Should highly successful platform creators be taxed more?",
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", {
        name: "How do AI agents, x402, and the agentic economy relate to Shadow Futures and UBI?",
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("link", {
        name: "x402 is an open internet payment standard",
      }),
    ).toHaveAttribute("href", "https://docs.x402.org/introduction");
    await expect(
      page.getByRole("link", {
        name: "complete paper and technical appendix",
      }),
    ).toHaveAttribute(
      "href",
      "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994",
    );

    const faqSchema = page.locator('script[type="application/ld+json"]');
    await expect(faqSchema).toHaveCount(1);
    const schemaText = await faqSchema.textContent();
    expect(schemaText).toContain('"@type":"FAQPage"');
    expect(schemaText).toContain('"@type":"Question"');
  });
});
