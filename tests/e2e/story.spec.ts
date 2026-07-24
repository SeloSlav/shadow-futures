import { expect, test } from "@playwright/test";

test.describe("interactive essay", () => {
  test("plays the creator graphs and reaches every chapter", async ({ page, isMobile }) => {
    test.skip(isMobile, "covered by the desktop narrative journey");
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Shadow Futures", exact: true }),
    ).toBeVisible();
    await expect(page.locator(".site-header .header-nav__paper")).toHaveAttribute(
      "href",
      "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6003994",
    );
    const firstSectionAfterHero = page.locator("#main-content > section").first();
    await expect(firstSectionAfterHero).toHaveAttribute("id", "novelty");
    await expect(
      firstSectionAfterHero.getByRole("heading", {
        name: "A million transactions can still be only one experiment.",
      }),
    ).toBeVisible();
    await expect(firstSectionAfterHero.getByText("Le Goff and Soulier's")).toBeVisible();
    await page.getByRole("link", { name: "See how the evidence disappears" }).click();
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
    await expect(page.getByText(/3\.0 times as much comparison/)).toBeVisible();

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
      page.getByRole("heading", { name: "One equation carries the whole argument." }),
    ).toBeVisible();
    await expect(page.getByText("Choose who receives the next opportunity")).toBeVisible();
    await expect(page.getByText("What one history cannot tell us")).toBeVisible();
  });

  test("runs the central equation in the comparison playground", async ({ page, isMobile }) => {
    test.skip(isMobile, "covered by the desktop playground journey");
    await page.goto("/playground");

    await expect(
      page.getByRole("heading", { name: "The Comparison Playground" }),
    ).toBeVisible();
    const surface = page.getByTestId("playground-surface");
    await expect(surface).toBeVisible();
    await expect(page.getByRole("link", { name: "Playground" })).toHaveAttribute(
      "href",
      "/playground",
    );
    await expect(page.getByLabel("Feedback strength ρ")).toHaveValue("1.65");

    const viewport = page.viewportSize();
    const labBounds = await page.locator(".playground-lab").boundingBox();
    expect(viewport).not.toBeNull();
    expect(labBounds).not.toBeNull();
    expect((labBounds?.y ?? 0) + (labBounds?.height ?? 0)).toBeLessThanOrEqual(
      (viewport?.height ?? 0) + 1,
    );

    await surface.hover();
    const scrollBeforeZoom = await page.evaluate(() => window.scrollY);
    const viewBeforeZoom = await surface.getAttribute("data-view");
    await page.mouse.wheel(0, 240);
    await expect(surface).not.toHaveAttribute("data-view", viewBeforeZoom ?? "");
    expect(await page.evaluate(() => window.scrollY)).toBe(scrollBeforeZoom);

    const canvasWidth = await surface.getAttribute("width");
    const canvasHeight = await surface.getAttribute("height");
    await page
      .getByRole("button", {
        name: "Protected discovery Reserve some exposure for alternatives.",
      })
      .click();
    await expect(page.getByLabel("Reserved discovery η")).toHaveValue("0.16");
    await expect(surface).toHaveAttribute("width", canvasWidth ?? "");
    await expect(surface).toHaveAttribute("height", canvasHeight ?? "");

    await page.locator("#playground-round").press("End");
    await expect(page.getByText("comparison units")).toBeVisible();
    await expect(page.getByText(/shadow routes have closed at this round/)).toBeVisible();
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
    await page.getByLabel("Shadow Futures home").click();
    await page.goto("/methodology");
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
        name: "How is Shadow Futures different from preferential attachment, increasing returns, or network effects?",
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
