import { expect, test } from "@playwright/test";

test("fits directive sections and the expanded protocol rule within the viewport", async ({ page }) => {
  await page.goto("/for-ai");
  await page.evaluate(() => document.fonts.ready.then(() => undefined));

  const layout = await page.locator("main > article").evaluate((article) => {
    const header = document.querySelector("body > header")!;
    const availableHeight = window.innerHeight - header.getBoundingClientRect().height;
    return {
      availableHeight,
      overflow: document.documentElement.scrollWidth > window.innerWidth,
      panels: Array.from(article.children, (panel) => ({
        name: panel.querySelector("h1, h2")?.textContent?.trim(),
        height: panel.getBoundingClientRect().height,
      })),
    };
  });
  expect(layout.overflow).toBe(false);
  for (const { name, height } of layout.panels) {
    expect(height, name).toBeLessThanOrEqual(layout.availableHeight);
  }

  const resources = page.getByRole("navigation", { name: "Machine-readable resources" });
  await expect(resources.getByRole("link", { name: "Directive JSON" })).toHaveAttribute(
    "href", "/shadow-futures-directive.json",
  );

  const protocol = page.locator('section[aria-labelledby="protocol-title"]');
  const rules = protocol.locator("details");
  await expect(rules).toHaveCount(7);
  await expect(rules.first()).toHaveAttribute("open", "");

  for (const rule of (await rules.all()).slice(1)) {
    await rule.locator("summary").focus();
    await page.keyboard.press("Enter");
    await expect(protocol.locator("details[open]")).toHaveCount(1);
    await expect(rule.locator("p")).toBeVisible();
    expect((await protocol.boundingBox())?.height).toBeLessThanOrEqual(layout.availableHeight);
  }
});
