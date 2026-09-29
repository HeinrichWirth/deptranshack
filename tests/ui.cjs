// Optional browser QA. PLAYWRIGHT_MODULE can point at an existing installation.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const assert = require("node:assert");
const fs = require("node:fs");
(async () => {
  const base = process.env.RAIL_URL || "http://127.0.0.1:8870";
  const jobs = await (await fetch(base + "/api/jobs")).json();
  const job = jobs.find((x) => x.status === "COMPLETE");
  assert(job, "Need one completed run");
  const browser = await chromium.launch({
    channel: process.env.BROWSER_CHANNEL || "msedge",
    headless: true,
  });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1100 },
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(base + "/?job=" + job.id);
  await page.waitForFunction(
    () => document.querySelectorAll("#objects tr").length > 0,
  );
  await page.screenshot({
    path: "local-results/dashboard.png",
    fullPage: true,
  });
  const state = await (await fetch(base + "/api/status?job=" + job.id)).json();
  const source = state.frames.find((s) => s > 300) || state.frames[0];
  await page.goto(
    base + `/predicted_train.html?job=${job.id}&source=${source}`,
  );
  await page.waitForFunction(
    () => document.querySelector("#sliceInfo").textContent.includes("Срез на"),
    { timeout: 60000 },
  );
  const before = await page.locator("#position").textContent();
  await page.locator("#forward").click();
  await page.waitForTimeout(200);
  assert.notEqual(await page.locator("#position").textContent(), before);
  await page.screenshot({ path: "local-results/viewer.png", fullPage: true });
  assert.deepEqual(errors, []);
  await browser.close();
  fs.writeFileSync(
    "local-results/ui-test.json",
    JSON.stringify({ pass: true, job: job.id, errors }),
  );
  console.log("PASS dashboard, result selection, frame viewer, virtual drive");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
