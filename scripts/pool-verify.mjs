import { chromium } from "playwright";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createStaticServer } from "./serve.mjs";
import path from "node:path";
const server = process.argv.includes("--dist")
  ? createStaticServer(path.resolve("dist"))
  : null;
if (server)
  await new Promise((resolve) => server.listen(5187, "127.0.0.1", resolve));
await mkdir("previews", { recursive: true });
const browser = await chromium.launch({
  channel: "msedge",
  headless: false,
  args: ["--enable-unsafe-webgpu"],
});
try {
  const page = await browser.newPage({
      viewport: { width: 1440, height: 900 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("response", (r) => {
    if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`);
  });
  await page.goto(
    process.env.POOL_URL ||
      (server ? "http://127.0.0.1:5187/pool/" : "http://127.0.0.1:5186/pool/"),
  );
  await page.waitForFunction(
    () =>
      window.poolDiagnostics?.ready || window.poolDiagnostics?.errors.length,
    {},
    { timeout: 120000 },
  );
  assert.deepEqual(
    await page.evaluate(() => window.poolDiagnostics.errors),
    [],
  );
  async function frames(n) {
    const start = await page.evaluate(
      () => window.poolDiagnostics.state.frames,
    );
    await page.waitForFunction(
      (s) => window.poolDiagnostics.state.frames >= s,
      start + n,
      { timeout: 120000 },
    );
  }
  await frames(25);
  assert.equal(
    await page.evaluate(() => window.poolDiagnostics.readbackBytes),
    0,
    "normal rendering must stay on GPU",
  );
  await page.screenshot({ path: "previews/pool-ui.png" });
  const calm = await page.evaluate(() => window.poolLab.inspect());
  assert(calm.finite);
  assert.equal(calm.peak, 0);
  await page.mouse.click(900, 540);
  await frames(5);
  const clicked = await page.evaluate(() => window.poolLab.inspect());
  assert(clicked.peak > 0.001, "click must disturb the water");
  await page.click("#hide");
  await frames(2);
  await page.screenshot({ path: "previews/pool.png" });
  await page.click("#restore");
  await page.click("#storm");
  await frames(180);
  const storm = await page.evaluate(() => window.poolLab.inspect());
  assert(storm.finite);
  assert(storm.peak > 0.0001);
  assert.equal(storm.borderPeak, 0);
  assert(storm.weather[0] > calm.weather[0]);
  assert(storm.weather[8] > 0);
  await page.screenshot({ path: "previews/pool-storm.png" });
  await page.click("#pause");
  const time = await page.evaluate(() => window.poolLab.state.time);
  await frames(5);
  assert.equal(await page.evaluate(() => window.poolLab.state.time), time);
  await page.click("#waterline");
  await frames(3);
  await page.screenshot({ path: "previews/pool-waterline.png" });
  await page.click("#tiles");
  await frames(3);
  await page.screenshot({ path: "previews/pool-tiles.png" });
  const distance = await page.evaluate(() => window.poolLab.state.distance);
  await page.mouse.move(1000, 600);
  await page.mouse.wheel(0, -200);
  await frames(2);
  assert((await page.evaluate(() => window.poolLab.state.distance)) < distance);
  await page.click("#hero");
  await page.click("#sunny");
  await page.click("#pause");
  await frames(80);
  const drying = await page.evaluate(() => window.poolLab.inspect());
  assert(drying.weather[8] > 0, "wet deck persists after rain stops");
  assert(drying.finite);
  await page.click("#auto");
  await page.evaluate(() => (window.poolLab.state.age = 890));
  await frames(20);
  const front = await page.evaluate(() => window.poolLab.inspect());
  assert(front.weather[6] >= 890);
  assert.deepEqual(errors, []);
  assert.deepEqual(
    await page.evaluate(() => window.poolDiagnostics.errors),
    [],
  );
  const report = { calm, clicked, storm, drying, front, errors };
  await writeFile(
    "previews/pool-verification.json",
    JSON.stringify(report, null, 2),
  );
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
  if (server) await new Promise((resolve) => server.close(resolve));
}
