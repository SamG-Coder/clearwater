import { chromium } from "playwright";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
const b = await chromium.launch({
  channel: "msedge",
  headless: false,
  args: ["--enable-unsafe-webgpu"],
});
try {
  const p = await b.newPage({ viewport: { width: 1440, height: 900 } }),
    errors = [];
  p.on("pageerror", (e) => errors.push(String(e)));
  p.on("response", (r) => {
    if (r.status() >= 400) errors.push(r.url());
  });
  await p.goto("http://127.0.0.1:5186/cube/");
  await p.waitForFunction(
    () =>
      window.poolDiagnostics?.ready || window.poolDiagnostics?.errors.length,
    {},
    { timeout: 120000 },
  );
  assert.deepEqual(await p.evaluate(() => poolDiagnostics.errors), []);
  async function frames(n) {
    const f = await p.evaluate(() => poolLab.state.frames);
    await p.waitForFunction((t) => poolLab.state.frames >= t, f + n);
  }
  await frames(20);
  assert.equal(await p.evaluate(() => poolDiagnostics.readbackBytes), 0);
  await p.screenshot({ path: "previews/cube-ui.png" });
  await p.mouse.click(800, 400);
  await frames(8);
  const clicked = await p.evaluate(() => poolLab.inspect());
  assert(clicked.finite);
  assert(clicked.peak > 0.0001);
  await p.click("#storm");
  await frames(100);
  const storm = await p.evaluate(() => poolLab.inspect());
  assert(storm.finite);
  assert(storm.peak > 0.0001);
  await p.click("#pause");
  const t = await p.evaluate(() => poolLab.state.time);
  await frames(5);
  assert.equal(await p.evaluate(() => poolLab.state.time), t);
  await p.click("#tiles");
  await frames(5);
  await p.screenshot({ path: "previews/cube-close.png" });
  await p.click("#hero");
  await p.click("#sunny");
  await p.click("#pause");
  await frames(50);
  assert.deepEqual(errors, []);
  const report = { clicked, storm, errors };
  await writeFile(
    "previews/cube-verification.json",
    JSON.stringify(report, null, 2),
  );
  console.log(report);
} finally {
  await b.close();
}
