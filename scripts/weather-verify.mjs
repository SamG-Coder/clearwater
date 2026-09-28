import { chromium } from "playwright";
import { writeFile } from "node:fs/promises";
import assert from "node:assert/strict";
const browser = await chromium.launch({
  channel: "msedge",
  headless: false,
  args: ["--enable-unsafe-webgpu"],
});
const errors = [];
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("response", (r) => {
    if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`);
  });
  await page.goto(process.env.CLEARWATER_URL || "http://127.0.0.1:5186/");
  await page.waitForFunction(
    () =>
      window.clearwaterDiagnostics?.ready ||
      window.clearwaterDiagnostics?.errors.length,
    null,
    { timeout: 120000 },
  );
  assert.deepEqual(await page.evaluate(() => clearwaterDiagnostics.errors), []);
  assert.equal(
    await page.evaluate(() => clearwaterDiagnostics.readbackBytes),
    0,
  );
  await page.evaluate(() => {
    clearwaterLab.pause();
    clearwaterLab.state.pitch = -0.17;
  });
  await page.evaluate(() => clearwaterLab.seek(5));
  const calm = await page.evaluate(() => clearwaterLab.inspect());
  await page.screenshot({ path: "previews/weather-calm.png" });
  // Rain in otherwise calm conditions must produce real ripple heights.
  await page.evaluate(() => {
    clearwaterLab.state.rain = 1;
  });
  await page.evaluate(() => clearwaterLab.weatherAdvance(5));
  const rain = await page.evaluate(() => clearwaterLab.inspect());
  assert.ok(rain.ripplePeak > 0.00001, "rain must excite the wave solver");
  assert.equal(await page.evaluate(() => clearwaterLab.state.weatherAge), -1);
  await page.evaluate(() => {
    clearwaterLab.state.rain = 0;
  });
  await page.locator("#storm").click();
  await page.evaluate(() => {
    clearwaterLab.pause();
    clearwaterLab.state.weatherAge = 0;
  });
  await page.evaluate(() => clearwaterLab.weatherAdvance(450));
  await page.screenshot({ path: "previews/weather-front.png" });
  await page.evaluate(() => clearwaterLab.weatherAdvance(450));
  const peak = await page.evaluate(() => clearwaterLab.weatherInspect());
  const peakWaves = await page.evaluate(() => clearwaterLab.inspect());
  const peakOptics = await page.evaluate(() => clearwaterLab.inspectOptics());
  assert.ok(
    peak.spectrum.finite &&
      peakWaves.finite &&
      peakOptics.hdrFinite &&
      peakOptics.glareFinite,
  );
  assert.ok(
    peak.weather[0] > 18 && peakWaves.bandRms[1] > calm.bandRms[1] * 1.5,
    "wind-wave band must grow independently of background swell",
  );
  assert.ok(peak.foamPeak > 0.001 && peak.foamPeak <= 1);
  await page.screenshot({ path: "previews/weather-storm.png" });
  const paused = await page.evaluate(() => ({
    time: clearwaterLab.state.time,
    age: clearwaterLab.state.weatherAge,
  }));
  await page.waitForTimeout(250);
  assert.deepEqual(
    await page.evaluate(() => ({
      time: clearwaterLab.state.time,
      age: clearwaterLab.state.weatherAge,
    })),
    paused,
  );
  await page.evaluate(() => clearwaterLab.seek(32.93713468687923)); // Seeded irregular lightning pulse.
  await page.screenshot({ path: "previews/weather-lightning.png" });
  await page.evaluate(() => clearwaterLab.weatherAdvance(1200));
  const aftermath = await page.evaluate(() => clearwaterLab.weatherInspect());
  await page.screenshot({ path: "previews/weather-aftermath.png" });
  assert.ok(
    aftermath.weather[0] < 7 &&
      aftermath.spectrum.max > 1.05 &&
      aftermath.spectrum.max < peak.spectrum.max,
    "wind must ease before stored wave energy",
  );
  // Weather controls are live inputs, and the buoy remains an optional scale marker.
  await page.locator("summary").click();
  await page.locator("#wind").fill("12");
  await page.locator("#wind").dispatchEvent("input");
  await page.locator("#direction").fill("-1.2");
  await page.locator("#direction").dispatchEvent("input");
  await page.locator("#rain").fill("0.3");
  await page.locator("#rain").dispatchEvent("input");
  await page.locator("#clouds").fill("0.7");
  await page.locator("#clouds").dispatchEvent("input");
  await page.locator("#buoy").check();
  await page.locator("#weatherRate").selectOption("1");
  const controls = await page.evaluate(() => ({ ...clearwaterLab.state }));
  assert.equal(controls.wind, 12);
  assert.equal(controls.direction, -1.2);
  assert.equal(controls.rain, 0.3);
  assert.equal(controls.clouds, 0.7);
  assert.equal(controls.weatherRate, 1);
  assert.equal(controls.buoy, true);
  await page.locator("#clearWeather").click();
  assert.equal(await page.evaluate(() => clearwaterLab.state.weatherAge), -1);
  assert.deepEqual(errors, []);
  assert.deepEqual(await page.evaluate(() => clearwaterDiagnostics.errors), []);
  const report = {
    passed: true,
    calm: { rms: calm.rms, bandRms: calm.bandRms },
    rain: { ripplePeak: rain.ripplePeak },
    peak,
    peakWaves: {
      rms: peakWaves.rms,
      bandRms: peakWaves.bandRms,
      min: peakWaves.min,
      max: peakWaves.max,
    },
    aftermath,
    paused: true,
    controls: true,
    finiteOptics: peakOptics.hdrFinite && peakOptics.glareFinite,
    errors,
  };
  await writeFile(
    "previews/weather-verification.json",
    JSON.stringify(report, null, 2),
  );
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
}
