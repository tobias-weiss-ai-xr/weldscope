const { test, expect } = require('@playwright/test');

// In-page canvas pixel summary: hash over all pixels + count of non-blank
// pixels (clearRect leaves all-zero RGBA, so nonzero > 0 proves drawing).
const pixels = (page) => (sel) =>
  page.evaluate((s) => {
    const c = document.querySelector(s);
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let hash = 0, nonzero = 0;
    for (let i = 0; i < d.length; i += 4) {
      const p = d[i] | d[i + 1] | d[i + 2] | d[i + 3];
      if (p) nonzero++;
      hash = (Math.imul(hash, 31) + p) | 0;
    }
    return { hash: hash >>> 0, nonzero };
  }, sel);

test('demo: wasm boot, canvases, controls, clean console', async ({ page }) => {
  const pageErrors = [];
  const consoleErrors = [];
  page.on('pageerror', (err) => pageErrors.push(String(err)));
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });

  await page.goto('./');
  await expect(page).toHaveTitle(/WeldScope/);

  // nav links to both docs pages
  await expect(page.locator('nav a[href="docs-fft.html"]')).toHaveCount(1);
  await expect(page.locator('nav a[href="docs-pipeline.html"]')).toHaveCount(1);

  // 2D canvases + 3D container exist
  await expect(page.locator('#ascan')).toHaveCount(1);
  await expect(page.locator('#trace')).toHaveCount(1);
  await expect(page.locator('#three')).toHaveCount(1);

  // WASM module loaded: spectra generated + 256 FFTs computed in-browser.
  await expect(page.locator('#status'))
    .toContainText('WASM fft computed 256 ascans', { timeout: 15000 });

  // #three receives the THREE.js renderer canvas
  await expect(page.locator('#three canvas')).toHaveCount(1, { timeout: 15000 });

  const px = pixels(page);

  // both 2D canvases actually drew something (non-blank pixels)
  expect((await px('#ascan')).nonzero, 'ascan non-blank').toBeGreaterThan(0);
  expect((await px('#trace')).nonzero, 'trace non-blank').toBeGreaterThan(0);

  // PLAY: recomputes a NEW weld run in-browser (fresh seed per click).
  // Regression coverage: the previous demo reused seed 7 on every click, so
  // Play showed zero visual change (identical pixels) — fixed by reseeding.
  const traceBefore = await px('#trace');
  const ascanBefore = await px('#ascan');
  await page.click('#play');
  await expect(page.locator('#status'))
    .toContainText(/recomputed 256 FFTs \(run 1\) in [\d.]+ ms/);
  const traceAfter = await px('#trace');
  const ascanAfter = await px('#ascan');
  expect(traceAfter.nonzero, 'trace non-blank after play').toBeGreaterThan(0);
  expect(ascanAfter.nonzero, 'ascan non-blank after play').toBeGreaterThan(0);
  expect(traceAfter.hash, 'play reseeded -> trace changed').not.toBe(traceBefore.hash);
  expect(ascanAfter.hash, 'play reseeded -> ascan changed').not.toBe(ascanBefore.hash);

  // frame slider within run 1: frame 200 redraws a different A-scan
  const ascanF0 = await px('#ascan');
  await page.evaluate(() => {
    const el = document.querySelector('#frame');
    el.value = '200';
    el.dispatchEvent(new Event('input', { bubbles: true }));
  });
  const ascanF200 = await px('#ascan');
  expect(ascanF200.nonzero, 'ascan non-blank on frame 200').toBeGreaterThan(0);
  expect(ascanF200.hash, 'frame 200 differs from frame 0 of same run')
    .not.toBe(ascanF0.hash);

  // second play -> run 2 with yet another seed: trace differs from run 1
  await page.click('#play');
  await expect(page.locator('#status')).toContainText(/\(run 2\)/);
  const traceRun2 = await px('#trace');
  expect(traceRun2.hash, 'run 2 differs from run 1').not.toBe(traceAfter.hash);
  // THREE canvas survives recomputes (cloud swapped, not page reload)
  await expect(page.locator('#three canvas')).toHaveCount(1);

  // no page errors; console errors allowed only for the known benign
  // THREE r160 computeBoundingSphere warning (headless-chromium quirk).
  // WASM panics surface as pageerror/console error — the filter must not
  // hide them.
  expect(pageErrors, 'uncaught page errors').toEqual([]);
  const realErrors = consoleErrors.filter((m) => !m.includes('computeBoundingSphere'));
  expect(realErrors, 'unexpected console errors').toEqual([]);
});
