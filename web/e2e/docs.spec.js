const { test, expect } = require('@playwright/test');

test('docs-fft: FFT explainer with live wasm demo', async ({ page }) => {
  const pageErrors = [];
  page.on('pageerror', (err) => pageErrors.push(String(err)));

  await page.goto('docs-fft.html');
  await expect(page.locator('h1')).toHaveText('How the OCT FFT pipeline works');
  await expect(page.locator('#demo')).toHaveCount(1);
  await expect(page.locator('#d_play')).toHaveCount(1);
  await expect(page.locator('#d_window')).toHaveCount(1);
  await expect(page.locator('#d_pad')).toHaveCount(1);

  // wasm pipeline booted (initial text is "load wasm…" / "loading…")
  await expect(page.locator('#d_status'))
    .toContainText(/WASM core runs|ready/, { timeout: 15000 });

  expect(pageErrors, 'uncaught page errors').toEqual([]);
});

test('docs-pipeline: architecture page documents the WS01 wire format', async ({ page }) => {
  await page.goto('docs-pipeline.html');
  await expect(page.locator('h1')).toHaveText(/Pipeline/);
  await expect(page.locator('body')).toContainText('WS01');
});

test('docs pages reachable from the main-page nav', async ({ page }) => {
  await page.goto('./');
  await page.click('nav a[href="docs-fft.html"]');
  await expect(page.locator('h1')).toHaveText('How the OCT FFT pipeline works');

  await page.goBack();
  await page.click('nav a[href="docs-pipeline.html"]');
  await expect(page.locator('h1')).toHaveText(/Pipeline/);
});
