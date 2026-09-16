// Playwright e2e against the live WeldScope site (GitHub Pages).
// Override with PLAYWRIGHT_BASE_URL to test another deployment.
module.exports = {
  testDir: '.',
  fullyParallel: false,
  retries: 0,
  timeout: 30000,
  reporter: 'list',
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || 'https://tobias-weiss-ai-xr.github.io/weldscope/',
    headless: true,
    trace: 'retain-on-failure',
  },
};
