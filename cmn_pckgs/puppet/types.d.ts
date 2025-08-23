declare module "puppeteer-extra" {
  import type { Browser } from "puppeteer";


  interface LaunchOptions {
    headless?: boolean;
    userDataDir?: string;
    browser?: "chrome" | "firefox";

    [key: string]: unknown;
  }

  interface PuppeteerExtra {
    use(plugin: unknown): this;
    launch(options?: LaunchOptions): Promise<Browser>;
    connect?(...args: any[]): Promise<Browser>;
  }

  const puppeteerExtra: PuppeteerExtra;
  export default puppeteerExtra;
}

declare module "puppeteer-extra-plugin-stealth" {
  const StealthPlugin: (...args: any[]) => unknown;
  export default StealthPlugin;
}