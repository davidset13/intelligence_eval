import pp from "puppeteer-extra";
import StealthPlugin from "puppeteer-extra-plugin-stealth";

async function login() {pp.use(StealthPlugin());

	const browser = await pp.launch({
	headless: false,
	userDataDir: "./profile",
	});

	const page = await browser.newPage();

	await page.goto("https://chatgpt.com/auth/login", { waitUntil: "networkidle0"});
	
}

login();