import express from "express";
import puppet from "./chatgpt_puppet.js";
import SchedulerQueue from "./scheduler_queue.js";
import pp from "puppeteer-extra";
import StealthPlugin from "puppeteer-extra-plugin-stealth";
import dotenv from "dotenv";
import morgan from "morgan";
import { ElementHandle } from "puppeteer";

dotenv.config({ path: ".private.env" });

const model = process.env.GPT_MODEL || "GPT-5-FAST";

const mod_key = process.platform === "darwin" ? "Meta" : "Control";

const app = express();
const PORT = 9000;

app.use(morgan("dev"));

const scheduler = new SchedulerQueue();

app.use(express.json());

pp.use(StealthPlugin());

const browser = await pp.launch({
    headless: false,
	userDataDir: "./profile",
});

let page_name = "https://chatgpt.com/";
switch (model) {
    case "GPT-5-FAST":
        page_name = page_name + "?model=gpt-5-instant";
        break;
    case "GPT-4o":
        page_name = page_name
        break;
    case "GPT-5-THINK":
        page_name = page_name + "?model=gpt-5-thinking";
        break;
}

const page = await browser.newPage();
await page.goto(page_name, { waitUntil: "networkidle0"});

if (model === "GPT-4o") {
    let frame = page.frames().find(f => f.url().includes(page_name))!;

    console.log("Frame URL", frame.url());

    await Promise.race([
        frame.waitForNavigation({ waitUntil: "networkidle0"}).catch(() => {}),
        frame.waitForFunction(() => document.readyState === "complete").catch(() => {}),
    ])

    const buttons = await frame.$$eval(".group.flex.cursor-pointer", (buttons) => buttons.map((button) => {
        return {
            text: button.textContent,
            id: button.id,
        };
    }));


    for (const button of buttons) {
        if (!button.text?.includes("ChatGPT 5")) {
            continue;
        }
        let id = button.id;
        try {
            const button_attempt: ElementHandle<HTMLButtonElement> | null = await frame.waitForSelector(`button[id="${id}"]`, { visible: true, timeout: 5000});
            await button_attempt!.click();
            break;
        } catch {
            continue;
        }
    }

    const legacy_model = await frame.waitForSelector('div[data-testid="Legacy models-submenu"]', { timeout: 5000});
    await legacy_model!.click();

    const fouro = await frame.waitForSelector('div[data-testid="model-switcher-gpt-4o"]', { timeout: 5000});
    await fouro!.click();
}

const frame = page.frames().find(f => f.url().includes(page_name))!;

async function processRequest(req: any) {
    const prompt = req.body?.prompt;

    await puppet.ClickNewChat(frame);

    await puppet.TypeInChat(page, frame, prompt, mod_key);

    const response = await puppet.ReturnResponse(frame);

    if (model === "GPT-4o") {
        await page.goBack({ waitUntil: "networkidle0"});
    } else {
        await page.goto(page_name, { waitUntil: "networkidle0"});
    }

    return response;
}

app.post("/", async (req: any, res: any) => {
    try {
        const result = await scheduler.add(() => processRequest(req));
        res.send(result);
    } catch (error) {
        console.log("Error", error);
        res.status(500).json({ message: "Internal server error", error: error });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});