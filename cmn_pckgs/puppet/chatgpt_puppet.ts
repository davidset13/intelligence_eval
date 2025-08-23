import type { ElementHandle, Frame, Page, KeyInput } from "puppeteer";
import clipboardy from "clipboardy";


async function ClickNewChat(frame: Frame) {
 
    const selector = "a[data-testid='create-new-chat-button']";

        try {
            const handle: ElementHandle<HTMLAnchorElement> | null = await frame.waitForSelector(selector, { visible: true, timeout: 2000});
            await handle!.click();
            return true;
        } catch {
            throw new Error("No new chat button found");
        }
}


async function TypeInChat(page: Page, frame: Frame, message: string, mod_key: KeyInput) {
    const selector = "div[id='prompt-textarea']";
    await clipboardy.write(message);

    try {
        const handle: ElementHandle<HTMLDivElement> | null = await frame.waitForSelector(selector, { visible: true, timeout: 2000});
        await handle!.click();
        await page.keyboard.down(mod_key);
        await page.keyboard.press("v");
        await page.keyboard.up(mod_key);
        await page.keyboard.press("Enter");
        return true;
    } catch {
        throw new Error("No chat input found");
    }
}


async function ReturnResponse(frame: Frame) {
    const button_to_find = "button[data-testid=composer-speech-button]"

    const text_div = "div.markdown.prose"

    try {
        console.log("Frames on page", frame.url());
        await Promise.race([
            frame.waitForNavigation({ waitUntil: "networkidle0"}).catch(() => {}),
            frame.waitForFunction(() => document.readyState === "complete").catch(() => {}),
        ])
        await frame.waitForSelector(button_to_find, { visible: true, timeout: 0});
        await frame.waitForSelector(text_div, { visible: true, timeout: 10000});
        const blocks = await frame.$eval(text_div, (container) => {
            const nodes = Array.from(
                container.querySelectorAll(':scope h1, :scope h2, :scope h3, :scope h4, :scope h5, :scope h6, :scope p')
            );
            return (nodes.map((node) => (node.textContent?.trim() || ""))).join("\n");
        });
        console.log("Text Evaluated", blocks);
        return blocks;
    } catch (err) {
        console.log("Error", err);
        throw new Error(err as string);
    }
}


export default { TypeInChat, ReturnResponse, ClickNewChat };