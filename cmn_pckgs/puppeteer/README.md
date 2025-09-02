# Puppeteer How to Use (ChatGPT)

**You do not need to use this. At the time you are reading this, I already have tested the ChatGPT agentic system. I am merely using this README for the sake of scientific reproducibility. If your goal is not reproducibility, there is no need to run this**

[Link for Results Coming Soon!]

# DISCLAIMER

**GIVING A CHATBOT ACCESS TO DATA THAT SHOULD NOT BE USED FOR TRAINING IS EXTREMELY IRRESPONSIBLE, ENSURE THAT DATA SHARING IS OFF PRIOR TO DOING THIS**

How to disable data sharing, if you do choose to run this server:

- Click your profile in the lower left corner, then click Settings
- Click on Data Controls
- Turn off the portion labeled "improve the model for everyone"

# Step 1: Start Chromium Puppet Browser

```bash
cd cmn_pckgs/puppeteer
npx tsc # Node.js required
node dist/chatgpt_login.js 
```

# Step 2: Login

A browser just opened, likely Chromium. From here, create an account or login to your own already. It is highly advised to use a burner account in case anything goes wrong here. For the same reason, I advise making an account with OpenAI rather than using Google's authentication, as OpenAI has a weaker anti-bot.

Once you have created an account and can see the ChatGPT screen and are able to prompt, close the browser. This will kill the console process of the login script. Proceed to next step once the process is killed.

# Step 3: Sending Puppeteer

* Start by creating a `.private.env` file from the `template.private.env` provided. Once created, as a copy of the template, insert one of the following three options as values:
    - GPT-4o
    - GPT-5-FAST
    - GPT-5-THINK

* Once you have selected your model, you can proceed to start both the evaluation server and the puppeteer server

```bash
cd cmn_pckgs/puppeteer # Assuming you have not done this yet, you already have done this if you are in the same terminal you used to log in.
node dist/chatgpt_srv.js
```

* Open up a new terminal and start up `intelligence_srv.py`, instructions are listed for this in the main README for this repo.

* Open the `chatgpt_eval.py` file and run it in whichever way you see fit (terminal, REPL, etc.), be wary that ChatGPT has rate limits prior to running (not publically disclosed, but believed to be about 80 per 3 hours).
    - If you want to send fewer requests in one batch to abide by rate limits, then just change the margin of error in `intelligence_srv.py` (instructions in main README)



