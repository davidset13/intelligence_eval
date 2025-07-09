# HLE Server

* This server does agentic HLE evaluation for any model/agent
* There will be multiple forms of such evaluation, with the most important one being the non-specific one (all LLM-based questions)

## Three Possible Directions

1. Professional Support

* Agentic startups do direct B2B work and get evaluations back within 1-2 business days
* Significantly less work on tech front, and personalized connections
* Downside is not fully agentic

2. Fully Agentic

* Agentic evaluation of an agentic system, essentially
* Less individual work, fully agentic, higher margins
* subject to being replicated, impossible to patent, loss of ethos

3. Not a Good Business Idea

* Make it open source and make it center of resume

## Getting started (Windows Powershell)

* `cd hle_server`
* `python -m venv .venv`
* `.venv/Scripts/activate`
* `python.exe -m pip install -r requirements.txt`
* `python src/get_hle_dataset.py`
* The dataset should save to the `utility` folder.
