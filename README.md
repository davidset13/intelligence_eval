# Intelligence Server

* This server does agentic HLE evaluation for any model/agent.
* There will be multiple forms of such evaluation, with the most important one being the non-specific one (all LLM-based questions).

# Setup (Windows)
```powershell
cd intelligence_server
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

* Make sure uv is added to your machine's PATH environment variable. The installation path should be printed in the console followiung the above commands.

```powershell
$env:PATH += ";C:\{your installation path here}"
```

* Next, we configure the virtual environment and package locks

```powershell
uv venv
.venv/Scripts/activate
uv sync
```

# Environment Variables (All Machines)
```
# Create .private.env file for environment variables
cp template.private.env
```


# Product 1: HLE

* Fully automated evaluation, hardest questions
* Partial dataset publically available online

# Product 2: GAIA

