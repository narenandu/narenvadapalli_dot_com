---
title: "Nous Research's Hermes Agent: Under the Hood"
date: 2026-07-14
template: blog
image: "./cover_image.jpg"
description: "Deep-dive into the configuration, CLI commands, and skill architecture of Nous Research's Hermes Agent. Learn to manage tool gateways and design custom skills."
tags: ["ai", "agents", "hermes-agent", "software-engineering"]
---

*Autonomous AI Agents & Frameworks Series: &larr; [The Landscape of Agentic AI: From Single-Agent Scripts to Multi-Agent Networks](/blog/landscape-of-agentic-ai/) (Previous) | [The Self-Hosted AI Butler: Modular Assistance with OpenClaw](/blog/openclaw-self-hosted-ai-butler/) (Next) &rarr;*

### Prior Reading Material
Before exploring self-improving code loops, we recommend reading the prerequisite posts in this series and the foundational LLM guides:
*   [The Landscape of Agentic AI: From Single-Agent Scripts to Multi-Agent Networks](/blog/landscape-of-agentic-ai/) — Demystifying the ReAct loop, context bloat, and hierarchical coordination graphs.
*   [Training vs. Inference Lifecycle: A Developer's Guide to Weights, Backpropagation, and Serving](/blog/training-vs-inference-lifecycle/) — An inside look at how static weights perform stateless token predictions during inference.

---

In the landscape of AI assistant systems, most terminal agents operate with a fixed list of pre-configured tools. If a user asks a static agent to perform a task outside its direct scope—such as converting a proprietary raw video codec or parsing an obscure XML format—the agent fails because it doesn't have the necessary functions registered in its system context.

To solve this, Nous Research introduced **[Hermes Agent](https://github.com/NousResearch/Hermes-Agent)**: a self-improving, autonomous terminal assistant designed to run locally, inspect its own tool deficits, write custom skill modules on-the-fly, test them in a sandbox, and save them to a persistent **Skill Library** for future runs.

In this second installment of the **Autonomous AI Agents & Frameworks Series**, we'll go under the hood of Hermes Agent. We will inspect its file system layout, walk through its CLI config interface, and build a custom compiled skill from scratch.

---

### Setting Up Hermes Agent Locally

Hermes Agent is designed to be installed quickly on local developer workstations:

#### 1. Quick Installation
Choose the command that matches your operating system:

*   **Linux / macOS / WSL2:** Run the following command in your terminal:
    ```bash
    curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
    ```
*   **Windows (PowerShell):** Run this in your native PowerShell:
    ```powershell
    iex (irm https://hermes-agent.nousresearch.com/install.ps1)
    ```

#### 2. Initial Configuration
Once installed, authenticate and initialize the tool gateway:
```bash
# Initialize zero-config authentication and link capabilities
hermes setup --portal
```

---

### The Hermes Agent Directory Structure

On installation, Hermes Agent initializes a configuration and workspace directory under your home folder (`~/.hermes/`). This acts as the runtime environment for the agent:

```
~/.hermes/
├── config.yaml          # Main configuration file (models, paths, interval behaviors)
├── .env                 # API keys and secret variables (takes precedence over config.yaml)
└── skills/              # The default local Skill Library folder
    ├── web-search/      # Bundled core skills
    └── git-manager/
```

#### Main Configuration: `config.yaml`

The main configuration file controls where the agent looks for custom skills and how it nudges the user to save newly generated skills. Let's look at a typical `~/.hermes/config.yaml` profile:

```yaml
# ~/.hermes/config.yaml
model: "hermes-3-llama-3.1-70b"
provider: "nous-portal"

skills:
  # Paths to scan for custom, user-defined skill directories
  external_dirs:
    - ~/.agents/skills
    - ~/work/my-custom-skills
  
  # How many turns before the agent prompts you to compile a successful script into a permanent skill
  creation_nudge_interval: 15
```

---

### Managing Configurations via the CLI

Instead of editing `config.yaml` manually, you can manage the agent's parameters directly using the built-in `hermes config` CLI commands:

```bash
# View the current active configuration
hermes config show

# Open config.yaml in your terminal's default editor (e.g. nano or vim)
hermes config edit

# Set a specific configuration parameter
hermes config set skills.creation_nudge_interval 20
```

---

### Designing Custom Skills in Hermes

In Hermes Agent, a **Skill** is structured as a directory containing a `SKILL.md` markdown file. This file provides the agent's planner with both metadata (YAML frontmatter) and execution steps (Markdown instructions).

Let's build a custom skill that parses hex strings, runs mathematical checks on them, and saves the output.

#### Step 1: Create the Skill Directory
Create a folder inside your Skill Library:
```bash
mkdir -p ~/.hermes/skills/hex-parser
```

#### Step 2: Write the `SKILL.md` File
Create `~/.hermes/skills/hex-parser/SKILL.md` and define the tool metadata:

```markdown
---
name: hex-parser
description: Parses comma-separated hex strings, converts them to integers, and sums the results.
---
# Task Instructions
When the user asks to sum, parse, or evaluate a string containing hexadecimal values (e.g., '0x0A, 0x14'):
1. Parse the comma-separated hex values from the input.
2. Execute the python helper script using the local terminal command:
   `python scripts/hex_parser.py --hex [input_string]`
3. Report the result back to the user.
```

#### Step 3: Write the Helper Python Script
Create a helper script at `scripts/hex_parser.py` that is executed by the agent's shell execution tool:

```python
# scripts/hex_parser.py
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hex", required=True, help="Comma-separated hex string")
    args = parser.parse_args()
    
    try:
        parts = [p.strip() for p in args.hex.split(",") if p.strip()]
        total = sum(int(p, 16) for p in parts)
        print(f"Success: Parsed {len(parts)} hex values. Total Sum = {total}")
    except Exception as e:
        print(f"Error parsing hex string: {str(e)}")

if __name__ == "__main__":
    main()
```

#### Step 4: Configure the Skill
If the skill requires specific API keys or variables, configure them interactively:
```bash
hermes skills config hex-parser
```

Once configured, the next time you type `hermes chat` and query:
> *"Sum these hex codes for me: 0x0A, 0x14, 0x05"*

The agent will query the `~/.hermes/skills` index, match the prompt to the `hex-parser` description, read the execution steps, run the script locally, and print the output!

---

### The Self-Improving Compilation Loop

If Hermes Agent encounters a task for which it has no registered skill, the outer loop initiates a self-improvement phase:
1.  **Drafting**: The agent writes a new Python script and a matching `SKILL.md` inside a temporary sandboxed directory.
2.  **Validation**: The agent executes test assertions against the generated script.
3.  **Reflection**: If the script throws an error, the agent parses the stacktrace, rewrites the code, and retries.
4.  **Cataloging**: Once tests pass, the agent prompts the user to save the new skill to the permanent `~/.hermes/skills/` library.

This runtime lifecycle allows Hermes Agent to scale its own capabilities dynamically as you run tasks in your workspace.

---

### What's Next?

Hermes Agent provides a persistent terminal assistant that compiles its own skills. However, setting up sandboxed code generation requires substantial computing overhead. What if we want a modular, self-hosted agent that runs locally with pre-configured tools and connects directly to our messaging channels?

In our next post, **[The Self-Hosted AI Butler: Modular Assistance with OpenClaw](/blog/openclaw-self-hosted-ai-butler/)**, we'll explore setting up the open-source **OpenClaw** framework, configuring custom tool gateways, and deploying a local assistant on your developer workstation!
