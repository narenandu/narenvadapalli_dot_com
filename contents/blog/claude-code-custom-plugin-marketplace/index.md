---
title: "Claude Code Custom Plugins: Build and Host Your Own Marketplace"
date: 2026-07-05
template: blog
image: "./claude_code_custom_plugin_marketplace.jpg"
description: "Learn how to build, structure, and host a custom, decentralized Claude Code plugin marketplace using a simple GitHub repository."
tags: ["ai", "agentic", "claude", "claude-code", "customization"]
---

*Claude Code Series: &larr; [Anthropic's Claude Model Family: Specs, Pros, Cons, and Use Cases](/blog/claude-models-comparison-guide/) (Previous)*

Claude Code’s modular architecture is designed to be highly extensible. Through its plugin system, you can package custom skills, automated shell hooks, specialized agent personas, and Model Context Protocol (MCP) servers.

While Anthropic provides an official marketplace by default, the CLI is built with decentralized distribution in mind. You can create, host, and share your own internal or community-driven **Claude Plugin Marketplace** using a standard GitHub repository.

In this guide, we’ll look at how Claude Code’s `/plugin` system works, explore the structure of a custom marketplace, and walk through building and hosting a sample marketplace with concrete code snippets.

---

### Claude Code Plugin CLI Commands

Claude Code provides a set of `/plugin` slash commands to manage custom sources, installations, and reloading:

*   `/plugin` or `/plugins` – Opens the interactive, terminal-based plugin manager UI (with Installed, Discover, and Settings tabs).
*   `/reload-plugins` – Reloads all installed plugins from disk. This is extremely useful for plugin developers, as it applies changes to local manifests, scripts, or skills instantly without needing to restart the active CLI session.
*   `/plugin marketplace add <owner/repo>` – Adds a custom Git repository as a plugin source.
*   `/plugin install <name>@<marketplace-name>` – Installs a specific plugin from a registered marketplace.
*   `/plugin uninstall <name>` – Removes an installed plugin.
*   `/plugin marketplace update <marketplace-name>` – Pulls the latest registry manifest from the custom source.

---

### Marketplace Repository Structure

A custom marketplace is simply a Git repository (hosted publicly or privately on GitHub, GitLab, or a self-hosted instance) containing a `marketplace.json` file at its root. 

Here is the recommended folder structure for a marketplace repository named `corporate-claude-plugins`:

```
corporate-claude-plugins/        <-- Repository Root
  ├── marketplace.json           <-- Main Marketplace Registry Manifest
  └── plugins/                   <-- Folder containing individual plugins
       └── git-summarizer/       <-- Our sample plugin
            ├── .claude-plugin/
            │    └── plugin.json <-- Individual Plugin Manifest
            ├── skills/
            │    └── SKILL.md    <-- Instructions for the agent
            └── scripts/
                 └── summarize.sh <-- Local automation script
```

---

### Step 1: Create the Marketplace Manifest (`marketplace.json`)

The `marketplace.json` file at the root of your repository serves as the registry index. It tells Claude Code what plugins are available and where their assets are located relative to the repository root.

Create `marketplace.json` in the root of your repo with the following structure:

```json
{
  "name": "corporate-plugins",
  "description": "Internal developer tooling plugins for Skynet Tech",
  "version": "1.0.0",
  "plugins": [
    {
      "name": "git-summarizer",
      "version": "1.2.0",
      "description": "Auto-summarizes git diffs into release notes",
      "path": "./plugins/git-summarizer",
      "author": "Skynet DevTeam"
    }
  ]
}
```

---

### Step 2: Define the Plugin Manifest (`plugin.json`)

Each plugin folder must contain a `.claude-plugin/` subdirectory with a `plugin.json` manifest. This manifest defines the plugin's entrypoints, required execution scopes, and custom slash commands.

Create `plugins/git-summarizer/.claude-plugin/plugin.json` with the following configuration:

```json
{
  "name": "git-summarizer",
  "version": "1.2.0",
  "description": "Auto-summarizes git diffs into release notes",
  "entrypoint": "scripts/summarize.sh",
  "scopes": [
    "command",
    "read_file"
  ],
  "commands": [
    {
      "name": "summarize-diff",
      "description": "Generates a release summary of uncommitted git changes"
    }
  ]
}
```

*   `scopes`: Defines what permissions the plugin requires (e.g. executing shell commands, reading files).
*   `commands`: Registers custom slash commands (e.g. `/summarize-diff`) that will be auto-completed inside the Claude Code session.

---

### Step 3: Implement the Plugin Logic

Now, let's build the script and instructions for the plugin.

#### The Automation Script (`scripts/summarize.sh`)
This script executes a standard git command to extract the unified diff of unstaged changes. Create `plugins/git-summarizer/scripts/summarize.sh`:

```bash
#!/bin/bash
# Expose the git diff of current unstaged changes
echo "=== CURRENT UNCOMMITTED CHANGES ==="
git diff --no-color
```
*(Make sure to commit the script with execution permissions: `chmod +x plugins/git-summarizer/scripts/summarize.sh`)*.

#### The Agent Instructions (`skills/SKILL.md`)
The `SKILL.md` file defines the rules and context the AI agent must adopt when the command is run. Create `plugins/git-summarizer/skills/SKILL.md`:

```markdown
# Git Summarizer Skill

When the user runs `/summarize-diff`, you will trigger the `summarize.sh` script to retrieve the uncommitted code diff. 

Analyze the diff and generate a concise, professional release summary structured as:
1. **Summary of Changes**: A 2-sentence high-level overview.
2. **Impact Scope**: Which files were affected.
3. **Draft Commit Message**: A recommended git commit message following Conventional Commits guidelines (e.g. `feat(auth): ...` or `fix(ui): ...`).
```

---

### Hosting and Installation

#### 1. Hosting the Repository
Push your `corporate-claude-plugins` repository to GitHub:

```bash
git init
git add .
git commit -m "Initialize plugin marketplace"
git remote add origin git@github.com:username/corporate-claude-plugins.git
git push -u origin main
```

> [!NOTE]
> **Private Repositories**: If your marketplace contains proprietary tooling, you can host it in a private GitHub repository. Claude Code uses your workstation’s local SSH agent and Git credentials to clone and update plugins, making authentication completely seamless.

#### 2. Installing the Marketplace
On a developer's machine, open Claude Code and register the custom marketplace:

```bash
/plugin marketplace add username/corporate-claude-plugins
```

#### 3. Installing the Plugin
Once the source is registered, install the custom tool:

```bash
/plugin install git-summarizer
```

Claude Code will pull the files locally, register the `/summarize-diff` slash command, and load the accompanying shell script and skill rules.

#### 4. Reloading Plugins During Local Development
If you make changes to your local plugin manifest (`plugin.json`), script logic, or rules, you do not need to restart your Claude Code session. Simply run the reload command to refresh your environment:

```bash
/reload-plugins
```
This forces Claude Code to re-scan the plugin directories, re-evaluate scopes, and apply your local updates instantly.

---

### Summary

Creating a custom marketplace allows engineering teams to centralize local development scripts, secure hooks, and common instructions. Because it is powered entirely by standard Git repositories and simple JSON configurations, you do not need to host complex package registries or databases.

In the next post, we will explore the **Model Context Protocol (MCP)** and learn how to connect your terminal-based agents directly to external APIs and databases!
