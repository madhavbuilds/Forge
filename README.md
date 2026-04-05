# Forge

**Offline vibe coding for simple websites.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-violet.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-local-teal.svg)](https://ollama.com)

Forge is an open-source, local-first AI coding tool built for a simple workflow:

- chat with the AI
- generate a website
- open a live preview
- keep iterating with follow-up prompts

It runs fully on your machine with Ollama, stays lightweight for normal laptops, and is currently focused on **plain HTML + CSS websites**.

## What Forge Does

Forge is designed to feel more like an offline vibe-coding tool than a complex autonomous agent.

Right now it can:

- chat in a persistent session
- create a new website project folder from a prompt
- generate `index.html` and `styles.css`
- open a local browser preview automatically
- keep editing the same project through follow-up prompts
- show simple progress states like `Planning`, `Thinking`, `Coding`, and `Answering`

## Current Scope

Forge is intentionally narrow right now.

- stack: **HTML + CSS only**
- local models: **Ollama**
- main workflow: **chat -> create -> preview -> iterate**

That narrow scope is deliberate. The goal is to make the flow reliable, fast, and easy before expanding the stack.

## Why Open Source

Forge is being built in public so anyone can:

- run it locally
- inspect how it works
- improve the prompts and workflow
- adapt it to their own local-first coding setup

The project is still evolving, and the workflow will keep improving over time.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-name/forge.git
cd forge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Pull recommended models

For lightweight local use:

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:3b
```

### 4. Set local-friendly model defaults

In `~/.forge/config.toml`:

```toml
[models]
fast = "ollama/qwen2.5-coder:1.5b"
smart = "ollama/qwen2.5-coder:3b"
```

If your laptop is struggling, you can use `1.5b` for both.

## Main Workflow

### Chat mode

This is the main vibe-coding workflow.

```bash
cd /path/to/forge
.venv/bin/forge chat --repo /path/to/forge
```

Example prompts:

- `make a premium landing page for a local AI coding tool in html and css`
- `make the hero section cleaner`
- `add a pricing section`
- `improve the mobile layout`
- `make the colors feel more premium`

What happens:

1. Forge keeps the chat open
2. It creates the website project
3. It opens a browser preview automatically
4. Future prompts keep editing the same website

### One-shot mode

If you just want a single run:

```bash
.venv/bin/forge run "make a portfolio landing page in html and css"
```

## Example Experience

```text
you: make a premium landing page for a local AI coding tool in html and css

Forge:
Planning...
Thinking...
Coding...
Answering...
```

Then Forge:

- creates a project folder
- writes `index.html`
- writes `styles.css`
- opens preview in your browser

Then you continue:

```text
you: make the CTA stronger
you: improve spacing
you: add a testimonial section
```

## Design Goals

Forge is aiming for:

- local-first
- open source
- low-lag on normal laptops
- simple UI
- direct prompts
- visible progress
- fast website iteration

Not aiming for:

- a giant autonomous agent
- hidden multi-step orchestration
- cloud-only workflows
- overly technical UX for normal users

## Product Direction

Forge is currently focused on being:

**an offline Lovable-style website builder for simple static sites**

The long-term direction is to make local vibe coding feel:

- visual
- conversational
- lightweight
- fast
- fun to iterate with

## Current Limitations

Forge is still early.

Some current limitations:

- website generation is limited to HTML + CSS
- generated designs are improving, but not perfect yet
- model quality depends on your local hardware and Ollama setup
- global `forge` may conflict with another tool on some machines, so using `.venv/bin/forge` is the safest option during development

## Development

### Run tests

```bash
.venv/bin/pytest -q
```

### Compile check

```bash
.venv/bin/python -m compileall -q forge
```

## Contributing

Issues, ideas, prompt improvements, UI upgrades, and workflow suggestions are welcome.

Useful areas to contribute:

- better HTML/CSS generation
- stronger preview workflow
- cleaner browser studio UX
- faster local model handling
- better edit iteration in chat mode

## License

MIT — see [LICENSE](LICENSE).
