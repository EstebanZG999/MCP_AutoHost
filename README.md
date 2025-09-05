# MCP_AutoHost

A CLI **host** that connects to an LLM via API, preserves **conversation context**, and writes **JSONL logs**.
This does **(1) LLM connection** and **(2) Context memory**, and it lays plumbing for **(3) MCP logs**.

> Recommended environment: **Ubuntu/WSL2**. Python **3.10+**.

## Features
- Connects to **Anthropic or OpenAI** (auto-detected from API keys; can be forced via env vars).
- **Per-session memory** so follow-up questions work (e.g., “Who was Alan Turing?” → “When was he born?”).
- **JSON Lines logs** at `logs/host.jsonl` for each LLM request/response (MCP entries will be added in Milestone 2).
- Simple REPL with helper commands: `/help`, `/reset`, `/logs [N]`, `/exit`.
- MCP integration: connects to local MCP servers via stdio.
- Filesystem + Git demo: can create a folder, write a file, initialize a Git repo and commit.

## Requirements
- Python 3.10+
- One API key: **Anthropic** _or_ **OpenAI**
- npm i -g @modelcontextprotocol/server-filesystem @cyanheads/git-mcp-server

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # edit with your API key
```

## Configuration (.env)
```bash
# Use ONLY ONE provider (set its API key):
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME
# OPENAI_API_KEY=sk-openai-REPLACE_ME

# (Optional) Force provider/model:
# LLM_PROVIDER=anthropic        # or openai
# LLM_MODEL=claude-3-5-sonnet-20241022   # Anthropic example
# LLM_MODEL=gpt-4o-mini                  # OpenAI example
```

## Run
```bash
python -m src.host.cli
```

## Try context behavior
```
> Who was Alan Turing?
> When was he born?
```
The second question should resolve using the memory from the first one.

## Logs
- File: `logs/host.jsonl` (JSON Lines).
- Show the last N events inside the REPL:
```
/logs 20
```

## Project Layout
```
MCP_AutoHost/
├── README.md
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── configs/
│   └── servers.yaml
├── logs/
│   └── host.jsonl
└── src/
    └── host/
        ├── __init__.py
        ├── cli.py
        ├── llm_client.py
        ├── memory.py
        ├── mcp_client.py
        └── logging_utils.py
```

## Notes
- If both keys are present and `LLM_PROVIDER` is unset, **Anthropic** is preferred by default.
- On Windows native, install **Npcap** if you plan to capture `localhost` traffic with Wireshark later.
- This README covers Milestone 1. In Milestone 2 we will integrate **Filesystem/Git MCP** and start logging
  `mcp_request/mcp_response` in the same file.
