# MCP_AutoHost

A CLI **host** that connects to an LLM via API, preserves **conversation context**, and writes **JSONL logs**.  
Implements:  
- **(1)** LLM connection  
- **(2)** Context memory  
- **(3)** MCP logging  
- **(4)** Integration with official MCP servers (Filesystem + Git)  
- **(5)** Local non-trivial MCP server (AutoAdvisor)  
- **(6)** Two peer MCP servers  
- **(7)** Remote MCP server (Echo)  

> Recommended environment: **Ubuntu/WSL2**. Python **3.10+**.

---

## Features
- Connects to **Anthropic or OpenAI** (auto-detected from API keys; can be forced via env vars).
- **Per-session memory** so follow-up questions work (e.g., “Who was Alan Turing?” → “When was he born?”).
- **JSON Lines logs** at `logs/host.jsonl` for each LLM and MCP request/response.
- Simple REPL with helper commands: `/help`, `/reset`, `/logs [N]`, `/exit`.
- **MCP integration:** connects to local and remote MCP servers via stdio/SSH.
- **Filesystem + Git demo:** create folder, write file, init repo and commit.
- **Local server:** integrates with AutoAdvisor MCP (cars dataset).
- **Remote server:** integrates with remote-echo MCP over SSH/cloud.

---

## Requirements
- Python 3.10+
- One API key: **Anthropic** _or_ **OpenAI**
- Node.js + npm
- Install official MCP servers:
```bash
npm i -g @modelcontextprotocol/server-filesystem @cyanheads/git-mcp-server
```

---

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # edit with your API key
```

---

## Configuration (.env)
```bash
# Use ONLY ONE provider (set its API key):
ANTHROPIC_API_KEY=sk-ant-REPLACE_ME
# OPENAI_API_KEY=sk-openai-REPLACE_ME

# (Optional) Force provider/model:
# LLM_PROVIDER=anthropic        # or openai
# LLM_MODEL=claude-3-5-sonnet-20241022   # Anthropic example
# LLM_MODEL=gpt-4o-mini                  # OpenAI example

# Workspace for Filesystem MCP
WORKSPACE=./demo_repo

# Remote SSH target for remote MCP
REMOTE_SSH=user@remote-vm
```

---

## Run
```bash
python -m src.host.cli
```

---

## Try context behavior
```
> Who was Alan Turing?
> When was he born?
```
The second question should resolve using the memory from the first one.

---

## Logs
- File: `logs/host.jsonl` (JSON Lines).
- Show the last N events inside the REPL:
```
/logs 20
```

Example log entries:
```json
{"ts":"2025-09-21T14:05:01Z","kind":"mcp_request","server":"filesystem","method":"tools/call","params":{"name":"writeFile","arguments":{"path":"README.md","content":"# Demo Repo\nCreado vía MCP"}}}
{"ts":"2025-09-21T14:05:01Z","kind":"mcp_response","server":"filesystem","method":"tools/call","result":{"ok":true}}
{"ts":"2025-09-21T14:05:03Z","kind":"mcp_request","server":"git","method":"tools/call","params":{"name":"commit","arguments":{"message":"Primer commit vía MCP"}}}
{"ts":"2025-09-21T14:05:03Z","kind":"mcp_response","server":"git","method":"tools/call","result":{"commit":"c0ffee1","message":"Primer commit vía MCP"}}
```

---

## Filesystem + Git Demo
Inside REPL, run:
```
tools filesystem
tools git

git init {}
filesystem writeFile {"path":"README.md","content":"# Demo Repo\nCreado vía MCP"}
git add {"path":"README.md"}
git commit {"message":"Primer commit vía MCP"}
filesystem listDirectory {"path":"."}
```

Expected result: `README.md` file created, added, and committed. Logs are recorded in `logs/host.jsonl`.

---

## Local Non-Trivial MCP Server (AutoAdvisor)
Server repo: [MCP_AutoAdvisor_Server](https://github.com/your-username/MCP_AutoAdvisor_Server)  
Add to `configs/servers.yaml`:
```yaml
auto_advisor:
  command: "python"
  args: ["server.py"]
  cwd: "../MCP_AutoAdvisor_Server"
  env: {}
```

Example usage:
```
Recommend honda cars with low mileage
What are the best value cars for my money if im in a 10,000 budget
Average price  of BMW 3 series 2018 and newer. 
```

---

## Peer MCP Servers
Add peers to `configs/servers.yaml` (adjust paths/commands):

```yaml
peer_math_tools:
  command: "python"
  args: ["-m","server.main"]
  cwd: "../peer-math-tools"
  env: {}

peer_text_utils:
  command: "node"
  args: ["server.js"]
  cwd: "../peer-text-utils"
  env: {}
```

```
Pokebuilder: 
Recommend a balanced team with trickroom

Trainer: 
Calculate my BMI if im 175 cm tall, 28 years old, 78 kg, and a male. 
```

---

## Remote MCP Server (Echo)
Repo: [remote-echo-server](https://github.com/your-username/remote-echo-server)  
Deployed on remote VM/cloud.

Add to `configs/servers.yaml`:
```yaml
remote_echo:
  command: "ssh"
  args: ["-T","${REMOTE_SSH}","bash","-lc","'cd ~/remote-echo && . .venv/bin/activate && python server.py'"]
  env: {}
```

```
remote_echo
remote_echo HELLO
sum 9 10 
result = 19
```

Expected log entries:
```json
{"ts":"2025-09-21T15:02:40Z","kind":"mcp_request","server":"remote_echo","method":"tools/call","params":{"name":"echo","arguments":{"text":"ping desde host"}}}
{"ts":"2025-09-21T15:02:40Z","kind":"mcp_response","server":"remote_echo","method":"tools/call","result":{"text":"ping desde host"}}
{"ts":"2025-09-21T15:02:41Z","kind":"mcp_request","server":"remote_echo","method":"tools/call","params":{"name":"sum","arguments":{"a":7,"b":35}}}
{"ts":"2025-09-21T15:02:41Z","kind":"mcp_response","server":"remote_echo","method":"tools/call","result":{"result":42}}
```

---

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
        ├── nl_router.py
        ├── parsers.py
        ├── server_manager.py
        ├── mcp_client.py
        └── logging_utils.py
```

---

## Notes
- If both keys are present and `LLM_PROVIDER` is unset, **Anthropic** is preferred by default.
- For Filesystem MCP, ensure `${WORKSPACE}` directory exists.
- For remote MCP via SSH, define `REMOTE_SSH` in `.env`.