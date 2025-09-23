# Title: MCP_AutoHost CLI (Milestone 2 — async REPL clean)
import os, sys, json, asyncio
import mcp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON as RichJSON
from rich.syntax import Syntax
from dotenv import load_dotenv

from .memory import Memory
from .llm_client import LLMClient
from .logging_utils import tail_logs
from .mcp_client import MCPManager  # MCP wrapper

# Load .env explicitly from CWD
load_dotenv(".env")
console = Console()

def print_help():
    tbl = Table(title="Available commands")
    tbl.add_column("Command")
    tbl.add_column("Description")
    tbl.add_row("/help", "Show this help")
    tbl.add_row("/reset", "Clear memory/context")
    tbl.add_row("/logs [N]", "Show last N log entries (default 50)")
    tbl.add_row("/mcp tools <server>", "List tools of a server (filesystem|git)")
    tbl.add_row("/mcp call <server> <tool> <json-args>", "Call a tool with JSON args")
    tbl.add_row("/mcp demo_repo <dir>", "Create folder + README + initial git commit")
    tbl.add_row("/exit", "Exit")
    console.print(tbl)

def render_mcp_output(out):
    """
    Normalize the output of an MCP tools/call to print it nicely.
    - If out = {"content":[{"type":"text","text":"{...json...}"}]}, try to parse that inner JSON string.
    - If parsing fails, return it as plain text.
    - In other cases, assume 'out' is already a dict/JSON object.
    Returns (kind, payload) where kind ∈ {"json", "text"}.
    """
    try:
        # Typical MCP case: {"content":[{"type":"text","text":"{...}"}]}
        if isinstance(out, dict) and "content" in out and isinstance(out["content"], list):
            texts = []
            for item in out["content"]:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    texts.append(item["text"])
            joined = "\n".join(texts).strip()
            if joined:
                # Try to parse as JSON
                try:
                    data = json.loads(joined)
                    return "json", data
                except Exception:
                    # Not valid JSON; return as plain text
                    return "text", joined

        # If not in the above format, maybe 'out' is already a JSON-like object
        if isinstance(out, (dict, list)):
            return "json", out

        # Last resort: if it's a string, try JSON parse; if it fails, return as text
        if isinstance(out, str):
            try:
                data = json.loads(out)
                return "json", data
            except Exception:
                return "text", out

        # Unknown structure: return a string representation
        return "text", repr(out)

    except Exception as e:
        return "text", f"[render_mcp_output error] {e}"

async def repl_async():
    console.rule("[bold cyan]MCP_AutoHost")
    provider_hint = os.getenv("LLM_PROVIDER", "auto")
    model_hint = os.getenv("LLM_MODEL", "(default)")
    console.print(f"[bold]Provider:[/bold] {provider_hint}   [bold]Model:[/bold] {model_hint}")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        console.print("[red]No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    mem = Memory(max_messages=20)
    client = LLMClient()

    # Single MCP manager & single async loop — no more asyncio.run() inside REPL
    mcp = MCPManager()
    mcp.load_configs(workspace=os.getcwd())

    print_help()
    try:
        while True:
            try:
                user = console.input("\n[bold green]> [/bold green]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\nClosing…")
                break

            if not user:
                continue
            if user == "/exit":
                break
            if user == "/help":
                print_help()
                continue
            if user.startswith("/logs"):
                parts = user.split()
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 50
                events = tail_logs(n)
                if not events:
                    console.print("[yellow]No logs yet.")
                else:
                    for e in events:
                        console.print(e)
                continue
            if user == "/reset":
                mem.reset()
                console.print("[cyan]Memory cleared.")
                continue

            # ----- MCP block (async/await) -----
            if user.startswith("/mcp "):
                parts = user.split(maxsplit=2)
                sub = parts[1] if len(parts) > 1 else ""
                try:
                    if sub == "tools":
                        # /mcp tools <server>
                        if len(parts) < 3:
                            console.print("[red]Usage:[/red] /mcp tools <server>")
                        else:
                            name = parts[2]
                            tools = await mcp.list_tools(name)
                            console.print(Panel.fit("\n".join(tools),
                                                    title=f"MCP tools: {name}",
                                                    border_style="magenta"))

                    elif sub == "call":
                        # /mcp call <server> <tool> <json-args>
                        # ex: /mcp call filesystem write_file '{"path":"README.md","content":"Hello"}'
                        if len(parts) < 3:
                            console.print("[red]Usage:[/red] /mcp call <server> <tool> <json-args>")
                        else:
                            rest = parts[2]
                            try:
                                srv, tool, raw = parts[2].split(" ", 2)
                            except ValueError:
                                console.print("[red]Usage:[/red] /mcp call <server> <tool> <json-args>")
                                continue
                            try:
                                args = json.loads(raw)
                            except json.JSONDecodeError as je:
                                console.print(f"[red]Invalid JSON args:[/red] {je}")
                                continue
                            out = await mcp.call_tool(srv, tool, args)
                            kind, payload = render_mcp_output(out)

                            if kind == "json":
                                body = RichJSON.from_data(payload, indent=2)
                            else:
                                lang = "json"
                                try:
                                    json.loads(payload)
                                except Exception:
                                    lang = "text"
                                body = Syntax(payload, lang, word_wrap=True)

                            console.print(Panel(body, title=f"{srv}:{tool}", border_style="magenta"))

                    elif sub == "demo_repo":
                        # /mcp demo_repo <dir>
                        if len(parts) < 3:
                            console.print("[red]Usage:[/red] /mcp demo_repo <dir>")
                        else:
                            target = parts[2]
                            # Filesystem
                            await mcp.call_tool("filesystem", "create_directory", {"path": target})
                            await mcp.call_tool("filesystem", "write_file",
                                                {"path": f"{target}/README.md", "content": "# Demo Repo\n"})
                            # Git
                            try:
                                await mcp.call_tool("git", "git_set_working_dir", {"path": target})
                            except Exception:
                                pass
                            await mcp.call_tool("git", "git_init", {"path": target})
                            await mcp.call_tool("git", "git_add", {"path": target, "patterns": ["README.md"]})
                            await mcp.call_tool("git", "git_commit",
                                                {"path": target, "message": "chore: initial commit"})
                            console.print(f"[green]Demo repo initialized at {target}[/green]")
                    else:
                        print_help()
                except Exception as ex:
                    console.print(f"[red]MCP error: {ex}[/red]")
                continue
            
            # Normal LLM flow
            mem.add_user(user)
            try:
                reply = client.chat(mem)
            except Exception as ex:
                console.print(f"[red]LLM Error: {ex}[/red]")
                continue
            mem.add_assistant(reply)
            console.print(Panel.fit(reply, title="Assistant", border_style="blue"))
    finally:
        # Graceful shutdown of MCP sessions/transports
        await mcp.close()

def repl():
    """Wrapper to run the async REPL with a clean event loop."""
    asyncio.run(repl_async())

if __name__ == "__main__":
    repl()
