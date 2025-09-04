# MCP_AutoHost CLI
import os, sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv
from .memory import Memory
from .llm_client import LLMClient
from .logging_utils import tail_logs

load_dotenv()
console = Console()

def print_help():
    tbl = Table(title="Available Commands")
    tbl.add_column("Command")
    tbl.add_column("Description")
    tbl.add_row("/help", "Show this help")
    tbl.add_row("/reset", "Clear memory/context")
    tbl.add_row("/logs [N]", "Show last N log entries (default 50)")
    tbl.add_row("/exit", "Exit")
    console.print(tbl)

def repl():
    console.rule("[bold cyan]MCP_AutoHost")
    provider_hint = os.getenv("LLM_PROVIDER", "auto")
    model_hint = os.getenv("LLM_MODEL", "(default)")
    console.print(f"[bold]Provider:[/bold] {provider_hint}   [bold]Model:[/bold] {model_hint}")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")):
        console.print("[red]No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")
        sys.exit(1)

    mem = Memory(max_messages=20)
    client = LLMClient()

    print_help()
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

        # Normal flow: add the user to memory and query the model
        mem.add_user(user)
        try:
            reply = client.chat(mem)
        except Exception as ex:
            console.print(f"[red]LLM Error: {ex}[/red]")
            continue
        mem.add_assistant(reply)
        console.print(Panel.fit(reply, title="Assistant", border_style="blue"))

if __name__ == "__main__":
    repl()
