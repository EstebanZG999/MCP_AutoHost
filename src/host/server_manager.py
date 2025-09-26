# src/host/server_manager.py
import os
import asyncio
from typing import Dict, Tuple, Any, Optional, List
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
import shutil


START_TIMEOUT = 30
INIT_TIMEOUT  = 20

def _expand_vars(value: Any, workspace: str) -> Any:
    """Expands ${VAR} and ${workspace} in strings (recursive in lists/dicts)."""
    if isinstance(value, str):
        # First ${workspace}
        value = value.replace("${workspace}", workspace)
        # Then environment variables ${VAR}
        # Note: os.path.expandvars uses $VAR but also accepts ${VAR}.
        value = os.path.expandvars(value)
        return value
    if isinstance(value, list):
        return [_expand_vars(v, workspace) for v in value]
    if isinstance(value, dict):
        return {k: _expand_vars(v, workspace) for k, v in value.items()}
    return value

def _looks_runnable(cmd: str) -> bool:
    if cmd in ("ssh", "npx", "bash", "python", "python3"):
        return True
    if os.path.isabs(cmd) and os.path.exists(cmd):
        return True
    return shutil.which(cmd) is not None


class ServerManager:
    def __init__(self, servers_cfg: Dict[str, Dict[str, Any]], workspace: Optional[str] = None):
        self.cfg_raw = servers_cfg
        self.workspace = workspace or os.getcwd()

        # Expanded configuration
        self.cfg: Dict[str, Dict[str, Any]] = {
            name: _expand_vars(conf, self.workspace) for name, conf in servers_cfg.items()
        }

        # Active resources
        self.sessions: Dict[str, ClientSession] = {}
        self._ctxmgrs = {}   # name -> AsyncContextManager of stdio_client
        self._conns = {}     # name -> (read, write)

    async def _start_one(self, name: str, s: Dict[str, Any]):
        env = s.get("env", {}) or {}
        cmd = s["command"]
        args: List[str] = s.get("args", []) or []
        cwd = s.get("cwd")

        if cwd:
            cmdline = " ".join([cmd] + [str(a) for a in args])
            cmd = "bash"
            args = ["-lc", f"cd {cwd} && {cmdline}"]

        params = StdioServerParameters(command=cmd, args=args, env=env)

        ctx = stdio_client(params)
        try:
            read, write = await asyncio.wait_for(ctx.__aenter__(), timeout=START_TIMEOUT)
        except Exception as e:
            raise RuntimeError(f"[{name}] Failed to start server: {type(e).__name__}: {e}") from e

        sess = ClientSession(read, write)
        try:
            # 👇 IMPORTANT: enter the session context manager
            await asyncio.wait_for(sess.__aenter__(), timeout=INIT_TIMEOUT)
            await asyncio.wait_for(sess.initialize(), timeout=INIT_TIMEOUT)
        except Exception as e:
            try:
                # close the session if it was opened
                try:
                    await sess.__aexit__(None, None, None)
                except Exception:
                    pass
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
            raise RuntimeError(f"[{name}] Failed to initialize MCP session: {type(e).__name__}: {e}") from e

        self.sessions[name] = sess
        self._ctxmgrs[name] = ctx
        self._conns[name] = (read, write)

    async def start_all(self):
        names = list(self.cfg.keys())
        # ⚠️ filters servers with empty or non-runnable command
        launch = []
        for n in names:
            cmd = str(self.cfg[n].get("command", "")).strip()
            if not cmd or not _looks_runnable(cmd):
                print(f"[warn] Server '{n}' skipped: command not runnable -> {cmd!r}")
                continue
            launch.append(n)

        coros = [self._start_one(n, self.cfg[n]) for n in launch]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for n, r in zip(launch, results):
            if isinstance(r, Exception):
                print(f"[warn] Server '{n}' did not start: {r}")

    async def stop_all(self):
        # Stops in reverse order: shutdown → __aexit__ → close stdio_client
        for name, sess in list(self.sessions.items()):
            try:
                await sess.shutdown()
            except Exception:
                pass
            try:
                await sess.__aexit__(None, None, None)
            except Exception:
                pass

        for name, ctx in list(self._ctxmgrs.items()):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass

        self.sessions.clear()
        self._ctxmgrs.clear()
        self._conns.clear()

    async def build_tool_index(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        idx: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for srv, sess in self.sessions.items():
            try:
                tools = await sess.list_tools()
                for t in tools.tools:
                    idx[(srv, t.name)] = t.inputSchema
            except Exception as e:
                # Continue, but report
                print(f"[warn] list_tools failed for {srv}: {e}")
        return idx

    async def call_tool(self, server: str, tool: str, arguments: Dict[str, Any]) -> str:
        if server not in self.sessions:
            raise RuntimeError(f"Server '{server}' not started")
        sess = self.sessions[server]
        res = await sess.call_tool(name=tool, arguments=arguments)
        texts = [c.text for c in res.content if c.type == "text"]
        return "\n".join(texts).strip()
