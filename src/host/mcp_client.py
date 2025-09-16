# MCP_AutoHost - Minimal MCP client wrapper (stdio)
from __future__ import annotations
import asyncio, json, os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

from .logging_utils import log_event

CONFIG_PATH = "configs/servers.yaml"

@dataclass
class MCPServerConfig:
    command: str
    args: list[str]
    env: dict[str, str]
    cwd: Optional[str] = None

def _expand(val: Any, workspace: str) -> Any:
    """Expands ${VARS}, ~ and ${workspace} in str / list / dict."""
    if isinstance(val, str):
        # first ${workspace}, then system variables and ~
        val = val.replace("${workspace}", workspace)
        val = os.path.expandvars(val)
        val = os.path.expanduser(val)
        return val
    if isinstance(val, list):
        return [_expand(v, workspace) for v in val]
    if isinstance(val, dict):
        return {k: _expand(v, workspace) for k, v in val.items()}
    return val

class MCPManager:
    """Manages multiple MCP server connections via stdio."""
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.processes: dict[str, Any] = {}
        self.configs: dict[str, MCPServerConfig] = {}
        self.workspace: Optional[str] = None

    def load_configs(self, workspace: Optional[str] = None):
        self.workspace = workspace or os.getcwd()
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_servers = (data.get("servers") or {})

        cfgs: dict[str, MCPServerConfig] = {}
        for name, cfg in raw_servers.items():
            # expand everything: command, args, env, cwd
            command = _expand(cfg.get("command", "npx"), self.workspace)
            args    = _expand(cfg.get("args", []), self.workspace)
            env     = _expand(cfg.get("env", {}) or {}, self.workspace)
            cwd     = _expand(cfg.get("cwd", None), self.workspace)
            cfgs[name] = MCPServerConfig(command=command, args=args, env=env, cwd=cwd)
        self.configs = cfgs

    async def connect(self, name: str):
        if name in self.sessions:
            return self.sessions[name]
        if name not in self.configs:
            raise RuntimeError(f"Unknown server '{name}'. Did you configure configs/servers.yaml?")

        cfg = self.configs[name]
        # Merge system env + server env (server env has priority)
        merged_env = dict(os.environ)
        merged_env.update(cfg.env or {})

        server_params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args,
            env=merged_env,
            cwd=cfg.cwd  # <- now we support cwd
        )
        log_event("mcp_request", server=name, op="connect",
                  command=cfg.command, args=cfg.args, cwd=cfg.cwd)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        reader, writer = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(reader, writer))
        await session.initialize()

        tools = (await session.list_tools()).tools
        log_event("mcp_response", server=name, op="connect", tools=[t.name for t in tools])
        self.sessions[name] = session
        return session

    async def list_tools(self, name: str):
        session = await self.connect(name)
        resp = await session.list_tools()
        return [t.name for t in resp.tools]

    async def call_tool(self, name: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        session = await self.connect(name)
        log_event("mcp_request", server=name, op="tools.call", tool=tool, input=args)
        result = await session.call_tool(tool, args)
        payload = {"content": [c.to_dict() if hasattr(c, "to_dict") else getattr(c, "__dict__", str(c)) for c in result.content]}
        log_event("mcp_response", server=name, op="tools.call", tool=tool, output=payload)
        return payload

    async def close(self):
        await self.exit_stack.aclose()
        self.sessions.clear()
