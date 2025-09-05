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
        cfgs = {}
        for name, cfg in (data.get("servers") or {}).items():
            args = []
            for a in cfg.get("args", []):
                a = a.replace("${workspace}", self.workspace)
                args.append(a)
            cfgs[name] = MCPServerConfig(
                command=cfg.get("command", "npx"),
                args=args,
                env=cfg.get("env", {}) or {}
            )
        self.configs = cfgs

    async def connect(self, name: str):
        if name in self.sessions:
            return self.sessions[name]
        if name not in self.configs:
            raise RuntimeError(f"Unknown server '{name}'. Did you configure configs/servers.yaml?")

        cfg = self.configs[name]
        server_params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env)
        log_event("mcp_request", server=name, op="connect", command=cfg.command, args=cfg.args)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        reader, writer = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(reader, writer))
        await session.initialize()

        # Optional: list tools once to confirm
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
        # result.content is a list of structured parts; stringify for log and return
        payload = {"content": [c.to_dict() if hasattr(c, "to_dict") else getattr(c, "__dict__", str(c)) for c in result.content]}
        log_event("mcp_response", server=name, op="tools.call", tool=tool, output=payload)
        return payload

    async def close(self):
        await self.exit_stack.aclose()
        self.sessions.clear()
