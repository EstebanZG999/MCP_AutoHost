# MCP_AutoHost - in-memory conversation buffer
from __future__ import annotations
from dataclasses import dataclass, field
import json

@dataclass
class Message:
    role: str  # 'system' | 'user' | 'assistant'
    content: str

@dataclass
class Memory:
    max_messages: int = 20
    system_prompt: str = (
        "You are a helpful assistant. Respond with factual accuracy and conciseness. "
        "Maintain context across turns."
    )
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.messages = [Message(role="system", content=self.system_prompt)]

    def add_user(self, content: str):
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant(self, content: str):
        self.messages.append(Message(role="assistant", content=content))
        self._trim()

    def export_for_openai(self):
        # OpenAI uses the system message inside 'messages'
        return [m.__dict__ for m in self.messages]

    def export_for_anthropic(self):
        # Anthropic receives 'system' separately; here we return only user/assistant
        return [m.__dict__ for m in self.messages if m.role != "system"]

    def _trim(self):
        # Always keep the first system and the last N
        if len(self.messages) > self.max_messages + 1:
            head = self.messages[0:1]
            tail = self.messages[-self.max_messages:]
            self.messages = head + tail

    def history(self):
        return list(self.messages)  # o como almacenes el historial [{"role":"user","content":...}, ...]

    def dump_json(self):
        return json.dumps(self.history(), ensure_ascii=False, indent=2)