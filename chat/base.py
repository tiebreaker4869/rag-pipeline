from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """聊天消息"""
    role: str  # "user", "assistant", "system"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """LLM 响应"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # token usage info
    metadata: Optional[Dict[str, Any]] = None  # 其他元数据


class BaseLLM(ABC):
    """LLM 基础接口"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        生成回复

        Args:
            messages: 消息列表
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成 token 数
            **kwargs: 其他模型特定参数

        Returns:
            ChatResponse: 生成的响应
        """
        pass

    def chat(self, user_message: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        简化的单轮对话接口

        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            **kwargs: 传递给 generate 的其他参数

        Returns:
            str: 生成的回复内容
        """
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))

        response = self.generate(messages, **kwargs)
        return response.content
