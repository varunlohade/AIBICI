"""
Module for handling Grok API templates and calls.
"""
import aiohttp
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Message:
    role: str
    content: str

class ChatTemplate:
    def __init__(self, messages: List[tuple[str, str | Callable]]):
        self.messages = messages
    
    def format_messages(self, **kwargs) -> List[Message]:
        formatted_messages = []
        for role, content in self.messages:
            if isinstance(content, Callable):
                # If content is a function, call it with kwargs
                message_content = content(kwargs)
            else:
                # If content is a string, format it with kwargs
                message_content = content.format(**kwargs)
            formatted_messages.append(Message(role=role, content=message_content))
        return formatted_messages

class ChatModel(ABC):
    @abstractmethod
    async def generate(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        pass

class GrokChatModel(ChatModel):
    def __init__(self, api_key: str, model: str = "grok-3-latest", temperature: float = 0):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.api_url = "https://api.x.ai/v1/chat/completions"

    async def generate(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        template = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "model": self.model,
            "stream": False,
            "temperature": self.temperature
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, json=template, headers=headers) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")
                return await response.json()

    async def summarize_conversation(self, messages: List[Message]) -> str:
        """Summarize a conversation using the model itself."""
        summary_prompt = Message(
            role="system",
            content="Please provide a brief, relevant summary of the following conversation that captures the key points and context needed for continuing the discussion. Focus on the most important information only."
        )
        
        # Convert messages to a readable format
        conversation = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        summary_request = Message(
            role="user",
            content=f"Please summarize this conversation:\n{conversation}"
        )
        
        response = await self.generate([summary_prompt, summary_request])
        if "choices" in response and response["choices"]:
            return response["choices"][0]["message"]["content"]
        return ""

class ChatExecutor:
    def __init__(self, 
                 model: ChatModel,
                 template: ChatTemplate,
                 tools: List[Any] = None,
                 verbose: bool = False):
        self.model = model
        self.template = template
        self.tools = tools or []
        self.verbose = verbose
        self.chat_history = []
        self.conversation_summary = None

    async def summarize_history(self) -> None:
        """Create a summary of the current chat history."""
        if isinstance(self.model, GrokChatModel) and self.chat_history:
            self.conversation_summary = await self.model.summarize_conversation(self.chat_history)
            if self.verbose:
                print("\nConversation Summary:", self.conversation_summary)

    async def invoke(self, input_text: str, summarize: bool = False) -> Dict[str, Any]:
        # Prepare messages including summary if available
        current_messages = []
        
        # Add the conversation summary as context if available
        if self.conversation_summary:
            current_messages.append(Message(
                role="system",
                content=f"Previous conversation context: {self.conversation_summary}"
            ))
        
        # Add the template messages
        current_messages.extend(self.template.format_messages(
            input=input_text,
            chat_history=self.chat_history
        ))
        
        if self.verbose:
            print("\nSending messages:", current_messages)
            
        response = await self.model.generate(current_messages)
        
        if self.verbose:
            print("\nReceived response:", response)
            
        # Add the interaction to chat history
        self.chat_history.append(Message(role="user", content=input_text))
        if "choices" in response and response["choices"]:
            assistant_message = response["choices"][0]["message"]["content"]
            self.chat_history.append(Message(role="assistant", content=assistant_message))
            
        # Create summary if requested
        if summarize:
            await self.summarize_history()
            
        return response 