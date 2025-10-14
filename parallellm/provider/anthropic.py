from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from pydantic import BaseModel
from parallellm.core.backend.async_backend import AsyncBackend
from parallellm.core.backend.sync_backend import SyncBackend
from parallellm.core.identity import LLMIdentity
from parallellm.core.response import (
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.provider.base import AsyncProvider, BaseProvider, SyncProvider
from parallellm.types import CallIdentifier, LLMDocument

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic


def _fix_docs_for_anthropic(
    documents: Union[LLMDocument, List[LLMDocument]],
) -> List[dict]:
    """Ensure documents are in the correct format for Anthropic API"""
    if not isinstance(documents, list):
        documents = [documents]

    formatted_docs = []
    for doc in documents:
        if isinstance(doc, str):
            msg = {
                "role": "user",
                "content": doc,
            }
            formatted_docs.append(msg)
        elif isinstance(doc, dict):
            # If it's already a proper message dict, keep it
            if "role" in doc and "content" in doc:
                formatted_docs.append(doc)
            elif "content" in doc:
                formatted_docs.append({"role": "user", "content": doc["content"]})
            elif "text" in doc:
                formatted_docs.append({"role": "user", "content": doc["text"]})
            else:
                formatted_docs.append({"role": "user", "content": str(doc)})
        else:
            formatted_docs.append({"role": "user", "content": str(doc)})

    return formatted_docs


class AnthropicProvider(BaseProvider):
    provider_type: str = "anthropic"

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("claude-3-haiku-20240307", provider=self.provider_type)

    def parse_response(
        self, raw_response: Union[BaseModel, dict]
    ) -> Tuple[str, Optional[str], dict]:
        """Parse Anthropic API response into common format"""
        if isinstance(raw_response, BaseModel):
            # Pydantic model (e.g., anthropic.types.Message)
            content = raw_response.content
            if isinstance(content, list) and len(content) > 0:
                text_content = (
                    content[0].text if hasattr(content[0], "text") else str(content[0])
                )
            else:
                text_content = str(content)

            res = text_content, raw_response.id
            obj = raw_response.model_dump(mode="json")
            obj.pop("id", None)
            return (*res, obj)
        elif isinstance(raw_response, dict):
            # Dict response
            resp_id = raw_response.pop("id", None)

            # Extract text from content
            content = raw_response.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                first_block = content[0]
                if isinstance(first_block, dict) and "text" in first_block:
                    text_content = first_block["text"]
                else:
                    text_content = str(first_block)
            else:
                text_content = str(content)

            return text_content, resp_id, raw_response
        else:
            raise ValueError(f"Unsupported response type: {type(raw_response)}")


class SyncAnthropicProvider(SyncProvider, AnthropicProvider):
    def __init__(self, client: "Anthropic"):
        self.client = client

    def prepare_sync_call(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """Prepare a synchronous callable for Anthropic API"""
        messages = _fix_docs_for_anthropic(documents)

        # Add system instruction if provided
        config = kwargs.copy()
        if instructions:
            config["system"] = instructions

        def sync_anthropic_call():
            return self.client.messages.create(
                model=llm.model_name,
                max_tokens=config.pop("max_tokens", 4096),
                messages=messages,
                **config,
            )

        return sync_anthropic_call


class AsyncAnthropicProvider(AsyncProvider, AnthropicProvider):
    def __init__(self, client: "AsyncAnthropic"):
        self.client = client

    def prepare_async_call(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """Prepare an async coroutine for Anthropic API"""
        messages = _fix_docs_for_anthropic(documents)

        # Add system instruction if provided
        config = kwargs.copy()
        if instructions:
            config["system"] = instructions

        coro = self.client.messages.create(
            model=llm.model_name,
            max_tokens=config.pop("max_tokens", 1024),
            messages=messages,
            **config,
        )

        return coro
