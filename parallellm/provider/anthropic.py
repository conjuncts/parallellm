from typing import TYPE_CHECKING, List, Optional, Union
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


class SyncAnthropicProvider(SyncProvider, AnthropicProvider):
    def __init__(self, client: "Anthropic", backend: SyncBackend):
        self.client = client
        self.backend = backend

    def submit_query_to_provider(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        call_id: CallIdentifier,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """Synchronously submit a query to Anthropic and return a ready response"""

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

        # Execute the call synchronously and store the result
        resp_text, _, _ = self.backend.submit_sync_call(
            call_id, sync_function=sync_anthropic_call
        )

        # Return a ready response since the operation completed immediately
        return ReadyLLMResponse(call_id=call_id, value=resp_text)


class AsyncAnthropicProvider(AsyncProvider, AnthropicProvider):
    def __init__(self, client: "AsyncAnthropic", backend: AsyncBackend):
        self.client = client
        self.backend = backend

    def submit_query_to_provider(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        call_id: CallIdentifier,
        llm: Optional[LLMIdentity] = None,
        _hoist_images=None,
        **kwargs,
    ):
        """Asynchronously submit a query to Anthropic and return a pending response"""

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

        # Submit to the backend for asynchronous execution
        self.backend.submit_coro(call_id=call_id, coro=coro)

        return PendingLLMResponse(
            call_id=call_id,
            backend=self.backend,
        )
