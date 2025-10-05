from typing import TYPE_CHECKING, List, Optional, Union
from parallellm.core.backend.async_backend import AsyncBackend
from parallellm.core.backend.sync_backend import SyncBackend
from parallellm.core.response import (
    LLMIdentity,
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.provider.base import AsyncProvider, OpenAIProvider, SyncProvider
from parallellm.types import CallIdentifier, LLMDocument

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.responses.response_input_param import Message

# client = OpenAI()


def _fix_docs_for_openai(
    documents: Union[LLMDocument, List[LLMDocument]],
) -> "List[Message]":
    """Ensure documents are in the correct format for OpenAI API"""
    if not isinstance(documents, list):
        documents = [documents]

    documents = documents.copy()
    for i, doc in enumerate(documents):
        if isinstance(doc, str):
            msg: "Message" = {
                "role": "user",
                "content": doc,
            }
            documents[i] = msg
    return documents


class SyncOpenAIProvider(SyncProvider, OpenAIProvider):
    def __init__(self, client: "OpenAI", backend: SyncBackend):
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
        """Synchronously submit a query to OpenAI and return a ready response"""

        documents = _fix_docs_for_openai(documents)

        def sync_openai_call():
            return self.client.responses.create(
                model=llm.to_str("openai") if llm else "gpt-4.1-nano",
                instructions=instructions,
                input=documents,
                **kwargs,
            )

        # Execute the call synchronously and store the result
        resp_text, _, _ = self.backend.submit_sync_call(
            call_id, sync_function=sync_openai_call
        )

        # Return a ready response since the operation completed immediately
        return ReadyLLMResponse(call_id=call_id, value=resp_text)


class AsyncOpenAIProvider(AsyncProvider, OpenAIProvider):
    def __init__(self, client: "AsyncOpenAI", backend: AsyncBackend):
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
        documents = _fix_docs_for_openai(documents)

        coro = self.client.responses.create(
            model=llm.to_str("openai") if llm else "gpt-4.1-nano",
            instructions=instructions,
            input=documents,
            **kwargs,
        )

        # Submit to the backend for asynchronous execution
        self.backend.submit_coro(call_id=call_id, coro=coro)

        return PendingLLMResponse(
            call_id=call_id,
            backend=self.backend,
        )
