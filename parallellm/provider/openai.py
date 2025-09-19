import asyncio
from typing import List, Optional, Union
from parallellm.core.backend.async_backend import AsyncBackend
from parallellm.core.backend.sync_backend import SyncBackend
from parallellm.core.hash import compute_hash
from parallellm.core.response import (
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.provider.base import AsyncProvider, SyncProvider

from openai import OpenAI, AsyncOpenAI
from openai.types.responses.response_input_param import Message

# client = OpenAI()


def _fix_docs_for_openai(
    documents: Union[LLMDocument, List[LLMDocument]],
) -> List[Message]:
    """Ensure documents are in the correct format for OpenAI API"""
    if isinstance(documents, list):
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                msg: Message = {
                    "role": "user",
                    "content": doc,
                }
                documents[i] = msg
    return documents


class SyncOpenAIProvider(SyncProvider):
    def __init__(self, client: OpenAI, backend: SyncBackend):
        self.client = client
        self.backend = backend

    def submit_query_to_provider(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        hashed: str,
        stage: str,
        seq_id: int,
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
            stage=stage, doc_hash=hashed, seq_id=seq_id, sync_function=sync_openai_call
        )

        # Return a ready response since the operation completed immediately
        return ReadyLLMResponse(
            stage=stage, seq_id=seq_id, doc_hash=hashed, value=resp_text
        )


class AsyncOpenAIProvider(AsyncProvider):
    def __init__(self, client: AsyncOpenAI, backend: AsyncBackend):
        self.client = client
        self.backend = backend

    def submit_query_to_provider(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        hashed: str,
        stage: str,
        seq_id: int,
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
        self.backend.submit_coro(stage=stage, doc_hash=hashed, seq_id=seq_id, coro=coro)

        return PendingLLMResponse(
            stage=stage,
            seq_id=seq_id,
            doc_hash=hashed,
            backend=self.backend,
        )
