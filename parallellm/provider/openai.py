import asyncio
from typing import List, Optional, Union
from parallellm.core.backend.async_backend import AsyncBackend
from parallellm.core.hash import compute_hash
from parallellm.core.response import (
    LLMDocument,
    LLMIdentity,
    LLMResponse,
    PendingLLMResponse,
)
from parallellm.provider.base import AsyncProvider, SyncProvider

from openai import AsyncOpenAI
from openai.types.responses.response import Response

# client = OpenAI()


class SyncOpenAIProvider(SyncProvider):
    pass


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
        # async def query_coro():
        #     res: Response = await
        coro = self.client.responses.create(
            model=llm.to_str("openai") if llm else "gpt-4.1-nano",
            instructions=instructions,
            input=documents,
            **kwargs,
        )
        # out = {
        #     **res.to_dict(),
        # }
        # out.update({
        #     "output_text": res.output_text
        # })
        # return out
        self.backend.submit_coro(stage=stage, doc_hash=hashed, seq_id=seq_id, coro=coro)
        return PendingLLMResponse(
            stage=stage, seq_id=seq_id, doc_hash=hashed, backend=self.backend
        )
