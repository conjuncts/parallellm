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
    from google import genai


def _fix_docs_for_gemini(
    documents: Union[LLMDocument, List[LLMDocument]],
) -> Union[str, List]:
    """Ensure documents are in the correct format for Gemini API"""
    if not isinstance(documents, list):
        documents = [documents]

    # For Gemini, we can pass strings directly or convert to proper format
    # The SDK will handle the conversion automatically
    if len(documents) == 1 and isinstance(documents[0], str):
        return documents[0]  # Single string can be passed directly

    # For multiple documents or mixed types, return as list
    formatted_docs = []
    for doc in documents:
        if isinstance(doc, str):
            formatted_docs.append(doc)
        elif isinstance(doc, dict):
            # If it's already a proper content dict, keep it
            if "parts" in doc and "role" in doc:
                formatted_docs.append(doc)
            elif "content" in doc:
                formatted_docs.append(doc["content"])
            elif "text" in doc:
                formatted_docs.append(doc["text"])
            else:
                formatted_docs.append(str(doc))
        else:
            formatted_docs.append(str(doc))

    return formatted_docs


class GeminiProvider(BaseProvider):
    provider_type: str = "google"

    def get_default_llm_identity(self) -> LLMIdentity:
        return LLMIdentity("gemini-2.5-flash", provider=self.provider_type)


class SyncGeminiProvider(SyncProvider, GeminiProvider):
    def __init__(self, client: "genai.Client", backend: SyncBackend):
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
        """Synchronously submit a query to Gemini and return a ready response"""

        contents = _fix_docs_for_gemini(documents)

        config = kwargs.copy()
        if instructions:
            config["system_instruction"] = instructions

        def sync_gemini_call():
            return self.client.models.generate_content(
                model=llm.model_name if llm else "gemini-2.5-flash",
                contents=contents,
                config=config,
            )

        resp_text, _, _ = self.backend.submit_sync_call(
            call_id, sync_function=sync_gemini_call
        )

        return ReadyLLMResponse(call_id=call_id, value=resp_text)


class AsyncGeminiProvider(AsyncProvider, GeminiProvider):
    def __init__(self, client: "genai.Client", backend: AsyncBackend):
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
        """Asynchronously submit a query to Gemini and return a pending response"""

        # Convert documents to Gemini format
        contents = _fix_docs_for_gemini(documents)

        # Prepare generation config with system instructions
        config = kwargs.copy()
        if instructions:
            config["system_instruction"] = instructions

        coro = self.client.aio.models.generate_content(
            model=llm.model_name if llm else "gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        self.backend.submit_coro(call_id=call_id, coro=coro)

        return PendingLLMResponse(
            call_id=call_id,
            backend=self.backend,
        )
