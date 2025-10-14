from typing import Any, List, Optional, Tuple, Union
from pydantic import BaseModel

from parallellm.core.identity import LLMIdentity
from parallellm.types import (
    BatchIdentifier,
    BatchResult,
    CallIdentifier,
    LLMDocument,
    ProviderType,
)


class BaseProvider:
    provider_type: Optional[ProviderType] = None
    """Must be set by subclasses to identify the provider type."""

    def is_compatible(self, other: ProviderType) -> bool:
        """Returns whether this provider accepts the given provider type."""
        return other is None or self.provider_type == other

    def get_default_llm_identity(self) -> LLMIdentity:
        """Returns a default LLMIdentity for this provider."""
        raise NotImplementedError

    def parse_response(
        self, raw_response: Union[BaseModel, dict]
    ) -> Tuple[str, Optional[str], dict]:
        """
        Parse a raw API response into a common format.

        :param raw_response: The raw response from the API (Pydantic model or dict)
        :return: Tuple of (response_text, response_id, metadata_dict)
        """
        raise NotImplementedError


class SyncProvider(BaseProvider):
    def prepare_sync_call(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """
        Prepare a synchronous callable for the backend to execute.

        :return: A callable that when invoked will make the API call and return the raw response
        """
        raise NotImplementedError


class AsyncProvider(BaseProvider):
    def prepare_async_call(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """
        Prepare an async coroutine for the backend to execute.

        :return: A coroutine that when awaited will make the API call and return the raw response
        """
        raise NotImplementedError


class BatchProvider(BaseProvider):
    def prepare_batch_call(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        llm: LLMIdentity,
        _hoist_images=None,
        **kwargs,
    ):
        """
        Prepare batch call data for the backend to bookkeep.

        :return: A dict/object representing the batch request format for this provider
        """
        raise NotImplementedError

    def submit_batch_to_provider(
        self, call_ids: list[CallIdentifier], stuff: list[Any]
    ) -> BatchIdentifier:
        """Submit a batch of calls to the provider."""
        raise NotImplementedError

    def download_batch(self, batch_uuid: str) -> List[BatchResult]:
        """Download the results of a batch from the provider.

        :return: A tuple of (content, batch_status)

            - content is either a string (entire file content), a list (one item per call_id),
            or None if pending.

            - batch_status is one of "pending", "ready", or "error".
        """
        raise NotImplementedError
