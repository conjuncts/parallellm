from typing import Any, List, Optional, Union

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
        raise NotImplementedError

    def is_compatible(self, other: ProviderType) -> bool:
        """Returns whether this provider accepts the given provider type."""
        return other is None or self.provider_type == other

    def get_default_llm_identity(self) -> LLMIdentity:
        """Returns a default LLMIdentity for this provider."""
        raise NotImplementedError


class SyncProvider(BaseProvider):
    pass


class AsyncProvider(BaseProvider):
    pass


class BatchProvider(BaseProvider):
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
