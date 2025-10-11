from typing import Any, List, Optional, Union

from parallellm.core.identity import LLMIdentity
from parallellm.types import BatchIdentifier, CallIdentifier, LLMDocument, ProviderType


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
