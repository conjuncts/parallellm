from typing import List, Optional, Union

from parallellm.core.identity import LLMIdentity
from parallellm.types import CallIdentifier, LLMDocument, ProviderType


class BaseProvider:
    provider_type: Optional[ProviderType] = None
    """Must be set by subclasses to identify the provider type."""

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
        raise NotImplementedError

    def is_compatible(self, other: ProviderType) -> bool:
        """Returns whether this provider accepts the given provider type."""
        return other is None or self.provider_type == other


class SyncProvider(BaseProvider):
    pass


class AsyncProvider(BaseProvider):
    pass


class BatchProvider(BaseProvider):
    pass


class OpenAIProvider(BaseProvider):
    provider_type: str = "openai"
