from typing import List, Optional, Union

from parallellm.core.response import LLMDocument, LLMIdentity
from parallellm.types import CallIdentifier


class BaseProvider:
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


class SyncProvider(BaseProvider):
    pass


class AsyncProvider(BaseProvider):
    pass


class BatchProvider(BaseProvider):
    pass
