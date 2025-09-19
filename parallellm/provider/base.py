from typing import List, Optional, Union

from parallellm.core.response import LLMDocument, LLMIdentity


class BaseProvider:
    def submit_query_to_provider(
        self,
        instructions,
        documents: Union[LLMDocument, List[LLMDocument]] = [],
        *,
        hashed: str,
        checkpoint: str,
        seq_id: int,
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
