from typing import Union
from PIL import Image

from parallellm.core.backend import BaseBackend

# Type alias for documents that can be either text or images
LLMDocument = Union[str, Image.Image]


class LLMIdentity:
    def __init__(self, identity: str):
        """
        Identify a specific LLM agent.
        """
        self.identity = identity


class LLMResponse:
    """
    Base class for a response from an LLM.
    """

    def __init__(self, stage, seq_id, doc_hash, backend: BaseBackend):
        self.stage = stage
        self.seq_id = seq_id
        self.doc_hash = doc_hash
        self._backend = backend

    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self._backend.retrieve(self.stage, self.doc_hash, self.seq_id)
