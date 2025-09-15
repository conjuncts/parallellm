from typing import Literal, Union
from PIL import Image

from parallellm.core.backend import BaseBackend
from parallellm.logging.dash_logger import DashboardLogger, HashStatus

# Type alias for documents that can be either text or images
LLMDocument = Union[str, Image.Image]


class LLMIdentity:
    def __init__(self, identity: str):
        """
        Identify a specific LLM agent.
        """
        self.identity = identity

    def to_str(self, provider: Union[Literal["openai"], None] = None) -> str:
        """
        Convert to a string representation for a specific provider.

        :param provider: A specific provider (ie. openai)
        """
        if provider == "openai":
            if self.identity is None:
                return "gpt-4.1-nano"
            return self.identity
        return self.identity


class LLMResponse:
    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        raise NotImplementedError()


class PendingLLMResponse(LLMResponse):
    """
    Base class for a response from an LLM.
    """

    def __init__(
        self,
        stage,
        seq_id,
        doc_hash,
        backend: BaseBackend,
        *,
        dash_logger: DashboardLogger = None,
    ):
        self.stage = stage
        self.seq_id = seq_id
        self.doc_hash = doc_hash
        self._backend = backend
        self.value = None
        self.dash_logger = dash_logger

    def resolve(self) -> str:
        if self.value is not None:
            return self.value

        self.value = self._backend.retrieve(self.stage, self.doc_hash, self.seq_id)
        if self.dash_logger is not None:
            self.dash_logger.update_hash(self.doc_hash, HashStatus.RECEIVED)
        return self.value


class ReadyLLMResponse(LLMResponse):
    """
    A response that is already resolved.
    """

    def __init__(self, stage, seq_id, doc_hash, value: str):
        self.stage = stage
        self.seq_id = seq_id
        self.doc_hash = doc_hash
        self.value = value

    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self.value
