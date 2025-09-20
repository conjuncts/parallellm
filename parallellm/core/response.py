from typing import TYPE_CHECKING, Literal, Union
from PIL import Image

from parallellm.core.backend import BaseBackend
from parallellm.logging.dash_logger import DashboardLogger, HashStatus
from parallellm.types import CallIdentifier

if TYPE_CHECKING:
    from parallellm.core.manager import BatchManager

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
    def __init__(self, value):
        self.value = value

    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self.value


class PendingLLMResponse(LLMResponse):
    """
    Base class for a response from an LLM.
    """

    def __init__(
        self,
        call_id: CallIdentifier,
        backend: BaseBackend,
    ):
        self.call_id = call_id
        self._backend = backend
        self.value = None

    def resolve(self) -> str:
        if self.value is not None:
            return self.value

        self.value = self._backend.retrieve(self.call_id)
        return self.value


class ReadyLLMResponse(LLMResponse):
    """
    A response that is already resolved.
    """

    def __init__(self, call_id: CallIdentifier, value: str):
        self.call_id = call_id
        self.value = value

    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self.value
