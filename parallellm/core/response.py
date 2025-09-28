from typing import TYPE_CHECKING, Literal, Union
from PIL import Image

from parallellm.core.backend import BaseBackend
from parallellm.types import CallIdentifier

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
    def __init__(self, value, call_id=None):
        self.value = value
        self.call_id: CallIdentifier = call_id

    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self.value

    def __getstate__(self):
        """
        Support for pickling. Only store the call_id since that uniquely identifies the response.
        """
        return {"call_id": self.call_id}

    def __setstate__(self, state):
        """
        Support for unpickling. Restore the call_id, but value will need to be resolved later.
        """
        self.call_id = state["call_id"]
        self.value = None


class PendingLLMResponse(LLMResponse):
    """
    Base class for a response from an LLM.
    """

    def __init__(
        self,
        call_id: CallIdentifier,
        backend: BaseBackend,
    ):
        super().__init__(value=None, call_id=call_id)
        self._backend = backend

    def resolve(self) -> str:
        if self.value is not None:
            return self.value

        self.value = self._backend.retrieve(self.call_id)
        return self.value

    def __getstate__(self):
        """
        Support for pickling. Only store the call_id since that uniquely identifies the response.
        The backend reference will need to be restored when unpickling.
        """
        return {"call_id": self.call_id}

    def __setstate__(self, state):
        """
        Support for unpickling. Restore the call_id, but backend will need to be set separately.
        """
        self.call_id = state["call_id"]
        self.value = None
        self._backend = (
            None  # This will need to be set by the system when the object is loaded
        )


class ReadyLLMResponse(LLMResponse):
    """
    A response that is already resolved.
    """

    def __init__(self, call_id: CallIdentifier, value: str):
        super().__init__(value=value, call_id=call_id)

    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self.value

    def __getstate__(self):
        """
        Support for pickling. Only store the call_id since that uniquely identifies the response.
        """
        return {"call_id": self.call_id}

    def __setstate__(self, state):
        """
        Support for unpickling. Restore the call_id, but value will need to be resolved later.
        """
        self.call_id = state["call_id"]
        self.value = None
