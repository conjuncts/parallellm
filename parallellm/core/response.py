import json
from typing import Optional
from parallellm.core.backend import BaseBackend
from parallellm.core.calls import _call_to_concise_dict, _concise_dict_to_call
from parallellm.types import (
    CallIdentifier,
    LLMDocument,
    ParsedResponse,
    FunctionCallRequest,
    FunctionCall,
)


class LLMResponse:
    def __init__(self, value: str, *, call_id: CallIdentifier = None):
        self.value = value
        self.call_id = call_id
        self._pr: Optional[ParsedResponse] = None

    def resolve(self) -> str:
        """
        Resolve the response to a string.

        :returns: The resolved string response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """
        return self.value

    def resolve_json(self) -> dict:
        """
        Resolve the response to a JSON object.

        :returns: The resolved JSON response. If this value is not available,
            execution should stop gracefully and proceed to the next batch.
        """

        value = self.resolve()
        return json.loads(value)

    def resolve_function_calls(self, to_dict=False) -> list[FunctionCall]:
        """
        Resolve function calls (tool calls to user-defined functions) associated with this response.

        :param to_dict: Whether to parse the function calls' arguments into dictionaries (if they're JSON strings)
        :returns: A list of FunctionCall objects
        """
        if self._pr and self._pr.function_calls:
            # cast and jsonify if needed
            return self._pr.function_calls
        return []

    def __getstate__(self):
        """
        Support for pickling. Only store the call_id since that uniquely identifies the response.
        """
        return {"call_id": _call_to_concise_dict(self.call_id)}

    def __setstate__(self, state):
        """
        Support for unpickling. Restore the call_id, but value will need to be resolved later.
        """
        self.call_id = _concise_dict_to_call(state["call_id"])
        self.value = None
        self._pr = None

    def __repr__(self):
        v = self.value
        if v and len(v) > 50:
            v = v[:47] + "..."
        return (
            f"{self.__class__.__name__}({v!r}, doc_hash={self.call_id['doc_hash'][:8]})"
        )

    def __str__(self):
        if self.value is not None:
            return self.value
        return repr(self)


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

        pr = self._backend.retrieve(self.call_id)
        self.value = pr.text if pr else None
        self._pr = pr
        return self.value

    def __setstate__(self, state):
        """
        Support for unpickling. Restore the call_id, but backend will need to be set separately.
        """
        self.call_id = state["call_id"]
        self.value = None
        self._backend = None  # Will be set later


class ReadyLLMResponse(LLMResponse):
    """
    A response that is already resolved.
    """

    def __init__(
        self, call_id: CallIdentifier, *, pr: ParsedResponse = None, value: str = None
    ):
        super().__init__(value=pr.text if pr else value, call_id=call_id)
        self._pr = pr
