from typing import Optional
from parallellm.types import CallIdentifier, ParsedResponse


class BaseBackend:
    """
    A backend is a data store, but also a way to poll
    """

    async def _poll_changes(self, call_id: CallIdentifier):
        """
        A chance to poll for changes and update the data store
        """
        raise NotImplementedError

    def persist(self):
        """Persist data and clean up resources"""
        pass

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        raise NotImplementedError

    def store(self, call_id: CallIdentifier, response: str) -> int:
        raise NotImplementedError


def _call_matches(c1: CallIdentifier, c2: CallIdentifier) -> bool:
    """
    Given c1 and c2, compares checkpoint, doc_hash, seq_id but NOT session_id.

    This is intentional - session_id is for tracking/auditing purposes,
    but calls are conceptually the same across different sessions if they have
    the same checkpoint, doc_hash, and seq_id.
    """
    keys = ["agent_name", "checkpoint", "doc_hash", "seq_id"]
    return all(c1[key] == c2[key] for key in keys)
