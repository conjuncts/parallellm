from typing import Optional
from parallellm.types import CallIdentifier


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
        pass

    def retrieve(self, call_id: CallIdentifier) -> Optional[str]:
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
    return (
        c1["checkpoint"] == c2["checkpoint"]
        and c1["doc_hash"] == c2["doc_hash"]
        and c1["seq_id"] == c2["seq_id"]
    )
