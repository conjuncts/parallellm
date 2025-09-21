from typing import TypedDict, Optional


class WorkingMetadata(TypedDict):
    latest_checkpoint: Optional[str]
    """
    Latest checkpoint is that which has last been reached
    """
    checkpoint_counter: int
    """
    Counter for checkpoint mode operations. Persisted across runs.
    """


class CallIdentifier(TypedDict):
    checkpoint: Optional[str]
    doc_hash: str
    seq_id: int
    session_id: int
    """Numeric ID of session. For tracking/metadata purposes only."""
