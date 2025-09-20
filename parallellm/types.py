from typing import TypedDict


class WorkingMetadata(TypedDict):
    current_checkpoint: str


class CallIdentifier(TypedDict):
    checkpoint: str
    doc_hash: str
    seq_id: int
