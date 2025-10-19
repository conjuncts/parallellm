from parallellm.types import CallIdentifier


def _call_to_concise_dict(call: CallIdentifier) -> dict:
    """Convert CallIdentifier to the minimum dict"""
    return {
        "agent_name": call.get("agent_name"),
        "checkpoint": call.get("checkpoint"),
        "doc_hash": call.get("doc_hash"),
        "seq_id": call.get("seq_id"),
    }


def _concise_dict_to_call(d: dict) -> CallIdentifier:
    """Convert concise dict back to CallIdentifier. Mutates in place."""
    d.setdefault("session_id", None)
    d.setdefault("provider_type", None)
    return d


def _call_matches(c1: CallIdentifier, c2: CallIdentifier) -> bool:
    """
    Given c1 and c2, compares checkpoint, doc_hash, seq_id but NOT session_id.

    This is intentional - session_id is for tracking/auditing purposes,
    but calls are conceptually the same across different sessions if they have
    the same checkpoint, doc_hash, and seq_id.
    """
    keys = ["agent_name", "checkpoint", "doc_hash", "seq_id"]
    return all(c1[key] == c2[key] for key in keys)
