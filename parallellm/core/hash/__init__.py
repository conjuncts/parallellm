import hashlib
from typing import List, Optional
from io import BytesIO
from PIL import Image

from parallellm.types import LLMDocument, ToolCallRequest, ToolCallOutput


def compute_hash(instructions: Optional[str], documents: List[LLMDocument]) -> str:
    """
    Compute a hash for the given instructions and documents.

    :param instructions: The instructions to hash.
    :param documents: The documents to hash.
    :returns: A SHA-256 hash representing the combined content, in hexadecimal format.
    """
    hasher = hashlib.sha256()
    if instructions:
        hasher.update(instructions.encode("utf-8"))
    for doc in documents:
        if isinstance(doc, str):
            hasher.update(doc.encode("utf-8"))
        elif isinstance(doc, Image.Image):
            with BytesIO() as img_buffer:
                doc.save(img_buffer, format="PNG")
                hasher.update(img_buffer.getvalue())
        elif isinstance(doc, ToolCallRequest):
            hasher.update(b"function_call")
            hasher.update(doc.prior.encode("utf-8"))
            for call in doc.calls:
                hasher.update(str(call).encode("utf-8"))
        elif isinstance(doc, ToolCallOutput):
            hasher.update(b"function_call_output")
            for item in [doc.content, doc.call_id]:
                hasher.update(str(item).encode("utf-8"))
        elif isinstance(doc, tuple) and len(doc) == 2:
            # Handle Tuple[Literal["user", "assistant", "system", "developer"], str]
            role, content = doc
            hasher.update(role.encode("utf-8"))
            if isinstance(content, str):
                hasher.update(content.encode("utf-8"))
            else:
                for item in content:
                    hasher.update(str(item).encode("utf-8"))
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")

    return hasher.hexdigest()
