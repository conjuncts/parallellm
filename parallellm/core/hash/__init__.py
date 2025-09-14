import hashlib
from typing import List
from io import BytesIO
from PIL import Image

from parallellm.core.response import LLMDocument


def compute_hash(instructions, documents: List[LLMDocument]) -> str:
    """
    Compute a hash for the given instructions and documents.

    :param instructions: The instructions to hash.
    :param documents: The documents to hash.
    :returns: A SHA-256 hash representing the combined content, in hexadecimal format.
    """
    hasher = hashlib.sha256()
    hasher.update(instructions.encode("utf-8"))
    for doc in documents:
        if isinstance(doc, str):
            hasher.update(doc.encode("utf-8"))
        elif isinstance(doc, Image.Image):
            with BytesIO() as img_buffer:
                doc.save(img_buffer, format="PNG")
                hasher.update(img_buffer.getvalue())
        else:
            raise ValueError(f"Unsupported document type: {type(doc)}")
    return hasher.hexdigest()
