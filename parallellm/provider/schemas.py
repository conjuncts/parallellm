from typing import Optional, Union
from pydantic import BaseModel


def guess_schema(
    inp: Union[BaseModel, dict], provider_type: str = None
) -> tuple[str, str, dict]:
    """

    Guess the schema of the given object and convert it to a dictionary.
    :param obj: The object to guess the schema for. Can be a Pydantic BaseModel or a dictionary.

    :returns: A tuple containing the response text, response ID, and metadata dictionary.
    """
    model = None
    if isinstance(inp, BaseModel):
        model = inp
        obj = inp.model_dump(mode="json")
    elif isinstance(inp, dict):
        obj = inp.copy()
    else:
        # Handle OpenAI ChatCompletion objects and other API response objects
        if hasattr(inp, "model_dump"):
            obj = inp.model_dump(mode="json")
        elif hasattr(inp, "__dict__"):
            obj = inp.__dict__.copy()
        else:
            raise ValueError("Invalid input type", inp)

    # Try to extract response text from various possible fields
    resp_text = None

    # Try to get output_text as attribute ()
    if resp_text is None:
        resp_text = getattr(model, "output_text", None)

    if resp_text is None:
        resp_text = obj.get("output_text") or obj.get("content")

    # Extract response ID from various possible fields
    resp_id = (
        obj.pop("id", None)
        or obj.pop("response_id", None)
        or obj.pop("responseId", None)
    )

    if not resp_text:
        raise ValueError("Could not find response text in object", obj)

    return resp_text, resp_id, obj
