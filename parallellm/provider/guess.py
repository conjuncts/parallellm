from typing import Union
from pydantic import BaseModel


def guess_schema(inp: Union[BaseModel, dict]) -> tuple[str, str, dict]:
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
            raise ValueError("Invalid input type")

    # Try to extract response text from various possible fields
    resp_text = None

    # Try to get output_text as attribute ()
    if resp_text is None:
        resp_text = getattr(model, "output_text", None)

    # Extract response ID from various possible fields
    resp_id = (
        obj.pop("id", None)
        or obj.pop("response_id", None)
        or obj.pop("responseId", None)
    )

    return resp_text, resp_id, obj
