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
    if isinstance(inp, BaseModel):
        # hardcoded
        try:
            if provider_type == "openai":
                res = inp.output_text, inp.id
                obj = inp.model_dump(mode="json")
                obj.pop("id")
                return (*res, obj)
            elif provider_type == "google":
                res = inp.text, inp.response_id
                obj = inp.model_dump(mode="json")
                obj.pop("response_id")
                return (*res, obj)
            elif provider_type == "anthropic":
                # Anthropic response typically has content as a list with text blocks
                content = inp.content
                if isinstance(content, list) and len(content) > 0:
                    text_content = (
                        content[0].text
                        if hasattr(content[0], "text")
                        else str(content[0])
                    )
                else:
                    text_content = str(content)
                res = text_content, inp.id
                obj = inp.model_dump(mode="json")
                obj.pop("id")
                return (*res, obj)
        except Exception as e:
            print(
                f"Unexpected error while processing known provider type {provider_type}",
                e,
            )

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
        resp_text = obj.get("output_text") or obj.get("text")

    # Handle Anthropic-style content (list of content blocks)
    if resp_text is None and "content" in obj:
        content = obj.get("content")
        if isinstance(content, list) and len(content) > 0:
            # Extract text from first content block
            first_block = content[0]
            if isinstance(first_block, dict) and "text" in first_block:
                resp_text = first_block["text"]
            elif hasattr(first_block, "text"):
                resp_text = first_block.text
            else:
                resp_text = str(first_block)
        elif isinstance(content, str):
            resp_text = content
        else:
            resp_text = str(content)

    # Extract response ID from various possible fields
    resp_id = (
        obj.pop("id", None)
        or obj.pop("response_id", None)
        or obj.pop("responseId", None)
    )

    if not resp_text:
        raise ValueError("Could not find response text in object", obj)

    return resp_text, resp_id, obj
