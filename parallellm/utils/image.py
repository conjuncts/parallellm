import base64
from io import BytesIO
from PIL import Image


def is_image(obj):
    """Check if the object is a PIL Image."""
    return isinstance(obj, Image.Image)


def get_image_type(obj: Image.Image):
    """Get preferred file type for this PIL Image"""
    out = obj.format
    if out is None:
        return "image/jpeg"
    return f"image/{out.lower()}"


def image_to_b64(obj: Image.Image, format=None):
    """Convert a PIL Image to a base64-encoded string."""
    if format is None:
        format = obj.format
    buffered = BytesIO()
    obj.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
