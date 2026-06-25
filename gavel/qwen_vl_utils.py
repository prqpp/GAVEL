"""Minimal helper for extracting image objects from a Qwen-style chat message list."""


def process_vision_info(messages):
    image_inputs = []
    for msg in messages:
        for c in msg.get("content", []):
            if isinstance(c, dict) and c.get("type") == "image":
                image_inputs.append(c.get("image"))
    return image_inputs, None
