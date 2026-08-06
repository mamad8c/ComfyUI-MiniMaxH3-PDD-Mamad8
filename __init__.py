"""ComfyUI-MiniMaxH3-PDD V3 custom node package."""


async def comfy_entrypoint():
    """Return the lazily-built ComfyUI V3 extension."""

    from .nodes import create_extension

    return create_extension()
