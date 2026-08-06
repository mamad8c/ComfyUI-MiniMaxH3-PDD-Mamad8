"""Runtime container for validated MiniMax H3 PDD displacement heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PDDHeads:
    """CPU FP32 head masters and lazily-created per-device inference copies."""

    nfe: int
    num_intervals: int
    block_size: int
    sigmas_video: list[float]
    sigmas_audio: list[float]
    dsum_video: list[float]
    dsum_audio: list[float]
    video_w: Any
    video_b: Any
    audio_w: Any
    audio_b: Any
    metadata: dict[str, Any]
    _device_cache: dict[str, tuple[Any, Any, Any, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def for_device(self, device: Any) -> tuple[Any, Any, Any, Any]:
        """Return cached FP32 heads on ``device`` without module registration."""

        import torch

        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = tuple(
                tensor.to(device=device, dtype=torch.float32, non_blocking=True)
                for tensor in (self.video_w, self.video_b, self.audio_w, self.audio_b)
            )
            self._device_cache[key] = cached
        return cached
