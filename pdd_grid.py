"""Dependency-free MiniMax H3 PDD grid and scheduler arithmetic.

All calculations use Python floats, which are IEEE-754 binary64 on supported
ComfyUI platforms.  Keeping this module free of Torch and ComfyUI makes the
artifact identity checks independently testable.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass


GRID_VERSION = "minimax_h3_ref2va_pdd_euler_v1"
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
BLOCK_TOLERANCE = 1.0e-6


def shift_sigma(base_sigma: float, shift: float) -> float:
    """Apply the H3 rational sigma shift to one base-grid value."""

    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def remap_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """Map a sigma between shifted schedules through their shared base grid."""

    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


@dataclass(frozen=True)
class PDDGrid:
    """Full PDD knot grid plus its canonical artifact identity."""

    num_intervals: int
    block_size: int
    nfe: int
    sigmas_video: tuple[float, ...]
    sigmas_audio: tuple[float, ...]
    deltas_video: tuple[float, ...]
    deltas_audio: tuple[float, ...]
    grid_sha256: str

    @property
    def boundary_sigmas_video(self) -> tuple[float, ...]:
        return tuple(self.sigmas_video[i] for i in range(0, self.num_intervals + 1, self.block_size))

    @property
    def boundary_sigmas_audio(self) -> tuple[float, ...]:
        return tuple(self.sigmas_audio[i] for i in range(0, self.num_intervals + 1, self.block_size))

    def metadata_payload(self) -> dict:
        """Return the exact canonical payload whose JSON bytes are hashed."""

        return canonical_grid_payload(
            self.num_intervals,
            self.block_size,
            self.sigmas_video,
            self.sigmas_audio,
        )


def canonical_grid_payload(
    num_intervals: int,
    block_size: int,
    sigmas_video: tuple[float, ...] | list[float],
    sigmas_audio: tuple[float, ...] | list[float],
) -> dict:
    """Build the frozen training-side grid metadata payload, without its hash."""

    return {
        "version": GRID_VERSION,
        "num_intervals": int(num_intervals),
        "block_size": int(block_size),
        "nfe": int(num_intervals) // int(block_size),
        "video_shift": VIDEO_SHIFT,
        "audio_shift": AUDIO_SHIFT,
        "sigmas_video": [float(value) for value in sigmas_video],
        "sigmas_audio": [float(value) for value in sigmas_audio],
        "velocity": "clean_minus_noise",
        "time": "t_equals_1_minus_sigma",
    }


def grid_sha256(payload: dict) -> str:
    """Hash a canonical grid payload exactly as the training exporter does."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_grid(num_intervals: int = 256, block_size: int = 64) -> PDDGrid:
    """Construct and validate the full float64 MiniMax H3 dual-sigma grid."""

    num_intervals = int(num_intervals)
    block_size = int(block_size)
    if num_intervals < 1 or block_size < 1 or num_intervals % block_size:
        raise ValueError("PDD requires positive num_intervals divisible by block_size")

    # This is algebraically identical to torch.linspace(1, 0, N+1, float64)
    # for the frozen N=256 grid and preserves the exact 1.0/0.0 endpoints.
    base = tuple((num_intervals - index) / num_intervals for index in range(num_intervals + 1))
    video = tuple(shift_sigma(value, VIDEO_SHIFT) for value in base)
    audio = tuple(remap_sigma(value, VIDEO_SHIFT, AUDIO_SHIFT) for value in video)
    video = (1.0, *video[1:-1], 0.0)
    audio = (1.0, *audio[1:-1], 0.0)

    for name, values in (("video", video), ("audio", audio)):
        if values[0] != 1.0 or values[-1] != 0.0:
            raise RuntimeError(f"{name} PDD schedule endpoints are not exactly 1 and 0")
        if any(left <= right for left, right in zip(values, values[1:])):
            raise RuntimeError(f"{name} PDD schedule is not strictly decreasing")

    deltas_video = tuple(left - right for left, right in zip(video, video[1:]))
    deltas_audio = tuple(left - right for left, right in zip(audio, audio[1:]))
    if any(value <= 0.0 for value in (*deltas_video, *deltas_audio)):
        raise RuntimeError("PDD grid contains a non-positive Euler interval")

    payload = canonical_grid_payload(num_intervals, block_size, video, audio)
    return PDDGrid(
        num_intervals=num_intervals,
        block_size=block_size,
        nfe=num_intervals // block_size,
        sigmas_video=video,
        sigmas_audio=audio,
        deltas_video=deltas_video,
        deltas_audio=deltas_audio,
        grid_sha256=grid_sha256(payload),
    )


def parse_partition_knots(text, num_intervals: int):
    """Parse comma-separated interior cut knots into a full boundary knot list.

    Returns None for empty input. The result always starts at 0 and ends at
    ``num_intervals`` and is strictly increasing, e.g. "64,128,192,240" ->
    [0, 64, 128, 192, 240, 256].
    """

    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    interior = []
    for piece in cleaned.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            interior.append(int(piece))
        except ValueError as exc:
            raise ValueError(
                f"partition entries must be integer grid knots, got {piece!r}"
            ) from exc
    if not interior:
        return None
    knots = [0, *interior, num_intervals]
    for left, right in zip(knots, knots[1:]):
        if right <= left:
            raise ValueError(
                "partition must list strictly increasing interior cut knots in "
                f"(0, {num_intervals}), got {interior}"
            )
    return knots


def auto_partition_knots(
    grid: PDDGrid,
    nfe: int,
    anchors: tuple[int, ...] = (0, 64, 128, 192, 256),
    video_weight: float = 0.7,
    audio_weight: float = 0.3,
) -> list[int]:
    """Derive cut knots for any block count from the trained anchor partition.

    Above the anchor count, the block with the largest loss-weighted sigma span
    is repeatedly split at the knot that best halves that span — extra calls
    therefore land where the trained trajectory covers the most weighted sigma
    (in practice the late blocks). Below it, the interior anchor whose removal
    creates the smallest merged span is repeatedly dropped. Anchor launch
    states are preserved whenever the count allows, matching how the bank was
    trained (blocks launched only from anchor knots).
    """

    nfe = int(nfe)
    if nfe < 1 or nfe > grid.num_intervals:
        raise ValueError(f"nfe must be within 1..{grid.num_intervals}, got {nfe}")

    def wspan(left: int, right: int) -> float:
        return video_weight * (
            grid.sigmas_video[left] - grid.sigmas_video[right]
        ) + audio_weight * (grid.sigmas_audio[left] - grid.sigmas_audio[right])

    knots = [k for k in anchors if 0 <= k <= grid.num_intervals]
    while len(knots) - 1 > nfe:
        candidates = range(1, len(knots) - 1)
        drop = min(candidates, key=lambda i: wspan(knots[i - 1], knots[i + 1]))
        del knots[drop]
    while len(knots) - 1 < nfe:
        widths = [
            (wspan(knots[i], knots[i + 1]), i)
            for i in range(len(knots) - 1)
            if knots[i + 1] - knots[i] >= 2
        ]
        if not widths:
            raise ValueError(f"cannot reach nfe={nfe}: no splittable block left")
        _, index = max(widths)
        left, right = knots[index], knots[index + 1]
        cut = min(
            range(left + 1, right),
            key=lambda c: abs(wspan(left, c) - wspan(c, right)),
        )
        knots.insert(index + 1, cut)
    return knots


def select_block(
    sigma: float,
    boundaries: tuple[float, ...] | list[float],
    on_out_of_grid: str = "clamp",
    tolerance: float = BLOCK_TOLERANCE,
) -> int:
    """Select ``b`` for ``s[b+1] < sigma <= s[b]``, snapping near knots.

    Interior values within ``tolerance`` of a knot are assigned to the block
    beginning at that knot.  Exact zero is assigned to the final block.
    """

    values = tuple(float(value) for value in boundaries)
    if len(values) < 2 or any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError("PDD block boundaries must be strictly decreasing")
    if on_out_of_grid not in {"clamp", "error"}:
        raise ValueError("on_out_of_grid must be 'clamp' or 'error'")

    sigma = float(sigma)
    top, bottom = values[0], values[-1]
    if sigma > top + tolerance or sigma < bottom - tolerance:
        if on_out_of_grid == "error":
            raise ValueError(
                f"sigma {sigma:.9g} is outside the PDD grid [{bottom:.9g}, {top:.9g}]"
            )
        return 0 if sigma > top else len(values) - 2

    for index, knot in enumerate(values):
        if abs(sigma - knot) <= tolerance:
            return min(index, len(values) - 2)

    insertion = bisect.bisect_left(tuple(-value for value in values), -sigma)
    return max(0, min(insertion - 1, len(values) - 2))


def subsampled_grid_sigmas(num_intervals: int, steps: int) -> list[float]:
    """Return rounded, de-duplicated video knots for an arbitrary step count."""

    num_intervals = int(num_intervals)
    steps = int(steps)
    if num_intervals < 1 or steps < 1:
        raise ValueError("num_intervals and steps must be positive")
    full = build_grid(num_intervals, 1).sigmas_video
    indices: list[int] = []
    for index in range(steps + 1):
        knot = int(round(index * num_intervals / steps))
        if not indices or knot != indices[-1]:
            indices.append(knot)
    sigmas = [full[index] for index in indices]
    if sigmas[-1] != 0.0 or any(left <= right for left, right in zip(sigmas, sigmas[1:])):
        raise RuntimeError("subsampled PDD schedule is not strictly descending to zero")
    return sigmas


def scheduler_sigmas(
    num_intervals: int,
    block_size: int,
    mode: str,
    steps: int,
    denoise: float,
) -> list[float]:
    """Compute the scheduler result without importing Torch.

    Partial denoise deliberately follows the frozen contract and is outside the
    artifact's trained full-trajectory distribution.
    """

    denoise = float(denoise)
    if not 0.0 < denoise <= 1.0:
        raise ValueError("denoise must be greater than 0 and at most 1")
    grid = build_grid(num_intervals, block_size)
    if mode == "trained_blocks":
        result = list(grid.boundary_sigmas_video)
        if denoise < 1.0:
            kept_steps = int(round(grid.nfe * denoise))
            result = result[-(kept_steps + 1):]
    elif mode == "subsampled_grid":
        steps = int(steps)
        total_steps = steps if denoise >= 1.0 else int(round(steps / denoise))
        total_steps = max(1, total_steps)
        result = subsampled_grid_sigmas(num_intervals, total_steps)
        result = result[-(steps + 1):]
    else:
        raise ValueError("mode must be 'trained_blocks' or 'subsampled_grid'")

    if result[-1] != 0.0 or any(left <= right for left, right in zip(result, result[1:])):
        raise RuntimeError("PDD scheduler result must descend strictly to exactly zero")
    return result
