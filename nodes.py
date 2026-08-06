"""Lazy V3 node definitions for the MiniMax H3 PDD custom node package."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

from .pdd_grid import (
    AUDIO_SHIFT,
    BLOCK_TOLERANCE,
    VIDEO_SHIFT,
    auto_partition_knots,
    build_grid,
    parse_partition_knots,
    scheduler_sigmas,
    select_block,
)
from .pdd_heads import PDDHeads


def _required_int(metadata: dict, name: str) -> int:
    value = metadata.get(name)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"PDD heads metadata '{name}' must be an integer string, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"PDD heads metadata '{name}' must be positive, got {parsed}")
    return parsed


def _validate_hash(metadata: dict, name: str) -> str:
    value = metadata.get(name)
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"PDD heads metadata '{name}' must be a lowercase 64-character SHA-256")
    return value


def _is_hash(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def create_extension():
    """Create the ComfyUI V3 extension after the host runtime is available."""

    import folder_paths
    import torch
    import torch.nn.functional as functional
    import comfy.patcher_extension
    import comfy.utils
    from comfy.ldm.minimax.model import MiniMaxH3Model, time_shift_slope
    from comfy_api.latest import ComfyExtension, io

    heads_dir = os.path.join(folder_paths.models_dir, "pdd_heads")
    folder_paths.add_model_folder_path("pdd_heads", heads_dir, is_default=True)
    folder_paths.folder_names_and_paths["pdd_heads"][1].add(".safetensors")
    pdd_type = io.Custom("PDD_HEADS")

    class MiniMaxH3PDDHeadsLoader(io.ComfyNode):
        """Load and fail-closed validate a fused displacement-head artifact."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="MiniMaxH3PDDHeadsLoader",
                display_name="MiniMax H3 PDD Heads Loader",
                category="loaders/minimax",
                description="Load validated FP32 PDD-Euler fused displacement heads.",
                inputs=[
                    io.Combo.Input(
                        "heads_name",
                        options=folder_paths.get_filename_list("pdd_heads"),
                    ),
                    io.Int.Input(
                        "blocks",
                        default=0,
                        min=0,
                        max=256,
                        optional=True,
                        tooltip=(
                            "Transformer calls to fuse the artifact into. 0 keeps a "
                            "fused file's native count (raw banks default to 4). On "
                            "a raw interval bank ANY integer 1..256 works: divisors "
                            "of 256 fuse uniformly like the shipped exports, other "
                            "counts derive an anchored partition that preserves the "
                            "trained launch knots and splits the largest weighted "
                            "sigma spans (blocks=5 gives 0|64|128|192|240|256). "
                            "Fused files only re-fuse to divisors of their own "
                            "count; other values need the bank and fail closed."
                        ),
                    ),
                    io.String.Input(
                        "partition",
                        default="",
                        optional=True,
                        tooltip=(
                            "Comma-separated interior grid-knot cuts (grid knots run "
                            "0..256), e.g. '64,128,192,240' = 5 calls with the final "
                            "block split at sigma 0.444. Overrides blocks when set. "
                            "Cuts on trained block starts (multiples of 64) preserve "
                            "the trained launch states; other cuts create untrained "
                            "launch points and are diagnostic. Fused files only "
                            "support cuts aligned to their own block size; the raw "
                            "interval bank supports any cut."
                        ),
                    ),
                ],
                outputs=[pdd_type.Output(display_name="pdd_heads")],
            )

        @classmethod
        def execute(cls, heads_name, blocks=0, partition=""):
            path = folder_paths.get_full_path_or_raise("pdd_heads", heads_name)
            if not path.lower().endswith(".safetensors"):
                raise ValueError(f"PDD heads must be a .safetensors file, got {heads_name!r}")
            state, metadata = comfy.utils.load_torch_file(
                path,
                safe_load=True,
                device=torch.device("cpu"),
                return_metadata=True,
            )
            metadata = dict(metadata or {})
            if set(state) == {"video_weight", "video_bias", "audio_weight", "audio_bias"}:
                return cls._execute_bank(heads_name, state, metadata, blocks, partition)
            artifact_format = metadata.get("format")
            if not isinstance(artifact_format, str) or not artifact_format.startswith(
                "minimax_h3_ref2va_pdd_euler"
            ):
                raise ValueError(
                    "PDD heads metadata 'format' must start with "
                    "'minimax_h3_ref2va_pdd_euler'"
                )
            if metadata.get("adapter_a_included") != "false":
                raise ValueError("PDD deployment heads must declare adapter_a_included='false'")

            nfe = _required_int(metadata, "nfe")
            num_intervals = _required_int(metadata, "num_intervals")
            block_size = _required_int(metadata, "block_size")
            if num_intervals % nfe:
                raise ValueError(
                    f"PDD num_intervals={num_intervals} is not divisible by nfe={nfe}"
                )
            derived_block_size = num_intervals // nfe
            if block_size != derived_block_size:
                raise ValueError(
                    f"PDD block_size={block_size} does not equal "
                    f"num_intervals//nfe={derived_block_size}"
                )

            expected_grid_hash = _validate_hash(metadata, "grid_sha256")
            source_hash = _validate_hash(metadata, "source_checkpoint_manifest_sha256")
            # A-free finetunes have no teacher adapter; their exports carry the
            # explicit sentinel "none" instead of a hash.
            if metadata.get("teacher_adapter_a_sha256") == "none":
                teacher_hash = "none"
            else:
                teacher_hash = _validate_hash(metadata, "teacher_adapter_a_sha256")
            completed_update_text = metadata.get("completed_update")
            try:
                completed_update = int(completed_update_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "PDD heads metadata 'completed_update' must be a non-negative integer string"
                ) from exc
            if completed_update < 0:
                raise ValueError(
                    "PDD heads metadata 'completed_update' must be a non-negative integer string"
                )
            grid = build_grid(num_intervals, block_size)
            if grid.grid_sha256 != expected_grid_hash:
                raise ValueError(
                    "PDD grid_sha256 mismatch: metadata has "
                    f"{expected_grid_hash}, recomputed {grid.grid_sha256}. "
                    "The heads and schedule metadata do not belong together."
                )

            expected_keys = {
                f"blocks.{block}.{modality}.{kind}"
                for block in range(nfe)
                for modality in ("video", "audio")
                for kind in ("weight", "bias")
            }
            actual_keys = set(state)
            if actual_keys != expected_keys:
                raise ValueError(
                    "PDD head tensor keys differ from the declared nfe; "
                    f"missing={sorted(expected_keys - actual_keys)[:8]}, "
                    f"extra={sorted(actual_keys - expected_keys)[:8]}"
                )

            expected_shapes = {
                "video.weight": (96, 5376),
                "video.bias": (96,),
                "audio.weight": (32, 5376),
                "audio.bias": (32,),
            }
            for block in range(nfe):
                for suffix, shape in expected_shapes.items():
                    key = f"blocks.{block}.{suffix}"
                    tensor = state[key]
                    if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
                        raise ValueError(
                            f"PDD tensor {key} has shape/dtype "
                            f"{tuple(tensor.shape)}/{tensor.dtype}; expected {shape}/torch.float32"
                        )

            video_w = torch.stack(
                [state[f"blocks.{block}.video.weight"] for block in range(nfe)]
            ).detach().cpu().to(torch.float32).contiguous()
            video_b = torch.stack(
                [state[f"blocks.{block}.video.bias"] for block in range(nfe)]
            ).detach().cpu().to(torch.float32).contiguous()
            audio_w = torch.stack(
                [state[f"blocks.{block}.audio.weight"] for block in range(nfe)]
            ).detach().cpu().to(torch.float32).contiguous()
            audio_b = torch.stack(
                [state[f"blocks.{block}.audio.bias"] for block in range(nfe)]
            ).detach().cpu().to(torch.float32).contiguous()

            knots = parse_partition_knots(partition, num_intervals)
            if knots is not None and blocks:
                raise ValueError("Set either blocks or partition, not both")
            native_knots = list(range(0, num_intervals + 1, block_size))
            if knots is None:
                if blocks in (0, nfe):
                    knots = native_knots
                elif blocks < 1 or blocks > nfe or nfe % blocks:
                    raise ValueError(
                        f"blocks={blocks} is not reachable from this artifact: it "
                        f"holds {nfe} fused displacement heads, so valid values are "
                        f"the divisors of {nfe} (or 0 for native). Counts above "
                        f"{nfe} need the raw interval bank."
                    )
                else:
                    knots = list(range(0, num_intervals + 1, num_intervals // blocks))
            misaligned = [k for k in knots if k % block_size]
            if misaligned:
                raise ValueError(
                    f"partition cuts {misaligned} do not align with this artifact's "
                    f"fused block size {block_size}; arbitrary cuts need the raw "
                    "interval bank"
                )
            if knots != native_knots:
                # Displacement heads are interval sums, so summing consecutive fused
                # heads re-fuses exactly onto any coarser aligned partition.
                runs = [(a // block_size, b // block_size) for a, b in zip(knots, knots[1:])]
                video_w = torch.stack(
                    [video_w[a:b].to(torch.float64).sum(0) for a, b in runs]
                ).to(torch.float32).contiguous()
                video_b = torch.stack(
                    [video_b[a:b].to(torch.float64).sum(0) for a, b in runs]
                ).to(torch.float32).contiguous()
                audio_w = torch.stack(
                    [audio_w[a:b].to(torch.float64).sum(0) for a, b in runs]
                ).to(torch.float32).contiguous()
                audio_b = torch.stack(
                    [audio_b[a:b].to(torch.float64).sum(0) for a, b in runs]
                ).to(torch.float32).contiguous()
            effective_nfe = len(knots) - 1

            spans = {right - left for left, right in zip(knots, knots[1:])}
            effective_block_size = spans.pop() if len(spans) == 1 else 0
            boundaries_video = [grid.sigmas_video[k] for k in knots]
            boundaries_audio = [grid.sigmas_audio[k] for k in knots]
            heads = PDDHeads(
                nfe=effective_nfe,
                num_intervals=num_intervals,
                block_size=effective_block_size,
                sigmas_video=boundaries_video,
                sigmas_audio=boundaries_audio,
                dsum_video=[left - right for left, right in zip(boundaries_video, boundaries_video[1:])],
                dsum_audio=[left - right for left, right in zip(boundaries_audio, boundaries_audio[1:])],
                video_w=video_w,
                video_b=video_b,
                audio_w=audio_w,
                audio_b=audio_b,
                metadata={
                    "format": artifact_format,
                    "completed_update": completed_update,
                    "grid_sha256": expected_grid_hash,
                    "teacher_adapter_a_sha256": teacher_hash,
                    "source_checkpoint_manifest_sha256": source_hash,
                    "diagnostic_only": str(metadata.get("diagnostic_only", "false")).lower() == "true"
                    or knots != list(range(0, num_intervals + 1, num_intervals // 4)),
                    "heads_name": heads_name,
                    "source_nfe": nfe,
                    "partition_knots": knots,
                },
            )
            return io.NodeOutput(heads)

        @classmethod
        def _execute_bank(cls, heads_name, state, metadata, blocks, partition):
            """Fuse a raw per-interval head bank (velocity units) on demand.

            Banks hold the complete training-side artifact: one head per grid
            interval, stored as four stacked tensors. Fusion delta-weights each
            interval head by its own per-modality sigma span, so any contiguous
            partition of the grid is derivable. Launch states were only trained
            at multiples of 64, so partitions with other cuts are diagnostic.
            """

            num_intervals = int(state["video_weight"].shape[0])
            if num_intervals != 256:
                raise ValueError(
                    f"PDD interval banks must hold 256 heads, got {num_intervals}"
                )
            expected_shapes = {
                "video_weight": (96, 5376),
                "video_bias": (96,),
                "audio_weight": (32, 5376),
                "audio_bias": (32,),
            }
            for key, tail in expected_shapes.items():
                tensor = state[key]
                if tuple(tensor.shape) != (num_intervals, *tail) or tensor.dtype != torch.float32:
                    raise ValueError(
                        f"PDD bank tensor {key} has shape/dtype "
                        f"{tuple(tensor.shape)}/{tensor.dtype}, expected "
                        f"{(num_intervals, *tail)}/float32"
                    )
            # Banks from the current trainer are always the L=64 lineage; their
            # embedded hash is the block_size-64 canonical grid hash.
            grid = build_grid(num_intervals, 64)
            expected_grid_hash = _validate_hash(metadata, "grid_sha256")
            if grid.grid_sha256 != expected_grid_hash:
                raise ValueError(
                    "PDD bank grid_sha256 mismatch: metadata has "
                    f"{expected_grid_hash}, recomputed {grid.grid_sha256}."
                )
            try:
                completed_update = int(metadata.get("completed_update"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "PDD bank metadata 'completed_update' must be an integer string"
                ) from exc
            a_value = metadata.get("a_sha256")
            if a_value in (None, "", "none", "None", "null"):
                teacher_hash = "none"
            elif _is_hash(a_value):
                teacher_hash = a_value
            else:
                raise ValueError(
                    f"PDD bank metadata 'a_sha256' must be a SHA-256 or absent, got {a_value!r}"
                )

            knots = parse_partition_knots(partition, num_intervals)
            if knots is not None and blocks:
                raise ValueError("Set either blocks or partition, not both")
            if knots is None:
                # Default a bank to the production 4-call fusion; 0 would mean a
                # surprising 256-call schedule.
                chosen = blocks or 4
                if chosen < 1 or chosen > num_intervals:
                    raise ValueError(
                        f"blocks={chosen} must be within 1..{num_intervals} for a bank"
                    )
                if num_intervals % chosen == 0:
                    # Divisors keep the uniform meaning established by the
                    # shipped 4/8/16 exports.
                    knots = list(range(0, num_intervals + 1, num_intervals // chosen))
                else:
                    # Any other integer derives an anchored partition: trained
                    # launch knots preserved, extra cuts split the largest
                    # loss-weighted sigma spans (the late blocks).
                    knots = auto_partition_knots(grid, chosen)

            deltas_video = torch.tensor(grid.deltas_video, dtype=torch.float64)
            deltas_audio = torch.tensor(grid.deltas_audio, dtype=torch.float64)

            def fuse(weight_key, bias_key, deltas):
                weights, biases = [], []
                for left, right in zip(knots, knots[1:]):
                    span = deltas[left:right]
                    weights.append(
                        (state[weight_key][left:right].to(torch.float64) * span.view(-1, 1, 1))
                        .sum(0)
                        .to(torch.float32)
                    )
                    biases.append(
                        (state[bias_key][left:right].to(torch.float64) * span.view(-1, 1))
                        .sum(0)
                        .to(torch.float32)
                    )
                return (
                    torch.stack(weights).contiguous(),
                    torch.stack(biases).contiguous(),
                )

            video_w, video_b = fuse("video_weight", "video_bias", deltas_video)
            audio_w, audio_b = fuse("audio_weight", "audio_bias", deltas_audio)

            spans = {right - left for left, right in zip(knots, knots[1:])}
            boundaries_video = [grid.sigmas_video[k] for k in knots]
            boundaries_audio = [grid.sigmas_audio[k] for k in knots]
            heads = PDDHeads(
                nfe=len(knots) - 1,
                num_intervals=num_intervals,
                block_size=spans.pop() if len(spans) == 1 else 0,
                sigmas_video=boundaries_video,
                sigmas_audio=boundaries_audio,
                dsum_video=[left - right for left, right in zip(boundaries_video, boundaries_video[1:])],
                dsum_audio=[left - right for left, right in zip(boundaries_audio, boundaries_audio[1:])],
                video_w=video_w,
                video_b=video_b,
                audio_w=audio_w,
                audio_b=audio_b,
                metadata={
                    "format": "minimax_h3_ref2va_pdd_interval_bank_v1",
                    "completed_update": completed_update,
                    "grid_sha256": expected_grid_hash,
                    "teacher_adapter_a_sha256": teacher_hash,
                    "source_checkpoint_manifest_sha256": metadata.get(
                        "manifest_sha256", "unknown"
                    ),
                    "diagnostic_only": knots != list(range(0, num_intervals + 1, 64)),
                    "heads_name": heads_name,
                    "source_nfe": num_intervals,
                    "partition_knots": knots,
                },
            )
            return io.NodeOutput(heads)

    class MiniMaxH3PDDModelPatch(io.ComfyNode):
        """Replace the native final projections with block-selected PDD heads."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="MiniMaxH3PDDModelPatch",
                display_name="MiniMax H3 PDD Model Patch",
                category="model/patch/minimax",
                description=(
                    "Patch MiniMax H3 with fused PDD displacement heads. Exact mode is "
                    "trained for Euler; block velocity is an experimental compatibility mode."
                ),
                inputs=[
                    io.Model.Input("model"),
                    pdd_type.Input("pdd_heads"),
                    io.Combo.Input(
                        "mode",
                        options=["exact_euler_step", "block_velocity"],
                        default="exact_euler_step",
                    ),
                    io.Combo.Input(
                        "on_out_of_grid",
                        options=["clamp", "error"],
                        default="clamp",
                    ),
                    io.Float.Input(
                        "head_strength",
                        default=1.0,
                        min=0.0,
                        max=10.0,
                        step=0.05,
                        round=False,
                        optional=True,
                        tooltip=(
                            "Scale only the learned PDD-head correction relative to "
                            "the native H3 output head: native + strength * (PDD - native). "
                            "1.0 is the trained PDD path; values other than 1.0 are experimental."
                        ),
                    ),
                ],
                outputs=[io.Model.Output()],
            )

        @classmethod
        def execute(
            cls,
            model,
            pdd_heads,
            mode="exact_euler_step",
            on_out_of_grid="clamp",
            head_strength=1.0,
        ):
            if not isinstance(pdd_heads, PDDHeads):
                raise TypeError("pdd_heads is not a validated PDD_HEADS object")
            if mode not in {"exact_euler_step", "block_velocity"}:
                raise ValueError(f"Unsupported PDD mode {mode!r}")
            head_strength = float(head_strength)
            if not math.isfinite(head_strength) or not 0.0 <= head_strength <= 10.0:
                raise ValueError(
                    f"PDD head_strength must be finite and between 0 and 10, got {head_strength!r}"
                )

            model_clone = model.clone()
            diffusion_model = model_clone.get_model_object("diffusion_model")
            if not isinstance(diffusion_model, MiniMaxH3Model):
                raise RuntimeError("MiniMaxH3PDDModelPatch can only patch a MiniMax H3 model")

            holder = SimpleNamespace(sigma_v=None, transformer_options=None)

            def diffusion_wrapper(executor, *args, **kwargs):
                timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
                transformer_options = (
                    args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
                )
                if timestep is None:
                    raise RuntimeError("PDD diffusion wrapper did not receive a timestep")
                transformer_options = transformer_options or {}
                shift_video = float(
                    transformer_options.get(
                        "minimax_h3_sigma_shift_video", diffusion_model.sigma_shift_video
                    )
                )
                shift_audio = float(
                    transformer_options.get(
                        "minimax_h3_sigma_shift_audio", diffusion_model.sigma_shift_audio
                    )
                )
                if not (
                    math.isclose(shift_video, VIDEO_SHIFT, rel_tol=0.0, abs_tol=1.0e-9)
                    and math.isclose(shift_audio, AUDIO_SHIFT, rel_tol=0.0, abs_tol=1.0e-9)
                ):
                    raise ValueError(
                        "MiniMax H3 PDD heads require MiniMaxH3SigmaShift values "
                        f"12.0/3.0, got {shift_video}/{shift_audio}"
                    )
                holder.sigma_v = (timestep.flatten()[0] / 1000.0).float()
                holder.transformer_options = transformer_options
                try:
                    return executor(*args, **kwargs)
                finally:
                    holder.sigma_v = None
                    holder.transformer_options = None

            def pdd_final_forward(self, x, t_emb, video_seg, audio_seg):
                if holder.sigma_v is None or holder.transformer_options is None:
                    raise RuntimeError(
                        "PDD final-layer patch was invoked without active diffusion-call state"
                    )

                # This is intentionally line-for-line equivalent to H3 FinalLayer.forward
                # through the fp32 casts; only the two projections are replaced.
                shift, scale = self.adaln_proj(t_emb)
                va, vb, vrow = video_seg
                aa, ab, arow = audio_seg
                hv = (
                    self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]
                ).to(torch.float32)
                ha = (
                    self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]
                ).to(torch.float32)

                sigma_float = float(holder.sigma_v.detach().item())
                block = select_block(
                    sigma_float,
                    pdd_heads.sigmas_video,
                    on_out_of_grid=on_out_of_grid,
                    tolerance=BLOCK_TOLERANCE,
                )
                video_w, video_b, audio_w, audio_b = pdd_heads.for_device(hv.device)
                displacement_video = functional.linear(hv, video_w[block], video_b[block])
                displacement_audio = functional.linear(ha, audio_w[block], audio_b[block])

                def blend_with_native(pdd_video_velocity, pdd_audio_velocity):
                    # The exported PDD projections are complete block velocities,
                    # not additive residuals.  Scaling them directly would scale the
                    # entire Euler displacement.  Interpolate/extrapolate only their
                    # learned correction from the native H3 velocity head instead.
                    if head_strength == 1.0:
                        # Preserve the validated production path bit-for-bit and
                        # avoid needless native-head evaluation at the default.
                        return pdd_video_velocity, pdd_audio_velocity
                    native_video_velocity = self.video_out(hv)
                    native_audio_velocity = self.audio_out(ha)
                    if head_strength == 0.0:
                        return native_video_velocity, native_audio_velocity
                    return (
                        native_video_velocity
                        + head_strength * (pdd_video_velocity - native_video_velocity),
                        native_audio_velocity
                        + head_strength * (pdd_audio_velocity - native_audio_velocity),
                    )

                if mode == "exact_euler_step":
                    sample_sigmas = holder.transformer_options.get("sample_sigmas")
                    if sample_sigmas is not None:
                        flat = sample_sigmas.flatten()
                        if flat.numel() >= 2:
                            matches = (flat[:-1] - holder.sigma_v.to(flat.device)).abs() <= BLOCK_TOLERANCE
                            match_indices = matches.nonzero(as_tuple=False)
                            if match_indices.numel():
                                index = int(match_indices[0].item())
                                dsig = flat[index] - flat[index + 1]
                                if float(dsig.detach().item()) > 0.0:
                                    # Guard: each exact-mode sampler step must span
                                    # exactly one trained block, else the same block
                                    # displacement fires on several sub-steps (or a
                                    # step swallows several blocks) and the output
                                    # silently overshoots or undershoots.
                                    bounds = pdd_heads.sigmas_video
                                    next_sigma = float(flat[index + 1].detach().item())
                                    nearest = min(
                                        range(len(bounds) - 1),
                                        key=lambda i: abs(sigma_float - bounds[i]),
                                    )
                                    if (
                                        abs(sigma_float - bounds[nearest]) > BLOCK_TOLERANCE
                                        or abs(next_sigma - bounds[nearest + 1]) > BLOCK_TOLERANCE
                                    ):
                                        raise ValueError(
                                            "exact_euler_step needs every sampler step to "
                                            "span exactly one trained block; step "
                                            f"{sigma_float:.6g} -> {next_sigma:.6g} does not "
                                            "match consecutive artifact boundaries "
                                            f"[{', '.join(f'{v:.6g}' for v in bounds)}]. Use "
                                            "the PDD scheduler in trained_blocks mode (or a "
                                            "matching blocks/partition on the loader), or "
                                            "switch the patch to block_velocity for "
                                            "arbitrary schedules."
                                        )
                                    dsig_video = dsig.to(
                                        device=displacement_video.device,
                                        dtype=displacement_video.dtype,
                                    )
                                    slope_audio = time_shift_slope(
                                        holder.sigma_v.to(displacement_audio.device),
                                        VIDEO_SHIFT,
                                        AUDIO_SHIFT,
                                    ).to(displacement_audio.dtype)
                                    return blend_with_native(
                                        displacement_video / dsig_video,
                                        displacement_audio / (dsig_video * slope_audio),
                                    )

                # Exact mode deliberately falls back here for inner evaluations from
                # multi-evaluation samplers or for missing/non-descending schedules.
                return blend_with_native(
                    displacement_video / pdd_heads.dsum_video[block],
                    displacement_audio / pdd_heads.dsum_audio[block],
                )

            final_layer = diffusion_model.final_layer
            bound_forward = pdd_final_forward.__get__(final_layer, final_layer.__class__)
            if hasattr(model_clone, "remove_wrappers_with_key"):
                model_clone.remove_wrappers_with_key(
                    comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                    "minimax_h3_pdd",
                )
            model_clone.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                "minimax_h3_pdd",
                diffusion_wrapper,
            )
            model_clone.add_object_patch("diffusion_model.final_layer.forward", bound_forward)
            return io.NodeOutput(model_clone)

    class MiniMaxH3PDDScheduler(io.ComfyNode):
        """Emit trained PDD block boundaries or rounded full-grid subsamples."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="MiniMaxH3PDDScheduler",
                display_name="MiniMax H3 PDD Scheduler",
                category="sampling/custom_sampling/schedulers",
                description=(
                    "PDD sigma schedule. Partial denoise is outside the artifact's "
                    "trained full-trajectory distribution."
                ),
                inputs=[
                    pdd_type.Input("pdd_heads"),
                    io.Combo.Input(
                        "mode",
                        options=["trained_blocks", "subsampled_grid"],
                        default="trained_blocks",
                    ),
                    io.Int.Input("steps", default=4, min=1, max=256),
                    io.Float.Input("denoise", default=1.0, min=0.01, max=1.0, step=0.01),
                ],
                outputs=[io.Sigmas.Output()],
            )

        @classmethod
        def execute(cls, pdd_heads, mode="trained_blocks", steps=4, denoise=1.0):
            if not isinstance(pdd_heads, PDDHeads):
                raise TypeError("pdd_heads is not a validated PDD_HEADS object")
            if mode == "trained_blocks":
                # Read boundaries from the artifact itself so non-uniform
                # partitions schedule correctly; identical to the grid-derived
                # values for uniform artifacts.
                denoise = float(denoise)
                if not 0.0 < denoise <= 1.0:
                    raise ValueError("denoise must be greater than 0 and at most 1")
                values = list(pdd_heads.sigmas_video)
                if denoise < 1.0:
                    kept_steps = int(round(pdd_heads.nfe * denoise))
                    values = values[-(kept_steps + 1):]
                if values[-1] != 0.0 or any(
                    left <= right for left, right in zip(values, values[1:])
                ):
                    raise RuntimeError(
                        "PDD scheduler result must descend strictly to exactly zero"
                    )
            else:
                if pdd_heads.block_size == 0:
                    raise ValueError(
                        "subsampled_grid needs a uniform-block artifact; this "
                        "PDD_HEADS object carries a non-uniform partition "
                        f"{pdd_heads.metadata.get('partition_knots')}"
                    )
                values = scheduler_sigmas(
                    pdd_heads.num_intervals,
                    pdd_heads.block_size,
                    mode,
                    steps,
                    denoise,
                )
            result = torch.tensor(values, dtype=torch.float32, device="cpu")
            result[-1] = 0.0
            return io.NodeOutput(result)

        get_sigmas = execute

    class MiniMaxH3PDDArtifactInfo(io.ComfyNode):
        """Expose artifact identity and schedule provenance as readable text."""

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="MiniMaxH3PDDArtifactInfo",
                display_name="MiniMax H3 PDD Artifact Info",
                category="utils",
                inputs=[pdd_type.Input("pdd_heads")],
                outputs=[io.String.Output(display_name="artifact_info")],
            )

        @classmethod
        def execute(cls, pdd_heads):
            if not isinstance(pdd_heads, PDDHeads):
                raise TypeError("pdd_heads is not a validated PDD_HEADS object")
            metadata = pdd_heads.metadata
            completed_update = metadata.get("completed_update", "unknown")
            lines = [
                f"format: {metadata.get('format', 'unknown')}",
                f"nfe: {pdd_heads.nfe}",
                f"source_file_nfe: {metadata.get('source_nfe', pdd_heads.nfe)}",
                f"partition_knots: {metadata.get('partition_knots', 'uniform')}",
                f"step: {completed_update}",
                f"completed_update: {completed_update}",
                f"grid_sha256: {metadata.get('grid_sha256', 'unknown')}",
                "source_checkpoint_manifest_sha256: "
                f"{metadata.get('source_checkpoint_manifest_sha256', 'unknown')}",
                f"teacher_adapter_a_sha256: {metadata.get('teacher_adapter_a_sha256', 'unknown')}",
                f"diagnostic_only: {metadata.get('diagnostic_only', False)}",
                "sigma_video_boundaries: "
                + ", ".join(f"{value:.12g}" for value in pdd_heads.sigmas_video),
                "sigma_audio_boundaries: "
                + ", ".join(f"{value:.12g}" for value in pdd_heads.sigmas_audio),
            ]
            return io.NodeOutput("\n".join(lines))

    class MiniMaxH3PDDExtension(ComfyExtension):
        async def get_node_list(self):
            return [
                MiniMaxH3PDDHeadsLoader,
                MiniMaxH3PDDModelPatch,
                MiniMaxH3PDDScheduler,
                MiniMaxH3PDDArtifactInfo,
            ]

    return MiniMaxH3PDDExtension()
