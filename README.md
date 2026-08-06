# ComfyUI MiniMax H3 PDD

Experimental ComfyUI nodes for running compatible MiniMax H3 Ref2VA PDD-Euler displacement-head artifacts. This repository contains inference integration only. It does not include model weights, LoRA weights, PDD heads, training code, or example media.

## Nodes

- **MiniMax H3 PDD Heads Loader** — loads a fused PDD artifact or raw 256-interval head bank, validates its tensors and metadata, and optionally derives a compatible partition.
- **MiniMax H3 PDD Model Patch** — patches the MiniMax H3 final video/audio projections with the selected PDD heads.
- **MiniMax H3 PDD Scheduler** — produces the artifact's trained block boundaries or a diagnostic grid subsample.
- **MiniMax H3 PDD Artifact Info** — reports the loaded artifact's schedule and provenance metadata.

## Requirements

- A current ComfyUI version with MiniMax H3 and the V3 extension API.
- A compatible MiniMax H3 model and its normal text/video/audio components.
- Matching PDD head and student-adapter artifacts produced for this inference path.

There are no additional Python package requirements beyond the dependencies already used by ComfyUI.

## Installation

Clone this repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mamad8c/ComfyUI-MiniMaxH3-PDD-Mamad8.git
```

Restart ComfyUI after installation or updates.

Place PDD head artifacts in:

```text
ComfyUI/models/pdd_heads/
```

## Basic use

1. Load the MiniMax H3 model and apply the matching student LoRA S through the normal ComfyUI model path.
2. Load the matching PDD head artifact with **MiniMax H3 PDD Heads Loader**.
3. Connect the model and heads to **MiniMax H3 PDD Model Patch**.
4. Connect the same heads to **MiniMax H3 PDD Scheduler** and pass its sigmas to the sampler.
5. For the trained path, use `exact_euler_step`, scheduler mode `trained_blocks`, `denoise=1.0`, and MiniMax H3 sigma shifts `12.0` for video and `3.0` for audio.

The loader's `blocks` option can exactly combine consecutive displacement heads when the requested count is reachable from the artifact. A raw interval bank can derive arbitrary contiguous grid partitions. Partitions that introduce launch points outside the trained block starts are marked diagnostic.

## Important behavior

- The default `head_strength=1.0` is the native trained PDD path. Other strengths are experimental extrapolations relative to the original H3 output head.
- `exact_euler_step` fails closed when sampler steps do not match consecutive artifact boundaries. Use the PDD scheduler for the intended trajectory.
- Partial denoising and arbitrary schedules are outside the trained full-trajectory distribution.
- Deployment artifacts must explicitly declare that Stage A is not included. The loader rejects incompatible metadata.
- Artifacts are validated for format, tensor keys, dimensions, FP32 head dtype, schedule consistency, and provenance hashes before use.

## Scope and status

This is experimental research software. Compatibility depends on the MiniMax H3 implementation in ComfyUI and on correctly matched external model artifacts. No official model weights or training outputs are distributed here.

This project is not affiliated with or endorsed by MiniMax or the ComfyUI project.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
