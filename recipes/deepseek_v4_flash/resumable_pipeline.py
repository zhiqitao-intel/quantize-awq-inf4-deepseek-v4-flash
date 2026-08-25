"""Distributed, layer-checkpointed llm-compressor sequential pipeline.

The distributed modifier keeps a complete calibration objective on every rank,
partitions layer-wide grid points and independent expert mappings, and
broadcasts the winning scales. llm-compressor's stock sequential pipeline does
not persist completed layers. This module keeps its order of operations and
adds atomic checkpoints at decoder-layer boundaries.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from compressed_tensors.offload import disable_offloading, set_onload_device
from compressed_tensors.utils import update_offload_parameter
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from llmcompressor.core import LifecycleCallbacks, active_session
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pipelines.registry import CalibrationPipeline
from llmcompressor.pipelines.sequential.helpers import trace_subgraphs
from llmcompressor.utils.dev import get_main_device
from llmcompressor.utils.helpers import DisableQuantization, calibration_forward_context
from llmcompressor.utils.pytorch.module import infer_sequential_targets


_LAYER_NUMBER = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class ResumeConfig:
    checkpoint_dir: Path
    identity: dict[str, Any]
    layer_class: str
    stop_after_layer: int | None = None


_CONFIG: ResumeConfig | None = None


def configure(
    checkpoint_dir: str,
    identity: dict[str, Any],
    layer_class: str,
    stop_after_layer: int | None = None,
) -> None:
    """Configure the process-local pipeline before calling ``oneshot``."""
    global _CONFIG
    _CONFIG = ResumeConfig(
        checkpoint_dir=Path(checkpoint_dir),
        identity=identity,
        layer_class=layer_class,
        stop_after_layer=stop_after_layer,
    )


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _prepare_manifest(config: ResumeConfig) -> dict[str, Any]:
    """Create or validate the run identity on rank zero, then synchronize."""
    manifest_path = config.checkpoint_dir / "progress.json"
    if _rank() == 0:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest = json.load(open(manifest_path))
            if manifest.get("identity") != config.identity:
                raise RuntimeError(
                    "checkpoint identity does not match this run; use a new "
                    f"checkpoint directory or restore the original arguments: {manifest_path}"
                )
        else:
            manifest = {
                "format_version": 1,
                "identity": config.identity,
                "completed_layers": [],
            }
            _atomic_json(manifest_path, manifest)
    _barrier()
    return json.load(open(manifest_path))


def _root_for_subgraph(model: torch.nn.Module, subgraph, layer_class: str):
    module_names = {id(module): name for name, module in model.named_modules()}
    roots = [
        module
        for module in subgraph.submodules(model, recurse=False)
        if type(module).__name__ == layer_class
    ]
    if not roots:
        return None
    if len(roots) != 1:
        names = [module_names.get(id(module), "<unknown>") for module in roots]
        raise RuntimeError(f"expected one {layer_class} in subgraph, got {names}")
    root = roots[0]
    root_name = module_names[id(root)]
    match = _LAYER_NUMBER.search(root_name)
    if match is None:
        raise RuntimeError(f"cannot derive layer number from {root_name}")
    return int(match.group(1)), root_name, root


def _checkpoint_paths(config: ResumeConfig, layer_number: int) -> tuple[Path, Path]:
    stem = config.checkpoint_dir / f"layer_{layer_number:03d}"
    return stem.with_suffix(".safetensors"), stem.with_suffix(".json")


def _selected_state(root: torch.nn.Module, root_name: str) -> dict[str, torch.Tensor]:
    """Capture every tensor changed by AWQ or needed for final weight packing."""
    session = active_session()
    awq = next(
        (
            modifier
            for modifier in session.lifecycle.recipe.modifiers
            if isinstance(modifier, AWQModifier)
        ),
        None,
    )

    selected_ids: set[int] = set()
    if awq is not None:
        for mapping in awq._resolved_mappings:
            if mapping.smooth_name == root_name or mapping.smooth_name.startswith(
                root_name + "."
            ):
                selected_ids.add(id(mapping.smooth_layer))
                selected_ids.update(id(module) for module in mapping.balance_layers)

    # Quantized modules need their transformed weight plus scale/zero point.
    for module in root.modules():
        if hasattr(module, "quantization_scheme"):
            selected_ids.add(id(module))

    tensors: dict[str, torch.Tensor] = {}
    for relative_name, module in root.named_modules():
        if id(module) not in selected_ids:
            continue
        prefix = relative_name + "." if relative_name else ""
        for name, tensor in module.named_parameters(recurse=False):
            tensors[prefix + name] = tensor.detach().cpu().contiguous().clone()
        for name, tensor in module.named_buffers(recurse=False):
            tensors[prefix + name] = tensor.detach().cpu().contiguous().clone()

    if not tensors:
        raise RuntimeError(f"no resumable state selected for {root_name}")
    return tensors


def _save_checkpoint(
    config: ResumeConfig,
    layer_number: int,
    root_name: str,
    root: torch.nn.Module,
) -> None:
    """Write data then metadata then manifest, each by atomic rename."""
    data_path, metadata_path = _checkpoint_paths(config, layer_number)
    if _rank() == 0:
        tensors = _selected_state(root, root_name)
        tmp_data = data_path.with_suffix(data_path.suffix + ".tmp")
        save_file(tensors, str(tmp_data), metadata={"format": "pt"})
        os.replace(tmp_data, data_path)
        _atomic_json(
            metadata_path,
            {
                "layer": layer_number,
                "root_name": root_name,
                "tensor_count": len(tensors),
                "data_file": data_path.name,
                "data_bytes": data_path.stat().st_size,
            },
        )

        manifest_path = config.checkpoint_dir / "progress.json"
        manifest = json.load(open(manifest_path))
        completed = sorted(set(manifest.get("completed_layers", [])) | {layer_number})
        manifest["completed_layers"] = completed
        _atomic_json(manifest_path, manifest)
        print(
            f"checkpointed layer {layer_number}: {len(tensors)} tensors, "
            f"{data_path.stat().st_size / 2**30:.2f} GiB"
        )
    _barrier()


def _restore_checkpoint(
    config: ResumeConfig,
    layer_number: int,
    root_name: str,
    root: torch.nn.Module,
) -> None:
    data_path, metadata_path = _checkpoint_paths(config, layer_number)
    if not data_path.exists() or not metadata_path.exists():
        raise RuntimeError(f"manifest names layer {layer_number}, but checkpoint is missing")
    metadata = json.load(open(metadata_path))
    if metadata.get("root_name") != root_name:
        raise RuntimeError(
            f"checkpoint root {metadata.get('root_name')} does not match {root_name}"
        )
    if data_path.stat().st_size != metadata.get("data_bytes"):
        raise RuntimeError(f"checkpoint size mismatch: {data_path}")

    with safe_open(data_path, framework="pt", device="cpu") as handle:
        for relative_key in handle.keys():
            if "." in relative_key:
                module_name, attribute = relative_key.rsplit(".", 1)
                module = root.get_submodule(module_name)
            else:
                module, attribute = root, relative_key
            if not hasattr(module, attribute):
                raise RuntimeError(
                    f"checkpoint tensor {root_name}.{relative_key} is absent in model"
                )
            update_offload_parameter(module, attribute, handle.get_tensor(relative_key))
    print(f"rank {_rank()} restored completed layer {layer_number}")


def _batches(activations, num_batches, input_names, desc, prefetch):
    source = activations.iter_prefetch(input_names) if prefetch else activations.iter(input_names)
    yield from tqdm(enumerate(source), total=num_batches, desc=desc)


@CalibrationPipeline.register("resumable_sequential")
class ResumableSequentialPipeline(CalibrationPipeline):
    """Stock sequential semantics with per-decoder-layer durable checkpoints."""

    @staticmethod
    def __call__(model: torch.nn.Module, dataloader, dataset_args) -> None:
        if _CONFIG is None:
            raise RuntimeError("resumable pipeline was not configured")
        config = _CONFIG
        manifest = _prepare_manifest(config)
        completed = set(manifest.get("completed_layers", []))

        session = active_session()
        onload_device = get_main_device()
        offload_device = torch.device(dataset_args.sequential_offload_device)
        set_onload_device(model, onload_device)

        targets = infer_sequential_targets(model, dataset_args.sequential_targets)
        sample_input = next(iter(dataloader))
        subgraphs = trace_subgraphs(
            model,
            sample_input,
            targets,
            dataset_args.tracing_ignore,
            dataset_args.sequential_targets_per_subgraph,
        )
        LifecycleCallbacks.calibration_start()

        # If every decoder layer already has a durable checkpoint, the model is
        # fully quantized.  Replaying 512 samples through every restored layer is
        # only needed to feed a later *uncompleted* layer; it cannot change any
        # weights or AWQ statistics (resume replay runs with hooks disabled).  A
        # no-op resume would otherwise spend another hour and can also trigger
        # llm-compressor's empty-metrics finalization path.  Restore all layer
        # states in order, finish the lifecycle, and let rank zero serialize.
        checkpoint_layers = set()
        for subgraph in subgraphs:
            root_info = _root_for_subgraph(model, subgraph, config.layer_class)
            if root_info is not None:
                checkpoint_layers.add(root_info[0])
        if checkpoint_layers and checkpoint_layers.issubset(completed):
            for subgraph in subgraphs:
                root_info = _root_for_subgraph(model, subgraph, config.layer_class)
                if root_info is not None:
                    layer_number, root_name, root = root_info
                    _restore_checkpoint(config, layer_number, root_name, root)
            LifecycleCallbacks.calibration_end()
            print(
                f"all {len(checkpoint_layers)} decoder layers restored; "
                "skipping redundant activation replay"
            )
            return

        with contextlib.ExitStack() as stack:
            stack.enter_context(calibration_forward_context(model))
            stack.enter_context(DisableQuantization(model))
            activations = IntermediatesCache.from_dataloader(
                dataloader, onload_device, offload_device
            )
            session.state.loss_masks = None
            prefetch = getattr(dataset_args, "sequential_prefetch", False)
            session.state.sequential_prefetch = prefetch

            for subgraph_index, subgraph in enumerate(subgraphs):
                root_info = _root_for_subgraph(model, subgraph, config.layer_class)
                layer_number = root_info[0] if root_info else None
                num_batches = len(dataloader)

                if layer_number in completed:
                    _, root_name, root = root_info
                    _restore_checkpoint(config, layer_number, root_name, root)
                    with disable_offloading(), HooksMixin.disable_hooks():
                        for batch_idx, inputs in _batches(
                            activations,
                            num_batches,
                            subgraph.input_names,
                            f"({subgraph_index + 1}/{len(subgraphs)}): Resuming",
                            prefetch,
                        ):
                            outputs = subgraph.forward(model, **inputs)
                            if subgraph_index < len(subgraphs) - 1:
                                activations.update(batch_idx, outputs)
                                activations.delete(batch_idx, subgraph.consumed_names)
                    continue

                with disable_offloading():
                    for batch_idx, inputs in _batches(
                        activations,
                        num_batches,
                        subgraph.input_names,
                        f"({subgraph_index + 1}/{len(subgraphs)}): Calibrating",
                        prefetch,
                    ):
                        session.state.current_batch_idx = batch_idx
                        subgraph.forward(model, **inputs)

                    LifecycleCallbacks.sequential_epoch_end(subgraph.submodules(model))

                    if root_info is not None:
                        layer_number, root_name, root = root_info
                        _save_checkpoint(config, layer_number, root_name, root)
                        completed.add(layer_number)

                    with HooksMixin.disable_hooks():
                        for batch_idx, inputs in _batches(
                            activations,
                            num_batches,
                            subgraph.input_names,
                            f"({subgraph_index + 1}/{len(subgraphs)}): Propagating",
                            prefetch,
                        ):
                            outputs = subgraph.forward(model, **inputs)
                            if subgraph_index < len(subgraphs) - 1:
                                activations.update(batch_idx, outputs)
                                activations.delete(batch_idx, subgraph.consumed_names)

                if (
                    config.stop_after_layer is not None
                    and layer_number is not None
                    and layer_number >= config.stop_after_layer
                ):
                    raise RuntimeError(
                        f"intentional stop after checkpointing layer {layer_number}"
                    )

            LifecycleCallbacks.calibration_end()
