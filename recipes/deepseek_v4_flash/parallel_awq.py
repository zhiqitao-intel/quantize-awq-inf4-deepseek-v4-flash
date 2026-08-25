"""True multi-XPU AWQ search parallelism for replicated calibration data."""

from __future__ import annotations

import contextlib
import re

import torch
import torch.distributed as dist
from pydantic import PrivateAttr

import llmcompressor.modifiers.transform.awq.base as awq_base
from llmcompressor.modifiers.transform.awq import AWQModifier


_EXPERT = re.compile(r"\.experts\.(\d+)\.")


class ParallelAWQModifier(AWQModifier):
    """Partition AWQ's grid and independent expert mappings across ranks.

    Every rank receives the complete calibration set. For a mapping shared by
    the whole MoE layer (post-attention norm -> expert gate/up), ranks evaluate
    disjoint grid points and select the global minimum. For an expert-local
    mapping (up -> down), expert ``N`` is searched only by rank ``N % world``.
    Expert owners search concurrently, winning scales are exchanged in one
    collective, and every replica applies them in stock mapping order. This
    leaves all models identical for error propagation and checkpointing.
    """

    _grid_override: list[tuple[float, bool]] | None = PrivateAttr(default=None)

    def _log_error_metrics(self) -> None:
        """Allow a fully resumed run to finalize without fresh AWQ metrics.

        llm-compressor's stock logger assumes at least one mapping was
        searched in the current process.  When every decoder layer is restored
        from a durable checkpoint, the transform is still valid but the list
        is intentionally empty; the stock ``sum(...) / len(...)`` would abort
        the otherwise complete quantization.
        """
        if not self._error_metrics:
            print("AWQ metrics: all mappings restored from checkpoints")
            return
        super()._log_error_metrics()

    def _get_grid_search_params(self) -> list[tuple[float, bool]]:
        if self._grid_override is not None:
            return self._grid_override
        return super()._get_grid_search_params()

    @contextlib.contextmanager
    def _local_objective(self):
        # Each participating rank holds the complete calibration set, so AWQ's
        # normal data-parallel all-reduces would duplicate identical loss here.
        original = awq_base.is_distributed
        awq_base.is_distributed = lambda: False
        try:
            yield
        finally:
            awq_base.is_distributed = original

    def _serial_best_scale(self, mapping, fp16_outputs, orig_layer_weights):
        with self._local_objective():
            return super()._compute_best_scale(
                mapping, fp16_outputs, orig_layer_weights
            )

    @staticmethod
    def _valid_outputs(fp16_outputs) -> bool:
        return bool(
            fp16_outputs
            and any(output.numel() for output in fp16_outputs)
            and all(output.isfinite().all() for output in fp16_outputs)
        )

    @staticmethod
    def _restore_balance_weights(mapping, orig_layer_weights) -> None:
        for layer in mapping.balance_layers:
            awq_base.update_offload_parameter(
                layer,
                "weight",
                orig_layer_weights[layer].to(layer.weight.device),
            )

    @staticmethod
    def _apply_scale(mapping, best_scales, orig_layer_weights) -> None:
        smooth_layer = mapping.smooth_layer
        balance_layers = mapping.balance_layers

        def smooth(module):
            scales = best_scales.to(module.weight.device)
            if module in balance_layers:
                awq_base.update_offload_parameter(
                    module,
                    "weight",
                    orig_layer_weights[module].to(module.weight.device)
                    * scales.view(1, -1),
                )
            elif module == smooth_layer:
                if module.weight.ndim == 1:
                    awq_base.update_offload_parameter(
                        module, "weight", module.weight.div_(scales)
                    )
                else:
                    weight = module.weight
                    weight[-scales.size(0) :].div_(scales.view(-1, 1))
                    awq_base.update_offload_parameter(module, "weight", weight)
                if hasattr(module, "bias") and module.bias is not None:
                    awq_base.update_offload_parameter(
                        module, "bias", module.bias.div_(scales)
                    )

        for layer in balance_layers:
            smooth(layer)
        smooth(smooth_layer)

    def _cleanup_mapping(self, mapping) -> None:
        self._smooth_activation_stats.pop(mapping.smooth_name, None)
        self._parent_args_cache.pop(mapping.parent, None)

    def _process_shared_mapping(self, model, mapping) -> None:
        """Process one layer-wide mapping with grid candidates split by rank."""
        cache = self._parent_args_cache[mapping.parent]
        for batch_index in range(len(cache)):
            cache.pin_memory(batch_index)

        with (
            awq_base.align_modules(
                [mapping.parent, mapping.smooth_layer, *mapping.balance_layers]
            ),
            awq_base.calibration_forward_context(model),
            awq_base.HooksMixin.disable_hooks(),
        ):
            fp16_outputs = self._run_samples(mapping.parent)
            if not self._valid_outputs(fp16_outputs):
                self._cleanup_mapping(mapping)
                return
            orig_layer_weights = {
                layer: layer.weight.clone() for layer in mapping.balance_layers
            }
            best_scales = self._compute_best_scale(
                mapping, fp16_outputs, orig_layer_weights
            )
            self._apply_scale(mapping, best_scales, orig_layer_weights)
        self._cleanup_mapping(mapping)

    def _search_owned_expert(self, model, mapping):
        """Search one expert locally and restore its temporary grid weights."""
        cache = self._parent_args_cache[mapping.parent]
        for batch_index in range(len(cache)):
            cache.pin_memory(batch_index)

        with (
            awq_base.align_modules(
                [mapping.parent, mapping.smooth_layer, *mapping.balance_layers]
            ),
            awq_base.calibration_forward_context(model),
            awq_base.HooksMixin.disable_hooks(),
        ):
            fp16_outputs = self._run_samples(mapping.parent)
            if not self._valid_outputs(fp16_outputs):
                return None
            orig_layer_weights = {
                layer: layer.weight.clone() for layer in mapping.balance_layers
            }
            best_scales = self._serial_best_scale(
                mapping, fp16_outputs, orig_layer_weights
            )
            # Grid search leaves the last fake-quantized candidate resident.
            # Restore the original now; the globally exchanged winner is
            # applied to every replica only after all ranks finish searching.
            self._restore_balance_weights(mapping, orig_layer_weights)
            return best_scales

    @torch.no_grad()
    def _apply_smoothing(self, model) -> None:
        """Run independent expert objectives concurrently, then exchange once.

        A collective inside the per-expert loop would serialize the ranks: rank
        0 would search expert 0 while seven ranks waited, then rank 1 would
        search expert 1, and so on. Instead every rank first searches all 32 of
        its owned experts without collectives. One all-reduce exchanges the 256
        winning scale vectors, after which every replica applies them in stock
        mapping order and remains identical for propagation.
        """
        mappings = [
            mapping
            for mapping in self._resolved_mappings
            if mapping.smooth_name in self._smooth_activation_stats
        ]
        expert_mappings = []
        for mapping in mappings:
            if _EXPERT.search(mapping.smooth_name):
                expert_mappings.append(mapping)
            else:
                self._process_shared_mapping(model, mapping)

        if not expert_mappings:
            self._assert_all_activations_consumed()
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        device = torch.device("xpu", torch.xpu.current_device())
        scale_sizes = {
            self._smooth_activation_stats[mapping.smooth_name][0].numel()
            for mapping in expert_mappings
        }
        if len(scale_sizes) != 1:
            raise RuntimeError(
                f"expert mappings have different scale sizes: {sorted(scale_sizes)}"
            )
        scale_size = scale_sizes.pop()
        scales = torch.zeros(
            (len(expert_mappings), scale_size), dtype=torch.float32, device=device
        )
        available = torch.zeros(
            len(expert_mappings), dtype=torch.int32, device=device
        )

        owned = 0
        for position, mapping in enumerate(expert_mappings):
            expert = int(_EXPERT.search(mapping.smooth_name).group(1))
            if expert % world_size != rank:
                continue
            best = self._search_owned_expert(model, mapping)
            if best is not None:
                scales[position].copy_(best.to(device=device, dtype=torch.float32))
                available[position] = 1
            owned += 1
        print(
            f"rank {rank}: completed {owned} independent expert AWQ searches "
            f"before scale exchange"
        )

        if dist.is_initialized():
            dist.all_reduce(scales, op=dist.ReduceOp.SUM)
            dist.all_reduce(available, op=dist.ReduceOp.SUM)
        if not torch.all((available == 0) | (available == 1)):
            raise RuntimeError("an expert AWQ scale was produced by multiple ranks")

        for position, mapping in enumerate(expert_mappings):
            if not available[position].item():
                self._cleanup_mapping(mapping)
                continue
            with (
                awq_base.align_modules(
                    [mapping.parent, mapping.smooth_layer, *mapping.balance_layers]
                ),
                awq_base.calibration_forward_context(model),
                awq_base.HooksMixin.disable_hooks(),
            ):
                orig_layer_weights = {
                    layer: layer.weight.clone() for layer in mapping.balance_layers
                }
                self._apply_scale(
                    mapping, scales[position].detach().cpu(), orig_layer_weights
                )
            self._cleanup_mapping(mapping)

        self._assert_all_activations_consumed()

    def _compute_best_scale(self, mapping, fp16_outputs, orig_layer_weights):
        if not (dist.is_available() and dist.is_initialized()):
            return super()._compute_best_scale(
                mapping, fp16_outputs, orig_layer_weights
            )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("xpu", torch.xpu.current_device())
        scale_size = self._smooth_activation_stats[mapping.smooth_name][0].numel()

        # Layer-wide mapping: split the candidate ratios, not the samples.
        full_grid = AWQModifier._get_grid_search_params(self)
        local_grid = full_grid[rank::world_size]
        local_scale = torch.zeros(scale_size, dtype=torch.float32, device=device)
        local_error = torch.tensor(float("inf"), dtype=torch.float32, device=device)
        if local_grid:
            self._grid_override = local_grid
            try:
                best = self._serial_best_scale(
                    mapping, fp16_outputs, orig_layer_weights
                )
            finally:
                self._grid_override = None
            local_scale.copy_(best.to(device=device, dtype=local_scale.dtype).view(-1))
            best_error = self._error_metrics[-1]["best_error"]
            local_error.fill_(
                best_error.item() if isinstance(best_error, torch.Tensor) else best_error
            )

        gathered_errors = [torch.empty_like(local_error) for _ in range(world_size)]
        gathered_scales = [torch.empty_like(local_scale) for _ in range(world_size)]
        dist.all_gather(gathered_errors, local_error)
        dist.all_gather(gathered_scales, local_scale)
        winner = min(range(world_size), key=lambda idx: gathered_errors[idx].item())
        return gathered_scales[winner].cpu()
