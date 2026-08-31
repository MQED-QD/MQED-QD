from __future__ import annotations

import math

import numpy as np


def scheduler_for_backend(
    parallel_cfg,
    backend: str,
    backend_default_scheduler: str | dict[str, str] | None = None,
) -> str:
    backend = str(backend).strip().lower()
    requested = "backend_default"
    if parallel_cfg:
        requested = str(parallel_cfg.get("scheduler", "backend_default")).strip().lower()
    if requested == "backend_default":
        if backend_default_scheduler is None:
            backend_default_scheduler = {"joblib": "energy", "mpi": "auto"}
        if isinstance(backend_default_scheduler, str):
            scheduler = backend_default_scheduler.strip().lower()
        else:
            scheduler = str(backend_default_scheduler.get(backend, "energy")).strip().lower()
        if scheduler not in {"energy", "flattened", "auto"}:
            raise ValueError(
                "backend_default_scheduler must resolve to 'energy', 'flattened', or 'auto'."
            )
        return scheduler
    if requested not in {"energy", "flattened", "auto"}:
        raise ValueError(
            "parallel.scheduler must be 'backend_default', 'energy', 'flattened', or 'auto'."
        )
    return requested


def rx_chunk_size(parallel_cfg) -> int | None:
    if not parallel_cfg:
        return None
    value = parallel_cfg.get("rx_chunk_size", None)
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("parallel.rx_chunk_size must be a positive integer or null.")
    chunk_size = int(value)
    if chunk_size < 1:
        raise ValueError("parallel.rx_chunk_size must be a positive integer or null.")
    return chunk_size


def task_slices(
    n_energy: int,
    n_rx: int,
    worker_count: int,
    integration_method: str,
    scheduler: str,
    rx_chunk_size: int | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Build ordered energy/Rx work units for local and MPI executors.

    The flattened scheduler uses one process pool or MPI communicator; it does
    not create nested parallel loops. Each task contains one energy and one
    contiguous slice of the global Rx grid. ``fixed_grid`` always receives the
    complete Rx row because its solver shares sampled q kernels across all Rx.
    """
    if n_energy < 1 or n_rx < 1:
        raise ValueError("Task decomposition requires at least one energy and one Rx point.")
    if worker_count < 1:
        raise ValueError("Task decomposition requires at least one worker.")
    scheduler = str(scheduler).strip().lower()
    integration_method = str(integration_method).strip().lower()
    if scheduler not in {"energy", "flattened", "auto"}:
        raise ValueError("scheduler must be 'energy', 'flattened', or 'auto'.")
    if rx_chunk_size is not None and (
        isinstance(rx_chunk_size, (bool, np.bool_))
        or not isinstance(rx_chunk_size, (int, np.integer))
        or int(rx_chunk_size) < 1
    ):
        raise ValueError("rx_chunk_size must be a positive integer or null.")
    rx_chunk_size = None if rx_chunk_size is None else int(rx_chunk_size)

    all_rx_indices = np.arange(n_rx, dtype=int)
    should_flatten = scheduler == "flattened" or (
        scheduler == "auto" and n_energy < worker_count
    )
    if integration_method == "fixed_grid" or not should_flatten:
        return [(energy_index, all_rx_indices.copy()) for energy_index in range(n_energy)]

    if rx_chunk_size is None:
        chunks_per_energy = math.ceil(worker_count / n_energy)
        if scheduler == "flattened" and n_rx > 1:
            chunks_per_energy = max(2, chunks_per_energy)
        chunks_per_energy = min(n_rx, chunks_per_energy)
        rx_chunks = [chunk for chunk in np.array_split(all_rx_indices, chunks_per_energy) if chunk.size]
    else:
        rx_chunks = [
            all_rx_indices[start : start + rx_chunk_size]
            for start in range(0, n_rx, rx_chunk_size)
        ]
    return [
        (energy_index, rx_indices.copy())
        for energy_index in range(n_energy)
        for rx_indices in rx_chunks
    ]


def assemble_sliced_results(
    all_results: list[list[tuple]],
    n_energy: int,
    n_rx: int,
    save_components: bool,
    worker_label: str,
) -> tuple[np.ndarray, ...]:
    """Validate sliced worker results and restore globally ordered tensors."""
    if n_energy < 1 or n_rx < 1:
        raise ValueError("Result assembly requires at least one energy and one Rx point.")
    results_total = np.zeros((n_energy, n_rx, 3, 3), dtype=complex)
    results_vacuum = np.zeros_like(results_total)
    coverage = np.zeros((n_energy, n_rx), dtype=np.uint16)
    if save_components:
        results_structure = np.zeros_like(results_total)
        results_scattering_te = np.zeros_like(results_total)
        results_scattering_tm = np.zeros_like(results_total)

    for rank_results in all_results:
        for result in rank_results:
            expected_fields = 7 if save_components else 4
            if len(result) != expected_fields:
                raise ValueError(
                    f"{worker_label} worker result must contain {expected_fields} fields; "
                    f"got {len(result)}."
                )
            energy_index, rx_indices, total, vacuum = result[:4]
            if (
                isinstance(energy_index, (bool, np.bool_))
                or not isinstance(energy_index, (int, np.integer))
                or not 0 <= int(energy_index) < n_energy
            ):
                raise ValueError(
                    f"{worker_label} worker returned invalid energy index {energy_index!r}; "
                    f"expected an integer in [0, {n_energy})."
                )
            energy_index = int(energy_index)
            rx_indices = np.asarray(rx_indices)
            if rx_indices.ndim != 1 or rx_indices.size == 0:
                raise ValueError(
                    f"{worker_label} worker Rx indices must be a non-empty one-dimensional array."
                )
            if not np.issubdtype(rx_indices.dtype, np.integer):
                raise ValueError(
                    f"{worker_label} worker Rx indices must contain integers without coercion."
                )
            rx_indices = rx_indices.astype(int, copy=False)
            if np.any(rx_indices < 0) or np.any(rx_indices >= n_rx):
                raise ValueError(
                    f"{worker_label} worker Rx indices must be in [0, {n_rx}); "
                    f"got {rx_indices.tolist()}."
                )
            if np.unique(rx_indices).size != rx_indices.size:
                raise ValueError(
                    f"{worker_label} worker Rx indices must be unique within each slice; "
                    f"got {rx_indices.tolist()}."
                )
            if total.shape != (len(rx_indices), 3, 3) or vacuum.shape != total.shape:
                raise ValueError(
                    f"{worker_label} worker returned incompatible Green-function slice shapes: "
                    f"total={total.shape}, vacuum={vacuum.shape}, Rx count={len(rx_indices)}."
                )
            results_total[energy_index, rx_indices] = total
            results_vacuum[energy_index, rx_indices] = vacuum
            coverage[energy_index, rx_indices] += 1
            if save_components:
                _, _, _, _, structure, scattering_te, scattering_tm = result
                if not (
                    structure.shape == total.shape
                    and scattering_te.shape == total.shape
                    and scattering_tm.shape == total.shape
                ):
                    raise ValueError(
                        f"{worker_label} polarization slices must match the total slice shape."
                    )
                results_structure[energy_index, rx_indices] = structure
                results_scattering_te[energy_index, rx_indices] = scattering_te
                results_scattering_tm[energy_index, rx_indices] = scattering_tm

    if not np.all(coverage == 1):
        missing = np.argwhere(coverage == 0)
        duplicate = np.argwhere(coverage > 1)
        raise RuntimeError(
            f"{worker_label} result coverage must contain every energy/Rx pair exactly once; "
            f"missing={missing.tolist()}, duplicate={duplicate.tolist()}."
        )

    if save_components:
        return (
            results_total,
            results_vacuum,
            results_structure,
            results_scattering_te,
            results_scattering_tm,
        )
    return results_total, results_vacuum


_scheduler_for_backend = scheduler_for_backend
_rx_chunk_size = rx_chunk_size
_task_slices = task_slices
_assemble_sliced_results = assemble_sliced_results
