"""Utilities for exposing shared and personal Hydra config trees together."""

from pathlib import Path
import shutil
import tempfile


def prepare_hydra_config_path(config_group: str, caller_file: str) -> str:
    """Return a merged Hydra config directory for one config group.

    The returned directory contains the shared configs from ``configs/<group>``
    overlaid with any personal configs from ``local/configs/<group>`` so a
    Hydra entrypoint can resolve both through a single ``config_path``.

    Args:
        config_group: Config subgroup such as ``plots`` or ``Lindblad``.
        caller_file: ``__file__`` from the module that will pass the path into
            ``@hydra.main``.

    Returns:
        An absolute path to the merged config directory.

    Raises:
        FileNotFoundError: If the shared config directory for ``config_group``
            does not exist.
    """

    caller_path = Path(caller_file).resolve()
    repo_root = caller_path.parents[2]
    shared_dir = repo_root / "configs" / config_group

    if not shared_dir.is_dir():
        raise FileNotFoundError(f"Shared Hydra config directory not found: {shared_dir}")

    merged_root = repo_root / "local" / "hydra_merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    merged_dir = Path(tempfile.mkdtemp(prefix=f"{config_group}-", dir=merged_root))

    shutil.copytree(shared_dir, merged_dir, dirs_exist_ok=True)

    local_dir = repo_root / "local" / "configs" / config_group
    if local_dir.is_dir():
        shutil.copytree(local_dir, merged_dir, dirs_exist_ok=True)

    return str(merged_dir)
