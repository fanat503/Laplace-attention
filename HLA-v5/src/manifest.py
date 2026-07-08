# Copyright 2026 Ivan Ivanov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.



"""Manifest and hashing utilities for auditable experiment runs.

These helpers are intentionally dependency-free. They are used by scripts to
produce JSON records that can be stored next to checkpoints/logs and included in
paper appendices or internal run reviews.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


def file_sha256(path: str | os.PathLike[str], *, chunk_size: int = 64 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_info(path: str | os.PathLike[str], *, hash_file: bool = False) -> Dict[str, Any]:
    p = Path(path)
    out: Dict[str, Any] = {
        "path": str(p),
        "abs_path": str(p.resolve()) if p.exists() else None,
        "exists": p.exists(),
    }
    if p.exists():
        st = p.stat()
        out.update(
            {
                "size_bytes": int(st.st_size),
                "mtime_unix": float(st.st_mtime),
            }
        )
        if hash_file:
            out["sha256"] = file_sha256(p)
    return out


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def environment_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "argv": sys.argv,
        "cwd": os.getcwd(),
    }
    try:
        import torch

        out["torch"] = torch.__version__
    except Exception as e:  # pragma: no cover - environment-dependent
        out["torch_import_error"] = repr(e)
    try:
        import torch_xla

        out["torch_xla"] = getattr(torch_xla, "__version__", "unknown")
    except Exception as e:  # pragma: no cover - environment-dependent
        out["torch_xla_import_error"] = repr(e)

    for k in sorted(os.environ):
        if k.startswith(("XLA", "XRT", "TPU", "PJRT", "LIBTPU", "TORCH")):
            out[f"env_{k}"] = os.environ[k]
    return out


def build_experiment_manifest(
    *,
    config: Dict[str, Any],
    config_path: Optional[str] = None,
    init_path: Optional[str] = None,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    hash_large_files: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    train_path = train_path or config.get("train_path")
    val_path = val_path or config.get("val_path")
    init_path = init_path or config.get("init_ckpt")
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": stable_hash(config),
        "config": config,
        "files": {
            "config": file_info(config_path, hash_file=True) if config_path else None,
            "init": file_info(init_path, hash_file=hash_large_files) if init_path else None,
            "train": file_info(train_path, hash_file=hash_large_files) if train_path else None,
            "val": file_info(val_path, hash_file=hash_large_files) if val_path else None,
        },
        "environment": environment_snapshot(),
    }
    if extra:
        manifest["extra"] = extra
    manifest["manifest_hash"] = stable_hash({k: v for k, v in manifest.items() if k != "manifest_hash"}, length=24)
    return manifest


__all__ = [
    "canonical_json",
    "stable_hash",
    "file_sha256",
    "file_info",
    "atomic_write_json",
    "load_json",
    "environment_snapshot",
    "build_experiment_manifest",
]
