from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_URL = "https://github.com/christian-hoang-04/Laplace-attention.git"
REPO_REF = "pr/hla-v5-repair"
WORKDIR = Path("/kaggle/working")
REPO_DIR = WORKDIR / "Laplace-attention"
HLA_DIR = REPO_DIR / "HLA-v5"
RESULT_FILE = WORKDIR / "hla_v5_kaggle_smoke_result.json"
LOG_FILE = WORKDIR / "hla_v5_kaggle_smoke_stdout.log"


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 7200) -> dict[str, object]:
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    result = {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else os.getcwd(),
        "returncode": completed.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        if completed.stdout:
            f.write(completed.stdout)
            if not completed.stdout.endswith("\n"):
                f.write("\n")
        if completed.stderr:
            f.write("[stderr]\n")
            f.write(completed.stderr)
            if not completed.stderr.endswith("\n"):
                f.write("\n")
        f.write("\n")
    return result


def check(result: dict[str, object]) -> dict[str, object]:
    if int(result["returncode"]) != 0:
        raise RuntimeError(
            f"command failed: {' '.join(result['cmd'])}\n"
            f"stdout:\n{result['stdout']}\n"
            f"stderr:\n{result['stderr']}"
        )
    return result


def main() -> None:
    started = time.time()
    LOG_FILE.write_text("", encoding="utf-8")
    summary: dict[str, object] = {
        "status": "running",
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "repo_url": REPO_URL,
        "repo_ref": REPO_REF,
        "steps": [],
    }

    try:
        check(run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO_DIR)]))
        check(run(["python", "-m", "pip", "install", "--upgrade", "pip"], cwd=REPO_DIR, timeout=1800))
        check(run(["python", "-m", "pip", "install", "-r", "requirements_tpu.txt"], cwd=HLA_DIR, timeout=3600))

        for cmd in [
            ["python", "-m", "pytest", "tests/", "-q", "--tb=short", "-W", "error::UserWarning"],
            ["python", "scripts/audit_sterility.py"],
            ["python", "scripts/check_environment.py", "--requirements", "requirements_tpu.txt", "--require-xla"],
            ["python", "scripts/make_dummy_data.py"],
            ["python", "src/make_init.py", "--config", "configs/smoke_hla_s42.json", "--out", "data/init_hla_smoke_s42.pt"],
            ["python", "scripts/validate_data_pair.py", "--config", "configs/smoke_hla_s42.json"],
            ["python", "scripts/check_dataloader.py", "--config", "configs/smoke_hla_s42.json", "--batches", "2"],
            ["python", "scripts/verify_run.py", "--config", "configs/smoke_hla_s42.json", "--init", "data/init_hla_smoke_s42.pt"],
            ["python", "src/train_xla.py", "--config", "configs/smoke_hla_s42.json"],
        ]:
            result = check(run(cmd, cwd=HLA_DIR))
            summary["steps"].append(
                {
                    "cmd": result["cmd"],
                    "returncode": result["returncode"],
                    "duration_seconds": result["duration_seconds"],
                }
            )

        final_resume = HLA_DIR / "runs" / "smoke_hla_s42" / "final_smoke_hla_s42_resume.pt"
        latest_resume = HLA_DIR / "runs" / "smoke_hla_s42" / "latest_smoke_hla_s42_resume.pt"
        summary["artifacts"] = {
            "log_file": str(LOG_FILE),
            "final_resume_exists": final_resume.exists(),
            "latest_resume_exists": latest_resume.exists(),
            "run_dir": str(HLA_DIR / "runs" / "smoke_hla_s42"),
        }
        summary["status"] = "ok"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
    finally:
        summary["duration_seconds"] = round(time.time() - started, 3)
        RESULT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
