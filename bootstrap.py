"""Create a local virtual environment and run EEG Viewer without admin rights."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def main() -> int:
    python = _venv_python()
    if not python.exists():
        print(f"Creating local environment at {VENV}", flush=True)
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"]
        )

    marker = VENV / ".eegvis-installed"
    project_files = [ROOT / "pyproject.toml"]
    needs_install = not marker.exists() or any(
        path.stat().st_mtime_ns > marker.stat().st_mtime_ns for path in project_files
    )
    if needs_install:
        print("Installing EEG Viewer into the local environment", flush=True)
        subprocess.check_call([str(python), "-m", "pip", "install", "-e", str(ROOT)])
        marker.touch()

    command = [str(python), "-m", "eegvis_benchmark", *sys.argv[1:]]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
