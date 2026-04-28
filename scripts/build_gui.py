from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_FILE = ROOT / "rt_filter_gui.spec"
DEFAULT_DIST = ROOT / "dist"
DEFAULT_WORK = ROOT / "build" / "pyinstaller"
APP_NAME = "rt-filter-gui"
CONFIG_FILE = ROOT / "rt_filter_gui.json"
CONFIG_FILENAME = CONFIG_FILE.name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the rt-filter PySide6 GUI into a standalone app.")
    parser.add_argument("--distpath", default=str(DEFAULT_DIST), help="PyInstaller dist directory.")
    parser.add_argument("--workpath", default=str(DEFAULT_WORK), help="PyInstaller work directory.")
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Build only and skip launching the packaged app with --smoke-test.",
    )
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _packaged_executable(distpath: Path) -> Path:
    if sys.platform == "darwin":
        return distpath / f"{APP_NAME}.app" / "Contents" / "MacOS" / APP_NAME
    if sys.platform.startswith("win"):
        return distpath / APP_NAME / f"{APP_NAME}.exe"
    return distpath / APP_NAME / APP_NAME


def _packaged_config_path(distpath: Path, packaged_executable: Path) -> Path:
    if sys.platform == "darwin":
        return distpath / CONFIG_FILENAME
    return packaged_executable.parent / CONFIG_FILENAME


def main() -> int:
    args = _parse_args()
    distpath = Path(args.distpath).resolve()
    workpath = Path(args.workpath).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(distpath),
            "--workpath",
            str(workpath),
            str(SPEC_FILE),
        ],
        cwd=ROOT,
        env=env,
    )

    packaged = _packaged_executable(distpath)
    if not packaged.exists():
        raise FileNotFoundError(f"packaged executable was not created: {packaged}")

    config_target: Path | None = None
    if CONFIG_FILE.exists():
        config_target = _packaged_config_path(distpath, packaged)
        config_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG_FILE, config_target)
        print(f"wrote config: {config_target}")

    if not args.skip_smoke_test:
        smoke_env = env.copy()
        smoke_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        if config_target is not None:
            smoke_env["RT_FILTER_GUI_CONFIG"] = str(config_target)
        _run([str(packaged), "--smoke-test"], cwd=ROOT, env=smoke_env)

    print(f"built: {packaged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
