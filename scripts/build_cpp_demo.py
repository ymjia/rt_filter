from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_SOURCE = ROOT / "output_alg" / "cpp_demo"
THIRD_BUILD_DEFAULT = Path.home() / "dev" / "3rd_build"

PRESETS = {
    "macos-xcode-release": {
        "generator": "Xcode",
        "binary_dir": ROOT / "build" / "cpp_demo" / "macos-xcode",
        "third_build_toolchain": "macos-xcode15-arm64",
    },
    "windows-vs-release": {
        "generator": "Visual Studio 17 2022",
        "binary_dir": ROOT / "build" / "cpp_demo" / "windows-vs2022",
        "third_build_toolchain": "vs2026",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the standalone C++ demo with local Conan packages.")
    parser.add_argument(
        "--preset",
        default=_default_preset(),
        choices=sorted(PRESETS),
        help="CMake preset defined in output_alg/cpp_demo/CMakePresets.json.",
    )
    parser.add_argument("--build-type", default="Release", help="CMake/Conan build type.")
    parser.add_argument(
        "--third-build-root",
        default=str(THIRD_BUILD_DEFAULT),
        help="Path to the local 3rd_build workspace.",
    )
    parser.add_argument(
        "--third-build-toolchain",
        help="Override the 3rd_build toolchain used to export the Eigen package.",
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip refreshing Eigen through 3rd_build and only run conan install/cmake build.",
    )
    return parser.parse_args()


def _default_preset() -> str:
    if sys.platform == "darwin":
        return "macos-xcode-release"
    if sys.platform.startswith("win"):
        return "windows-vs-release"
    raise SystemExit("build_cpp_demo.py currently supports macOS and Windows only.")


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _conan_env(third_build_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    if sys.platform.startswith("win"):
        conan_bin_dir = third_build_root / ".tools" / "conan2_venv" / "Scripts"
        conan1_exe = third_build_root / "scripts" / "conan1_exe.cmd"
    else:
        conan_bin_dir = third_build_root / ".tools" / "conan2_venv" / "bin"
        conan1_exe = third_build_root / "scripts" / "conan1_exe.sh"
    env["CONAN_HOME"] = str(third_build_root / "conan_root")
    env["CONAN1_EXE"] = str(conan1_exe)
    env["CONAN1_USER_HOME"] = str(third_build_root / "conan_root_v1")
    env["PATH"] = str(conan_bin_dir) + os.pathsep + env.get("PATH", "")
    return env


def _conan_executable(third_build_root: Path) -> Path:
    if sys.platform.startswith("win"):
        return third_build_root / ".tools" / "conan2_venv" / "Scripts" / "conan.exe"
    return third_build_root / ".tools" / "conan2_venv" / "bin" / "conan"


def main() -> int:
    args = _parse_args()
    preset = PRESETS[args.preset]
    third_build_root = Path(args.third_build_root).expanduser().resolve()
    binary_dir = preset["binary_dir"]
    generator = str(preset["generator"])
    third_build_toolchain = args.third_build_toolchain or str(preset["third_build_toolchain"])
    env = _conan_env(third_build_root)
    conan = _conan_executable(third_build_root)

    if not args.skip_deps:
        _run(
            [
                sys.executable,
                str(third_build_root / "scripts" / "thirdparty_manager.py"),
                "pipeline",
                "eigen",
                "--toolchain",
                third_build_toolchain,
                "--build-type",
                args.build_type,
            ],
            cwd=third_build_root,
            env=env,
        )

    _run(
        [
            str(conan),
            "install",
            str(DEMO_SOURCE),
            "-of",
            str(binary_dir),
            "-s",
            f"build_type={args.build_type}",
            "--build=missing",
            "-c",
            f"tools.cmake.cmaketoolchain:generator={generator}",
        ],
        cwd=ROOT,
        env=env,
    )
    _run(["cmake", "--preset", args.preset], cwd=DEMO_SOURCE, env=env)
    _run(["cmake", "--build", "--preset", args.preset], cwd=DEMO_SOURCE, env=env)

    executable_name = "rt_filter_cpp_demo.exe" if sys.platform.startswith("win") else "rt_filter_cpp_demo"
    executable = binary_dir / args.build_type / executable_name
    if executable.exists():
        print(f"built: {executable}")
    else:
        print(f"build finished; expected executable under {binary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
