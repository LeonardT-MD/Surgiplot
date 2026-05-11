from __future__ import annotations

import shutil
import sys
from pathlib import Path


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[3]
    extension_root = repo_root / "extensions" / "SurgiplotToolkit"
    if len(argv) > 1:
        target_root = Path(argv[1]).expanduser().resolve()
    else:
        target_root = repo_root.parent / "SurgiplotToolkit"

    target_root.mkdir(parents=True, exist_ok=True)

    for item in extension_root.iterdir():
        destination = target_root / item.name
        if item.is_dir():
            copy_tree(item, destination)
        else:
            shutil.copy2(item, destination)

    vendored_backend = target_root / "SurgiplotToolkit" / "SurgiplotToolkitLib" / "External" / "surgiplot_backend"
    vendored_backend.mkdir(parents=True, exist_ok=True)
    copy_tree(repo_root / "surgiplot", vendored_backend / "surgiplot")

    print(f"Standalone SurgiplotToolkit repo exported to: {target_root}")
    print("Next suggested steps:")
    print("1. cd into the exported directory")
    print("2. git init")
    print("3. add remote origin for the new standalone extension repository")
    print("4. test the module in Slicer using that exported path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
