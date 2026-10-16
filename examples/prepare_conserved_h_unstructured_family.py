#!/usr/bin/env python3
"""Generate a nested split-refined unstructured periodic-box mesh family."""

from __future__ import annotations

import argparse
from pathlib import Path

import gmsh

from prepare_hp_isoresolution import make_family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--quality-min", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        meshes = make_family(
            gmsh,
            args.output_dir,
            "plane_wave",
            3,
            args.levels - 1,
            args.quality_min,
            args.seed,
        )
    finally:
        gmsh.finalize()

    for level in range(1, args.levels + 1):
        resolution = 3 * 2 ** (level - 1)
        source = Path(meshes[resolution]["mesh"])
        destination = args.output_dir / f"unstructured_m{level}.vtk"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        source.replace(destination)
        print(f"M{level}: q={resolution}, {destination}")


if __name__ == "__main__":
    main()
