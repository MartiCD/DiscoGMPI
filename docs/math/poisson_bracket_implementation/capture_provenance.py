"""Print the inspected working-tree identity and hashes, without changing it."""

import argparse
import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    args = parser.parse_args()
    root = args.repository.resolve()
    print("DiscoGMPI Poisson-bracket reconstruction provenance")
    print("Captured:", datetime.now(timezone.utc).isoformat())
    print("Repository:", root)
    print("Platform:", platform.platform())
    for command in (
        ["git", "branch", "--show-current"],
        ["git", "log", "-1", "--format=%H%n%cI%n%s"],
        ["git", "status", "--short"],
        ["git", "diff", "--stat", "--", "src", "examples", "test", "Project.toml", "Manifest.toml"],
    ):
        print("\n$", " ".join(command))
        print(subprocess.check_output(command, cwd=root, text=True), end="")
    paths = sorted((root / "src").rglob("*.jl"))
    paths += [root / path for path in (
        "Project.toml",
        "Manifest.toml",
        "examples/distributed_periodic_poisson_bracket_maxwell.jl",
        "examples/distributed_metallic_sphere_scattering.jl",
        "test/test_reference_tet_nodes.jl",
        "test/test_mpi_magnetic_charge_null_invariant.jl",
        "docs/math/maxwell_poisson_bracket_formulation.tex",
        "docs/numerics/magnetic_charge_longtime_analysis.md",
    )]
    print("\nSHA-256  repository-relative-path")
    for path in paths:
        print(hashlib.sha256(path.read_bytes()).hexdigest(), path.relative_to(root))


if __name__ == "__main__":
    main()
