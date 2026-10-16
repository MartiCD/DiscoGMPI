#!/usr/bin/env python3
"""Prepare, preview/submit, and plot the long-time fixed-p h campaign."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
RESOLUTIONS = ((3, 2, 2), (6, 4, 4), (12, 8, 8), (24, 16, 16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'submit', 'plot', 'status'))
    parser.add_argument('--root', type=Path, default=Path('output/conserved_quantities_h'))
    parser.add_argument('--families', nargs='+', choices=('structured', 'unstructured'),
                        default=['structured', 'unstructured'])
    parser.add_argument('--fluxes', nargs='+', choices=('centered', 'alternating'),
                        default=['centered', 'alternating'])
    parser.add_argument('--levels', nargs='+', type=int, choices=range(1, 5), default=[1, 2, 3, 4])
    parser.add_argument('--elements-per-rank', type=int, default=500)
    parser.add_argument('--max-ranks', type=int, default=448)
    parser.add_argument('--ranks-per-node', type=int, default=112)
    parser.add_argument('--submit', action='store_true', help='Actually submit; default is preview.')
    args = parser.parse_args()
    if min(args.elements_per_rank, args.max_ranks, args.ranks_per_node) < 1:
        parser.error('Resource settings must be positive')
    root = args.root.resolve()
    manifest_path = root / 'manifest.json'
    if args.action == 'prepare':
        root.mkdir(parents=True, exist_ok=False)
        (root / 'meshes').mkdir()
        meshes = []
        families = tuple(dict.fromkeys(args.families))
        if 'unstructured' in families:
            subprocess.run([
                sys.executable,
                'examples/prepare_conserved_h_unstructured_family.py',
                '--output-dir', str(root / 'meshes'),
                '--levels', str(max(args.levels)),
            ], cwd=REPO, check=True)
        for family in families:
            for level in sorted(set(args.levels)):
                counts = RESOLUTIONS[level-1]
                mesh = root / 'meshes' / f'{family}_m{level}.vtk'
                if family == 'structured':
                    subprocess.run(['julia', '--project=.', 'examples/prepare_conserved_h_mesh.jl',
                                    family, *map(str, counts), str(mesh)], cwd=REPO, check=True)
                elif not mesh.is_file():
                    raise FileNotFoundError(f'Missing split-refined mesh: {mesh}')
                # Legacy ASCII VTK emitted by the common writer: one cell type per line.
                lines = mesh.read_text().splitlines()
                start = next(i for i, line in enumerate(lines) if line.startswith('CELL_TYPES '))
                count = int(lines[start].split()[1])
                nelements = sum(int(v) == 10 for v in lines[start+1:start+1+count])
                if not nelements:
                    raise ValueError(f'No tetrahedra in {mesh}')
                meshes.append(dict(family=family, level=level, resolution=counts, mesh=str(mesh),
                                   nelements=nelements, sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),
                                   ranks=min(args.max_ranks, max(1, math.ceil(nelements/args.elements_per_rank)))))
        manifest_path.write_text(json.dumps(dict(
            order=2, esprk_order=3, final_time=10000., cfl=.2,
            initialization='interpolate',
            structured_refinement='nested directional subdivision',
            unstructured_refinement='Gmsh split refinement from M1',
            meshes=meshes), indent=2)+'\n')
        print(f'Prepared {len(meshes)} meshes: {manifest_path}')
        return
    if not manifest_path.is_file():
        parser.error(f'Missing {manifest_path}; run prepare first')
    manifest = json.loads(manifest_path.read_text())
    meshes = [m for m in manifest['meshes'] if m['family'] in args.families and m['level'] in args.levels]
    if not meshes:
        parser.error('No meshes matched')
    if args.action == 'plot':
        import plot_conserved_quantities_convergence_panels as plots
        plots.configure_style()
        for family in dict.fromkeys(args.families):
            selected = sorted((m for m in meshes if m['family'] == family), key=lambda m: m['level'])
            if not selected:
                continue
            for flux in dict.fromkeys(args.fluxes):
                directory = root / family / flux
                labels = [rf'$M_{m["level"]}$ ({"×".join(map(str, m["resolution"]))})' for m in selected]
                series = plots.load_family(directory, 'm', [m['level'] for m in selected], labels,
                                           'invariant_diagnostics.csv')
                for plotter, name in ((plots.plot_signed, 'h_signed_quantities'),
                                      (plots.plot_logy, 'h_logy_errors')):
                    plotter(series, directory/'plots'/name, (12., 14.),
                            f'{family}, {flux}, p=2', 600)
                print(f'Plots: {directory / "plots"}')
        return
    for mesh in meshes:
        for flux in dict.fromkeys(args.fluxes):
            output = root / mesh['family'] / flux / f'm{mesh["level"]}'
            if args.action == 'status':
                csv = output / 'diagnostics/invariant_diagnostics.csv'
                metadata = output / 'config/run_metadata.toml'
                print(f'{output}: diagnostics={csv.is_file()}, metadata={metadata.is_file()}')
                continue
            if hashlib.sha256(Path(mesh['mesh']).read_bytes()).hexdigest() != mesh['sha256']:
                raise ValueError(f'Mesh changed: {mesh["mesh"]}')
            logs = root / 'slurm'
            logs.mkdir(exist_ok=True)
            command = [os.environ.get('SBATCH_COMMAND', 'sbatch'), '--parsable',
                       '--account='+os.environ.get('ACCOUNT', 'bsc115'),
                       '--qos='+os.environ.get('QOS', 'gp_bsccase'),
                       '--time='+os.environ.get('WALLTIME', '72:00:00'),
                       f'--nodes={math.ceil(mesh["ranks"]/args.ranks_per_node)}',
                       f'--ntasks={mesh["ranks"]}', '--cpus-per-task=1',
                       f'--job-name=inv_h_{mesh["family"]}_{flux}_m{mesh["level"]}',
                       f'--output={logs}/%x_%j.out', f'--error={logs}/%x_%j.err',
                       'examples/run_mn5_conserved_quantities_h_convergence.slurm',
                       mesh['mesh'], flux, str(output)]
            print(shlex.join(command), flush=True)
            if args.submit:
                if output.exists():
                    raise FileExistsError(f'Refusing to resubmit into existing {output}')
                job = subprocess.check_output(command, cwd=REPO, text=True).strip()
                with (root/'submissions.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(job_id=job, output=str(output), command=command))+'\n')


if __name__ == '__main__':
    main()
