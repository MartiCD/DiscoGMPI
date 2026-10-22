#!/usr/bin/env python3
"""Prepare nested unstructured meshes and the MN5 hp campaign (no submission).

Requires numpy and the Gmsh Python module. Each output directory is immutable:
choose a fresh --output-dir to change campaign settings.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def inspect_mesh(gmsh, periodic, lengths):
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(xyz).reshape(-1, 3)
    lookup = np.empty(int(max(tags)) + 1, dtype=np.int64)
    lookup[np.asarray(tags, dtype=int)] = np.arange(len(tags))
    _, connectivity = gmsh.model.mesh.getElementsByType(4)
    connectivity = np.asarray(connectivity, dtype=int).reshape(-1, 4)
    h_min, quality_min, h_max, quality_sum = math.inf, math.inf, 0., 0.
    # Bound temporary memory even for the finest multi-million-element meshes.
    for start in range(0, len(connectivity), 100000):
        tets = xyz[lookup[connectivity[start:start+100000]]]
        volumes = np.abs(np.linalg.det(tets[:, 1:] - tets[:, :1])) / 6
        area = np.zeros(len(tets))
        for i, j, k in itertools.combinations(range(4), 3):
            area += np.linalg.norm(np.cross(tets[:, j] - tets[:, i], tets[:, k] - tets[:, i]), axis=1) / 2
        edges2 = np.stack([np.sum((tets[:, i] - tets[:, j])**2, axis=1)
                           for i, j in itertools.combinations(range(4), 2)])
        quality = 12 * (3 * volumes)**(2/3) / edges2.sum(axis=0)
        if not np.all(volumes > 0):
            raise ValueError('Degenerate tetrahedron')
        h_min = min(h_min, float(np.min(3*volumes/area)))
        h_max = max(h_max, float(np.sqrt(edges2.max())))
        quality_min = min(quality_min, float(quality.min()))
        quality_sum += float(quality.sum())
    if periodic:
        # Check actual translated TRIANGLES, not just coincident vertex sets.
        _, triangles = gmsh.model.mesh.getElementsByType(2)
        triangles = xyz[lookup[np.asarray(triangles, dtype=int).reshape(-1, 3)]]
        for axis, length in enumerate(lengths):
            cuts = []
            for side in (0., length):
                faces = triangles[np.all(np.isclose(triangles[:, :, axis], side, atol=1e-10, rtol=0), axis=1)].copy()
                faces[:, :, axis] -= side
                cuts.append({tuple(sorted(tuple(point) for point in face)) for face in np.round(faces, 10)})
            if not cuts[0] or cuts[0] != cuts[1]:
                raise ValueError(f'Periodic triangle matching failed on axis {axis}')
    return dict(nelements=len(connectivity), cfl_h_min=h_min,
                h_max=h_max, mean_ratio_min=quality_min,
                mean_ratio_avg=quality_sum/len(connectivity), periodic_faces_verified=periodic)


def make_family(gmsh, directory, mode, base, max_depth, threshold, seed):
    gmsh.clear()
    gmsh.model.add(f'{mode}_base{base}')
    lengths = (2., 1., 1.) if mode == 'plane_wave' else (1., 1., 1.)
    gmsh.model.occ.addBox(0, 0, 0, *lengths)
    gmsh.model.occ.synchronize()
    for name, value in {'General.NumThreads': 1, 'Mesh.RandomSeed': seed,
                        'Mesh.Algorithm': 6, 'Mesh.Algorithm3D': 1,
                        'Mesh.MeshSizeMin': lengths[0]/base,
                        'Mesh.MeshSizeMax': lengths[0]/base,
                        'Mesh.MeshSizeFromCurvature': 0, 'Mesh.ElementOrder': 1,
                        'Mesh.SaveAll': 1, 'Mesh.Binary': 0}.items():
        gmsh.option.setNumber(name, value)
    # Prescribe segments only on box edges; surfaces and volume stay unstructured.
    for dim, tag in gmsh.model.getEntities(1):
        length = gmsh.model.occ.getMass(dim, tag)
        gmsh.model.mesh.setTransfiniteCurve(tag, max(1, round(base*length/lengths[0])) + 1)
    for axis, length in enumerate(lengths):
        sides = []
        for side in (0., length):
            sides.append(next(tag for dim, tag in gmsh.model.getEntities(2)
                              if abs(gmsh.model.occ.getCenterOfMass(dim, tag)[axis] - side) < 1e-9))
        for side_index, tag in enumerate(sides):
            gmsh.model.addPhysicalGroup(2, [tag], 2*axis + side_index + 1)
        if mode == 'plane_wave':
            transform = np.eye(4)
            transform[axis, 3] = length
            gmsh.model.mesh.setPeriodic(2, [sides[1]], [sides[0]], transform.ravel().tolist())
    gmsh.model.addPhysicalGroup(3, [1], 7)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize('Netgen')
    result = {}
    for depth in range(max_depth + 1):
        if depth:
            gmsh.model.mesh.refine()
        metrics = inspect_mesh(gmsh, mode == 'plane_wave', lengths)
        if metrics['mean_ratio_min'] < threshold:
            raise ValueError(f'{mode} base{base} depth{depth}: mean ratio {metrics["mean_ratio_min"]} < {threshold}')
        resolution = base * 2**depth
        path = directory / f'{mode}_q{resolution}.vtk'
        gmsh.write(str(path))
        metrics.update(mesh=str(path), mesh_sha256=sha256(path), resolution=resolution,
                       base=base, split_depth=depth, lengths=lengths)
        result[resolution] = metrics
        print(f'{mode} q={resolution}: {metrics["nelements"]} tetrahedra, minimum quality={metrics["mean_ratio_min"]:.4f}', flush=True)
    return result


def structured_base_and_depth(resolution):
    """Return the nested q=3 or q=4 hierarchy containing ``resolution``."""
    for base in (3, 4):
        ratio = resolution / base
        if ratio >= 1 and ratio.is_integer():
            ratio = int(ratio)
            if ratio & (ratio - 1) == 0:
                return base, ratio.bit_length() - 1
    raise ValueError(
        f'Structured resolution q={resolution} is outside the q=3 and q=4 '
        'dyadic refinement hierarchies')


def make_structured_family(gmsh, directory, mode, resolutions, threshold, seed):
    """Generate nested straight-sided structured tetrahedral box meshes.

    Every directional cell count is doubled within each of the q=3 and q=4
    hierarchies.  In particular, the plane-wave q=3 hierarchy is
    (3,2,2), (6,4,4), (12,8,8), ... rather than independently rounded grids.
    """
    lengths = (2., 1., 1.) if mode == 'plane_wave' else (1., 1., 1.)
    result = {}
    for resolution in sorted(resolutions):
        base, split_depth = structured_base_and_depth(resolution)
        refinement = 2**split_depth
        gmsh.clear()
        gmsh.model.add(f'{mode}_structured_q{resolution}')
        gmsh.model.occ.addBox(0, 0, 0, *lengths)
        gmsh.model.occ.synchronize()
        for name, value in {'General.NumThreads': 1, 'Mesh.RandomSeed': seed,
                            'Mesh.ElementOrder': 1, 'Mesh.SaveAll': 1,
                            'Mesh.Binary': 0}.items():
            gmsh.option.setNumber(name, value)
        for dim, tag in gmsh.model.getEntities(1):
            length = gmsh.model.occ.getMass(dim, tag)
            base_cells = max(1, round(base * length / lengths[0]))
            cells = base_cells * refinement
            gmsh.model.mesh.setTransfiniteCurve(tag, cells + 1)
        for _, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setTransfiniteVolume(1)
        for axis, length in enumerate(lengths):
            sides = []
            for side in (0., length):
                sides.append(next(tag for dim, tag in gmsh.model.getEntities(2)
                                  if abs(gmsh.model.occ.getCenterOfMass(dim, tag)[axis] - side) < 1e-9))
            for side_index, tag in enumerate(sides):
                gmsh.model.addPhysicalGroup(2, [tag], 2*axis + side_index + 1)
            if mode == 'plane_wave':
                transform = np.eye(4)
                transform[axis, 3] = length
                gmsh.model.mesh.setPeriodic(2, [sides[1]], [sides[0]], transform.ravel().tolist())
        gmsh.model.addPhysicalGroup(3, [1], 7)
        gmsh.model.mesh.generate(3)
        metrics = inspect_mesh(gmsh, mode == 'plane_wave', lengths)
        if metrics['mean_ratio_min'] < threshold:
            raise ValueError(
                f'{mode} structured q{resolution}: mean ratio '
                f'{metrics["mean_ratio_min"]} < {threshold}')
        path = directory / f'{mode}_structured_q{resolution}.vtk'
        gmsh.write(str(path))
        metrics.update(mesh=str(path), mesh_sha256=sha256(path),
                       resolution=resolution, base=base, split_depth=split_depth,
                       lengths=lengths)
        result[resolution] = metrics
        print(f'{mode} structured q={resolution}: {metrics["nelements"]} tetrahedra, '
              f'minimum quality={metrics["mean_ratio_min"]:.4f}', flush=True)
    return result


def main(discretization='equal'):
    mixed = discretization == 'mixed'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('output/mixed_hp_isoresolution' if mixed else 'output/hp_isoresolution'))
    parser.add_argument('--levels', type=int, default=4)
    parser.add_argument('--cfl', type=float, default=.1 if mixed else .2)
    parser.add_argument('--periods', type=float, default=1.)
    parser.add_argument('--quality-min', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--elements-per-rank', type=int, default=1000)
    parser.add_argument('--max-ranks', type=int, default=None if mixed else 448,
                        help=('Maximum ranks per case. Mixed campaigns are uncapped by default; '
                              'the scheduler may still impose site limits.'))
    parser.add_argument('--cfl-degree-rule', choices=('quadratic', 'linear'),
                        default='linear' if mixed else 'quadratic',
                        help='Mixed only: linear (default) uses p_H; quadratic uses (p_H+1)^2.')
    parser.add_argument('--mesh-families', nargs='+', choices=('unstructured', 'structured'),
                        default=('unstructured', 'structured'))
    parser.add_argument('--time-factors', nargs='+', type=int, default=(1, 2, 4),
                        help='Step-count multipliers for the fixed-mesh time study.')
    parser.add_argument('--temporal-levels', nargs='+', type=int, default=None,
                        help='Mesh levels receiving factors greater than one.')
    args = parser.parse_args()
    if not mixed and args.cfl_degree_rule != 'quadratic':
        parser.error('--cfl-degree-rule applies only to mixed campaigns')
    if args.temporal_levels is None:
        args.temporal_levels = [args.levels]
    if (not 1 <= args.levels <= 4 or
            not all(math.isfinite(v) for v in (args.cfl, args.periods, args.quality_min)) or
            min(args.cfl, args.periods, args.elements_per_rank) <= 0 or
            (args.max_ranks is not None and args.max_ranks <= 0) or
            not 0 <= args.quality_min <= 1 or
            any(value < 1 for value in args.time_factors) or
            1 not in args.time_factors or
            any(level < 1 or level > args.levels for level in args.temporal_levels)):
        parser.error('Invalid levels, CFL, periods, quality threshold, or rank settings.')
    import gmsh
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    meshes_dir = root / 'meshes'
    meshes_dir.mkdir()
    (root / 'cases').mkdir()
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    cases = []
    try:
        for mode in ('plane_wave', 'pec_cavity'):
            required_resolutions = {
                12 * 2**(level-1) // p
                for level in range(1, args.levels+1) for p in (1, 2, 3, 4)
            }
            families = {}
            if 'unstructured' in args.mesh_families:
                meshes = make_family(gmsh, meshes_dir, mode, 3, args.levels+1,
                                     args.quality_min, args.seed)
                meshes.update(make_family(gmsh, meshes_dir, mode, 4, args.levels-1,
                                          args.quality_min, args.seed))
                families['unstructured'] = meshes
            if 'structured' in args.mesh_families:
                families['structured'] = make_structured_family(
                    gmsh, meshes_dir, mode, required_resolutions,
                    args.quality_min, args.seed)
            period = 2. if mode == 'plane_wave' else 2/math.sqrt(3)
            final_time = args.periods * period
            for level in range(1, args.levels+1):
                for family, meshes in families.items():
                    # dt_p = p/4 * dt4, bounded by each p-specific estimate.
                    # Mixed: inradius <= minimum altitude, so this is conservative
                    # relative to mixed_cfl_dt with the selected degree rule.
                    # Keeping dt_p=p*dt4/4 requires dt4 <= 4*limit_p/p.
                    dt4 = min(4*args.cfl*meshes[12*2**(level-1)//p]['cfl_h_min'] /
                              (p*((p+2)**2 if args.cfl_degree_rule == 'quadratic' else p+1) if mixed else p**2)
                              for p in (1, 2, 3, 4))
                    n4 = 3 * math.ceil(final_time/dt4/3)
                    factors = args.time_factors if level in args.temporal_levels else (1,)
                    for p, flux, nodal, time_factor in itertools.product(
                            (1, 2, 3, 4), ('centered', 'alternating'),
                            ('equispaced', 'warp-blend'), factors):
                        mesh = meshes[12*2**(level-1)//p]
                        target_ranks = max(1, math.ceil(
                            mesh['nelements']/args.elements_per_rank))
                        ranks = (target_ranks if args.max_ranks is None else
                                 min(args.max_ranks, target_ranks))
                        name = (f'{mode}_{flux}_{nodal}_{family}_p{p}_m{level}'
                                f'_dt{time_factor}')
                        if mixed:
                            name = 'mixed_' + name
                        case = dict(mesh, name=name, mode=mode, order=p, level=level,
                                    flux=flux, nodal_set=nodal, mesh_family=family,
                                    time_factor=time_factor,
                                    nsteps=(4*n4//p)*time_factor,
                                    final_time=final_time, cfl=args.cfl, ranks=ranks,
                                    output=str(root/'results'/f'{name}.csv'),
                                    scalar_dofs=mesh['nelements']*math.comb(p+3, 3),
                                    field_dofs=6*mesh['nelements']*math.comb(p+3, 3),
                                    effective_resolution=12*2**(level-1), seed=args.seed)
                        if mixed:
                            ne = mesh['nelements']
                            case.update(discretization='mixed', p_E=p, p_H=p+1,
                                        cfl_degree_rule=args.cfl_degree_rule,
                                        esprk_order=p+2, Np_E=math.comb(p+3, 3),
                                        Np_H=math.comb(p+4, 3),
                                        electric_dofs=3*ne*math.comb(p+3, 3),
                                        magnetic_dofs=3*ne*math.comb(p+4, 3),
                                        field_dofs=3*ne*(math.comb(p+3, 3)+math.comb(p+4, 3)),
                                        effective_resolution_H=(p+1)*mesh['resolution'],
                                        cubature_degree=4*(p+1)+2)
                            del case['scalar_dofs']
                        casefile = root / 'cases' / f'{name}.toml'
                        # Flat scalar manifest: JSON string quoting is valid TOML here.
                        with casefile.open('w') as stream:
                            for key, value in case.items():
                                stream.write(f'{key} = {json.dumps(value)}\n')
                        cases.append(dict(case, config=str(casefile)))
    finally:
        gmsh.finalize()
    (root/'manifest.json').write_text(json.dumps(dict(
        discretization=discretization, gmsh_version=gmsh.__version__, time_factors=args.time_factors,
        elements_per_rank=args.elements_per_rank, max_ranks=args.max_ranks,
        temporal_levels=args.temporal_levels, mesh_families=args.mesh_families,
        cases=cases), indent=2)+'\n')
    print(f'Prepared {len(cases)} cases in {root}; no jobs submitted.')


if __name__ == '__main__':
    main()
