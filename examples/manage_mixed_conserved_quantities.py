#!/usr/bin/env python3
"""Prepare, preview/submit, inspect and plot mixed-order six-quantity h/p studies."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
PHYSICS_KEYS = ('name','p_E','p_H','esprk_order','mesh_sha256','nelements','flux',
                'final_time','cfl','cfl_degree_rule','electric_initialization',
                'magnetic_initialization','cubature_degree','nodal_set','print_every')


def recommended_ranks(case, args):
    """Dense rectangular RHS cost, normalized to p_E=2, plus a node memory floor.

    This selects a pilot, not a predicted speedup. One node holds no more mesh
    elements than the old 56*500 policy; extra ranks add runtime overhead.
    """
    ne, p = case['nelements'], case['p_E']
    weight = math.comb(p+3,3)*math.comb(p+4,3)*(p+2)/(10*20*4)
    compute = math.ceil(ne*weight/args.elements_per_rank)
    memory = 112*math.ceil(ne/args.elements_per_node)
    ranks = 112*math.ceil(max(compute,memory)/112)
    return min(ranks, 112*(ne//112) if ne>=112 else ne)


def actual_ranks(case, args):
    ranks = args.ntasks or recommended_ranks(case,args)
    if ranks>case['nelements']:
        raise ValueError('MPI ranks exceed elements')
    if ranks>112 and ranks%112:
        raise ValueError('Multi-node jobs require a multiple of 112 ranks; partial single-node jobs may share a node')
    return ranks


def benchmark_identity(case,args):
    source = hashlib.sha256()
    files = sorted((REPO/'src').rglob('*.jl')) + [
        REPO/'examples/run_mixed_conserved_quantities.jl',
        REPO/'examples/mixed_invariant_checkpoint.jl',
        REPO/'examples/run_mn5_mixed_conserved_quantities.slurm']
    for path in files:
        source.update(str(path.relative_to(REPO)).encode())
        source.update(path.read_bytes())
    data = {k:case[k] for k in PHYSICS_KEYS}
    data.update(ranks=actual_ranks(case,args),sample_every=args.sample_every,
                checkpoint_every=args.checkpoint_every,source=source.hexdigest())
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def benchmark_output(case,args,root):
    return root/'benchmarks'/case['name']/benchmark_identity(case,args)


def measured_estimate(case,args,root):
    path=benchmark_output(case,args,root)/'benchmark.csv'
    if not path.is_file():
        return None
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=1:
        raise ValueError(f'Invalid benchmark report: {path}')
    row=rows[0]
    if (row['benchmark_id']!=benchmark_identity(case,args) or
            int(row['actual_ranks'])!=actual_ranks(case,args) or
            int(row['sample_every'])!=args.sample_every or
            int(row['checkpoint_every'])!=args.checkpoint_every):
        raise ValueError(f'Benchmark settings mismatch: {path}')
    elapsed=float(row['measured_seconds'])
    steps=int(row['measured_steps'])
    nsteps=int(row['nsteps'])
    rate=float(row['seconds_per_step'])
    if (steps<args.benchmark_steps or int(row['warmup_steps'])<args.warmup_steps or
            nsteps<steps or not math.isfinite(elapsed) or elapsed<=0 or
            not math.isfinite(rate) or rate<=0 or not math.isclose(elapsed/steps,rate,rel_tol=1e-10)):
        raise ValueError(f'Insufficient/invalid benchmark: {path}; require {args.benchmark_steps} measured steps')
    return rate*nsteps


def wall_seconds(text):
    days,_,clock=text.rpartition('-')
    fields=[int(v) for v in clock.split(':')]
    if len(fields)==1:
        seconds=fields[0]*(3600 if days else 60)
    elif len(fields)==2:
        seconds=(3600*fields[0]+60*fields[1]) if days else (60*fields[0]+fields[1])
    elif len(fields)==3:
        seconds=3600*fields[0]+60*fields[1]+fields[2]
    else:
        raise ValueError('Invalid WALLTIME')
    return seconds+(int(days)*86400 if days else 0)


def selected_cases(manifest, args):
    return [c for c in manifest['cases']
            if c['study'] in args.studies and c['mesh_family'] in args.families
            and c['flux'] in args.fluxes
            and (c['study'] != 'h' or c['level'] in args.levels)
            and (c['study'] != 'p' or c['p_E'] in args.orders)
            and all(s in c['name'] for s in args.match)]


def submission_command(case, args, root):
    ranks = actual_ranks(case,args)
    logs = root / 'slurm'
    walltime=os.environ.get('WALLTIME','02:00:00' if args.action=='benchmark' else '36:00:00')
    seconds=wall_seconds(walltime)
    if seconds<=args.checkpoint_margin:
        raise ValueError('Walltime must exceed the checkpoint margin')
    options=['--restart=auto',f'--sample-every={args.sample_every}',
             f'--checkpoint-every={args.checkpoint_every}']
    if args.action=='benchmark':
        options += [f'--output-dir={benchmark_output(case,args,root)}',
                    f'--benchmark-steps={args.benchmark_steps}',f'--warmup-steps={args.warmup_steps}',
                    f'--benchmark-id={benchmark_identity(case,args)}']
    return [os.environ.get('SBATCH_COMMAND', 'sbatch'), '--parsable',
            '--account='+os.environ.get('ACCOUNT', 'bsc115'),
            '--qos='+os.environ.get('QOS', 'gp_bsccase'),
            '--time='+walltime,
            f'--nodes={math.ceil(ranks/args.ranks_per_node)}',
            f'--ntasks={ranks}', f'--ntasks-per-node={min(ranks,args.ranks_per_node)}',
            '--cpus-per-task=1','--hint=nomultithread',
            '--job-name='+('inv_bench_' if args.action=='benchmark' else 'inv_')+case['name'],
            f'--output={logs}/%x_%j.out', f'--error={logs}/%x_%j.err',
            (f'--export=ALL,MIXED_INV_WALL_SECONDS={seconds},MIXED_INV_CHECKPOINT_MARGIN={args.checkpoint_margin}'
             f',MIXED_INV_WATCHDOG_SECONDS={args.watchdog_seconds},MIXED_INV_STARTUP_TIMEOUT={args.startup_timeout}'),
            'examples/run_mn5_mixed_conserved_quantities.slurm', case['config'],*options]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','benchmark','report','submit','status','plot'))
    parser.add_argument('--root', type=Path, default=Path('output/mixed_conserved_quantities'))
    parser.add_argument('--studies', nargs='+', choices=('h','p'), default=['h','p'])
    parser.add_argument('--families', nargs='+', choices=('structured','unstructured'),
                        default=['structured','unstructured'])
    parser.add_argument('--fluxes', nargs='+', choices=('centered','alternating'),
                        default=['centered','alternating'])
    parser.add_argument('--levels', nargs='+', type=int, choices=range(1,5), default=[1,2,3,4])
    parser.add_argument('--orders', nargs='+', type=int, choices=range(1,5), default=[1,2,3,4])
    parser.add_argument('--h-order', type=int, choices=range(1,5), default=2)
    parser.add_argument('--p-mesh-level', type=int, choices=range(1,5), default=2)
    parser.add_argument('--mesh-manifest', type=Path,
                        help='Reuse an existing equal-order h mesh manifest for identical-mesh comparisons.')
    parser.add_argument('--p-mesh', type=Path,
                        help='Existing ASCII tetrahedral VTK mesh; requires p-only and one family.')
    parser.add_argument('--final-time', type=float, default=10000.)
    parser.add_argument('--cfl', type=float, default=.1)
    parser.add_argument('--cfl-degree-rule', choices=('linear','quadratic'), default='linear')
    parser.add_argument('--electric-initialization', choices=('interpolate','curl-potential'), default='interpolate')
    parser.add_argument('--magnetic-initialization', choices=('interpolate','curl-potential'), default='interpolate')
    parser.add_argument('--sample-every', type=int, default=100,
                        help='Effective sampling for new cases and submitted continuations.')
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--print-every', type=int, default=1000)
    parser.add_argument('--elements-per-rank', type=int, default=500,
                        help='p_E=2 equivalent dense-operator work per rank; benchmark validates the heuristic.')
    parser.add_argument('--elements-per-node', type=int, default=28000,
                        help='Conservative mesh memory floor, matching old 56*500 elements/node.')
    parser.add_argument('--ranks-per-node', type=int, choices=(112,), default=112)
    parser.add_argument('--ntasks', type=int)
    parser.add_argument('--match', action='append', default=[])
    parser.add_argument('--max-points', type=int, default=50000)
    parser.add_argument('--benchmark-steps',type=int,default=1000)
    parser.add_argument('--warmup-steps',type=int,default=20)
    parser.add_argument('--target-days',type=float,default=7.)
    parser.add_argument('--estimate-margin',type=float,default=1.2,
                        help='Safety factor on measured runtime for benchmark admission and chaining.')
    parser.add_argument('--checkpoint-margin',type=int,default=600)
    parser.add_argument('--watchdog-seconds',type=int,default=2700)
    parser.add_argument('--startup-timeout',type=int,default=7200)
    parser.add_argument('--segments',default='1',help='Number of dependent allocations, or auto from the benchmark.')
    parser.add_argument('--resume',action='store_true',help='Continue existing outputs/recorded submissions; same MPI count required.')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    if (not all(math.isfinite(v) and v>0 for v in (args.final_time,args.cfl)) or
            min(args.elements_per_rank,args.elements_per_node,args.sample_every,args.print_every,
                args.benchmark_steps,args.warmup_steps,args.checkpoint_margin,args.startup_timeout)<1 or
            args.checkpoint_every<0 or (args.ntasks is not None and args.ntasks<1) or
            args.watchdog_seconds<0 or args.max_points<2 or
            not math.isfinite(args.target_days) or args.target_days<=0 or
            not math.isfinite(args.estimate_margin) or args.estimate_margin<1 or
            (args.segments!='auto' and (not args.segments.isdigit() or int(args.segments)<1))):
        parser.error('Invalid time, sampling or resource setting')
    root=args.root.resolve()
    if args.action=='prepare':
        if args.p_mesh and (set(args.studies)!={'p'} or len(set(args.families))!=1
                            or args.mesh_manifest):
            parser.error('--p-mesh requires --studies p, one --families value and no --mesh-manifest')
        root.mkdir(parents=True,exist_ok=False)
        levels=set(args.levels if 'h' in args.studies else [])
        if 'p' in args.studies:
            levels.add(args.p_mesh_level)
        # Reuse the existing nested structured and Gmsh split-refined families.
        bank=root/'mesh_bank'
        if args.p_mesh:
            mesh=args.p_mesh.resolve()
            lines=mesh.read_text().splitlines()
            start=next(i for i,line in enumerate(lines) if line.startswith('CELL_TYPES '))
            count=int(lines[start].split()[1])
            types=[]
            for line in lines[start+1:]:
                if len(types)>=count:
                    break
                types.extend(int(v) for v in line.split())
            if len(types)!=count or not types or any(v!=10 for v in types):
                raise ValueError('--p-mesh must contain only tetrahedral cells')
            meshes=[dict(family=args.families[0],level=args.p_mesh_level,mesh=str(mesh),
                         nelements=count,sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())]
        elif args.mesh_manifest is None:
            subprocess.run([sys.executable,'examples/manage_conserved_quantities_h.py','prepare',
                            '--root',str(bank),'--families',*dict.fromkeys(args.families),
                            '--levels',*map(str,sorted(levels))],cwd=REPO,check=True)
        if not args.p_mesh:
            meshes=json.loads((args.mesh_manifest or bank/'manifest.json').read_text())['meshes']
        meshes=[m for m in meshes if m['family'] in args.families and m['level'] in levels]
        if {(m['family'],m['level']) for m in meshes} != {
                (family,level) for family in args.families for level in levels}:
            raise ValueError('Mesh manifest does not cover the selected families/levels')
        for mesh in meshes:
            mesh['mesh']=str(Path(mesh['mesh']).resolve())
            if hashlib.sha256(Path(mesh['mesh']).read_bytes()).hexdigest()!=mesh['sha256']:
                raise ValueError(f'Mesh hash mismatch: {mesh["mesh"]}')
        (root/'cases').mkdir()
        cases=[]
        for mesh in meshes:
            for study in dict.fromkeys(args.studies):
                if study=='h' and mesh['level'] not in args.levels:
                    continue
                if study=='p' and mesh['level']!=args.p_mesh_level:
                    continue
                for p in sorted(set(args.orders if study=='p' else [args.h_order])):
                    for flux in dict.fromkeys(args.fluxes):
                        family,level=mesh['family'],mesh['level']
                        name=f'mixed_inv_{study}_{family}_{flux}_p{p}_m{level}'
                        output=root/study/family/flux/(f'p{p}' if study=='p' else f'm{level}')
                        config=root/'cases'/f'{name}.toml'
                        c=dict(name=name,study=study,discretization='mixed',mesh_family=family,
                               level=level,p_E=p,p_H=p+1,esprk_order=p+2,flux=flux,
                               nodal_set='equispaced',mesh=mesh['mesh'],mesh_sha256=mesh['sha256'],
                               nelements=mesh['nelements'],
                               final_time=args.final_time,cfl=args.cfl,cfl_degree_rule=args.cfl_degree_rule,
                               electric_initialization=args.electric_initialization,
                               magnetic_initialization=args.magnetic_initialization,
                               sample_every=args.sample_every,checkpoint_every=args.checkpoint_every,
                               print_every=args.print_every,cubature_degree=4*(p+1)+2,
                               output=str(output),config=str(config))
                        c['ranks']=recommended_ranks(c,args)
                        # Flat TOML containing only strings, ints and finite floats.
                        config.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k,v in c.items()))
                        cases.append(c)
        (root/'manifest.json').write_text(json.dumps(dict(discretization='mixed',
            experiment='periodic-six-quantities',resource_policy='rectangular-work-p2-reference',
            elements_per_rank=args.elements_per_rank,elements_per_node=args.elements_per_node,
            cases=cases),indent=2)+'\n')
        print(f'Prepared {len(cases)} cases in {root}')
        return
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('discretization')!='mixed' or manifest.get('experiment')!='periodic-six-quantities':
        parser.error('Wrong campaign manifest')
    cases=selected_cases(manifest,args)
    if not cases:
        parser.error('No cases matched')
    if args.action=='plot':
        import plot_conserved_quantities_convergence_panels as plots
        plots.configure_style()
        for study,family,flux in sorted({(c['study'],c['mesh_family'],c['flux']) for c in cases}):
            group=sorted((c for c in cases if (c['study'],c['mesh_family'],c['flux'])==(study,family,flux)),
                         key=lambda c:c['p_E'] if study=='p' else c['level'])
            series=[]
            for c in group:
                path=Path(c['output'])
                if not (path/'completed.toml').exists():
                    raise RuntimeError(f'Incomplete run: {path}; narrow the selection to completed runs')
                label=rf'$p_E={c["p_E"]}, p_H={c["p_H"]}$' if study=='p' else rf'$M_{c["level"]}$'
                series.append(plots.read_series(path/'diagnostics/invariant_diagnostics.csv',
                    c['p_E'] if study=='p' else c['level'],label,args.max_points))
            directory=root/study/family/flux/'plots'
            title=f'Mixed DG, {family}, {flux}, ESPRK(p_E+2)'
            if study=='h':
                title+=f', p_E={group[0]["p_E"]}, p_H={group[0]["p_H"]}'
            for plotter,name in ((plots.plot_signed,'signed_quantities'),(plots.plot_logy,'logy_errors')):
                plotter(series,directory/f'{study}_{name}',(12.,14.),title,300)
            print(directory)
        return
    submitted=set()
    records=[]
    ledger=root/'submissions.jsonl'
    if ledger.exists():
        records=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        submitted={r['case'] for r in records if r.get('action','submit')=='submit'}
    if args.action=='report':
        for c in cases:
            estimate=measured_estimate(c,args,root)
            ranks=actual_ranks(c,args)
            if estimate is None:
                print(f'{c["name"]}: ranks={ranks}, benchmark required')
                continue
            days=estimate*args.estimate_margin/86400
            decision='PASS' if days<=args.target_days else 'OVER TARGET; benchmark another layout'
            print(f'{c["name"]}: ranks={ranks}, measured={estimate/86400:.3f} days, '
                  f'with margin={days:.3f} days: {decision}')
        return
    if args.action in ('submit','benchmark'):
        active=set()
        if args.submit and records:
            # Query this user's queue once, including chained pending jobs.
            queue=subprocess.run(['squeue','--noheader','--user='+os.environ['USER'],'--format=%i'],
                                 check=True,text=True,capture_output=True)
            active=set(queue.stdout.split())
        selected=[]
        for c in cases:
            output=Path(c['output']) if args.action=='submit' else benchmark_output(c,args,root)
            available=(output/'completed.toml').is_file()
            if args.action=='benchmark' and (output/'benchmark.csv').is_file():
                try:
                    available=measured_estimate(c,args,root) is not None
                except ValueError:
                    print(f'Incomplete benchmark; continuing: {output}',flush=True)
                    available=False
            if available:
                print(f'Skipping available result: {output}',flush=True)
                continue
            if args.action=='submit' and not args.resume and (output.exists() or c['name'] in submitted):
                print(f'Skipping existing/submitted case: {c["name"]}; use --resume to continue',flush=True)
                continue
            relevant=[r for r in records if r['case']==c['name'] and
                      r.get('action','submit')==args.action and
                      (args.action=='submit' or r.get('benchmark_id')==benchmark_identity(c,args))]
            if any(str(r['job_id']).split(';')[0] in active for r in relevant):
                print(f'Skipping active/queued case: {c["name"]}',flush=True)
                continue
            # Rank changes require a fresh output, even if mesh/physics match.
            for path in (output/'checkpoints').glob('*/mixed_manifest.toml'):
                match=re.search(r'^mpi_ranks\s*=\s*(\d+)\s*$',path.read_text(),re.M)
                if match and int(match[1])!=actual_ranks(c,args):
                    parser.error(f'{c["name"]}: existing checkpoints use {match[1]} ranks; '
                                 'pass that --ntasks to continue, or prepare a fresh root for a new rank count')
            estimate=None
            if args.action=='submit':
                estimate=measured_estimate(c,args,root)
                if estimate is None:
                    parser.error(f'{c["name"]}: benchmark required for these ranks, sampling and source; '
                                 'run benchmark --submit, then report')
                if estimate*args.estimate_margin>args.target_days*86400:
                    parser.error(f'{c["name"]}: estimated runtime exceeds {args.target_days} days with margin; '
                                 'benchmark another rank count before production')
            selected.append((c,estimate))
        # Validate every selected case before submitting any job.
        for c,_ in selected:
            if hashlib.sha256(Path(c['mesh']).read_bytes()).hexdigest()!=c['mesh_sha256']:
                raise ValueError(f'Mesh changed: {c["mesh"]}')
            submission_command(c,args,root)
        (root/'slurm').mkdir(exist_ok=True)
        for c,estimate in selected:
            command=submission_command(c,args,root)
            count=1
            if args.action=='submit':
                capacity=wall_seconds(os.environ.get('WALLTIME','36:00:00'))-args.checkpoint_margin
                count=max(1,math.ceil(estimate*args.estimate_margin/capacity)) if args.segments=='auto' else int(args.segments)
            previous=None
            for segment in range(count):
                cmd=command.copy()
                if previous is not None:
                    cmd[1:1]=[f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes']
                print(shlex.join(cmd),flush=True)
                if args.submit:
                    job=subprocess.check_output(cmd,cwd=REPO,text=True).strip()
                    previous=job.split(';')[0]
                    record=dict(case=c['name'],job_id=job,command=cmd,action=args.action,
                                benchmark_id=benchmark_identity(c,args),segment=segment+1)
                    with ledger.open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                else:
                    previous=f'JOB_ID_SEGMENT_{segment+1}'
        return
    for c in cases:
        output=Path(c['output'])
        if args.action=='status':
            state='completed' if (output/'completed.toml').is_file() else (
                'partial' if output.exists() else 'submitted' if c['name'] in submitted else 'not-submitted')
            print(f'{c["name"]}: {state}')
            continue


if __name__=='__main__':
    main()
