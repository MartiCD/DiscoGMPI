#!/usr/bin/env python3
"""Preview/submit prepared MN5 cases or collect measured convergence rates."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


INITIAL_QUANTITIES = ('electric', 'magnetic', 'total', 'ex', 'ey', 'ez',
                      'hx', 'hy', 'hz')


def initial_key(case):
    return (case['mode'], case['nodal_set'],
            case.get('mesh_family', 'unstructured'), case['order'],
            case['level'], case['mesh_sha256'])


def has_initial_errors(row):
    return all(row.get(f'initial_l2_space_{quantity}_error', '').strip()
               for quantity in INITIAL_QUANTITIES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'collect'))
    parser.add_argument('--root', type=Path, default=Path('output/hp_isoresolution'))
    parser.add_argument('--submit', action='store_true', help='Actually submit; otherwise preview commands.')
    parser.add_argument('--initial-only', action='store_true',
                        help='Submit one deduplicated t=0 interpolation job per physical discretization.')
    parser.add_argument('--match', action='append', default=[],
                        help='Require this substring in case names; repeat to combine filters.')
    parser.add_argument('--ranks-per-node', type=int, default=112)
    args = parser.parse_args()
    if args.ranks_per_node < 1:
        parser.error('--ranks-per-node must be positive')
    manifest = json.loads((args.root/'manifest.json').read_text())
    cases = [c for c in manifest['cases']
             if all(pattern in c['name'] for pattern in args.match)]
    if not cases:
        parser.error('No cases matched')
    if args.action == 'submit':
        matched_cases = cases
        if args.initial_only:
            canonical = {}
            for case in cases:
                key = initial_key(case)
                score = (case.get('time_factor', 1) != 1, case['flux'] != 'centered',
                         case['name'])
                if key not in canonical or score < canonical[key][0]:
                    canonical[key] = (score, case)
            cases = [item[1] for item in canonical.values()]
            cases.sort(key=lambda c: (c['mode'], c['nodal_set'],
                                      c.get('mesh_family', 'unstructured'),
                                      c['order'], c['level']))
        log_dir = (args.root/'slurm').resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        for case in cases:
            initial_path = (args.root/'initial_results'/
                            f"{case['mode']}_{case['nodal_set']}_"
                            f"{case.get('mesh_family', 'unstructured')}_"
                            f"p{case['order']}_m{case['level']}.csv")
            if args.initial_only:
                existing = initial_path.is_file()
                if not existing:
                    for candidate in matched_cases:
                        if initial_key(candidate) != initial_key(case):
                            continue
                        result_path = Path(candidate['output'])
                        if not result_path.is_file():
                            continue
                        with result_path.open() as stream:
                            data = list(csv.DictReader(stream))
                        if len(data) == 1 and has_initial_errors(data[0]):
                            existing = True
                            break
                if existing:
                    print(f'Skipping available initial result: {initial_path.name}', flush=True)
                    continue
            command = [os.environ.get('SBATCH_COMMAND', 'sbatch'), '--parsable',
                       f'--account={os.environ.get("ACCOUNT", "bsc115")}',
                       f'--qos={os.environ.get("QOS", "gp_bsccase")}',
                       f'--time={os.environ.get("WALLTIME", "36:00:00")}',
                       f'--nodes={math.ceil(case["ranks"]/args.ranks_per_node)}',
                       f'--ntasks={case["ranks"]}', '--cpus-per-task=1',
                       f'--job-name={"hp_init_" if args.initial_only else "hp_"}{case["name"]}',
                       f'--output={log_dir}/%x_%j.out',
                       f'--error={log_dir}/%x_%j.err',
                       f'--export=ALL,INITIAL_ONLY={1 if args.initial_only else 0}',
                       'examples/run_mn5_hp_isoresolution.slurm', case['config']]
            print(shlex.join(command), flush=True)
            if args.submit:
                job_id = subprocess.check_output(command, text=True).strip()
                with (args.root/'submissions.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(job_id=job_id, case=case['name'], command=command))+'\n')
        return

    initial_results = {}
    for path in sorted((args.root/'initial_results').glob('*.csv')):
        with path.open() as stream:
            data = list(csv.DictReader(stream))
        if len(data) != 1:
            print(f'warning: invalid initial result (expected one row): {path}')
            continue
        record = data[0]
        key = (record['mode'], record['nodal_set'], record['mesh_family'],
               int(record['order']), int(record['mesh_level']), record['mesh_sha256'])
        initial_results[key] = record

    # Older completed campaigns may contain t=0 errors in only one duplicate
    # (flux or time factor).  Treat any such row as the shared interpolation
    # result for every matching physical discretization.
    for case in cases:
        path = Path(case['output'])
        if not path.is_file():
            continue
        with path.open() as stream:
            data = list(csv.DictReader(stream))
        if len(data) == 1 and has_initial_errors(data[0]):
            initial_results.setdefault(initial_key(case), data[0])

    rows = []
    missing = []
    quantities = INITIAL_QUANTITIES
    previous = {}
    for case in sorted(cases, key=lambda c: (
            c['mode'], c['flux'], c['nodal_set'], c.get('mesh_family', 'unstructured'),
            c['order'], c.get('time_factor', 1), c['level'])):
        path = Path(case['output'])
        if not path.is_file():
            missing.append(case['name'])
            continue
        with path.open() as stream:
            data = list(csv.DictReader(stream))
        if not data:
            print(f'warning: incomplete result (no data row): {path}')
            missing.append(case['name'])
            continue
        if len(data) != 1:
            raise ValueError(f'{path}: expected exactly one result, found {len(data)}')
        row = data[0]
        for key, expected in [('order', case['order']), ('nsteps', case['nsteps'])]:
            if int(row[key]) != expected:
                raise ValueError(f'{path}: {key} differs from manifest')
        expected_boundary = 'periodic' if case['mode'] == 'plane_wave' else 'pec'
        if (row['flux'] != case['flux'] or
                row['nodal_set'] != case['nodal_set'].replace('-', '_') or
                row['boundary_condition'] != expected_boundary or
                row['error_mode'] != 'max-time' or
                not math.isclose(float(row['final_time']), case['final_time'])):
            raise ValueError(f'{path}: physical case, flux, nodes, or sampling differs from manifest')
        h_eff = (float(row['volume_total']) / int(row['nelements']))**(1/3)
        initial = initial_results.get(initial_key(case), {})
        for quantity in quantities:
            field = f'initial_l2_space_{quantity}_error'
            if not row.get(field, '').strip() and initial.get(field, '').strip():
                row[field] = initial[field]
            else:
                row.setdefault(field, '')
        row.update(campaign_name=case['name'], resolution=case['resolution'],
                   campaign_level=case['level'],
                   scalar_dofs=case['scalar_dofs'], field_dofs=case['field_dofs'],
                   temporal_rule='dt_p=p*dt4/(4*time_factor)', campaign_mode=case['mode'],
                   campaign_mesh_family=case.get('mesh_family', 'unstructured'),
                   time_factor=case.get('time_factor', 1), h_eff=h_eff)
        key = (case['mode'], case['flux'], case['nodal_set'],
               case.get('mesh_family', 'unstructured'), case['order'],
               case.get('time_factor', 1))
        coarse = previous.get(key)
        for quantity in quantities:
            row[f'rate_{quantity}'] = ''
            row[f'rate_hmax_{quantity}'] = ''
            row[f'rate_heff_{quantity}'] = ''
            row[f'initial_rate_hmax_{quantity}'] = ''
            row[f'initial_rate_heff_{quantity}'] = ''
            # Rates are measured; do not apply an assumed p+1 pass/fail policy.
            if coarse and int(row['mesh_level']) == int(coarse['mesh_level'])+1:
                e0 = float(coarse[f'linf_time_l2_space_{quantity}_error'])
                e1 = float(row[f'linf_time_l2_space_{quantity}_error'])
                initial_pair = (
                    coarse.get(f'initial_l2_space_{quantity}_error', ''),
                    row.get(f'initial_l2_space_{quantity}_error', ''))
                for metric, output_name in (('h_max', 'hmax'), ('h_eff', 'heff')):
                    h0, h1 = float(coarse[metric]), float(row[metric])
                    if min(e0, e1, h1) > 0 and h0 > h1:
                        rate = math.log(e0/e1)/math.log(h0/h1)
                        row[f'rate_{output_name}_{quantity}'] = rate
                        if metric == 'h_max':
                            row[f'rate_{quantity}'] = rate
                    if all(initial_pair):
                        i0, i1 = map(float, initial_pair)
                        if min(i0, i1, h1) > 0 and h0 > h1:
                            row[f'initial_rate_{output_name}_{quantity}'] = (
                                math.log(i0/i1)/math.log(h0/h1))
        previous[key] = row
        rows.append(row)
    if not rows:
        raise SystemExit('No completed cases to collect.')
    output = args.root/'convergence.csv'
    # Omit legacy verdict columns: this campaign reports measured rates only.
    keys = [k for k in rows[0] if not k.startswith(('status_', 'expected_rate_')) and k != 'rate_verdict']
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} cases to {output}; {len(missing)} cases pending.')
    (args.root/'pending_cases.json').write_text(json.dumps(missing, indent=2)+'\n')


if __name__ == '__main__':
    main()
