import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import manage_mixed_conserved_quantities as campaign


class MixedInvariantWorkflow(unittest.TestCase):
    def prepare(self, directory):
        mesh = directory/'mesh.vtk'
        mesh.write_text('test fixture: mesh generation is covered by the Julia smoke run')
        manifest = directory/'meshes.json'
        manifest.write_text(json.dumps({'meshes': [
            dict(family=family, level=level, mesh=str(mesh), nelements=60000,
                 sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())
            for family in ('structured','unstructured') for level in range(1,5)]}))
        root = directory/'campaign'
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.main(['prepare','--root',str(root),'--mesh-manifest',str(manifest)])
        return root,json.loads((root/'manifest.json').read_text())['cases']

    def test_h_and_p_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,cases=self.prepare(Path(tmp))
            self.assertEqual(len(cases),32)
            for c in cases:
                self.assertEqual(c['p_H'],c['p_E']+1)
                self.assertEqual(c['esprk_order'],c['p_E']+2)
                self.assertEqual(c['sample_every'],100)
                if c['study']=='h':
                    self.assertEqual(c['p_E'],2)
                else:
                    self.assertEqual(c['level'],2)
                self.assertTrue(Path(c['config']).is_file())
            self.assertEqual(len({c['output'] for c in cases}),32)

    def test_submission_guard_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,cases=self.prepare(Path(tmp))
            args=['benchmark','--root',str(root),'--studies','h','--families','structured',
                  '--fluxes','centered','--levels','1']
            with patch.object(campaign.subprocess,'check_output',return_value='123\n') as submit:
                output=io.StringIO()
                with contextlib.redirect_stdout(output):
                    campaign.main(args)
                submit.assert_not_called()
                self.assertIn('--nodes=3 --ntasks=336 --ntasks-per-node=112',output.getvalue())
                self.assertIn('--sample-every=100',output.getvalue())
                self.assertIn('--hint=nomultithread',output.getvalue())
                with contextlib.redirect_stdout(io.StringIO()):
                    campaign.main(args+['--submit'])
                self.assertEqual(submit.call_count,1)
            self.assertEqual(len((root/'submissions.jsonl').read_text().splitlines()),1)

    def test_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,cases=self.prepare(Path(tmp))
            Path(cases[0]['mesh']).write_text('changed')
            with self.assertRaises(ValueError):
                campaign.main(['benchmark','--root',str(root)])

    def test_fixed_p_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'campaign'
            mesh=Path(tmp)/'mesh.vtk'
            mesh.write_text('CELL_TYPES 2\n10 10\nCELL_DATA 2\n')
            with contextlib.redirect_stdout(io.StringIO()):
                campaign.main(['prepare','--root',str(root),'--studies','p',
                               '--families','structured','--p-mesh',str(mesh)])
            cases=json.loads((root/'manifest.json').read_text())['cases']
            self.assertEqual(len(cases),8)
            self.assertTrue(all(c['mesh']==str(mesh) and c['nelements']==2 for c in cases))

    def test_benchmark_gate_chaining_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,cases=self.prepare(Path(tmp))
            select=['--root',str(root),'--studies','p','--families','structured',
                    '--fluxes','centered','--orders','4','--ntasks','112']
            # A fresh production submission cannot bypass the benchmark.
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                campaign.main(['submit',*select])
            # Capture the effective benchmark arguments and its exact identity.
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                campaign.main(['benchmark',*select])
            import shlex
            command=shlex.split(out.getvalue().strip())
            directory=Path(next(x.split('=',1)[1] for x in command if x.startswith('--output-dir=')))
            identity=next(x.split('=',1)[1] for x in command if x.startswith('--benchmark-id='))
            directory.mkdir(parents=True)
            fields=['benchmark_id','actual_ranks','sample_every','checkpoint_every','nsteps',
                    'measured_steps','measured_seconds','seconds_per_step','estimated_total_seconds','warmup_steps']
            def report(rate=1.):
                with (directory/'benchmark.csv').open('w') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(dict(zip(fields,[identity,112,100,1000,200000,1000,
                                                       1000*rate,rate,200000*rate,20])))
            report()
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                campaign.main(['submit',*select,'--segments','auto'])
            self.assertEqual(len(out.getvalue().splitlines()),2)
            self.assertIn('--dependency=afterok:JOB_ID_SEGMENT_1',out.getvalue())
            with patch.object(campaign.subprocess,'check_output',side_effect=['101\n','102\n']) as sbatch:
                with contextlib.redirect_stdout(io.StringIO()):
                    campaign.main(['submit',*select,'--segments','auto','--submit'])
                self.assertEqual(sbatch.call_count,2)
                self.assertIn('--dependency=afterok:101',sbatch.call_args_list[1].args[0])
            # A pending chained successor blocks another manual continuation.
            from types import SimpleNamespace
            with patch.object(campaign.subprocess,'run',return_value=SimpleNamespace(stdout='102\n')):
                with patch.object(campaign.subprocess,'check_output') as sbatch:
                    with contextlib.redirect_stdout(io.StringIO()):
                        campaign.main(['submit',*select,'--resume','--submit'])
                    sbatch.assert_not_called()
            # Changed cadence selects a new fingerprint and needs another pilot.
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                campaign.main(['submit',*select,'--resume','--sample-every','1'])
            report(rate=10.)
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                campaign.main(['submit',*select,'--resume'])

    def test_polynomial_cost_and_rank_validation(self):
        from types import SimpleNamespace
        args=SimpleNamespace(ntasks=None,elements_per_rank=500,elements_per_node=28000)
        ranks=[campaign.recommended_ranks(dict(nelements=60000,p_E=p),args) for p in range(1,5)]
        self.assertEqual(ranks,sorted(ranks))
        self.assertGreater(ranks[3],ranks[1])
        self.assertTrue(all(n%112==0 for n in ranks))
        args.ntasks=120
        with self.assertRaises(ValueError):
            campaign.actual_ranks(dict(nelements=60000,p_E=2),args)
        self.assertEqual(campaign.wall_seconds('36:00:00'),129600)
        self.assertEqual(campaign.wall_seconds('1-12'),129600)
        self.assertEqual(campaign.wall_seconds('1-12:00'),129600)
        self.assertEqual(campaign.wall_seconds('120'),7200)

    def test_existing_checkpoint_rank_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,cases=self.prepare(Path(tmp))
            c=next(c for c in cases if c['study']=='h' and c['mesh_family']=='structured'
                   and c['flux']=='centered' and c['level']==1)
            slot=Path(c['output'])/'checkpoints'/'slot-a'
            slot.mkdir(parents=True)
            (slot/'mixed_manifest.toml').write_text('mpi_ranks = 4\nstep = 1000\n')
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                campaign.main(['submit','--root',str(root),'--resume','--studies','h',
                               '--families','structured','--fluxes','centered','--levels','1'])


if __name__=='__main__':
    unittest.main()
