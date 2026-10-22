"""Campaign contract checks with synthetic mesh metadata, no convergence runs."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import prepare_hp_isoresolution as prepare
import manage_hp_isoresolution as manage
import postprocess_convergence_csv_tables as spatial
import postprocess_hp_temporal_sensitivity_tables as temporal


def fake_meshes(gmsh, directory, mode, resolutions, *unused):
    return {q: dict(mesh=str(directory/f'{mode}_{q}.vtk'), mesh_sha256='synthetic',
                    resolution=q, nelements=6*q**3, cfl_h_min=0.1/q,
                    h_max=2./q, mean_ratio_min=.5, mean_ratio_avg=.6,
                    lengths=[2.,1.,1.] if mode=='plane_wave' else [1.,1.,1.])
            for q in resolutions}


class MixedCampaignContract(unittest.TestCase):
    def prepare(self, root, mixed=True, extra=()):
        fake = types.SimpleNamespace(initialize=lambda: None, finalize=lambda: None,
            option=types.SimpleNamespace(setNumber=lambda *args: None), __version__='synthetic')
        argv = ['prepare', '--output-dir', str(root), '--mesh-families', 'structured', *extra]
        with patch.dict(sys.modules, gmsh=fake), patch.object(sys, 'argv', argv), \
                patch.object(prepare, 'make_structured_family', fake_meshes), \
                contextlib.redirect_stdout(io.StringIO()):
            prepare.main(discretization='mixed' if mixed else 'equal')
        return json.loads((root/'manifest.json').read_text())

    def test_mixed_dimensions_cfl_and_time_ratios(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.prepare(Path(tmp)/'campaign')
            # Half of the full two-family 384-case campaign.
            self.assertEqual(len(manifest['cases']), 192)
            baseline = {}
            for c in manifest['cases']:
                p, ne = c['p_E'], c['nelements']
                self.assertEqual(c['p_H'], p+1)
                self.assertEqual(c['esprk_order'], p+2)
                self.assertEqual(c['electric_dofs'], 3*ne*math.comb(p+3,3))
                self.assertEqual(c['magnetic_dofs'], 3*ne*math.comb(p+4,3))
                self.assertEqual(c['field_dofs'], c['electric_dofs']+c['magnetic_dofs'])
                self.assertEqual(c['ranks'], math.ceil(ne/1000))
                self.assertEqual(p*c['resolution'], 12*2**(c['level']-1))
                dt = c['final_time']/c['nsteps']
                self.assertEqual(c['cfl_degree_rule'], 'linear')
                self.assertLessEqual(dt, c['cfl']*c['cfl_h_min']/c['p_H'])
                key = (c['mode'], c['level'])
                scaled = dt*c['time_factor']/p
                self.assertAlmostEqual(scaled, baseline.setdefault(key, scaled))

    def test_preview_dispatch_initial_dedup_and_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'campaign'
            self.prepare(root)
            output = io.StringIO()
            with patch.object(sys, 'argv', ['manage', 'submit', '--root', str(root), '--initial-only']), \
                    patch.object(manage.subprocess, 'check_output') as submit, \
                    contextlib.redirect_stdout(output):
                manage.main(discretization='mixed')
            submit.assert_not_called()
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 64)
            self.assertTrue(all('run_mn5_mixed_hp_isoresolution.slurm' in line for line in lines))
            self.assertFalse((root/'submissions.jsonl').exists())
            with patch.object(sys, 'argv', ['manage', 'submit', '--root', str(root)]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                manage.main()

    def test_equal_order_default_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = self.prepare(Path(tmp)/'equal', mixed=False)['cases']
            for c in cases:
                self.assertNotIn('p_H', c)
                self.assertEqual(c['field_dofs'], 6*c['scalar_dofs'])

    def test_quadratic_cfl_override_and_rank_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'quadratic'
            cases = self.prepare(root, extra=('--cfl-degree-rule','quadratic'))['cases']
            for c in cases:
                self.assertEqual(c['cfl_degree_rule'], 'quadratic')
                self.assertLessEqual(c['final_time']/c['nsteps'], c['cfl']*c['cfl_h_min']/(c['p_H']+1)**2)
            output = io.StringIO()
            with patch.object(sys, 'argv', ['manage','submit','--root',str(root),'--ntasks','2']), contextlib.redirect_stdout(output):
                manage.main(discretization='mixed')
            self.assertTrue(all('--ntasks=2' in line for line in output.getvalue().splitlines()))
            self.assertTrue(all('--ntasks-per-node=2' in line for line in output.getvalue().splitlines()))

            output = io.StringIO()
            with patch.object(sys, 'argv', ['manage','submit','--root',str(root),
                    '--ntasks','113','--ranks-per-node','56']), contextlib.redirect_stdout(output):
                manage.main(discretization='mixed')
            self.assertTrue(all('--nodes=3' in line for line in output.getvalue().splitlines()))
            self.assertTrue(all('--ntasks-per-node=56' in line for line in output.getvalue().splitlines()))

    def test_explicit_rank_cap_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.prepare(Path(tmp)/'capped', extra=('--max-ranks','448'))
            self.assertEqual(manifest['max_ranks'], 448)
            self.assertTrue(all(c['ranks'] <= 448 for c in manifest['cases']))
            self.assertTrue(any(c['ranks'] == 448 for c in manifest['cases']))

    def test_checkpoint_and_watchdog_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'campaign'
            self.prepare(root)
            output=io.StringIO()
            with patch.object(sys,'argv',['manage','submit','--root',str(root),
                    '--checkpoint-every','500','--watchdog-seconds','1800',
                    '--startup-timeout','10800','--restart','auto']), contextlib.redirect_stdout(output):
                manage.main(discretization='mixed')
            lines=output.getvalue().splitlines()
            self.assertTrue(lines)
            self.assertTrue(all('MIXED_HP_CHECKPOINT_EVERY=500' in line and
                               'MIXED_HP_WATCHDOG_SECONDS=1800' in line and
                               'MIXED_HP_STARTUP_TIMEOUT=10800' in line and
                               'MIXED_HP_RESTART=auto' in line for line in lines))

    def test_tables_label_mixed_spaces_and_do_not_bridge_levels(self):
        a = spatial.Row(2, 1, 'equispaced', '1.', '.01', '.1', '.2')
        b = spatial.Row(2, 3, 'equispaced', '.25', '.0025', '.001', '.002')
        self.assertEqual(spatial.recomputed_rate(b,a,'electric_error'), '--')
        rendered = spatial.render_table('plane_wave','final',[a],[a],Path('synthetic.csv'),
                                        'structured','h_max',1,True)
        self.assertIn('p_H=p+1', rendered)
        rows = {(2,f): {'dt': .01/f, 'final_l2_space_electric_error': .1,
                        'final_l2_space_magnetic_error': .2} for f in (1,2,4)}
        rendered = temporal.render('plane_wave','centered','final','structured',
                                   'equispaced',4,rows,Path('synthetic.csv'),True)
        self.assertIn('ESPRK order $p+2$', rendered)


if __name__ == '__main__':
    unittest.main()
