import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "integrated_plot", Path(__file__).resolve().parents[1] /
    "examples/plot_integrated_fields_and_charge.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "data.csv"

    def write(self, times=(5, 6, 7), missing=False, nan=False):
        columns = plot.REQUIRED[:-1] if missing else plot.REQUIRED
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for i, time in enumerate(times):
                row = {name: float(i + 2) for name in columns}
                row["time"] = time
                if nan:
                    row["electric_charge"] = float("nan")
                writer.writerow(row)

    def test_baseline_and_statistics(self):
        self.write()
        data = plot.read_data(self.path)
        np.testing.assert_array_equal(plot.drift(data["magnetic_charge"]), [0, 1, 2])
        row = plot.summary_rows(data)[0]
        self.assertEqual(row["reference_time"], 5)
        self.assertEqual(row["max_abs_sampled_drift"], 2)
        self.assertAlmostEqual(row["sample_rms_drift"], np.sqrt(5/3))

    def test_reject_bad_input(self):
        for options in (dict(missing=True), dict(nan=True),
                        dict(times=(0, 0, 1)), dict(times=(0,))):
            self.write(**options)
            with self.assertRaises(ValueError):
                plot.read_data(self.path)

    def test_signed_component_cancellation(self):
        data = {name: np.zeros(3) for name in plot.REQUIRED}
        data["time"] = np.arange(3)
        data["integrated_b_x"] = np.array([1., 2., 3.])
        data["integrated_b_y"] = np.array([2., 1., 0.])
        np.testing.assert_array_equal(plot.integrated_sum(data, "b"), [3., 3., 3.])
        np.testing.assert_array_equal(plot.drift(plot.integrated_sum(data, "b")), [0., 0., 0.])

    def test_zero_drift_and_formats(self):
        data = {name: np.zeros(3) for name in plot.REQUIRED}
        data["time"] = np.arange(3)
        figures = plot.make_figures(data, "")
        self.assertEqual([len(fig.axes) for fig in figures], [6, 4])
        for i, figure in enumerate(figures):
            target = Path(self.directory.name) / f"figure{i}.pdf"
            figure.savefig(target)
            self.assertGreater(target.stat().st_size, 0)
            plot.plt.close(figure)
        np.testing.assert_array_equal(data["integrated_b_x"], np.zeros(3))

    def test_integrated_fields_sum_panel(self):
        data = {name: np.zeros(3) for name in plot.REQUIRED}
        data["time"] = np.arange(3)
        data["integrated_d_x"] = np.array([1., 2., 3.])
        data["integrated_d_y"] = np.array([2., 1., 0.])
        data["integrated_b_z"] = np.array([3., 4., 5.])
        full = plot.integrated_fields_panel(data, "")
        sums = plot.integrated_fields_sum_panel(data, "")
        self.assertEqual(len(full.axes), 8)
        self.assertEqual(len(sums.axes), 4)
        np.testing.assert_array_equal(sums.axes[0].lines[0].get_ydata(), [3., 3., 3.])
        np.testing.assert_array_equal(sums.axes[1].lines[0].get_ydata(), [3., 4., 5.])
        np.testing.assert_array_equal(sums.axes[2].lines[0].get_ydata(), [0., 0., 0.])
        np.testing.assert_array_equal(sums.axes[3].lines[0].get_ydata(), [0., 0., 0.])
        plot.plt.close(full)
        plot.plt.close(sums)

    def test_directional_reconstruction_and_plot(self):
        data = {name: np.zeros(3) for name in plot.REQUIRED + plot.DIRECTIONAL}
        data["time"] = np.array([5., 6., 7.])
        for field in ("electric", "magnetic"):
            data[f"{field}_charge_x"] = np.array([2., 3., 4.])
            data[f"{field}_charge_y"] = np.array([-1., -2., -3.])
            data[f"{field}_charge_z"] = np.array([0., 0., 1.])
            data[f"{field}_charge"] = np.array([1., 1., 2.])
        rows = plot.reconstruction_rows(data)
        for row in rows:
            self.assertEqual(row["electric_sum_minus_charge"], 0)
            self.assertEqual(row["magnetic_sum_minus_charge"], 0)
        self.assertEqual(len(plot.summary_rows(data)), 16)
        figure = plot.directional_figure(data, "")
        self.assertEqual(len(figure.axes), 6)
        np.testing.assert_array_equal(figure.axes[2].lines[0].get_ydata(), [0, 1, 2])
        figure.savefig(Path(self.directory.name) / "directional.pdf")
        plot.plt.close(figure)

    def test_directional_columns_complete_or_absent(self):
        for columns in (plot.REQUIRED + plot.DIRECTIONAL[:1],
                        plot.REQUIRED + plot.DIRECTIONAL):
            with self.path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for time in (0, 1):
                    row = dict.fromkeys(columns, 0)
                    row["time"] = time
                    writer.writerow(row)
            if len(columns) == len(plot.REQUIRED) + 1:
                with self.assertRaisesRegex(ValueError, "Incomplete directional"):
                    plot.read_data(self.path)
            else:
                self.assertEqual(set(plot.read_data(self.path)), set(columns))

    def test_legacy_columns_and_figures(self):
        for columns in (plot.REQUIRED + plot.LEGACY[:1], plot.REQUIRED + plot.LEGACY):
            with self.path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for time in (0, 1, 2):
                    row = dict.fromkeys(columns, time * 1e-12)
                    row["time"] = time
                    writer.writerow(row)
            if len(columns) == len(plot.REQUIRED) + 1:
                with self.assertRaisesRegex(ValueError, "Incomplete DiscoG"):
                    plot.read_data(self.path)
            else:
                data = plot.read_data(self.path)
                self.assertEqual(len(plot.summary_rows(data)), 26)
                figs = plot.discog_figures(data, "")
                self.assertEqual([len(fig.axes) for fig in figs], [6, 4])
                for i, fig in enumerate(figs):
                    fig.savefig(Path(self.directory.name) / f"legacy{i}.pdf")
                    plot.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
