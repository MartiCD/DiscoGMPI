#!/usr/bin/env python3
"""Validate audit CSV/TOML contracts on an existing short+fixed suite output."""
import csv
import math
import sys
from decimal import Decimal, localcontext
from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def metadata(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def verify(root):
    checked = 0
    for campaign_name in ("short", "fixed"):
        campaign = root / campaign_name
        config = metadata(campaign / "campaign.toml")["options"]
        paths = sorted(campaign.glob("*/history.csv"))
        assert paths, campaign
        hashes = {}
        for path in paths:
            m = metadata(path.parent / "metadata.toml")
            done = metadata(path.parent / "completed.toml")
            rows = table(path)
            final = rows[-1]
            key = path.parent.name.rsplit("_cfl", 1)[0]
            assert hashes.setdefault(key, m["initial_sha256"]) == m["initial_sha256"]
            assert done["initial_state_reused_exactly"]
            assert int(final["step"]) == done["steps"] == m["steps"]
            assert float(final["time"]) == done["final_time"] == m["final_time"]
            assert all(float(a["time"]) < float(b["time"]) for a, b in zip(rows, rows[1:]))
            assert int(final["diagnostic_rhs"]) == done["diagnostic_rhs_evaluations"] == len(rows)
            assert m["epsilon"] == m["mu"] == 1
            assert m["tetrahedra"] == 96 and sum(m["face_counts"]) == 192
            assert m["weak_tests_selected"] == m["weak_tests_total"] == 16
            for field in ("electric", "magnetic"):
                assert int(final[field + "_updates"]) == m["steps"] * m[field + "_updates_per_step"]
            assert int(final["integration_rhs"]) == int(final["electric_updates"]) + int(final["magnetic_updates"])
            for row in rows[1:]:
                assert 0 < float(row["dt"]) <= m["dt_limit"] * (1 + 1e-14)
                for field in ("e", "h"):
                    assert math.isclose(float(row["dq" + field]),
                                        float(row["q" + field]) - float(rows[0]["q" + field]),
                                        rel_tol=1e-12, abs_tol=1e-30)
            if campaign_name == "fixed":
                assert m["steps"] == int(config["steps"])
                assert all(float(row["dt"]) == m["dt_limit"] for row in rows[1:])
            updates = table(path.parent / "updates.csv")
            for row in rows:
                for field in ("e", "h"):
                    face_sum = sum(float(row[f"face_q{field}_{kind}"]) for kind in ("interior", "periodic"))
                    assert math.isclose(float(row["q" + field]), face_sum, rel_tol=1e-10, abs_tol=1e-10)
            for row in table(path.parent / "weak.csv"):
                assert float(row["rhs_gradient_pairing"]) == -float(row["rhs_w"])
            stage_every = int(config["stage-every"])
            sampled = {int(r["step"]) for r in updates}
            assert sampled == set(range(stage_every, m["steps"] + 1, stage_every)) if stage_every else not updates
            for row in updates:
                assert math.isclose(float(row["partition_time_after"]) - float(row["partition_time_before"]),
                                    float(row["coefficient"]), rel_tol=1e-10, abs_tol=1e-18)
                if row["functional"].startswith("momentum"):
                    continue
                with localcontext() as ctx:
                    ctx.prec = 100
                    delta, predicted, defect = (Decimal(row[k]) for k in ("delta256", "predicted256", "defect256"))
                    scale = max(abs(delta), abs(predicted), abs(defect))
                    assert abs(delta - predicted - defect) <= Decimal("1e-65") * scale
            checked += 1
    print(f"PASS: {checked} case contracts, identical initial states, counts, times and update records")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
