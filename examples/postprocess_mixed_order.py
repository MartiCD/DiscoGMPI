"""Report measured E/H slopes without imposing an expected convergence order."""
import argparse
import csv
import math
from collections import defaultdict


def summarize(path):
    groups = defaultdict(list)
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            p = int(row["p_E"])
            if int(row["p_H"]) != p + 1 or int(row["esprk_order"]) != p + 2:
                raise ValueError("Inconsistent mixed-order metadata")
            groups[(p, row["flux"], row["study"], row["final_time"], row["error_norm"], row["initialization"])].append(row)
    for (p, flux, study, time, norm, initialization), rows in sorted(groups.items()):
        rows.sort(key=lambda row: -float(row["h"]))
        print(f"p_E={p} p_H={p+1} ESPRK={p+2} {flux} {study} t={time} {norm} {initialization}")
        previous = None
        for row in rows:
            h, e, m = (float(row[key]) for key in ("h", "error_E", "error_H"))
            if not all(math.isfinite(x) and x > 0 for x in (h, e, m)):
                raise ValueError("Errors and h must be finite and positive")
            if previous is not None and h == previous[0]:
                raise ValueError("Duplicate mesh size within a study")
            rates = (math.nan, math.nan) if previous is None else (
                math.log(previous[1] / e) / math.log(previous[0] / h),
                math.log(previous[2] / m) / math.log(previous[0] / h),
            )
            print(f"  n={row['n']:>4} E={e:.7e} H={m:.7e} rates=({rates[0]:.3f}, {rates[1]:.3f})")
            previous = h, e, m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    summarize(parser.parse_args().csv)
