"""Mine the congestion-origin target sensors of the paper (Section 4.1).

A sensor is an *active origin* at a timestamp at which
  (i)   it is congested,
  (ii)  none of its downstream neighbors is congested, and
  (iii) at least one of its upstream neighbors is congested.
The origin score of a sensor is the number of such timestamps, and the top-k
sensors by this score form the target set of the experiments (k = 10 per
dataset in the paper).

The labels are the exact congestion indicators of the experiment pipeline
(load_speed_data: per-sensor tau * q90 threshold computed on the training
split, zero speeds treated as missing), so the mining is aligned with the
evaluation protocol. The fixed target lists used by the reported experiments
are embedded in shared.py as TARGET_SENSOR_IDS_BY_DATASET.

Usage:
    python method/origins.py --dataset pems
    python method/origins.py --dataset metr
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description='Mine the top-k congestion-origin sensors of a dataset (paper Section 4.1).'
    )
    parser.add_argument(
        '--dataset',
        choices=['pems', 'metr'],
        required=True,
        help='Dataset to mine.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of origins to report (default: 10, the paper value).',
    )
    return parser.parse_args()


ARGS = parse_args()
os.environ['DATASET'] = ARGS.dataset
sys.path.insert(0, str(METHOD_DIR))

from shared import DATASET, load_adjacency, load_speed_data  # noqa: E402


def compute_origin_counts(congested, adj):
    """Origin score of every sensor for a boolean (time, sensor) congestion matrix.

    adj[i, j] > 0 encodes the physical direction in which sensor i is
    *upstream* of sensor j, so the downstream neighbors of a sensor are given
    by its row of adj and its upstream neighbors by its column. Both public
    datasets carry self-loops (adj[i, i] > 0), which are excluded so that a
    sensor's own congestion does not count as a congested neighbor.
    """
    n = congested.shape[1]
    adj_bin = np.asarray(adj, dtype=bool)
    np.fill_diagonal(adj_bin, False)

    origin_count = np.zeros(n, dtype=np.int64)
    for t, row in enumerate(congested):
        active = np.flatnonzero(row)
        if active.size == 0:
            continue
        has_downstream = (adj_bin[active] & row[None, :]).any(axis=1)
        has_upstream = (adj_bin.T[active] & row[None, :]).any(axis=1)
        origins = active[(~has_downstream) & has_upstream]
        origin_count[origins] += 1
        if (t + 1) % 5000 == 0:
            print(f'  processed {t + 1:,}/{congested.shape[0]:,} timestamps')
    return origin_count


def main():
    print(f'=== Congestion-origin mining (dataset={DATASET}) ===')
    smooth, smooth_congested = load_speed_data()
    sensor_ids, sensor_id_to_ind, adj = load_adjacency()
    if adj is None:
        raise SystemExit('Adjacency matrix unavailable for this dataset; cannot mine origins.')

    sensor_ids = [str(sensor_id) for sensor_id in sensor_ids]
    # Reindex so the congestion columns are provably aligned with the adjacency rows.
    congested = smooth_congested.loc[:, sensor_ids].to_numpy(dtype=bool)

    origin_count = compute_origin_counts(congested, adj)
    order = np.argsort(-origin_count, kind='stable')
    top = [int(i) for i in order if origin_count[i] > 0][: ARGS.top_k]

    print(f'\nTop-{len(top)} congestion origins for {DATASET}:')
    for rank, idx in enumerate(top, start=1):
        print(f'  {rank:2d}. {sensor_ids[idx]}  (origin score {origin_count[idx]:,})')


if __name__ == '__main__':
    main()