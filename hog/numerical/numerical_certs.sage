"""
Numerical (SDP) conformal rigidity survey of the HoG database.

For each connected graph with n >= 2 in raw_hog_data.jsonl:
  - m <= 150 : full per-edge SDP  (find_edge_isometric_embedding)  — eq. (13)
  - m > 150  : symmetry-reduced SDP (find_orbit_isometric_embedding) — eq. (25)

Writes numerical_certs.csv containing only graphs that are LCR or UCR.
The file is flushed every FLUSH_EVERY rows so progress is preserved if the
script is interrupted; re-running resumes from the last completed row.
Graphs that exceed GRAPH_TIMEOUT seconds are skipped and logged to
timed_out.jsonl.

Run from this directory:
    sage numerical_certs.sage
"""

import csv
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath('../../cr'))
from solvers import (
    find_edge_isometric_embedding,
    find_orbit_isometric_embedding,
    get_eigenspace,
)
from orbits import get_edge_orbits

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = os.path.abspath('../raw_hog_data.jsonl')
OUTPUT_CSV = 'numerical_certs.csv'
TIMED_OUT_FILE = 'timed_out.jsonl'

# Use full per-edge SDP for m <= this; symmetry-reduced SDP above.
EDGE_SDP_THRESHOLD = 150

# Per-graph timeout in seconds (hard kill via subprocess).
GRAPH_TIMEOUT = 300

# Flush the output CSV to disk after this many CR rows written.
FLUSH_EVERY = 100

FIELDS = [
    "id", "order", "size", "cr",
    "lambda_2", "lambda_2_mult", "lcr",
    "ucr", "lambda_n", "lambda_n_mult",
    "edge_orbits", "vertex_orbits", "group_size",
    "bipartite", "regular", "1_walk_regular",
]


# ---------------------------------------------------------------------------
# Per-graph worker (runs in a forked subprocess)
# ---------------------------------------------------------------------------

def _graph_task(g6, threshold):
    """
    Process one graph. Called in a forked subprocess so all Sage/numpy
    imports are already available. Returns a result dict, or None if the
    graph should be skipped (disconnected / too small).
    """
    G = Graph(g6)
    G.relabel()
    n, m = G.order(), G.size()

    if n < 2 or not G.is_connected():
        return None

    L_np = np.asarray(
        G.laplacian_matrix().change_ring(RDF).numpy(), dtype=float
    )
    eigvals = np.sort(np.linalg.eigh(L_np)[0])
    l2_val  = float(eigvals[1])
    ln_val  = float(eigvals[-1])
    l2_mult = int(np.sum(np.isclose(eigvals, l2_val, atol=1e-8)))
    ln_mult = int(np.sum(np.isclose(eigvals, ln_val, atol=1e-8)))

    if m > threshold:
        aut_group   = G.automorphism_group()
        edge_orbits = get_edge_orbits(G, list(aut_group.gens()))
    else:
        edge_orbits = None

    def _sdp(which):
        _, B = get_eigenspace(L_np, which=which)
        if edge_orbits is None:
            edges = [(int(u), int(v)) for u, v in G.edges(labels=False)]
            status, _, _ = find_edge_isometric_embedding(B, edges)
        else:
            status, _, _ = find_orbit_isometric_embedding(B, edge_orbits)
        return status in ("optimal", "optimal_inaccurate")

    is_lcr = _sdp('lambda2')
    is_ucr = _sdp('lambdan')

    walk_reg = False
    if is_lcr and is_ucr:
        if G.is_regular():
            edges = list(G.edges(labels=False))
            A = G.adjacency_matrix()
            curr = A
            walk_reg = True
            for _ in range(n):
                vals = {curr[u, v] for u, v in edges}
                if len(vals) > 1:
                    walk_reg = False
                    break
                curr = curr * A

    return {
        'n': n, 'm': m,
        'l2_val': l2_val, 'l2_mult': l2_mult,
        'ln_val': ln_val, 'ln_mult': ln_mult,
        'is_lcr': is_lcr, 'is_ucr': is_ucr,
        'walk_reg': walk_reg,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Resume: skip HoG IDs already written to the output CSV
# ---------------------------------------------------------------------------

print(f"Data  : {DATA_PATH}")
print(f"Output: {OUTPUT_CSV}")

already_done = set()
resume_mode = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
if resume_mode:
    with open(OUTPUT_CSV) as _f:
        already_done = {int(row['id']) for row in csv.DictReader(_f)}
    print(f"Resuming: skipping {len(already_done)} already-processed graphs.")

# Pre-count lines for progress fraction
with open(DATA_PATH) as _f:
    n_total = sum(1 for _ in _f)
print(f"Graphs: {n_total}  |  remaining: {n_total - len(already_done)}\n")

# ---------------------------------------------------------------------------
# Main survey loop
# ---------------------------------------------------------------------------

file_mode = 'a' if resume_mode else 'w'

# Use fork so the child inherits all Sage/numpy globals — no re-import cost.
_mp_ctx = mp.get_context('fork')

def _new_pool():
    return _mp_ctx.Pool(processes=1)

with open(DATA_PATH) as f_in, \
     open(OUTPUT_CSV, file_mode, newline='') as f_out, \
     open(TIMED_OUT_FILE, 'a') as f_timeout:

    writer = csv.DictWriter(f_out, fieldnames=FIELDS)
    if not resume_mode:
        writer.writeheader()

    pool = _new_pool()
    total = cr_count = error_count = timeout_count = rows_since_flush = 0
    t_start = time.time()

    for raw in f_in:
        data   = json.loads(raw)
        hog_id = data['id']
        inv    = data['inv']
        total += 1

        if hog_id in already_done:
            continue

        n = None  # filled in on success
        try:
            future = pool.apply_async(_graph_task, (data['g6'], EDGE_SDP_THRESHOLD))
            result = future.get(timeout=GRAPH_TIMEOUT)

            if result is None:
                # disconnected or n < 2 — skip silently
                pass
            else:
                n = result['n']
                if result['is_lcr'] or result['is_ucr']:
                    writer.writerow({
                        "id":             hog_id,
                        "order":          n,
                        "size":           result['m'],
                        "cr":             result['is_lcr'] and result['is_ucr'],
                        "lambda_2":       round(result['l2_val'], 6),
                        "lambda_2_mult":  result['l2_mult'],
                        "lcr":            result['is_lcr'],
                        "ucr":            result['is_ucr'],
                        "lambda_n":       round(result['ln_val'], 6),
                        "lambda_n_mult":  result['ln_mult'],
                        "edge_orbits":    inv.get('NumberOfEdgeOrbits'),
                        "vertex_orbits":  inv.get('NumberOfVertexOrbits'),
                        "group_size":     inv.get('GroupSize'),
                        "bipartite":      inv.get('Bipartite'),
                        "regular":        inv.get('Regular'),
                        "1_walk_regular": result['walk_reg'],
                    })
                    cr_count += 1
                    rows_since_flush += 1
                    if rows_since_flush >= FLUSH_EVERY:
                        f_out.flush()
                        rows_since_flush = 0

        except mp.TimeoutError:
            timeout_count += 1
            pool.terminate()
            pool.join()
            pool = _new_pool()
            f_timeout.write(json.dumps({"id": hog_id, "n": inv.get('Order')}) + '\n')
            f_timeout.flush()
            print(f"\n  Timeout on HoG {hog_id} (>{GRAPH_TIMEOUT}s)")

        except Exception as exc:
            error_count += 1
            print(f"\n  Error on HoG {hog_id}: {type(exc).__name__}: {exc}")

        elapsed = time.time() - t_start
        rate    = total / elapsed if elapsed > 0 else 0
        eta     = _fmt_time((n_total - total) / rate) if rate > 0 else "?"
        n_str   = str(n) if n is not None else "?"
        sys.stdout.write(
            f"\r[{total:>6}/{n_total}] {100.0*total/n_total:5.1f}%"
            f"  HoG {hog_id:<6} n={n_str:<4}"
            f"  |  {cr_count} CR  {error_count} err  {timeout_count} timeout"
            f"  |  {_fmt_time(elapsed)}  ETA {eta}   "
        )
        sys.stdout.flush()

    pool.terminate()
    pool.join()

print(f"\n[{total}/{n_total}] Done — {cr_count} CR found, {error_count} errors, "
      f"{timeout_count} timeouts, {_fmt_time(time.time()-t_start)} total.")
