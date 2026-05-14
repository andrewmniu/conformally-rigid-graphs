"""
Exact conformal rigidity certification for the HoG survey.

Reads numerical_certs.csv and certifies each LCR/UCR graph. Directions
covered by a structural result are recorded immediately:
  - edge_transitive  (edge_orbits == 1)     covers LCR and UCR  (Thm 4.10)
  - 1_walk_regular                          covers LCR and UCR  (Thm 3.2)
  - reg_bipartite    (regular + bipartite)  covers UCR only     (Prop 3.11)

Remaining directions are certified via the exact pipeline in cr/exact_cert.sage
(isotypic decomposition + polyhedral LP).

Outputs
-------
needs_exact.csv     Rows from numerical_certs.csv requiring exact cert in at
                    least one direction. Written before the main loop starts.
exact_certs.csv     All rows of numerical_certs.csv with lcr_cert inserted
                    after lcr and ucr_cert inserted after ucr. Cert reasons:
                    edge_transitive, 1_walk_regular, reg_bipartite,
                    exact, infeasible, mult_issue, timed_out, error, ''
exact_certs.jsonl   One record per graph with an exact cert: {id, lcr, ucr}
                    where each value is a serialize_certificate dict or null.
infeasible.jsonl    {id, direction} for LP-infeasible graphs.
mult_issues.jsonl   {id, direction} for graphs with irrep multiplicity issues.
timeouts.jsonl      {id, n} for graphs that exceeded CERT_TIMEOUT.

Run from this directory:
    sage exact_certs.sage
"""

import csv
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.abspath('../../cr'))
from orbits import get_edge_orbits

load('../../cr/exact_cert.sage')


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV        = '../numerical/numerical_certs.csv'
DATA_PATH        = os.path.abspath('../raw_hog_data.jsonl')
NEEDS_EXACT_CSV  = 'needs_exact.csv'
OUTPUT_CSV       = 'exact_certs.csv'
PHI_JSONL        = 'exact_certs.jsonl'
INFEASIBLE_JSONL = 'infeasible.jsonl'
MULT_ISSUES_JSONL = 'mult_issues.jsonl'
TIMEOUT_JSONL    = 'timeouts.jsonl'

CERT_TIMEOUT = 300
FLUSH_EVERY  = 20

# numerical_certs.csv fieldnames in order
_NUM_FIELDS = [
    "id", "order", "size", "cr",
    "lambda_2", "lambda_2_mult", "lcr",
    "ucr", "lambda_n", "lambda_n_mult",
    "edge_orbits", "vertex_orbits", "group_size",
    "bipartite", "regular", "1_walk_regular",
]

# Insert cert columns adjacent to their direction
OUTPUT_FIELDS = [
    "id", "order", "size", "cr",
    "lambda_2", "lambda_2_mult", "lcr", "lcr_cert",
    "ucr", "ucr_cert", "lambda_n", "lambda_n_mult",
    "edge_orbits", "vertex_orbits", "group_size",
    "bipartite", "regular", "1_walk_regular",
]


# ---------------------------------------------------------------------------
# Structural cert helper
# ---------------------------------------------------------------------------

def _parse_bool(val):
    return str(val).strip() == 'True'

def _parse_float_or_none(val):
    if val in (None, '', 'None'):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _structural_cert(row):
    """
    Return (lcr_cert, ucr_cert) based on structural shortcuts.
    '' means direction is not active (False).
    A non-None string means the direction is covered structurally.
    None means the direction needs an exact certificate.
    """
    is_lcr = _parse_bool(row['lcr'])
    is_ucr = _parse_bool(row['ucr'])

    eo = _parse_float_or_none(row.get('edge_orbits'))
    edge_trans = (eo is not None and eo == 1.0)
    is_1wr     = _parse_bool(row.get('1_walk_regular', False))
    reg        = _parse_float_or_none(row.get('regular'))
    bip        = _parse_float_or_none(row.get('bipartite'))
    reg_bip    = (reg == 1.0 and bip == 1.0)

    if not is_lcr:
        lcr_cert = ''
    elif edge_trans:
        lcr_cert = 'edge_transitive'
    elif is_1wr:
        lcr_cert = '1_walk_regular'
    else:
        lcr_cert = None  # needs exact

    if not is_ucr:
        ucr_cert = ''
    elif edge_trans:
        ucr_cert = 'edge_transitive'
    elif is_1wr:
        ucr_cert = '1_walk_regular'
    elif reg_bip:
        ucr_cert = 'reg_bipartite'
    else:
        ucr_cert = None  # needs exact

    return lcr_cert, ucr_cert


# ---------------------------------------------------------------------------
# Worker (forked subprocess)
# ---------------------------------------------------------------------------

def _cert_task(g6, do_lcr, do_ucr):
    """
    Run _certify_exact for each requested direction. Automorphism group and
    edge orbits are computed once and shared. Returns a dict with per-direction
    (reason, serialized_cert) pairs.
    """
    import sys, os
    # Suppress prints from the exact cert pipeline inside the worker
    _devnull = open(os.devnull, 'w')
    _orig_stdout = sys.stdout
    sys.stdout = _devnull

    try:
        G = Graph(g6)
        G.relabel()
        aut_group   = G.automorphism_group()
        edge_orbits = get_edge_orbits(G, list(aut_group.gens()))

        out = {
            'lcr_reason': '', 'lcr_cert': None,
            'ucr_reason': '', 'ucr_cert': None,
        }

        for direction, key in (('lambda2', 'lcr'), ('lambdan', 'ucr')):
            if (key == 'lcr' and not do_lcr) or (key == 'ucr' and not do_ucr):
                continue
            try:
                phi_vectors, weights, lam, feasible = _certify_exact(
                    G, direction, aut_group, edge_orbits
                )
                if feasible is None:
                    out[f'{key}_reason'] = 'mult_issue'
                elif feasible is False:
                    out[f'{key}_reason'] = 'infeasible'
                else:
                    out[f'{key}_reason'] = 'exact'
                    out[f'{key}_cert'] = serialize_certificate(phi_vectors, weights, lam)
            except Exception as exc:
                out[f'{key}_reason'] = 'error'
                out[f'{key}_error'] = f"{type(exc).__name__}: {exc}"

        return out

    finally:
        sys.stdout = _orig_stdout
        _devnull.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

print(f"Loading g6 strings from {DATA_PATH} ...")
g6_map = {}
with open(DATA_PATH) as f:
    for raw in f:
        d = json.loads(raw)
        g6_map[d['id']] = d['g6']
print(f"  {len(g6_map)} graphs loaded.\n")

print(f"Reading {INPUT_CSV} ...")
with open(INPUT_CSV) as f:
    all_rows = list(csv.DictReader(f))
print(f"  {len(all_rows)} graphs.\n")


# ---------------------------------------------------------------------------
# Pre-pass: write needs_exact.csv
# ---------------------------------------------------------------------------

needs_exact_rows = []
for row in all_rows:
    lcr_cert, ucr_cert = _structural_cert(row)
    if lcr_cert is None or ucr_cert is None:
        needs_exact_rows.append(row)

print(f"Writing {NEEDS_EXACT_CSV} ...")
with open(NEEDS_EXACT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=_NUM_FIELDS)
    writer.writeheader()
    writer.writerows(needs_exact_rows)
print(f"  {len(needs_exact_rows)} graphs need exact certification.\n")


# ---------------------------------------------------------------------------
# Resume: skip IDs already in exact_certs.csv
# ---------------------------------------------------------------------------

already_done = set()
resume_mode = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
if resume_mode:
    with open(OUTPUT_CSV) as f:
        already_done = {int(row['id']) for row in csv.DictReader(f)}
    print(f"Resuming: {len(already_done)} rows already done.\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_mp_ctx = mp.get_context('fork')

def _new_pool():
    return _mp_ctx.Pool(processes=1)

file_mode = 'a' if resume_mode else 'w'

total = len(all_rows)
done = exact_count = struct_count = infeas_count = mult_count = timeout_count = error_count = 0
t_start = time.time()

with open(OUTPUT_CSV,       file_mode, newline='') as f_csv,  \
     open(PHI_JSONL,        'a')                   as f_phi,  \
     open(INFEASIBLE_JSONL, 'a')                   as f_infeas, \
     open(MULT_ISSUES_JSONL,'a')                   as f_mult,  \
     open(TIMEOUT_JSONL,    'a')                   as f_timeout:

    writer = csv.DictWriter(f_csv, fieldnames=OUTPUT_FIELDS)
    if not resume_mode:
        writer.writeheader()

    pool = _new_pool()
    rows_since_flush = 0

    for row in all_rows:
        hog_id = int(row['id'])
        n      = int(row['order'])
        done  += 1

        if hog_id in already_done:
            continue

        lcr_cert, ucr_cert = _structural_cert(row)
        needs_lcr = (lcr_cert is None)
        needs_ucr = (ucr_cert is None)

        out_row = {f: row[f] for f in _NUM_FIELDS}
        out_row['lcr_cert'] = lcr_cert if lcr_cert is not None else ''
        out_row['ucr_cert'] = ucr_cert if ucr_cert is not None else ''

        phi_lcr = phi_ucr = None

        if not needs_lcr and not needs_ucr:
            # Fully covered by structural results
            struct_count += 1
            writer.writerow(out_row)
        else:
            g6 = g6_map[hog_id]
            try:
                future = pool.apply_async(_cert_task, (g6, needs_lcr, needs_ucr))
                res    = future.get(timeout=CERT_TIMEOUT)

            except mp.TimeoutError:
                pool.terminate(); pool.join(); pool = _new_pool()
                if needs_lcr: out_row['lcr_cert'] = 'timed_out'
                if needs_ucr: out_row['ucr_cert'] = 'timed_out'
                writer.writerow(out_row)
                f_timeout.write(json.dumps({"id": hog_id, "n": n}) + '\n')
                f_timeout.flush()
                timeout_count += 1

            except Exception as exc:
                if needs_lcr: out_row['lcr_cert'] = 'error'
                if needs_ucr: out_row['ucr_cert'] = 'error'
                writer.writerow(out_row)
                print(f"\n  Error on HoG {hog_id}: {type(exc).__name__}: {exc}")
                error_count += 1

            else:
                for key in ('lcr', 'ucr'):
                    needs = needs_lcr if key == 'lcr' else needs_ucr
                    if not needs:
                        continue
                    reason = res[f'{key}_reason']
                    out_row[f'{key}_cert'] = reason
                    if reason == 'exact':
                        exact_count += 1
                        if key == 'lcr':
                            phi_lcr = res['lcr_cert']
                        else:
                            phi_ucr = res['ucr_cert']
                    elif reason == 'infeasible':
                        infeas_count += 1
                        f_infeas.write(json.dumps({"id": hog_id, "direction": key}) + '\n')
                    elif reason == 'mult_issue':
                        mult_count += 1
                        f_mult.write(json.dumps({"id": hog_id, "direction": key}) + '\n')
                    elif reason == 'error':
                        error_count += 1
                        print(f"\n  Error on HoG {hog_id} ({key}): {res.get(f'{key}_error', '')}")

                writer.writerow(out_row)

                if phi_lcr is not None or phi_ucr is not None:
                    f_phi.write(json.dumps({
                        "id":  hog_id,
                        "lcr": phi_lcr,
                        "ucr": phi_ucr,
                    }) + '\n')

        rows_since_flush += 1
        if rows_since_flush >= FLUSH_EVERY:
            f_csv.flush(); f_phi.flush(); f_mult.flush(); f_infeas.flush()
            rows_since_flush = 0

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta  = _fmt_time((total - done) / rate) if rate > 0 else "?"
        sys.stdout.write(
            f"\r[{done:>6}/{total}] {100.0*done/total:5.1f}%"
            f"  HoG {hog_id:<6} n={n:<4}  |"
            f"  struct {struct_count}  exact {exact_count}"
            f"  infeas {infeas_count}  mult {mult_count}"
            f"  to {timeout_count}  err {error_count}"
            f"  |  {_fmt_time(elapsed)}  ETA {eta}   "
        )
        sys.stdout.flush()

    pool.terminate(); pool.join()

print(f"\n[{done}/{total}] Done —"
      f"  struct {struct_count}  exact {exact_count}"
      f"  infeasible {infeas_count}  mult_issue {mult_count}"
      f"  timeout {timeout_count}  error {error_count}"
      f"  |  {_fmt_time(time.time()-t_start)} total.")
print(f"  CSV          -> {OUTPUT_CSV}")
print(f"  Needs exact  -> {NEEDS_EXACT_CSV}")
print(f"  Phi vectors  -> {PHI_JSONL}")
print(f"  Infeasible   -> {INFEASIBLE_JSONL}")
print(f"  Mult issues  -> {MULT_ISSUES_JSONL}")
print(f"  Timed out    -> {TIMEOUT_JSONL}")
