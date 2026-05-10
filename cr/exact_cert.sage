"""
Exact algebraic certification for conformally rigid graphs.

Produces a certificate vector phi over AA (algebraic reals) satisfying
  L * phi = lambda * phi
  sum_{(u,v) in O_i} (phi[u] - phi[v])^2 = |O_i|   for all edge orbits O_i

Works over QQbar throughout. Suitable for graphs up to ~250 vertices.

Functions
---------
eigenspace_exact       -- exact eigenspace basis over QQbar
build_subrep           -- subrep matrices Mg for each group element, via BFS from generators
subrep_character       -- character of the subrep (class reps only)
irrep_decomposition    -- multiplicities of each irrep
pair_complex_conjugates -- group active irreps into singletons and conjugate pairs
isotypic_projectors    -- exact projectors onto each real isotypic component
get_isotypic_representative -- extract one representative vector per isotypic component
orbit_energies         -- total squared-distance contribution per orbit per vector
exact_weights          -- Polyhedron over AA to find non-negative weights
certify_exact_lcr      -- full LCR pipeline (lambda_2)
certify_exact_ucr      -- full UCR pipeline (lambda_n)
combine_certificate    -- build phi = sum_i sqrt(w_i) * phi_i over AA
verify_certificate     -- check eigenvector equation and orbit energy constraints
certify_exact_lcr      -- full pipeline
"""



# ---------------------------------------------------------------------------
# Step 1: Exact eigenspace
# ---------------------------------------------------------------------------

def eigenspace_exact(G, which='lambda2'):
    """
    Compute an exact eigenspace of the graph Laplacian over QQbar.

    Works by factoring the characteristic polynomial over QQ first (cheap),
    then constructing Q(lambda) as a NumberField and solving the kernel there
    (fast concrete linear algebra), and finally embedding into QQbar.
    This avoids the heavy PARI stack usage of computing eigenvalues directly
    over QQbar for large graphs.

    Parameters
    ----------
    G : Sage graph (vertices labeled 0..n-1)
    which : {'lambda2', 'lambdan'}
        'lambda2' returns the smallest positive eigenvalue's eigenspace.
        'lambdan' returns the largest eigenvalue's eigenspace.

    Returns
    -------
    lam : AA element
        The exact eigenvalue.
    B : matrix over AA, shape (d, n)
        Row basis for the eigenspace (raw kernel basis, not normalized).
    """
    n = G.order()
    L = G.laplacian_matrix()   # over ZZ

    # Factor charpoly over QQ — cheap, avoids PARI stack pressure
    cp      = L.charpoly()
    factors = cp.factor()      # list of (irreducible poly over QQ, multiplicity)

    # Collect (approx_root, factor) pairs over RDF for identification
    candidates = []
    for f, _ in factors:
        for r in f.roots(RDF, multiplicities=False):
            candidates.append((float(r), f))
    candidates.sort(key=lambda x: x[0])

    pos = [(r, f) for r, f in candidates if r > 1e-10]
    if which == 'lambda2':
        target_r, target_f = pos[0]
    elif which == 'lambdan':
        target_r, target_f = candidates[-1]
    else:
        raise ValueError(f"which must be 'lambda2' or 'lambdan', got {which!r}")

    if target_f.degree() == 1:
        # Rational eigenvalue — solve over QQ directly
        lam_qq = QQ(-target_f[0] / target_f[1])
        LQ     = L.change_ring(QQ)
        B_qq   = (LQ - lam_qq * identity_matrix(QQ, n)).right_kernel().basis_matrix()
        return AA(lam_qq), B_qq.change_ring(AA)

    # Irrational eigenvalue — construct Q(lambda) and solve there
    R   = QQ['x']
    K   = NumberField(R(target_f), 'lam')
    lam_K = K.gen()

    # Pick the embedding K -> AA that sends lam_K closest to target_r
    emb = min(K.embeddings(AA), key=lambda e: abs(RR(e(lam_K)) - target_r))

    LK  = L.change_ring(K)
    B_K = (LK - lam_K * identity_matrix(K, n)).right_kernel().basis_matrix()

    # Embed lam and B into AA
    lam_aa = emb(lam_K)
    B_aa   = B_K.apply_map(emb)
    return lam_aa, B_aa


# ---------------------------------------------------------------------------
# Step 2: Action matrices
# ---------------------------------------------------------------------------

def build_subrep(B, aut_group):
    """
    Compute the d×d representation matrix for every group element.

    Solves Mg * B = B_perm only for the generators of aut_group, then
    propagates to the rest of the group by d×d matrix multiplication via BFS.

    Parameters
    ----------
    B : matrix over any ring, shape (d, n)
    aut_group : Sage PermutationGroup

    Returns
    -------
    subrep : dict mapping each group element g → matrix of shape (d, d).
    """
    d, n = B.nrows(), B.ncols()
    R = B.base_ring()
    gens = aut_group.gens()

    Mg_gens = {}
    for g in gens:
        B_perm = matrix(R, d, n)
        for i in range(n):
            B_perm[:, g(i)] = B[:, i]
        Mg_gens[g] = B.solve_left(B_perm)

    identity = aut_group.identity()
    subrep = {identity: matrix.identity(R, d)}
    frontier = [identity]

    while frontier:
        curr = frontier.pop(0)
        for gen in gens:
            new_g = curr * gen
            if new_g not in subrep:
                subrep[new_g] = subrep[curr] * Mg_gens[gen]
                frontier.append(new_g)

    return subrep


# ---------------------------------------------------------------------------
# Step 3: Character of the subrep
# ---------------------------------------------------------------------------

def subrep_character(subrep, aut_group):
    """
    Compute the character of the subrepresentation.

    Evaluates the trace of Mg at one representative per conjugacy class.

    Parameters
    ----------
    subrep : dict mapping group element → d×d matrix over QQbar
    aut_group : Sage PermutationGroup

    Returns
    -------
    list of QQbar elements, one trace per conjugacy class (in Sage's class ordering).
    """
    return [subrep[C.representative()].trace()
            for C in aut_group.conjugacy_classes()]


# ---------------------------------------------------------------------------
# Step 4: Irrep decomposition
# ---------------------------------------------------------------------------

def irrep_decomposition(chi, aut_group, ct, classes):
    """
    Decompose the subrepresentation into irreducible components.

    Uses the inner product formula:
        m_i = (1/|G|) * sum_C |C| * chi[C] * conj(chi_i[C])

    Parameters
    ----------
    chi : list of QQbar
        Character of the subrep (output of subrep_character).
    aut_group : Sage PermutationGroup
    ct : character table (aut_group.character_table())
    classes : conjugacy classes (aut_group.conjugacy_classes())

    Returns
    -------
    list of (irrep_idx, dim, mult) for all irreps with mult > 0.
    """
    order = aut_group.order()

    result = []
    for idx, chi_i in enumerate(ct):
        inner = sum(
            classes[j].cardinality() * QQbar(chi[j]) * QQbar(chi_i[j]).conjugate()
            for j in range(len(classes))
        )
        mult = ZZ((inner / order).real())
        if mult > 0:
            result.append((idx, int(chi_i[0]), int(mult)))

    return result


# ---------------------------------------------------------------------------
# Step 4b: Group irreps for real representation theory
# ---------------------------------------------------------------------------

def frobenius_schur_indicators(aut_group, ct, classes, decomp):
    """
    Compute the Frobenius-Schur indicator for each active irrep.

    FS(chi_i) = (1/|G|) * sum_{g in G} chi_i(g^2)
              = (1/|G|) * sum_C |C| * chi_i(g_C^2)

    For each conjugacy class C with representative g_C, we find which class
    g_C^2 belongs to and look up that column of the character table.

    Parameters
    ----------
    aut_group : Sage PermutationGroup
    ct : character table (aut_group.character_table())
    classes : conjugacy classes (aut_group.conjugacy_classes())
    decomp : list of (irrep_idx, dim, mult) from irrep_decomposition

    Returns
    -------
    dict mapping irrep_idx → FS indicator (-1, 0, or +1) for active irreps.
    """
    order = aut_group.order()

    # For each class C_j, find index k such that g_C_j^2 lies in C_k.
    # Use libgap for class membership to avoid pexpect line-length overflow
    # when serializing permutation elements for large graphs.
    from sage.libs.gap.libgap import libgap
    lgap_group = libgap(aut_group)
    lgap_classes = lgap_group.ConjugacyClasses()
    squared_class_idx = []
    for C_lgap in lgap_classes:
        rep2 = C_lgap.Representative() ** 2
        for k, C2_lgap in enumerate(lgap_classes):
            if lgap_group.IsConjugate(rep2, C2_lgap.Representative()):
                squared_class_idx.append(k)
                break

    result = {}
    for idx, dim, mult in decomp:
        chi_i = ct[idx]
        fs_val = sum(
            classes[j].cardinality() * QQbar(chi_i[squared_class_idx[j]])
            for j in range(len(classes))
        ) / order
        result[idx] = ZZ(fs_val.real())

    return result

def pair_complex_conjugates(decomp, ct, fs_indicators):
    """
    Group active irreps into singletons (real/quaternionic) and conjugate pairs (complex).

    Real type (FS=+1) and quaternionic type (FS=-1) are passed through as singletons.
    Complex type (FS=0) irreps are paired with their conjugate partner.

    Parameters
    ----------
    decomp : list of (irrep_idx, dim, mult) from irrep_decomposition
    ct : character table (aut_group.character_table())
    fs_indicators : dict mapping irrep_idx → FS indicator, from frobenius_schur_indicators()

    Returns
    -------
    list of lists of int — each inner list is either [i] (singleton) or [i, j] (conjugate pair).
    Key used downstream is always the first element of the group.
    """
    handled = set()
    pairs = []

    for idx, _, _ in decomp:
        if idx in handled:
            continue
        if fs_indicators[idx] == 0:  # complex type: find conjugate partner
            partner = None
            for jdx, _, _ in decomp:
                if jdx != idx and jdx not in handled:
                    if all(ct[jdx][k] == ct[idx][k].conjugate()
                           for k in range(ct.ncols())):
                        partner = jdx
                        break
            if partner is None:
                raise ValueError(
                    f"Complex-type irrep {idx} has no conjugate partner among active irreps"
                )
            pairs.append([idx, partner])
            handled.update([idx, partner])
        else:  # real (FS=+1) or quaternionic (FS=-1): singleton
            pairs.append([idx])
            handled.add(idx)

    return pairs


# ---------------------------------------------------------------------------
# Step 5: Isotypic projectors
# ---------------------------------------------------------------------------

def isotypic_projectors(subrep, aut_group, ct, classes, paired_irreps):
    """
    Compute the exact projector onto each real isotypic component.

    For each group of irrep indices:
      - Singleton [i] (real or quaternionic): P = (d_i/|G|) * sum_g conj(chi_i(g)) * Mg
      - Conjugate pair [i, j] (complex): P = P_i + P_j  (this sum is real-valued)

    Summed by conjugacy class for efficiency.

    Parameters
    ----------
    subrep : dict mapping group element → d×d matrix over QQbar
    aut_group : Sage PermutationGroup
    ct : character table (aut_group.character_table())
    classes : conjugacy classes (aut_group.conjugacy_classes())
    paired_irreps : list of lists of int (output of pair_complex_conjugates)

    Returns
    -------
    dict mapping first irrep index of each group → d×d matrix over QQbar.
    """
    order = aut_group.order()
    d = next(iter(subrep.values())).nrows()

    # Precompute class sums of Mg matrices (shared across all irreps)
    class_sums = [sum(subrep[g] for g in C) for C in classes]

    projectors = {}
    for group in paired_irreps:
        P = matrix(QQbar, d, d)
        for idx in group:
            chi_i = ct[idx]
            d_i = QQbar(chi_i[0])
            P_i = matrix(QQbar, d, d)
            for j, cs in enumerate(class_sums):
                P_i += QQbar(chi_i[j]).conjugate() * cs
            P += (d_i / order) * P_i
        projectors[group[0]] = P

    return projectors


# ---------------------------------------------------------------------------
# Step 6: Lift projector vectors
# ---------------------------------------------------------------------------

def get_isotypic_representative(projectors, B):
    """
    Extract a vector from each isotypic component and lift it to n-space.

    Scans rows of each projector for the first one that lifts to a nonzero
    vector in n-space, then takes the real part. P[0] can be zero if the first
    subrep basis vector has no component in a given isotypic component.

    Parameters
    ----------
    projectors : dict mapping irrep_idx → d×d matrix over QQbar
    B : matrix over QQbar, shape (d, n)

    Returns
    -------
    dict mapping irrep_idx → vector over AA of length n.
    """
    result = {}
    for idx, P in projectors.items():
        phi = None
        for i in range(P.nrows()):
            v = vector(AA, [x.real() for x in (P[i] * B)])
            if any(x != 0 for x in v):
                phi = v
                break
        if phi is None:
            raise ValueError(f"Projector for irrep {idx} lifts to the zero vector on all rows")
        result[idx] = phi
    return result


# ---------------------------------------------------------------------------
# Step 7: Orbit energies
# ---------------------------------------------------------------------------

def orbit_energies(phi_vectors, edge_orbits):
    """
    Compute the total squared-distance contribution of each vector across each orbit.

    For each vector phi_i and each orbit O_j:
        E[i][j] = sum_{(u,v) in O_j} (phi_i[u] - phi_i[v])^2

    Parameters
    ----------
    phi_vectors : dict mapping irrep_idx → vector over QQbar of length n
    edge_orbits : list of list of (int, int)

    Returns
    -------
    dict mapping irrep_idx → list of orbit energies (length = number of orbits).
    """
    return {
        idx: [sum((phi[u] - phi[v])**2 for u, v in orbit) for orbit in edge_orbits]
        for idx, phi in phi_vectors.items()
    }


# ---------------------------------------------------------------------------
# Step 8: Exact weights via PPL LP
# ---------------------------------------------------------------------------

def exact_weights(energy_dict, edge_orbits):
    """
    Find non-negative weights w_i (over AA) such that:

        sum_i w_i * energy_dict[i][j] = |O_j|   for all orbits j

    Uses Sage's exact Polyhedron over AA — no rational LP decomposition needed.

    Parameters
    ----------
    energy_dict : dict mapping irrep_idx → list of orbit energies (AA elements)
    edge_orbits : list of list of (int, int)

    Returns
    -------
    dict mapping irrep_idx → AA weight, or None if infeasible.
    """
    keys = list(energy_dict.keys())
    orbit_sizes = [len(orbit) for orbit in edge_orbits]
    k = len(keys)

    # Inequalities: w_i >= 0
    # Format: [b, a_0, ..., a_{k-1}] means b + sum a_i * x_i >= 0
    ieqs = []
    for i in range(k):
        row = [AA(0)] * (k + 1)
        row[i + 1] = AA(1)
        ieqs.append(row)

    # Equalities: sum_i w_i * E_i[j] = |O_j|
    # Format: [b, a_0, ..., a_{k-1}] means b + sum a_i * x_i == 0
    eqns = []
    for j, size in enumerate(orbit_sizes):
        row = [-AA(size)] + [AA(energy_dict[key][j]) for key in keys]
        eqns.append(row)

    P = Polyhedron(ieqs=ieqs, eqns=eqns, base_ring=AA)
    if P.is_empty():
        print("FAIL: no feasible weights found.")
        return None

    pt = P.an_element()
    return {key: pt[i] for i, key in enumerate(keys)}


# ---------------------------------------------------------------------------
# Step 9: Combine certificate vector
# ---------------------------------------------------------------------------

def combine_certificate(phi_vectors, weights):
    """
    Build the exact certificate vector phi = sum_i sqrt(w_i) * phi_i over AA.

    Parameters
    ----------
    phi_vectors : dict mapping irrep_idx → vector over QQbar of length n
    weights : dict mapping irrep_idx → non-negative rational weight

    Returns
    -------
    vector over AA of length n.
    """
    n = next(iter(phi_vectors.values())).degree()
    phi = vector(AA, n)
    for idx, phi_i in phi_vectors.items():
        c = AA(weights[idx]).sqrt()
        phi += c * vector(AA, [AA(x) for x in phi_i])
    return phi


# ---------------------------------------------------------------------------
# Step 10: Verify certificate
# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def deserialize_phi(records):
    """
    Reconstruct an exact AA vector from a list of {minpoly, approx} dicts
    as produced by serialize_phi in exact_certs.sage.

    For each entry, finds the unique root of the minimal polynomial (over AA)
    nearest to the given 50-digit decimal approximation.

    Parameters
    ----------
    records : list of dicts with keys 'minpoly' (list of rational strings)
              and 'approx' (50-digit decimal string)

    Returns
    -------
    vector over AA
    """
    R = QQ['x']
    phi = []
    for rec in records:
        p = R([QQ(c) for c in rec['minpoly']])
        target = RR(rec['approx'])
        roots = p.roots(AA, multiplicities=False)
        x = min(roots, key=lambda r: abs(RR(r) - target))
        phi.append(x)
    return vector(AA, phi)


# ---------------------------------------------------------------------------

def verify_certificate(phi, G, lam, edge_orbits):
    """
    Check that phi is an exact conformal rigidity certificate.

    Verifies:
      1. L * phi = lam * phi  (phi lies in the correct eigenspace)
      2. sum_{(u,v) in O_i} (phi[u] - phi[v])^2 = |O_i|  for all orbits O_i

    Parameters
    ----------
    phi : vector over AA of length n
    G : Sage graph (vertices labeled 0..n-1)
    lam : exact eigenvalue (QQbar or AA), from eigenspace_exact
    edge_orbits : list of list of (int, int)

    Returns
    -------
    bool : True if all checks pass.
    """
    L = G.laplacian_matrix().change_ring(AA)
    lam = AA(lam)
    ok = True

    residual = L * phi - lam * phi
    if any(x != 0 for x in residual):
        print("FAIL: phi is not an eigenvector of L")
        ok = False

    for i, orbit in enumerate(edge_orbits):
        energy = sum((phi[u] - phi[v])**2 for u, v in orbit)
        if energy != len(orbit):
            print(f"FAIL: orbit {i} energy = {energy}, expected {len(orbit)}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Step 11: Serialization and independent verification
# ---------------------------------------------------------------------------

def _ser_aa(x):
    """Serialize a single AA element as {minpoly, approx}."""
    p = x.minpoly()
    return {
        "minpoly": [str(c) for c in p.list()],
        "approx":  str(x.numerical_approx(digits=50)),
    }

def _ser_weight(w):
    """Serialize a weight: as a rational string if possible, else {minpoly, approx}."""
    w = AA(w)
    try:
        return str(QQ(w))
    except (TypeError, ValueError):
        return _ser_aa(w)

def serialize_certificate(phi_vectors, weights, lam):
    """
    Serialize a certificate as (lam, phi_i components, weights) for external
    verification.  Avoids serializing the assembled phi = sum_i sqrt(w_i)*phi_i,
    whose entries have high algebraic degree and expensive minpoly computation.

    Format
    ------
    {
        "lam": {"minpoly": [...], "approx": "..."},
        "components": [
            {
                "phi":    [{"minpoly": [...], "approx": "..."}, ...],
                "weight": "p/q"   (rational string, or {minpoly, approx} if irrational)
            },
            ...
        ]
    }

    Verification protocol (independent of the solver):
      1. Reconstruct lam and each phi_i from their serializations.
      2. Check L * phi_i = lam * phi_i for each component i.
      3. Check sum_i w_i * sum_{(u,v) in O_j} (phi_i[u]-phi_i[v])^2 = |O_j|
         for every edge orbit O_j.
    """
    return {
        "lam": _ser_aa(lam),
        "components": [
            {
                "phi":    [_ser_aa(AA(x)) for x in phi_vectors[key]],
                "weight": _ser_weight(weights[key]),
            }
            for key in phi_vectors
        ],
    }


def _deser_aa(d):
    """Reconstruct an AA element from a {minpoly, approx} dict."""
    R = QQ['x']
    p = R([QQ(c) for c in d['minpoly']])
    target = RR(d['approx'])
    roots  = p.roots(AA, multiplicities=False)
    return min(roots, key=lambda r: abs(RR(r) - target))

def _deser_weight(w):
    """Reconstruct a weight from its serialized form (rational string or dict)."""
    if isinstance(w, str):
        return AA(QQ(w))
    return _deser_aa(w)

def deserialize_and_verify(record, G):
    """
    Independently verify a certificate produced by serialize_certificate.

    Requires only the serialized record and the graph G (vertices labeled 0..n-1).
    Computes the automorphism group and edge orbits from G directly.

    Checks
    ------
    1. L * phi_i = lam * phi_i  (exact, over AA) for each component i.
    2. sum_i w_i * E_i[j] = |O_j|  (exact, over AA) for every edge orbit j,
       where E_i[j] = sum_{(u,v) in O_j} (phi_i[u] - phi_i[v])^2.

    Returns True if all checks pass.
    """
    lam   = _deser_aa(record['lam'])
    L     = G.laplacian_matrix().change_ring(AA)
    lam_aa = AA(lam)

    aut_group   = G.automorphism_group()
    edge_orbits = get_edge_orbits(G, list(aut_group.gens()))

    ok = True
    components = []
    for comp in record['components']:
        phi_i = vector(AA, [_deser_aa(e) for e in comp['phi']])
        w_i   = _deser_weight(comp['weight'])
        components.append((phi_i, w_i))

        # Check eigenvector condition per component. A linear combination of
        # eigenvectors is still an eigenvector, so this implies L*phi = lam*phi
        # for the assembled phi without needing to compute it.
        residual = L * phi_i - lam_aa * phi_i
        if any(x != 0 for x in residual):
            print("FAIL: component phi_i is not an eigenvector of L")
            ok = False

    # Assemble phi = sum_i sqrt(w_i) * phi_i and check the orbit energy
    # condition directly. Checking sum_i w_i * E_i[j] = |O_j| individually
    # would require knowing the cross terms cancel (isotypic orthogonality),
    # which is not obvious to an independent verifier.
    n   = G.order()
    phi = vector(AA, n)
    for phi_i, w_i in components:
        phi += w_i.sqrt() * phi_i

    for j, orbit in enumerate(edge_orbits):
        energy = sum((phi[u] - phi[v])**2 for u, v in orbit)
        if energy != AA(len(orbit)):
            print(f"FAIL: orbit {j} energy = {energy}, expected {len(orbit)}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Step 12: Full pipeline
# ---------------------------------------------------------------------------

def _certify_exact(G, which, aut_group, edge_orbits, verbose=False):
    """
    Shared pipeline for exact LCR/UCR certification.

    Parameters
    ----------
    G          : Sage graph (vertices labeled 0..n-1)
    which      : 'lambda2' or 'lambdan'
    aut_group  : PermutationGroup (already resolved by caller)
    edge_orbits: list of list of (int, int) (already resolved by caller)
    verbose    : if True, print each pipeline stage as it runs

    Returns
    -------
    phi_vectors : dict mapping irrep_idx -> AA vector, or None on failure.
    weights     : dict mapping irrep_idx -> AA weight, or None on failure.
    lam         : exact eigenvalue (AA), or None on failure.
    feasible    : True if LP succeeded, False if LP infeasible,
                  None if eigenspace has multiplicity issues.
    """
    def log(msg):
        if verbose:
            print(f"    [{which}] {msg}", flush=True)

    log("character_table + conjugacy_classes ...")
    ct = aut_group.character_table()
    classes = aut_group.conjugacy_classes()

    log("eigenspace_exact ...")
    lam, B = eigenspace_exact(G, which=which)
    log(f"eigenspace done: dim={B.nrows()}")

    log("build_subrep ...")
    subrep = build_subrep(B, aut_group)
    log("subrep done")

    log("subrep_character + irrep_decomposition + FS indicators ...")
    chi = subrep_character(subrep, aut_group)
    decomp = irrep_decomposition(chi, aut_group, ct, classes)
    fs_indicators = frobenius_schur_indicators(aut_group, ct, classes, decomp)
    log("decomposition done")

    multiplicity_issues = []
    for idx, dim, mult in decomp:
        fs = fs_indicators[idx]
        expected_mult = 2 if fs == -1 else 1
        if mult != expected_mult:
            multiplicity_issues.append({
                "irrep_idx": int(idx),
                "fs":        int(fs),
                "mult":      int(mult),
                "expected":  int(expected_mult),
            })

    if multiplicity_issues:
        print(f"MULTIPLICITY ISSUES ({which}): {multiplicity_issues}")
        return None, None, None, None

    log("isotypic_projectors ...")
    paired_irreps = pair_complex_conjugates(decomp, ct, fs_indicators)
    projectors = isotypic_projectors(subrep, aut_group, ct, classes, paired_irreps)
    log("projectors done")

    log("get_isotypic_representative + orbit_energies ...")
    phi_vectors = get_isotypic_representative(projectors, B)
    energies = orbit_energies(phi_vectors, edge_orbits)

    log("exact_weights (LP) ...")
    weights = exact_weights(energies, edge_orbits)
    if weights is None:
        log("LP infeasible")
        return None, None, None, False

    log("done!")
    return phi_vectors, weights, lam, True


def certify_exact_lcr(G, aut_group=None, edge_orbits=None):
    """
    Exact lower conformal rigidity certificate for G (uses lambda_2).

    Parameters
    ----------
    G           : Sage graph (vertices labeled 0..n-1)
    aut_group   : PermutationGroup or None (computed if not provided)
    edge_orbits : list of list of (int, int) or None (computed if not provided)

    Returns
    -------
    phi : vector over AA of length n, or None if certification fails.
    """
    if aut_group is None:
        aut_group = G.automorphism_group()
    if edge_orbits is None:
        edge_orbits = get_edge_orbits(G, aut_group.gens())
    phi_vectors, weights, lam, feasible = _certify_exact(G, 'lambda2', aut_group, edge_orbits)
    if phi_vectors is None:
        return None
    return combine_certificate(phi_vectors, weights)


def certify_exact_ucr(G, aut_group=None, edge_orbits=None):
    """
    Exact upper conformal rigidity certificate for G (uses lambda_n).

    Parameters
    ----------
    G           : Sage graph (vertices labeled 0..n-1)
    aut_group   : PermutationGroup or None (computed if not provided)
    edge_orbits : list of list of (int, int) or None (computed if not provided)

    Returns
    -------
    phi : vector over AA of length n, or None if certification fails.
    """
    if aut_group is None:
        aut_group = G.automorphism_group()
    if edge_orbits is None:
        edge_orbits = get_edge_orbits(G, aut_group.gens())
    phi_vectors, weights, lam, feasible = _certify_exact(G, 'lambdan', aut_group, edge_orbits)
    if phi_vectors is None:
        return None
    return combine_certificate(phi_vectors, weights)
