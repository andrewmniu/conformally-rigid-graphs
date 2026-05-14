"""
Conformal rigidity SDP solvers.

Implements the feasibility SDPs from the paper:

  (11)  find  Φ ≽ 0
        s.t.  LΦ = λΦ
              ⟨L^{uv}, Φ⟩_F = 1   for all uv ∈ E

  (12)  find  Z ≽ 0
        s.t.  ⟨L̄^{uv}, Z⟩_F = 1  for all uv ∈ E

where L̄^{uv} = (B[u] - B[v])(B[u] - B[v])^T and B is an orthonormal
basis for E_λ. Any solution Z to (12) lifts to Φ = B Z B^T solving (11).

The symmetry-reduced version replaces per-edge constraints with per-orbit
constraints ⟨L̄^{O_i}, Z⟩_F = |O_i|.

Architecture
------------
Inner layer (no Sage types):
  get_eigenspace            -- numpy dense eigendecomposition
  find_edge_isometric_embedding  -- eq. (12), one constraint per edge
  find_orbit_isometric_embedding -- symmetry-reduced, one constraint per orbit
  verify_solution           -- checks constraints on a returned Z

Outer layer (imports sage.all, bridges Sage → numpy):
  is_lcr(G_sage)  -- wraps find_edge_isometric_embedding with λ₂ eigenspace
  is_ucr(G_sage)  -- wraps find_edge_isometric_embedding with λ_n eigenspace

Eigenvalue computation: np.linalg.eigh on the dense Laplacian.
Not suitable for n ≳ 2000; use scipy.sparse.linalg.eigsh for large graphs
(see the Big Example notebook).

Solver: MOSEK via CVXPY, with SCS fallback on solver error.
"""

import numpy as np
import cvxpy as cp


# ---------------------------------------------------------------------------
# Inner layer
# ---------------------------------------------------------------------------

def get_eigenspace(L_np: np.ndarray, which: str = "lambda2") -> tuple[float, np.ndarray]:
    """
    Compute an eigenspace of the graph Laplacian.

    Parameters
    ----------
    L_np : np.ndarray, shape (n, n)
        Dense symmetric Laplacian matrix.
    which : {'lambda2', 'lambdan'}
        'lambda2' returns the eigenspace of the smallest positive eigenvalue.
        'lambdan' returns the eigenspace of the largest eigenvalue.

    Returns
    -------
    lam : float
        The eigenvalue.
    B : np.ndarray, shape (n, d)
        Orthonormal basis for the eigenspace (columns are eigenvectors).
    """
    evals, evecs = np.linalg.eigh(L_np)

    if which == "lambda2":
        target = evals[1]
    elif which == "lambdan":
        target = evals[-1]
    else:
        raise ValueError(f"which must be 'lambda2' or 'lambdan', got {which!r}")

    idx = np.where(np.abs(evals - target) < 1e-7)[0]
    return float(target), evecs[:, idx]


def _solve_feasibility_sdp(
    L_bars: list[np.ndarray],
    rhs: list[float],
    tol: float = 1e-8,
    solver: str = 'MOSEK',
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """
    Solve the feasibility SDP:

        find  Z ≽ 0
        s.t.  tr(L_bars[i] @ Z) == rhs[i]   for all i

    Parameters
    ----------
    L_bars : list of np.ndarray, each shape (d, d)
        Compressed Laplacian operators.
    rhs : list of float
        Right-hand side values (edge counts or orbit sizes).
    tol : float, default 1e-8
        Solver tolerance. Passed to MOSEK as MSK_DPAR_INTPNT_CO_TOL_PFEAS,
        TOL_DFEAS, TOL_INFEAS, TOL_REL_GAP; to SCS as eps.
    solver : {'MOSEK', 'SCS'}, default 'MOSEK'
        Solver to use. MOSEK falls back to SCS on error.

    Returns
    -------
    status : str
        CVXPY problem status string, e.g. 'optimal' or 'infeasible'.
    Z : np.ndarray or None
        The solution matrix if a solution is returned, else None.
    residuals : np.ndarray or None
        Per-constraint residuals tr(L_bars[i] Z) - rhs[i] on the returned Z.
        None when Z is None.
    """
    d = L_bars[0].shape[0]
    Z = cp.Variable((d, d), symmetric=True)
    constraints = [Z >> 0]
    for L_bar, r in zip(L_bars, rhs):
        constraints.append(cp.trace(L_bar @ Z) == r)

    # Minimize trace rather than 0: gives MOSEK a non-degenerate dual and
    # avoids UNKNOWN status on highly underdetermined feasibility problems.
    prob = cp.Problem(cp.Minimize(cp.trace(Z)), constraints)

    solver_upper = solver.upper()
    if solver_upper == 'MOSEK':
        try:
            prob.solve(solver=cp.MOSEK, mosek_params={
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   tol,
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   tol,
                'MSK_DPAR_INTPNT_CO_TOL_INFEAS':  tol,
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
            })
            solver_used = "MOSEK"
        except cp.SolverError as e:
            print(f"MOSEK failed ({type(e).__name__}): {e}")
            prob.solve(solver=cp.SCS, eps=tol)
            solver_used = "SCS (fallback)"
    elif solver_upper == 'SCS':
        prob.solve(solver=cp.SCS, eps=tol)
        solver_used = "SCS"
    else:
        raise ValueError(f"solver must be 'MOSEK' or 'SCS', got {solver!r}")

    print(f"Solver used: {solver_used}")

    if prob.status in ("optimal", "optimal_inaccurate") or (
        prob.status == "unknown" and Z.value is not None
    ):
        Z_val = Z.value
        residuals = np.array([
            float(np.trace(L_bar @ Z_val)) - r
            for L_bar, r in zip(L_bars, rhs)
        ])
        return prob.status, Z_val, residuals
    return prob.status, None, None


def find_edge_isometric_embedding(
    B: np.ndarray,
    edges: list[tuple[int, int]],
    tol: float = 1e-8,
    solver: str = 'MOSEK',
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """
    Feasibility SDP matching eq. (12): one constraint per edge.

        find  Z ≽ 0
        s.t.  ⟨L̄^{uv}, Z⟩_F = 1   for all uv ∈ edges

    where L̄^{uv} = (B[u] - B[v])(B[u] - B[v])^T.

    Parameters
    ----------
    B : np.ndarray, shape (n, d)
        Orthonormal basis for E_λ.
    edges : list of (int, int)
        Edge list with integer vertex labels 0..n-1.
    tol : float, default 1e-8
        Solver tolerance forwarded to _solve_feasibility_sdp.

    Returns
    -------
    status : str
        'optimal' if G is (L/U)CR, 'infeasible' otherwise.
    Z : np.ndarray or None
        Solution matrix if optimal, else None. Lifts to Φ = B @ Z @ B.T.
    residuals : np.ndarray or None
        Per-edge residuals ⟨L̄^{uv}, Z⟩_F - 1 on the returned Z, in the
        same order as `edges`. None when Z is None. Callers can report
        max|residuals| or sqrt(mean(residuals**2)) as a numerical
        feasibility quality on the returned solution.
    """
    L_bars = [np.outer(B[u] - B[v], B[u] - B[v]) for u, v in edges]
    rhs = [1.0] * len(edges)
    return _solve_feasibility_sdp(L_bars, rhs, tol=tol, solver=solver)


def find_orbit_isometric_embedding(
    B: np.ndarray,
    edge_orbits: list[list[tuple[int, int]]],
    tol: float = 1e-8,
    solver: str = 'MOSEK',
) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    """
    Symmetry-reduced feasibility SDP: one constraint per edge orbit.

        find  Z ≽ 0
        s.t.  ⟨L̄^{O_i}, Z⟩_F = |O_i|   for all orbits O_i

    where L̄^{O_i} = Σ_{uv ∈ O_i} (B[u] - B[v])(B[u] - B[v])^T.

    This is the special case of find_edge_isometric_embedding when
    edge_orbits contains singleton lists (trivial symmetry group).

    Parameters
    ----------
    B : np.ndarray, shape (n, d)
        Orthonormal basis for E_λ.
    edge_orbits : list of list of (int, int)
        Partition of edges into orbits under the symmetry group.
    tol : float, default 1e-8
        Solver tolerance forwarded to _solve_feasibility_sdp.
    solver : {'MOSEK', 'SCS'}, default 'MOSEK'
        Solver to use. Forwarded to _solve_feasibility_sdp.

    Returns
    -------
    status : str
    Z : np.ndarray or None
    residuals : np.ndarray or None
        Per-orbit residuals ⟨L̄^{O_i}, Z⟩_F - |O_i| on the returned Z,
        in the same order as `edge_orbits`. None when Z is None.
    """
    L_bars = []
    rhs = []
    for orbit in edge_orbits:
        L_bar = sum(np.outer(B[u] - B[v], B[u] - B[v]) for u, v in orbit)
        L_bars.append(L_bar)
        rhs.append(float(len(orbit)))
    return _solve_feasibility_sdp(L_bars, rhs, tol=tol, solver=solver)


def _verify_solution(
    B: np.ndarray,
    Z: np.ndarray,
    edge_orbits: list[list[tuple[int, int]]],
    tol: float = 1e-5,
) -> bool:
    """
    Check that tr(L̄^{O_i} Z) = |O_i| for all orbits.

    Works for both the unsymmetrized case (one singleton orbit per edge)
    and the symmetrized case.

    Parameters
    ----------
    B : np.ndarray, shape (n, d)
    Z : np.ndarray, shape (d, d)
    edge_orbits : list of list of (int, int)
    tol : float

    Returns
    -------
    bool
        True if all constraints are satisfied within tol.
    """
    ok = True
    for i, orbit in enumerate(edge_orbits):
        L_bar = sum(np.outer(B[u] - B[v], B[u] - B[v]) for u, v in orbit)
        val = float(np.trace(L_bar @ Z))
        target = float(len(orbit))
        if abs(val - target) > tol:
            print(f"Orbit {i}: tr(L̄ Z) = {val:.8f}, expected {target:.1f}  [FAIL]")
            ok = False
    return ok


def verify_compressed_edge_isometry(
    B: np.ndarray,
    Z: np.ndarray,
    edges: list[tuple[int, int]],
    tol: float = 1e-5,
) -> bool:
    """
    Check that tr(L̄^{uv} Z) = 1 for every edge individually.

    Parameters
    ----------
    B : np.ndarray, shape (n, d)
    Z : np.ndarray, shape (d, d)
    edges : list of (int, int)
    tol : float

    Returns
    -------
    bool
        True if every per-edge constraint is satisfied within tol.
    """
    ok = True
    for u, v in edges:
        L_bar = np.outer(B[u] - B[v], B[u] - B[v])
        val = float(np.trace(L_bar @ Z))
        if abs(val - 1.0) > tol:
            print(f"Edge ({u}, {v}): tr(L̄ Z) = {val:.8f}, expected 1.0  [FAIL]")
            ok = False
    return ok


def verify_compressed_orbit_isometry(
    B: np.ndarray,
    Z: np.ndarray,
    edge_orbits: list[list[tuple[int, int]]],
    tol: float = 1e-5,
) -> bool:
    """
    Check that tr(L̄^{O_i} Z) = |O_i| for every edge orbit.

    Parameters
    ----------
    B : np.ndarray, shape (n, d)
    Z : np.ndarray, shape (d, d)
    edge_orbits : list of list of (int, int)
    tol : float

    Returns
    -------
    bool
        True if every per-orbit constraint is satisfied within tol.
    """
    return _verify_solution(B, Z, edge_orbits, tol=tol)


# ---------------------------------------------------------------------------
# Embedding lift and verification
# ---------------------------------------------------------------------------

def lift_embedding(B: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Lift the SDP solution Z to the PSD matrix Φ from equation (11).

    Computes Φ = B Z Bᵀ ∈ S⁺ⁿ. Any solution Z to the parametrized
    SDP (12) lifts to a solution Φ of the full SDP (11) this way.

    Parameters
    ----------
    B : np.ndarray, shape (n, d)
        Orthonormal eigenspace basis.
    Z : np.ndarray, shape (d, d)
        SDP solution matrix.

    Returns
    -------
    Phi : np.ndarray, shape (n, n)
    """
    return B @ Z @ B.T


def _verify_embedding(
    Phi: np.ndarray,
    edge_orbits: list[list[tuple[int, int]]],
    tol: float = 1e-5,
) -> bool:
    """
    Check that the total squared distance across each orbit equals |O_i|.

    For a valid certificate the squared distance for edge uv is
    Φ[u,u] + Φ[v,v] - 2Φ[u,v], and the orbit total must equal |O_i|.

    Parameters
    ----------
    Phi : np.ndarray, shape (n, n)
    edge_orbits : list of list of (int, int)
    tol : float

    Returns
    -------
    bool
        True if all orbits satisfy the energy constraint within tol.
    """
    ok = True
    for i, orbit in enumerate(edge_orbits):
        total = sum(float(Phi[u, u] + Phi[v, v] - 2 * Phi[u, v]) for u, v in orbit)
        target = float(len(orbit))
        if abs(total - target) > tol:
            print(f"Orbit {i}: total sq dist = {total:.8f}, expected {target:.1f}  [FAIL]")
            ok = False
    return ok


def verify_edge_isometry(
    Phi: np.ndarray,
    edges: list[tuple[int, int]],
    tol: float = 1e-5,
) -> bool:
    """
    Check that Φ[u,u] + Φ[v,v] - 2Φ[u,v] = 1 for every edge.

    Parameters
    ----------
    Phi : np.ndarray, shape (n, n)
    edges : list of (int, int)
    tol : float

    Returns
    -------
    bool
        True if every per-edge constraint is satisfied within tol.
    """
    ok = True
    for u, v in edges:
        val = float(Phi[u, u] + Phi[v, v] - 2 * Phi[u, v])
        if abs(val - 1.0) > tol:
            print(f"Edge ({u}, {v}): sq dist = {val:.8f}, expected 1.0  [FAIL]")
            ok = False
    return ok


def verify_orbit_isometry(
    Phi: np.ndarray,
    edge_orbits: list[list[tuple[int, int]]],
    tol: float = 1e-5,
) -> bool:
    """
    Check that the total squared distance across each orbit equals |O_i|.

    Parameters
    ----------
    Phi : np.ndarray, shape (n, n)
    edge_orbits : list of list of (int, int)
    tol : float

    Returns
    -------
    bool
        True if every per-orbit constraint is satisfied within tol.
    """
    return _verify_embedding(Phi, edge_orbits, tol=tol)


# ---------------------------------------------------------------------------
# Outer layer  (Sage-facing)
# ---------------------------------------------------------------------------

def is_lcr(G_sage, solver: str = 'MOSEK') -> bool:
    """
    Test whether G is lower-conformally rigid.

    Computes the λ₂ eigenspace via numpy and solves the feasibility SDP (12).
    G_sage must have integer vertices labeled 0..n-1; call G.relabel() first.

    Parameters
    ----------
    G_sage : sage.graphs.graph.Graph
    solver : {'MOSEK', 'SCS'}, default 'MOSEK'

    Returns
    -------
    bool
        True if the SDP is feasible (G is LCR).
    """
    L_np = _sage_laplacian(G_sage)
    _, B = get_eigenspace(L_np, which="lambda2")
    edges = [(int(u), int(v)) for u, v in G_sage.edges(labels=False)]
    status, _, _ = find_edge_isometric_embedding(B, edges, solver=solver)
    return status == "optimal"


def is_ucr(G_sage, solver: str = 'MOSEK') -> bool:
    """
    Test whether G is upper-conformally rigid.

    Computes the λ_n eigenspace via numpy and solves the feasibility SDP (12).
    G_sage must have integer vertices labeled 0..n-1; call G.relabel() first.

    Parameters
    ----------
    G_sage : sage.graphs.graph.Graph
    solver : {'MOSEK', 'SCS'}, default 'MOSEK'

    Returns
    -------
    bool
        True if the SDP is feasible (G is UCR).
    """
    L_np = _sage_laplacian(G_sage)
    _, B = get_eigenspace(L_np, which="lambdan")
    edges = [(int(u), int(v)) for u, v in G_sage.edges(labels=False)]
    status, _, _ = find_edge_isometric_embedding(B, edges, solver=solver)
    return status == "optimal"


def _sage_laplacian(G_sage) -> np.ndarray:
    """Convert a Sage graph's Laplacian to a dense numpy float64 array."""
    from sage.rings.real_double import RDF
    verts = G_sage.vertices(sort=True)
    n = len(verts)
    assert verts == list(range(n)), (
        "Vertices must be labeled 0..n-1. Call G.relabel() before passing to is_lcr/is_ucr."
    )
    return G_sage.laplacian_matrix(vertices=verts).change_ring(RDF).numpy()
