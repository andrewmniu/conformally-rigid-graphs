"""
Visualization utilities for conformally rigid graph certificates.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pca2d(M):
    """Project rows of M to 2D via PCA."""
    M_centered = M - M.mean(axis=0)
    _, _, Vt = np.linalg.svd(M_centered, full_matrices=False)
    return M_centered @ Vt[:2].T


def pca3d(M):
    """Project rows of M to 3D via PCA."""
    M_centered = M - M.mean(axis=0)
    _, _, Vt = np.linalg.svd(M_centered, full_matrices=False)
    return M_centered @ Vt[:3].T


def _orbit_colors(edge_orbits, colors=None, cmap_name='tab10'):
    """Return a dict mapping each edge -> color, one color per orbit.

    Parameters
    ----------
    colors : list of colors, optional
        If provided, cycle through this list instead of the colormap.
    """
    if colors is None:
        cmap = plt.get_cmap(cmap_name)
        n_orbs = len(edge_orbits)
        colors = [cmap(i % 10) if n_orbs <= 10 else cmap(i / n_orbs)
                  for i in range(n_orbs)]
    edge_color_map = {}
    for i, orbit in enumerate(edge_orbits):
        c = colors[i % len(colors)]
        for edge in orbit:
            edge_color_map[edge] = c
            edge_color_map[(edge[1], edge[0])] = c
    return edge_color_map


def _orbit_legend(edge_orbits, colors, ax):
    patches = [
        mpatches.Patch(color=colors[orbit[0]], label=f"Orbit {i+1} ({len(orbit)} edges)")
        for i, orbit in enumerate(edge_orbits)
    ]
    ax.legend(handles=patches, fontsize=7,
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


def _draw_graph_2d(G_nx, pos, edge_orbits, edge_color_map, ax, title, vertex_labels=False):
    """Draw graph with orbit-colored edges onto a 2D ax."""
    nx.draw_networkx_nodes(G_nx, pos, node_size=40, node_color='#333333', ax=ax)

    for orbit in edge_orbits:
        nx.draw_networkx_edges(
            G_nx, pos,
            edgelist=orbit,
            edge_color=[edge_color_map[orbit[0]]],
            width=1.8, ax=ax,
        )

    if vertex_labels:
        nx.draw_networkx_labels(G_nx, pos, font_size=8, font_color='white', ax=ax)

    ax.set_title(title, fontsize=10)
    ax.axis('off')
    _orbit_legend(edge_orbits, edge_color_map, ax)


def _equal_axes_3d(ax, coords):
    """Set equal axis ranges on a 3D axes so all dimensions share the same scale."""
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2
    half = (maxs - mins).max() / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _draw_graph_3d(G_nx, coords, edge_orbits, edge_color_map, ax, title, vertex_labels=False):
    """Draw graph with orbit-colored edges onto a 3D ax."""
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    for orbit in edge_orbits:
        color = edge_color_map[orbit[0]]
        for u, v in orbit:
            ax.plot([xs[u], xs[v]], [ys[u], ys[v]], [zs[u], zs[v]],
                    color=color, linewidth=1.8)

    ax.scatter(xs, ys, zs, s=40, c='#333333', zorder=5)

    if vertex_labels:
        for i in G_nx.nodes():
            ax.text(xs[i], ys[i], zs[i], str(i), fontsize=8)

    _equal_axes_3d(ax, coords)
    ax.set_title(title, fontsize=10)
    _orbit_legend(edge_orbits, edge_color_map, ax)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_embedding(B, Z, tol=1e-8):
    """
    Compute the spectral embedding from the SDP certificate.

    Forms Phi = B Z Bᵀ, eigendecomposes it, keeps eigenvalues above tol,
    and returns P where P @ P.T == Phi and each row is a vertex position.

    Parameters
    ----------
    B : np.ndarray, shape (n, d)
    Z : np.ndarray, shape (d, d)
    tol : float

    Returns
    -------
    P : np.ndarray, shape (n, r)
    """
    Phi = B @ Z @ B.T
    lambdas, U = np.linalg.eigh(Phi)
    mask = lambdas > tol
    return U[:, mask] * np.sqrt(lambdas[mask])


def plot_embedding(G, positions, edge_orbits=None, pca=True, dim=2,
                   colors=None, vertex_labels=False, ax=None, title=None, figsize=(6, 5)):
    """
    Plot a graph using a pre-computed vertex embedding.

    Parameters
    ----------
    G : networkx.Graph or Sage graph
    positions : np.ndarray, shape (n, d)
    edge_orbits : list of list of (int, int), optional
        If provided, edges are colored by orbit. If None, all edges are gray.
    pca : bool
        If True and d > dim, project to `dim` dimensions via PCA before plotting.
    dim : int (2 or 3)
        Whether to produce a 2D or 3D plot.
    colors : list of colors, optional
        One color per orbit. Defaults to tab10 colormap.
    vertex_labels : bool
    ax : matplotlib Axes, optional
    title : str, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    if hasattr(G, 'networkx_graph'):
        G = G.networkx_graph()

    positions = np.asarray(positions, dtype=float)
    d = positions.shape[1] if positions.ndim == 2 else 1

    if dim == 3:
        if d > 3:
            coords = pca3d(positions) if pca else positions[:, :3]
        elif d == 3:
            coords = positions
        elif d == 2:
            coords = np.column_stack([positions, np.zeros(len(positions))])
        else:
            coords = np.column_stack([positions.ravel(),
                                      np.zeros(len(positions)),
                                      np.zeros(len(positions))])

        fig = None
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

        if edge_orbits is None:
            xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
            for u, v in G.edges():
                ax.plot([xs[u], xs[v]], [ys[u], ys[v]], [zs[u], zs[v]],
                        color='gray', linewidth=1.5)
            ax.scatter(xs, ys, zs, s=40, c='#333333')
            if vertex_labels:
                for i in G.nodes():
                    ax.text(xs[i], ys[i], zs[i], str(i), fontsize=8)
            _equal_axes_3d(ax, coords)
            if title is not None:
                ax.set_title(title, fontsize=10)
        else:
            edge_color_map = _orbit_colors(edge_orbits, colors=colors)
            _draw_graph_3d(G, coords, edge_orbits, edge_color_map, ax,
                           title=title or '', vertex_labels=vertex_labels)

        return fig, ax

    else:
        if d > 2:
            coords2d = pca2d(positions) if pca else positions[:, :2]
        elif d == 2:
            coords2d = positions
        else:
            coords2d = np.column_stack([positions.ravel(), np.zeros(len(positions))])

        pos = {v: (coords2d[v, 0], coords2d[v, 1]) for v in G.nodes()}

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if edge_orbits is None:
            nx.draw_networkx_nodes(G, pos, node_size=40, node_color='#333333', ax=ax)
            nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, ax=ax)
            if vertex_labels:
                nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', ax=ax)
            if title is not None:
                ax.set_title(title, fontsize=10)
        else:
            edge_color_map = _orbit_colors(edge_orbits, colors=colors)
            _draw_graph_2d(G, pos, edge_orbits, edge_color_map, ax,
                           title=title or '', vertex_labels=vertex_labels)

        ax.set_aspect('equal')
        ax.axis('off')
        return fig, ax


def plot_symmetrized_embedding(G, phi, aut_perms, edge_orbits=None, dim=2,
                               colors=None, vertex_labels=False, ax=None, title=None, figsize=(6, 5)):
    """
    Build the Ψ-symmetrized embedding (Def. 5.7) from an eigenfunction φ,
    project to 2D or 3D via PCA, and plot the graph.

    The symmetrized embedding P_Ψ has columns {σ·φ : σ ∈ Ψ}, where the action
    is (σ·φ)(v) = φ(σ⁻¹(v)).

    Parameters
    ----------
    G : networkx.Graph or Sage graph
    phi : array-like, shape (n,)
        A single λ-eigenfunction.
    aut_perms : list of array-like
        Each element is a permutation array perm where perm[i] = σ(i).
        In Sage: [list(sigma) for sigma in aut_group]
    edge_orbits : list of list of (int, int), optional
    dim : int (2 or 3)
    vertex_labels : bool
    ax : matplotlib Axes, optional
    title : str, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    phi = np.asarray(phi, dtype=float)

    columns = []
    for perm in aut_perms:
        perm = np.asarray(perm, dtype=int)
        inv_perm = np.argsort(perm)
        columns.append(phi[inv_perm])
    P_sym = np.column_stack(columns)  # shape (n, |Ψ|)

    coords = pca3d(P_sym) if dim == 3 else pca2d(P_sym)
    return plot_embedding(G, coords, edge_orbits=edge_orbits, pca=False, dim=dim,
                          colors=colors, vertex_labels=vertex_labels, ax=ax,
                          title=title or 'Symmetrized embedding', figsize=figsize)


def plot_cert_embedding(G, B, Z, edge_orbits=None, tol=1e-8, pca=True, dim=2,
                        colors=None, vertex_labels=False, ax=None, title=None, figsize=(6, 5)):
    """
    Compute the certificate embedding Phi = BZBᵀ and plot the graph.

    Parameters
    ----------
    G : networkx.Graph or Sage graph
    B : np.ndarray, shape (n, d)
    Z : np.ndarray, shape (d, d)
    edge_orbits : list of list of (int, int), optional
    tol : float
    pca : bool
    dim : int (2 or 3)
    vertex_labels : bool
    ax : matplotlib Axes, optional
    title : str, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    positions = extract_embedding(B, Z, tol)
    return plot_embedding(G, positions, edge_orbits=edge_orbits, pca=pca, dim=dim,
                          colors=colors, vertex_labels=vertex_labels, ax=ax,
                          title=title, figsize=figsize)
