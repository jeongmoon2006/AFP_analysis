"""
rdf.py

Computing radial distribution functions (RDF, g(r))
from MD trajectories using MDAnalysis.

Typical usage
-------------
from analysis_lib.rdf import compute_rdf, coordination_number

u = mda.Universe("prod.gro", "prod_centered.xtc")
r, gr = compute_rdf(
    u,
    sel1="name OW and resid 1-100",
    sel2="name OW and not resid 1-100",
    r_range=(0.0, 1.0),
    nbins=200,
)

# Optionally compute coordination number up to first minimum, etc.
rho = number_density(u, sel="name OW")  # particles per volume
cn = coordination_number(r, gr, rho, r_max=0.35)
"""

from __future__ import annotations

from typing import Tuple, Optional, Union

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF


AtomGroupLike = Union[str, mda.core.groups.AtomGroup]


def _to_atomgroup(
    u: mda.Universe,
    sel: AtomGroupLike,
    name: str = "selection",
) -> mda.core.groups.AtomGroup:
    """
    Convert a selection (string or AtomGroup) into an AtomGroup.
    """
    if isinstance(sel, mda.core.groups.AtomGroup):
        return sel
    elif isinstance(sel, str):
        ag = u.select_atoms(sel)
        if len(ag) == 0:
            raise ValueError(f"{name!r} selection matched 0 atoms: {sel!r}")
        return ag
    else:
        raise TypeError(
            f"{name} must be a selection string or AtomGroup, got {type(sel)}"
        )


def compute_rdf(
    u: mda.Universe,
    sel1: AtomGroupLike,
    sel2: AtomGroupLike,
    r_range: Tuple[float, float] = (0.0, 1.0),
    nbins: int = 200,
    exclusion_block: Optional[Tuple[int, int]] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the radial distribution function g(r) between two atom sets
    using MDAnalysis.InterRDF.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Trajectory universe (topology + traj).
    sel1, sel2 : str or AtomGroup
        Atom selection for group 1 and 2. Can be:
        - MDAnalysis selection string (e.g., "name OW")
        - AtomGroup (e.g., u.atoms[...])
    r_range : (float, float), optional
        Minimum and maximum distance (same units as trajectory coordinates,
        e.g. nm or Å).
    nbins : int, optional
        Number of bins for the histogram.
    exclusion_block : (int, int), optional
        If provided, passed directly to MDAnalysis.InterRDF to exclude
        pairs within the same residue / molecule. See MDAnalysis docs.
    start, stop, step : int or None, optional
        Frame range for analysis. Same meaning as u.trajectory[start:stop:step].
    verbose : bool, optional
        If True, prints a short summary.

    Returns
    -------
    r : np.ndarray, shape (nbins,)
        Bin centers (distance).
    g_r : np.ndarray, shape (nbins,)
        Radial distribution function values.

    Notes
    -----
    - Make sure your trajectory is properly centered / imaged if needed
      before computing RDF.
    - For symmetric RDF (A–A), sel1 and sel2 can be the same.
    """
    ag1 = _to_atomgroup(u, sel1, name="sel1")
    ag2 = _to_atomgroup(u, sel2, name="sel2")

    # Setup frame slicing
    if any(v is not None for v in (start, stop, step)):
        traj_slice = slice(start, stop, step)
    else:
        traj_slice = slice(None, None, None)

    if verbose:
        print("=== RDF calculation ===")
        print(f"  Group 1: {len(ag1)} atoms")
        print(f"  Group 2: {len(ag2)} atoms")
        print(f"  r_range : {r_range[0]:.3f} – {r_range[1]:.3f}")
        print(f"  nbins   : {nbins}")
        print(f"  frames  : {u.trajectory.n_frames} (slice={traj_slice})")
        if exclusion_block is not None:
            print(f"  exclusion_block: {exclusion_block}")
        print("  Running InterRDF ...")

    rdf = InterRDF(
        ag1,
        ag2,
        nbins=nbins,
        range=r_range,
        exclusion_block=exclusion_block,
        verbose=verbose,
    )

    # Limit frames considered
    with u.trajectory[traj_slice]:
        rdf.run()

    if verbose:
        print("  Done.")
        print("=======================")

    # MDAnalysis stores:
    # rdf.bins : r values (bin centers)
    # rdf.rdf  : g(r)
    return rdf.bins.copy(), rdf.rdf.copy()


def number_density(
    u: mda.Universe,
    sel: Optional[AtomGroupLike] = None,
    frame: int = 0,
) -> float:
    """
    Estimate number density (particles per volume) for given selection.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Universe containing system and box information.
    sel : str or AtomGroup or None, optional
        Selection for counting particles. If None, uses all atoms in u.
    frame : int, optional
        Frame index to take box dimensions from.

    Returns
    -------
    rho : float
        Number density = N / V, in (1 / box_units^3),
        where box_units are whatever units the coordinates use (nm or Å).

    Notes
    -----
    This assumes a homogeneous system in a single periodic box.
    For mixtures, you may want to select only the species of interest.
    """
    if sel is None:
        ag = u.atoms
    else:
        ag = _to_atomgroup(u, sel, name="sel")

    # Move to target frame to get correct box
    u.trajectory[frame]
    box = u.dimensions  # [lx, ly, lz, alpha, beta, gamma]
    lx, ly, lz = box[:3]
    volume = lx * ly * lz

    if volume == 0:
        raise ValueError("Box volume is zero – check if your trajectory has box info.")

    n = len(ag)
    rho = n / volume
    return rho


def coordination_number(
    r: np.ndarray,
    g_r: np.ndarray,
    rho: float,
    r_max: Optional[float] = None,
) -> float:
    """
    Compute coordination number by integrating g(r):

        n(r_max) = 4π ρ ∫_0^{r_max} g(r) r^2 dr

    Parameters
    ----------
    r : np.ndarray
        Distances (same units as in RDF).
    g_r : np.ndarray
        RDF values g(r), same length as r.
    rho : float
        Number density (particles per volume).
    r_max : float, optional
        Upper limit of integration. If None, integrates up to max(r).

    Returns
    -------
    n : float
        Coordination number up to r_max.

    Notes
    -----
    - Typically r_max is chosen as the first minimum after the first peak
      in g(r).
    - r and g_r should be 1D arrays of the same shape.
    """
    r = np.asarray(r)
    g_r = np.asarray(g_r)

    if r.shape != g_r.shape:
        raise ValueError("r and g_r must have the same shape")

    if r_max is not None:
        mask = r <= r_max
        if not np.any(mask):
            raise ValueError("r_max is smaller than the minimum r value")
        r_int = r[mask]
        g_int = g_r[mask]
    else:
        r_int = r
        g_int = g_r

    integrand = g_int * r_int**2
    integral = np.trapz(integrand, r_int)
    n = 4.0 * np.pi * rho * integral
    return float(n)
