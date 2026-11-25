"""Block structure for the NUTS mass matrix, derived from the model configuration.

A fully dense mass matrix has O(n^2) entries to estimate from a few hundred
warmup draws, most of which describe parameters that are not correlated.
:func:`mass_blocks` instead groups the free parameters that the configuration
couples (shared kinematics, a line's own parameters, a continuum region's
parameters, dependent priors) and returns those groups in the form NumPyro's
``dense_mass`` expects; parameters left out are given a diagonal mass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from unite.prior import Fixed

if TYPE_CHECKING:
    from unite.model import ModelArgs

#: Default maximum number of parameters in one dense block.
DEFAULT_MAX_BLOCK_SIZE: int = 16


class _Blocks:
    """Union-find over site names that tracks component sizes."""

    def __init__(self, names: list[str]) -> None:
        self._parent = {n: n for n in names}
        self._size = dict.fromkeys(names, 1)

    def find(self, name: str) -> str:
        root = name
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[name] != root:
            self._parent[name], name = root, self._parent[name]
        return root

    def merge(self, sites: list[str], cap: int | None) -> None:
        """Merge the components of *sites*, unless the result exceeds *cap*."""
        roots = list(dict.fromkeys(self.find(s) for s in sites))
        if len(roots) < 2:
            return
        if cap is not None and sum(self._size[r] for r in roots) > cap:
            return
        head = roots[0]
        for r in roots[1:]:
            self._parent[r] = head
            self._size[head] += self._size[r]

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name in self._parent:
            out.setdefault(self.find(name), []).append(name)
        return out


def _line_sites(args: ModelArgs, free: set[str]) -> list[dict[str, list[str]]]:
    """Free site names for every line, split into ``own`` and ``shared`` roles.

    ``amp`` holds the line's flux/tau site, ``other`` the redshift, widths,
    shape and covering-fraction sites.
    """
    cm = args.matrices
    n_lines = int(cm.wavelengths.shape[0])
    slots = {
        'amp': [(cm.flux_names, cm.flux_matrix), (cm.tau_names, cm.tau_matrix)],
        'other': [
            (cm.z_names, cm.z_matrix),
            (cm.p0_names, cm.p0_matrix),
            (cm.p1v_names, cm.p1v_matrix),
            (cm.p1d_names, cm.p1d_matrix),
            (cm.p2v_names, cm.p2v_matrix),
            (cm.p2d_names, cm.p2d_matrix),
            (cm.cover_names, cm.cover_matrix),
        ],
    }
    lines: list[dict[str, list[str]]] = [
        {'amp': [], 'other': []} for _ in range(n_lines)
    ]
    for role, pairs in slots.items():
        for names, matrix in pairs:
            mat = np.asarray(matrix)
            for i, name in enumerate(names):
                if name not in free:
                    continue
                for j in np.flatnonzero(mat[i]):
                    lines[int(j)][role].append(name)
    return lines


def mass_blocks(
    args: ModelArgs, max_block_size: int = DEFAULT_MAX_BLOCK_SIZE
) -> list[tuple[str, ...]]:
    """Group free parameters into dense mass-matrix blocks.

    Couplings are merged in order of strength, and a merge that would push a
    block past ``max_block_size`` is skipped (except dependent priors, which
    are always kept together, since a dependent bound makes the two
    parameters strongly correlated in unconstrained space):

    1. Dependent priors: a parameter and the parameters its prior depends on.
    2. Continuum regions: all free parameters of one region.
    3. Lines: each line's flux (or tau) with its redshift, widths, shape and
       covering-fraction parameters.  Lines that share a token therefore end
       up in the same block.
    4. Line-continuum: each line's flux (or tau) with the parameters of every
       continuum region containing it.
    5. Calibration: the calibration parameters of one disperser.

    Only blocks with at least two parameters are returned; every other free
    parameter is left to NumPyro's diagonal mass.

    Parameters
    ----------
    args : ModelArgs
        Output of :meth:`unite.model.ModelBuilder.build`.
    max_block_size : int, optional
        Maximum number of parameters in a block (default: 16).

    Returns
    -------
    list of tuple of str
        Sorted tuples of site names, suitable for NumPyro's ``dense_mass``.
    """
    if max_block_size < 1:
        msg = f'max_block_size must be >= 1, got {max_block_size}.'
        raise ValueError(msg)

    free = [
        name
        for name in args.dependency_order
        if not isinstance(args.all_priors[name], Fixed)
    ]
    free_set = set(free)
    blocks = _Blocks(free)

    # Token identity -> site name, including continuum tokens.
    tok_to_name: dict[int, str] = {id(t): n for n, t in args.name_to_token.items()}
    region_sites: list[list[str]] = []
    for resolved in args.cont_resolved_params or []:
        names = []
        for tok in resolved.values():
            tok_to_name[id(tok)] = tok.name
            if tok.name in free_set:
                names.append(tok.name)
        region_sites.append(names)

    # 1. Dependent priors (always merged).
    for name in free:
        deps = [
            tok_to_name[id(d)]
            for d in args.all_priors[name].dependencies()
            if id(d) in tok_to_name
        ]
        blocks.merge([name, *[d for d in deps if d in free_set]], cap=None)

    cap = max_block_size

    # 2. Continuum regions.
    for names in region_sites:
        blocks.merge(names, cap)

    # 3. Lines.
    lines = _line_sites(args, free_set)
    for line in lines:
        blocks.merge([*line['amp'], *line['other']], cap)

    # 4. Lines inside continuum regions.
    if args.cont_low is not None and args.cont_high is not None:
        wavelengths = np.asarray(args.matrices.wavelengths)
        for names, low, high in zip(
            region_sites, args.cont_low, args.cont_high, strict=True
        ):
            if not names:
                continue
            for line, wl in zip(lines, wavelengths, strict=True):
                if low <= wl <= high and line['amp']:
                    blocks.merge([names[0], *line['amp']], cap)

    # 5. Calibration parameters of each disperser.
    seen: set[int] = set()
    for spectrum in args.spectra:
        disp = spectrum.disperser
        if id(disp) in seen:
            continue
        seen.add(id(disp))
        names = [t.name for _, t in disp.calibration_tokens() if t.name in free_set]
        blocks.merge(names, cap)

    return sorted(
        tuple(sorted(group)) for group in blocks.groups().values() if len(group) > 1
    )
