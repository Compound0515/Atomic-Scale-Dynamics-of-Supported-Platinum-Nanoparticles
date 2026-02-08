import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
from math import cos, sin, radians, sqrt


# ------------------- I/O helpers: read XYZ frames -------------------


def iter_xyz_frames(path):
    """Yield (comment, atom_symbols_list, coords_array) for each frame.
    Assumes standard XYZ: N_atoms line, comment line, then N atom lines.
    """
    with open(path, "r") as f:
        while True:
            nline = f.readline()
            if not nline:
                break
            n_atoms = int(nline.strip())
            comment = f.readline().rstrip("\n")
            syms = []
            coords = np.empty((n_atoms, 3), dtype=np.float64)
            for i in range(n_atoms):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Bad atom line at frame, got: {parts}")
                syms.append(parts[0])
                coords[i, 0:3] = (float(parts[1]), float(parts[2]), float(parts[3]))
            yield comment, syms, coords


# ------------------- Box utilities for triclinic cell -------------------


def box_matrix_from_lengths_angles(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha = radians(alpha_deg)
    beta = radians(beta_deg)
    gamma = radians(gamma_deg)
    ca = cos(alpha)
    cb = cos(beta)
    cg = cos(gamma)
    sg = sin(gamma)
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * cg, b * sg, 0.0])
    cx = c * cb
    denom = sg
    if abs(denom) < 1e-12:
        raise ValueError("gamma angle too close to 0 or 180 deg; sin(gamma) ~ 0.")
    cy = c * (ca - cb * cg) / denom
    cz_sq = 1.0 - cb * cb - ((ca - cb * cg) / denom) ** 2
    if cz_sq < -1e-12:
        raise ValueError("Box angles produce negative cz^2 (invalid box).")
    cz = c * sqrt(max(0.0, cz_sq))
    c_vec = np.array([cx, cy, cz])
    L = np.column_stack((a_vec, b_vec, c_vec))
    invL = np.linalg.inv(L)
    return L, invL


def min_image_delta(delta, L, invL):
    """
    Apply minimum image to a displacement vector delta.
    """
    orig_shape = delta.shape
    flat = delta.reshape(-1, 3).T
    frac = invL.dot(flat)
    frac = frac - np.round(frac)
    cart = L.dot(frac).T
    return cart.reshape(orig_shape)


def dist_sq_min_image(r1, r2, L=None, invL=None):
    """
    Return squared minimum-image distance between two cartesian vectors r1 and r2.
    """
    d = r2 - r1
    if L is not None:
        d = min_image_delta(d, L, invL)
    return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]


def pairs_within_cutoff_pbc(
    coordsA, indsA, coordsB, indsB, cutoff, L, invL, self_pairs=False
):
    """
    Return list of unordered pairs where min-image distance <= cutoff.
    """
    cutoff_square = cutoff * cutoff
    pairs = []
    Na = len(indsA)
    Nb = len(indsB)
    chunk = max(1, int(1e6 // (Nb + 1)))
    chunk = min(chunk, Na)
    for i0 in range(0, Na, chunk):
        i1 = min(Na, i0 + chunk)
        subA = coordsA[i0:i1]
        d = subA[:, None, :] - coordsB[None, :, :]
        d_flat = d.reshape(-1, 3)
        d_mic = min_image_delta(d_flat, L, invL).reshape(d.shape)
        d_square = (d_mic**2).sum(axis=2)
        hit_idx = np.nonzero(d_square <= cutoff_square)
        for ia, ib in zip(hit_idx[0], hit_idx[1]):
            a_global = indsA[i0 + ia]
            b_global = indsB[ib]
            if self_pairs:
                if a_global < b_global:
                    pairs.append((a_global, b_global))
            else:
                pairs.append((a_global, b_global))
    return pairs


# ------------------- Scanning and matrix building -------------------


def collect_pairs(
    trajfile,
    pt_ti_cutoff=2.95,
    pt_o_cutoff=2.60,
    pt_pt_cutoff=3.00,
    use_pbc=False,
    L=None,
    invL=None,
    nbottom=None,
):
    """Scan trajectory:
    - collect Pt-Ti, Pt-O pairs that ever come within cutoffs
    - collect Pt-Pt pairs that ever come within pt_pt_cutoff
    - determine interfacial Pt set from first frame.
      If nbottom is set: the 'nbottom' Pt atoms with lowest Z coordinates.
      Else: Pt atoms with at least one O or Ti within their cutoffs.
    """
    pt_ti_pairs = set()
    pt_o_pairs = set()
    pt_pt_pairs = set()
    atom_syms = None
    nframes = 0
    interfacial_set = set()
    first_frame_handled = False

    for _, syms, coords in tqdm(
        iter_xyz_frames(trajfile), desc="Scanning frames for pairs"
    ):
        if atom_syms is None:
            atom_syms = syms[:]

        pt_idx = [i for i, s in enumerate(syms) if s.lower() == "pt"]
        ti_idx = [i for i, s in enumerate(syms) if s.lower() == "ti"]
        o_idx = [i for i, s in enumerate(syms) if s.lower() == "o"]

        if len(pt_idx) == 0:
            raise RuntimeError("No Pt atoms found in frame.")

        if not first_frame_handled:
            first_frame_handled = True
            if nbottom is not None:
                if len(pt_idx) > 0:
                    pt_coords = coords[pt_idx]
                    sorted_local_indices = np.argsort(pt_coords[:, 2])
                    count = min(nbottom, len(pt_idx))
                    for k in range(count):
                        global_id = pt_idx[sorted_local_indices[k]]
                        interfacial_set.add(global_id)
            else:
                if use_pbc:
                    if len(ti_idx) > 0:
                        pt_coords = coords[pt_idx]
                        ti_coords = coords[ti_idx]
                        pairs = pairs_within_cutoff_pbc(
                            pt_coords,
                            pt_idx,
                            ti_coords,
                            ti_idx,
                            pt_ti_cutoff,
                            L,
                            invL,
                            self_pairs=False,
                        )
                        for p, _ in pairs:
                            interfacial_set.add(p)
                    if len(o_idx) > 0:
                        pt_coords = coords[pt_idx]
                        o_coords = coords[o_idx]
                        pairs = pairs_within_cutoff_pbc(
                            pt_coords,
                            pt_idx,
                            o_coords,
                            o_idx,
                            pt_o_cutoff,
                            L,
                            invL,
                            self_pairs=False,
                        )
                        for p, _ in pairs:
                            interfacial_set.add(p)
                else:
                    if len(ti_idx) > 0:
                        ti_coords = coords[ti_idx]
                        tree_ti = cKDTree(ti_coords)
                    else:
                        tree_ti = None
                    if len(o_idx) > 0:
                        o_coords = coords[o_idx]
                        tree_o = cKDTree(o_coords)
                    else:
                        tree_o = None
                    for p in pt_idx:
                        ppos = coords[p]
                        connected = False
                        if tree_ti is not None:
                            if len(tree_ti.query_ball_point(ppos, pt_ti_cutoff)) > 0:
                                connected = True
                        if (not connected) and tree_o is not None:
                            if len(tree_o.query_ball_point(ppos, pt_o_cutoff)) > 0:
                                connected = True
                        if connected:
                            interfacial_set.add(p)

        # Build pair sets for this frame
        if use_pbc:
            if len(ti_idx) > 0:
                pairs = pairs_within_cutoff_pbc(
                    coords[pt_idx],
                    pt_idx,
                    coords[ti_idx],
                    ti_idx,
                    pt_ti_cutoff,
                    L,
                    invL,
                    self_pairs=False,
                )
                for pair in pairs:
                    pt_ti_pairs.add(pair)

            if len(o_idx) > 0:
                pairs = pairs_within_cutoff_pbc(
                    coords[pt_idx],
                    pt_idx,
                    coords[o_idx],
                    o_idx,
                    pt_o_cutoff,
                    L,
                    invL,
                    self_pairs=False,
                )
                for pair in pairs:
                    pt_o_pairs.add(pair)

            if len(pt_idx) > 1:
                pairs = pairs_within_cutoff_pbc(
                    coords[pt_idx],
                    pt_idx,
                    coords[pt_idx],
                    pt_idx,
                    pt_pt_cutoff,
                    L,
                    invL,
                    self_pairs=True,
                )
                for pair in pairs:
                    pt_pt_pairs.add(pair)
        else:
            if len(ti_idx) > 0:
                ti_coords = coords[ti_idx]
                tree_ti = cKDTree(ti_coords)
                for p in pt_idx:
                    ppos = coords[p]
                    neigh_local = tree_ti.query_ball_point(ppos, pt_ti_cutoff)
                    for nl in neigh_local:
                        ti_global = ti_idx[nl]
                        pt_ti_pairs.add((p, ti_global))

            if len(o_idx) > 0:
                o_coords = coords[o_idx]
                tree_o = cKDTree(o_coords)
                for p in pt_idx:
                    ppos = coords[p]
                    neigh_local = tree_o.query_ball_point(ppos, pt_o_cutoff)
                    for nl in neigh_local:
                        o_global = o_idx[nl]
                        pt_o_pairs.add((p, o_global))

            if len(pt_idx) > 1:
                pt_coords = coords[pt_idx]
                tree_pt = cKDTree(pt_coords)
                for _, p in enumerate(pt_idx):
                    ppos = coords[p]
                    neigh_local = tree_pt.query_ball_point(ppos, pt_pt_cutoff)
                    for nl in neigh_local:
                        q_local = nl
                        q = pt_idx[q_local]
                        if q <= p:
                            continue
                        pt_pt_pairs.add((p, q))

        nframes += 1

    return (
        sorted(pt_ti_pairs),
        sorted(pt_o_pairs),
        sorted(pt_pt_pairs),
        atom_syms,
        nframes,
        interfacial_set,
    )


def build_time_matrix(
    trajfile,
    pairs,
    pair_type_name,
    nframes,
    cutoff=None,
    use_pbc=False,
    L=None,
    invL=None,
):
    """
    Build boolean matrix shape (n_pairs, nframes) marking bond presence.
    If use_pbc True, uses minimum-image distances from box L/invL.
    """
    if len(pairs) == 0:
        return np.zeros((0, nframes), dtype=np.uint8)
    pair_to_row = {pair: idx for idx, pair in enumerate(pairs)}
    n_pairs = len(pairs)
    mat = np.zeros((n_pairs, nframes), dtype=np.uint8)

    # prepare per-first-index mapping
    firsts = sorted({p for p, q in pairs})
    partner_map = {p: [] for p in firsts}
    for p, q in pairs:
        partner_map[p].append(q)

    frame_idx = 0
    for _, _, coords in tqdm(
        iter_xyz_frames(trajfile), desc=f"Building {pair_type_name} matrix"
    ):
        if use_pbc:
            for p in firsts:
                ppos = coords[p]
                partners = partner_map[p]
                if len(partners) == 0:
                    continue
                qpos = coords[partners]
                d = qpos - ppos
                d_mi = min_image_delta(d, L, invL)
                dsq = (d_mi**2).sum(axis=1)
                for k, q in enumerate(partners):
                    bonded = dsq[k] <= cutoff * cutoff
                    if bonded:
                        ridx = pair_to_row[(p, q)]
                        mat[ridx, frame_idx] = 1
        else:
            for p in firsts:
                ppos = coords[p]
                for q in partner_map[p]:
                    qpos = coords[q]
                    d2 = (
                        (ppos[0] - qpos[0]) ** 2
                        + (ppos[1] - qpos[1]) ** 2
                        + (ppos[2] - qpos[2]) ** 2
                    )
                    if d2 <= cutoff * cutoff:
                        ridx = pair_to_row[(p, q)]
                        mat[ridx, frame_idx] = 1
        frame_idx += 1

    return mat


# ------------------- Correlation computation: intermittent and continuous -------------------


def compute_C_from_bool_matrix_intermit(mat):
    """
    Bond intermittent survival: counts h(t)*h(t+lag) allowing re-formation.
    """
    if mat.size == 0:
        return np.array([])
    T = mat.shape[1]
    denom = mat.sum()
    if denom == 0:
        return np.zeros(T, dtype=float)
    max_lag = T - 1
    C = np.zeros(max_lag + 1, dtype=float)
    for lag in range(0, max_lag + 1):
        if lag == 0:
            numerator = (mat & mat).sum()
        else:
            numerator = (mat[:, : T - lag] & mat[:, lag:]).sum()
        C[lag] = (T / (T - lag)) * (numerator / denom)
    return C


def compute_C_from_bool_matrix_continuous(mat):
    """
    Bond continuous survival: uses run-length encoding across each pair.
    """
    if mat.size == 0:
        return np.array([])
    T = mat.shape[1]
    denom = mat.sum()
    if denom == 0:
        return np.zeros(T, dtype=float)
    max_lag = T - 1
    C = np.zeros(max_lag + 1, dtype=float)
    # For each pair, find runs of ones and accumulate contributions.
    # For a binary vector v: find indices where v==1 segments start and their lengths.
    for row in range(mat.shape[0]):
        v = mat[row]
        if not v.any():
            continue
        # find run starts/lengths
        # RLE: split by zeros
        # Find indices where v transitions 0->1 (starts) and 1->0 (ends)
        dif = np.diff(np.concatenate(([0], v.astype(int), [0])))
        starts = np.nonzero(dif == 1)[0]
        ends = np.nonzero(dif == -1)[0]
        lengths = ends - starts
        for r in lengths:
            # this run contributes for lags L = 0..r-1 with multiplicity (r - L)
            # aggregate numerator contributions across lags
            # We'll accumulate directly into C[0..r-1] by adding (r - L)
            C[:r] += np.arange(r, 0, -1)
        # denom is total ones, already computed
    # After summing across all rows, apply normalization factor T/(T-lag) and divide by denom
    for lag in range(0, max_lag + 1):
        C[lag] = (T / (T - lag)) * (C[lag] / denom)
    return C


# ------------------- Output helpers -------------------


def save_and_plot(lags_ps, results_dict, out_prefix="bondcorr"):
    for label, C in results_dict.items():
        if C.size:
            fname = f"{out_prefix}_{label}.dat"
            np.savetxt(
                fname,
                np.vstack([lags_ps, C]).T,
                header=f"tau\tC_tau ({label})",
                fmt="%.6e",
            )
    plt.figure(figsize=(6, 4))
    plotted = False
    for label, C in results_dict.items():
        if C.size:
            plt.plot(lags_ps, C, label=label)
            plotted = True
    if not plotted:
        print("No correlation data to plot.")
        return
    plt.xlabel("lag time (ps)")
    plt.ylabel("C(tau)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=300)
    plt.close()


# ------------------- Main CLI -------------------


def main():
    p = argparse.ArgumentParser(
        description="Bond correlation functions (Pt-O, Pt-Ti, Pt-Pt). PBC optional."
    )
    p.add_argument(
        "traj",
        help="XYZ trajectory file (frames ordered, same atom ordering)",
    )
    p.add_argument(
        "--dt_fs",
        type=float,
        default=10.0,
        help="frame interval in fs (default 10 fs)",
    )
    p.add_argument(
        "--max_lag_ps",
        type=float,
        default=1.0,
        help="max lag time in ps (default 1.0 ps)",
    )
    p.add_argument(
        "--pt_ti_cutoff",
        type=float,
        default=2.85,
        help="Pt-Ti cutoff (Å)",
    )
    p.add_argument(
        "--pt_o_cutoff",
        type=float,
        default=2.05,
        help="Pt-O cutoff (Å)",
    )
    p.add_argument(
        "--pt_pt_cutoff",
        type=float,
        default=2.80,
        help="Pt-Pt cutoff (Å)",
    )
    p.add_argument(
        "--out_prefix",
        default="bcf",
        help="output filename prefix",
    )
    p.add_argument(
        "--box",
        nargs=6,
        type=float,
        metavar=("a", "b", "c", "alpha", "beta", "gamma"),
        help="triclinic box: a b c alpha beta gamma (angles in degrees). If provided, PBC/minimum-image is used.",
    )
    p.add_argument(
        "--continuous",
        action="store_true",
        help="compute continuous survival BCF instead of intermittent.",
    )
    p.add_argument(
        "--nbottom",
        type=int,
        default=None,
        help="If set, define interfacial Pt as the N atoms with lowest Z coordinates (overrides geometric definition).",
    )

    args = p.parse_args()

    use_pbc = False
    L = invL = None
    if args.box is not None:
        a, b, c, alpha, beta, gamma = args.box
        L, invL = box_matrix_from_lengths_angles(a, b, c, alpha, beta, gamma)
        use_pbc = True
        print(f"PBC enabled. Box matrix L:\n{L}")

    max_lag_frames = int(round(args.max_lag_ps * 1000.0 / args.dt_fs))
    if max_lag_frames < 1:
        max_lag_frames = 1

    print(
        "First pass: scanning trajectory to collect Pt-Ti, Pt-O and Pt-Pt pairs (ever within cutoff)."
    )

    pt_ti_pairs, pt_o_pairs, pt_pt_pairs, atom_syms, nframes, interfacial_set = (
        collect_pairs(
            args.traj,
            pt_ti_cutoff=args.pt_ti_cutoff,
            pt_o_cutoff=args.pt_o_cutoff,
            pt_pt_cutoff=args.pt_pt_cutoff,
            use_pbc=use_pbc,
            L=L,
            invL=invL,
            nbottom=args.nbottom,
        )
    )

    print(f"Frames found: {nframes}")
    print(f"Pt-Ti unique pairs found (total): {len(pt_ti_pairs)}")
    print(f"Pt-O  unique pairs found (total): {len(pt_o_pairs)}")
    print(f"Pt-Pt unique pairs found (total): {len(pt_pt_pairs)}")
    print(
        f"Interfacial Pt atoms (based on first frame): {len(interfacial_set)} indices"
    )
    if args.nbottom is not None:
        print(f"  (Method: Bottommost {args.nbottom} atoms based on Z-coord)")

    if nframes <= max_lag_frames:
        raise RuntimeError("Trajectory too short for requested max lag.")

    pt_ti_interfacial = [pair for pair in pt_ti_pairs if pair[0] in interfacial_set]
    pt_o_interfacial = [pair for pair in pt_o_pairs if pair[0] in interfacial_set]
    pt_pt_interfacial = [
        pair
        for pair in pt_pt_pairs
        if (pair[0] in interfacial_set and pair[1] in interfacial_set)
    ]
    pt_pt_non_interfacial = [
        pair
        for pair in pt_pt_pairs
        if (pair[0] not in interfacial_set and pair[1] not in interfacial_set)
    ]

    print(f"Pt-Ti pairs with interfacial Pt: {len(pt_ti_interfacial)}")
    print(f"Pt-O  pairs with interfacial Pt: {len(pt_o_interfacial)}")
    print(f"Pt-Pt pairs (interfacial both): {len(pt_pt_interfacial)}")
    print(f"Pt-Pt pairs (non-interfacial both): {len(pt_pt_non_interfacial)}")

    # Build boolean presence matrices (second pass)
    print("Second pass: building boolean time-matrices for chosen pair sets...")
    mat_pt_ti_interfacial = build_time_matrix(
        args.traj,
        pt_ti_interfacial,
        "Pt-Ti (interfacial Pt)",
        nframes,
        cutoff=args.pt_ti_cutoff,
        use_pbc=use_pbc,
        L=L,
        invL=invL,
    )
    mat_pt_o_interfacial = build_time_matrix(
        args.traj,
        pt_o_interfacial,
        "Pt-O  (interfacial Pt)",
        nframes,
        cutoff=args.pt_o_cutoff,
        use_pbc=use_pbc,
        L=L,
        invL=invL,
    )
    mat_pt_pt_interfacial = build_time_matrix(
        args.traj,
        pt_pt_interfacial,
        "Pt-Pt (interfacial group)",
        nframes,
        cutoff=args.pt_pt_cutoff,
        use_pbc=use_pbc,
        L=L,
        invL=invL,
    )
    mat_pt_pt_non_interfacial = build_time_matrix(
        args.traj,
        pt_pt_non_interfacial,
        "Pt-Pt (non-interfacial group)",
        nframes,
        cutoff=args.pt_pt_cutoff,
        use_pbc=use_pbc,
        L=L,
        invL=invL,
    )

    # Compute bond survival (intermittent or continuous based on args.continuous)
    print("Computing bond-correlation / survival functions...")
    if args.continuous:
        C_full_pt_ti_interfacial = compute_C_from_bool_matrix_continuous(
            mat_pt_ti_interfacial
        )
        C_full_pt_o_interfacial = compute_C_from_bool_matrix_continuous(
            mat_pt_o_interfacial
        )
        C_full_pt_pt_interfacial = compute_C_from_bool_matrix_continuous(
            mat_pt_pt_interfacial
        )
        C_full_pt_pt_non_interfacial = compute_C_from_bool_matrix_continuous(
            mat_pt_pt_non_interfacial
        )
    else:
        C_full_pt_ti_interfacial = compute_C_from_bool_matrix_intermit(
            mat_pt_ti_interfacial
        )
        C_full_pt_o_interfacial = compute_C_from_bool_matrix_intermit(
            mat_pt_o_interfacial
        )
        C_full_pt_pt_interfacial = compute_C_from_bool_matrix_intermit(
            mat_pt_pt_interfacial
        )
        C_full_pt_pt_non_interfacial = compute_C_from_bool_matrix_intermit(
            mat_pt_pt_non_interfacial
        )

    # Truncate to requested max lag
    wanted_len = max_lag_frames + 1

    def trunc(arr):
        return arr[:wanted_len] if arr.size else np.array([])

    C_pt_ti_interfacial = trunc(C_full_pt_ti_interfacial)
    C_pt_o_interfacial = trunc(C_full_pt_o_interfacial)
    C_pt_pt_interfacial = trunc(C_full_pt_pt_interfacial)
    C_pt_pt_non_interfacial = trunc(C_full_pt_pt_non_interfacial)

    lags = np.arange(wanted_len)
    lags_ps = lags * args.dt_fs / 1000.0

    results = {}
    if C_pt_ti_interfacial.size:
        results["Pt-Ti_interfacialPt"] = C_pt_ti_interfacial
    if C_pt_o_interfacial.size:
        results["Pt-O_interfacialPt"] = C_pt_o_interfacial
    if C_pt_pt_interfacial.size:
        results["Pt-Pt_interfacialGroup"] = C_pt_pt_interfacial
    if C_pt_pt_non_interfacial.size:
        results["Pt-Pt_nonInterfacialGroup"] = C_pt_pt_non_interfacial

    save_and_plot(lags_ps, results, out_prefix=args.out_prefix)

    print("Done. Outputs:")
    for label in results.keys():
        print(f"  {args.out_prefix}_{label}.dat")
    print(f"  {args.out_prefix}.png")


if __name__ == "__main__":
    main()
