import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import cos, sin, radians, sqrt


# ------------------- I/O and Geometry Helpers -------------------


def iter_xyz_frames(path):
    """Yields atoms and coordinates from an XYZ file."""
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
                syms.append(parts[0])
                coords[i, 0:3] = [float(x) for x in parts[1:4]]
            yield comment, syms, coords


def box_matrix_from_lengths_angles(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Returns 3x3 box matrix L and its inverse."""
    alpha, beta, gamma = map(radians, [alpha_deg, beta_deg, gamma_deg])
    ca, cb, cg, sg = cos(alpha), cos(beta), cos(gamma), sin(gamma)
    a_vec = np.array([a, 0.0, 0.0])
    b_vec = np.array([b * cg, b * sg, 0.0])
    cx, cy = c * cb, c * (ca - cb * cg) / sg
    cz = c * sqrt(max(0.0, 1.0 - cb**2 - ((ca - cb * cg) / sg) ** 2))
    c_vec = np.array([cx, cy, cz])
    L = np.column_stack((a_vec, b_vec, c_vec))
    return L, np.linalg.inv(L)


def min_image_delta(delta, L, invL):
    """Applies minimum image convention to displacement vectors."""
    orig_shape = delta.shape
    flat = delta.reshape(-1, 3).T
    frac = invL.dot(flat)
    frac = frac - np.round(frac)
    cart = L.dot(frac).T
    return cart.reshape(orig_shape)


# ------------------- Core Calculation Logic -------------------


def count_total_bonds(coords, idx_a, idx_b, cutoff, L, invL, use_pbc, self_pairs=False):
    """Counts the total number of unique bonds between two sets of atoms."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return 0

    total_bonds = 0
    cutoff_square = cutoff**2
    coords_a = coords[idx_a]
    coords_b = coords[idx_b]

    # This simple loop is O(Na * Nb), fine for small clusters/interfaces.
    # For large systems, KDTree or cell lists should be used.
    for _, pos_a in enumerate(coords_a):
        diffs = coords_b - pos_a
        if use_pbc:
            diffs = min_image_delta(diffs, L, invL)
        dists_square = np.sum(diffs**2, axis=1)

        if self_pairs:
            # For Pt-Pt within the same group, only count each pair once
            # This implementation assumes we are passing the exact same list for idx_a and idx_b
            # So we can just divide the final result by 2 to account for double counting.
            # However, we must exclude the self-interaction.
            valid_bonds = np.sum(
                (dists_square <= cutoff_square) & (dists_square > 1e-6)
            )
            total_bonds += valid_bonds
        else:
            total_bonds += np.sum(dists_square <= cutoff_square)

    return total_bonds // 2 if self_pairs else total_bonds


def main():
    p = argparse.ArgumentParser(
        description="Calculate total bond numbers for interfacial Pt."
    )
    p.add_argument(
        "traj",
        help="XYZ trajectory file",
    )
    p.add_argument(
        "--dt_fs",
        type=float,
        default=10.0,
        help="Time step in fs",
    )
    p.add_argument(
        "--pt_ti_cutoff",
        type=float,
        default=2.85,
    )
    p.add_argument(
        "--pt_o_cutoff",
        type=float,
        default=2.05,
    )
    p.add_argument(
        "--pt_pt_cutoff",
        type=float,
        default=2.80,
    )
    p.add_argument(
        "--box",
        nargs=6,
        type=float,
        help="a b c alpha beta gamma",
    )
    p.add_argument(
        "--out",
        default="total_interfacial_bonds",
        help="Output prefix",
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
    if args.box:
        L, invL = box_matrix_from_lengths_angles(*args.box)
        use_pbc = True

    interfacial_pt = []
    time_series = []

    print("Starting trajectory analysis...")

    for frame_idx, (comment, syms, coords) in enumerate(
        tqdm(iter_xyz_frames(args.traj))
    ):
        pt_idx = [i for i, s in enumerate(syms) if s.lower() == "pt"]
        ti_idx = [i for i, s in enumerate(syms) if s.lower() == "ti"]
        o_idx = [i for i, s in enumerate(syms) if s.lower() == "o"]

        if frame_idx == 0:
            if args.nbottom is not None:
                pt_coords = coords[pt_idx]
                if len(pt_coords) > 0:
                    sorted_local_indices = np.argsort(pt_coords[:, 2])
                    count = min(args.nbottom, len(pt_idx))
                    for k in range(count):
                        global_id = pt_idx[sorted_local_indices[k]]
                        interfacial_pt.append(global_id)
                print(
                    f"Frame 0: Defined interfacial Pt as bottommost {len(interfacial_pt)} atoms."
                )
            else:
                for p_idx in pt_idx:
                    p_pos = coords[p_idx]
                    if len(ti_idx) > 0:
                        d_ti = coords[ti_idx] - p_pos
                        if use_pbc:
                            d_ti = min_image_delta(d_ti, L, invL)
                        if np.any(np.sum(d_ti**2, axis=1) <= args.pt_ti_cutoff**2):
                            interfacial_pt.append(p_idx)
                            continue
                    if len(o_idx) > 0:
                        d_o = coords[o_idx] - p_pos
                        if use_pbc:
                            d_o = min_image_delta(d_o, L, invL)
                        if np.any(np.sum(d_o**2, axis=1) <= args.pt_o_cutoff**2):
                            interfacial_pt.append(p_idx)
                print(f"Frame 0: Found {len(interfacial_pt)} interfacial Pt atoms")

        # Calculate Total Bonds for the current frame
        time_ps = (frame_idx * args.dt_fs) / 1000.0
        total_ti = count_total_bonds(
            coords,
            interfacial_pt,
            ti_idx,
            args.pt_ti_cutoff,
            L,
            invL,
            use_pbc,
        )
        total_o = count_total_bonds(
            coords,
            interfacial_pt,
            o_idx,
            args.pt_o_cutoff,
            L,
            invL,
            use_pbc,
        )
        total_pt = count_total_bonds(
            coords,
            interfacial_pt,
            interfacial_pt,
            args.pt_pt_cutoff,
            L,
            invL,
            use_pbc,
            self_pairs=True,
        )

        time_series.append([time_ps, total_ti, total_o, total_pt])

    if not time_series:
        print("No data collected.")
        return

    data = np.array(time_series)
    np.savetxt(
        f"{args.out}.dat",
        data,
        fmt="%.4f",
        header="Time(ps) Total_Pt-Ti Total_Pt-O Total_Pt-Pt",
    )

    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1], label="Interfacial Pt-Ti", color="blue")
    plt.plot(data[:, 0], data[:, 2], label="Interfacial Pt-O", color="red")
    plt.plot(data[:, 0], data[:, 3], label="Interfacial Pt-Pt", color="black")
    plt.xlabel("Time (ps)")
    plt.ylabel("Total Number of Bonds")
    plt.title("Total Bonding Evolution of Interfacial Pt Atoms")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{args.out}.png", dpi=300)
    print(f"Results saved to {args.out}.dat and {args.out}.png")


if __name__ == "__main__":
    main()
