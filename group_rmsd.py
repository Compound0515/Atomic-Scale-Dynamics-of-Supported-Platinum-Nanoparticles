import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    from ase.io import read as ase_read

    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False


def read_dump_frames(pattern):
    """
    Reads dump files.
    Sorts atoms by ID to ensure group indices remain validacross the entire trajectory.
    """
    files = glob.glob(pattern)

    # Sort files by time
    def extract_time(fname):
        match = re.search(r"pt-([\d.]+)ns\.dump", fname)
        if match:
            return float(match.group(1))
        # Fallback for general numbers
        match_gen = re.search(r"([\d.]+)", fname)
        if match_gen:
            return float(match_gen.group(1))
        return 0

    files = sorted(files, key=extract_time)
    print(f"Found trajectory files: {files}")

    if len(files) == 0:
        raise FileNotFoundError(f"No files match pattern '{pattern}'")

    all_frames = []
    for fname in files:
        with open(fname, "r") as f:
            while True:
                # Read 9-line header
                header = []
                for _ in range(9):
                    line = f.readline()
                    if not line:
                        break
                    header.append(line.rstrip("\n"))
                if len(header) == 0:
                    break

                try:
                    natoms = int(header[3].split()[0])
                except Exception as e:
                    raise RuntimeError(f"Cannot parse natoms from {fname}: {e}")

                # Read atoms (ID, type, x, y, z)
                raw_data = np.zeros((natoms, 4), dtype=float)

                for i in range(natoms):
                    line = f.readline()
                    parts = line.split()
                    try:
                        atom_id = float(parts[0])
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        raw_data[i] = [atom_id, x, y, z]
                    except Exception as e:
                        raise RuntimeError(f"Bad atom line in {fname}: {e}")

                # Sort by Atom ID so indices are stable for grouping
                raw_data = raw_data[raw_data[:, 0].argsort()]
                all_frames.append(raw_data[:, 1:])

    return all_frames


def match_group_to_first_frame(group_coords, frame_coords, tol=1.0):
    """
    Map atoms from a CIF group to the first frame of the dump using KDTree.
    """
    if group_coords.shape[0] == 0:
        return []

    tree = cKDTree(frame_coords)
    dists, idxs = tree.query(group_coords, k=1)

    matched_indices = []
    for i, d in enumerate(dists):
        if d <= tol:
            matched_indices.append(idxs[i])
        else:
            pass

    return sorted(list(set(matched_indices)))


def kabsch_align(ref, cur):
    """
    Align current frame to reference frame using Kabsch algorithm.
    """
    if ref.shape != cur.shape:
        raise ValueError("ref and cur must have the same shape")
    N = ref.shape[0]
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0, np.zeros(3, dtype=float)

    ref_centroid = ref.mean(axis=0)
    cur_centroid = cur.mean(axis=0)
    ref_centered = ref - ref_centroid
    cur_centered = cur - cur_centroid

    H = cur_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T

    # ensure right-handed rotation (determinant +1)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    t_opt = ref_centroid - R_opt @ cur_centroid
    moved = (R_opt @ cur.T).T + t_opt
    diff = ref - moved
    rmsd = np.sqrt((diff * diff).sum() / float(N))

    # extract Euler angles in ZYX order (yaw, pitch, roll)
    try:
        yaw, pitch, roll = Rotation.from_matrix(R_opt).as_euler("ZYX", degrees=True)
    except Exception:
        roll = pitch = yaw = 0.0

    return float(roll), float(pitch), float(yaw), float(rmsd), t_opt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze rotation/RMSD using original Kabsch core with Grouping."
    )
    parser.add_argument(
        "--pattern",
        default="pt-*ns.dump",
        help="glob pattern to find dump files",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="List of CIF files for groups (e.g. group1.cif group2.cif)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=10.0,
        help="time between frames (number)",
    )
    parser.add_argument(
        "--unit",
        choices=["fs", "ps", "ns"],
        default="fs",
        help="unit for --dt",
    )
    parser.add_argument(
        "--outdir",
        default="analysis",
        help="output folder",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.5,
        help="Tolerance (Å) for matching CIF atoms to Dump",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="do not generate PNG figures",
    )
    return parser.parse_args()


def dt_to_ns(dt, unit):
    if unit == "fs":
        return dt * 1e-6
    elif unit == "ps":
        return dt * 1e-3
    elif unit == "ns":
        return dt * 1.0
    return dt


def main():
    args = parse_args()
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Read Frames (Sorted by ID)
    print("Reading frames...")
    frames = read_dump_frames(args.pattern)
    if len(frames) == 0:
        raise RuntimeError("No frames read.")

    print(f"Total frames read: {len(frames)}")
    nframes = len(frames)

    # Define Groups
    tasks = []

    if args.groups:
        if not ASE_AVAILABLE:
            raise ImportError(
                "ASE is required for --groups. Install with `pip install ase`."
            )

        print(f"Matching groups from: {args.groups}")
        first_frame_coords = frames[0]

        for cif_file in args.groups:
            group_name = os.path.splitext(os.path.basename(cif_file))[0]
            atoms = ase_read(cif_file)
            group_coords = atoms.get_positions()
            indices = match_group_to_first_frame(
                group_coords, first_frame_coords, tol=args.tol
            )

            if indices:
                tasks.append((group_name, indices))
                print(f"  Group '{group_name}': Matched {len(indices)} atoms")
            else:
                print(f"  Warning: Group '{group_name}' matched 0 atoms!")
    else:
        # If no groups, use all atoms
        print("No groups specified. Using all atoms.")
        tasks.append(("system", np.arange(frames[0].shape[0])))

    dt_ns = dt_to_ns(args.dt, args.unit)
    times_ns = np.arange(nframes) * dt_ns

    for group_name, indices in tasks:
        print(f"\nProcessing Group: {group_name}")
        roll = np.zeros(nframes)
        pitch = np.zeros(nframes)
        yaw = np.zeros(nframes)
        rmsd_arr = np.zeros(nframes)
        translation_vecs = np.zeros((nframes, 3))
        ref = frames[0][indices, :].copy()
        roll[0] = pitch[0] = yaw[0] = rmsd_arr[0] = 0.0

        for i in range(1, nframes):
            frame = frames[i][indices, :]
            r, p, ydeg, rmsd, t_opt = kabsch_align(ref, frame)
            roll[i], pitch[i], yaw[i] = r, p, ydeg
            rmsd_arr[i] = rmsd
            translation_vecs[i, :] = t_opt

            if (i % 50 == 0) or (i == nframes - 1):
                pass

        np.savetxt(
            os.path.join(outdir, f"{group_name}_RMSD.dat"),
            np.column_stack((np.arange(nframes), rmsd_arr)),
            fmt="%d %.6f",
            header="Frame RMSD(Å)",
        )

        np.savetxt(
            os.path.join(outdir, f"{group_name}_rotation.dat"),
            np.column_stack((np.arange(nframes), roll, pitch, yaw)),
            fmt="%d %.6f %.6f %.6f",
            header="Frame Roll(deg) Pitch(deg) Yaw(deg)",
        )

        magnitudes = np.linalg.norm(translation_vecs, axis=1)
        np.savetxt(
            os.path.join(outdir, f"{group_name}_translation.dat"),
            np.column_stack((np.arange(nframes), translation_vecs, magnitudes)),
            fmt="%d %.6f %.6f %.6f %.6f",
            header="Frame t_x(Å) t_y(Å) t_z(Å) |t|(Å)",
        )

        if not args.no_plot:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 3, 1)
            plt.plot(times_ns, rmsd_arr)
            plt.title(f"{group_name} RMSD")
            plt.xlabel("Time (ns)")
            plt.ylabel("RMSD (Å)")

            plt.subplot(1, 3, 2)
            plt.plot(times_ns, roll, label="Roll (X)")
            plt.plot(times_ns, pitch, label="Pitch (Y)")
            plt.plot(times_ns, yaw, label="Yaw (Z)")
            plt.title(f"{group_name} Rotation")
            plt.xlabel("Time (ns)")
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.plot(times_ns, magnitudes)
            plt.title(f"{group_name} Translation")
            plt.xlabel("Time (ns)")
            plt.ylabel("|t| (Å)")

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{group_name}_results.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(8, 5))
            plt.plot(times_ns, translation_vecs[:, 0], label="t_x")
            plt.plot(times_ns, translation_vecs[:, 1], label="t_y")
            plt.plot(times_ns, translation_vecs[:, 2], label="t_z")
            plt.xlabel("Time (ns)")
            plt.ylabel("Translation (Å)")
            plt.title(f"{group_name} Translation")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(outdir, f"{group_name}_translation.png"),
                dpi=300,
            )
            plt.close()

    print(f"All outputs written to {outdir}")


if __name__ == "__main__":
    main()
