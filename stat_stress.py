import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread


def compute_radial_stress_profile(
    trajectory_file="computed.xyz",
    output_dat="stress_profile.dat",
    output_png="stress_profile.png",
    bin_width=0.5,
    max_radius=20.0,
):
    """
    Calculates the radially averaged von Mises stress for Pt atoms averaged over the entire trajectory.
    The method accumulates sum(sigma) and count(N) across all frames.
    """
    # Setup radial bins
    bin_edges = np.arange(0, max_radius + bin_width, bin_width)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    n_bins = len(bin_centers)
    total_counts = np.zeros(n_bins, dtype=np.float64)
    total_stress = np.zeros(n_bins, dtype=np.float64)
    total_stress_sq = np.zeros(n_bins, dtype=np.float64)

    print(f"Reading trajectory from: {trajectory_file}")
    print(f"Parameters: bin_width={bin_width} A, max_radius={max_radius} A")

    frame_count = 0

    # Iterate trajectory
    for i, atoms in enumerate(iread(trajectory_file, index=":")):
        # Select only platinum atoms
        pt_indices = [atom.index for atom in atoms if atom.symbol == "Pt"]
        if not pt_indices:
            continue

        pt_atoms = atoms[pt_indices]

        # Extract "Myproperty" (von Mises stress)
        if "Myproperty" in pt_atoms.arrays:
            stress = pt_atoms.get_array("Myproperty")
        else:
            keys = list(pt_atoms.arrays.keys())
            stress = pt_atoms.get_array(keys[-1])

        # Calculate distance from center of mass (COM)
        com = pt_atoms.get_center_of_mass()
        positions = pt_atoms.get_positions()
        distances = np.linalg.norm(positions - com, axis=1)

        # Binning using numpy histograms
        counts_frame, _ = np.histogram(distances, bins=bin_edges)
        stress_frame, _ = np.histogram(distances, bins=bin_edges, weights=stress)
        stress_sq_frame, _ = np.histogram(distances, bins=bin_edges, weights=stress**2)
        total_counts += counts_frame
        total_stress += stress_frame
        total_stress_sq += stress_sq_frame
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed frame {frame_count}...", end="\r")

    print(f"\nFinished processing {frame_count} frames.")

    # Normalization
    # Mean = Sum(Sigma) / Sum(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_stress = total_stress / total_counts
        mean_sq = total_stress_sq / total_counts
        variance = mean_sq - (mean_stress**2)
        std_stress = np.sqrt(np.maximum(variance, 0.0))

    mean_stress = np.nan_to_num(mean_stress)
    std_stress = np.nan_to_num(std_stress)

    # Output data
    data = np.column_stack((bin_centers, mean_stress, std_stress, total_counts))
    header = "Distance(A) Mean_Stress StdDev Total_Atom_Count"
    np.savetxt(output_dat, data, header=header, fmt="%10.5f %12.6f %12.6f %10d")
    print(f"Statistics saved to {output_dat}")
    plot_results(bin_centers, mean_stress, std_stress, total_counts, output_png)


def plot_results(r, mean, std, counts, output_file):
    mask = counts > 0
    r_plot = r[mask]
    mean_plot = mean[mask]
    std_plot = std[mask]
    counts_plot = counts[mask]
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color_stress = "tab:blue"
    ax1.set_xlabel("Distance from Centroid ($\AA$)", fontsize=12)
    ax1.set_ylabel("von Mises Stress", color=color_stress, fontsize=12)
    ax1.plot(r_plot, mean_plot, color=color_stress, linewidth=2, label="Mean Stress")
    ax1.fill_between(
        r_plot,
        mean_plot - std_plot,
        mean_plot + std_plot,
        color=color_stress,
        alpha=0.2,
        label="Std Dev",
    )
    ax1.tick_params(axis="y", labelcolor=color_stress)

    ax2 = ax1.twinx()
    color_dens = "tab:gray"
    ax2.set_ylabel("Atom Count (Structure)", color=color_dens, fontsize=12)
    ax2.plot(
        r_plot,
        counts_plot,
        color=color_dens,
        linestyle="--",
        alpha=0.5,
        label="Atom Count",
    )
    ax2.tick_params(axis="y", labelcolor=color_dens)

    plt.title("Radial Stress Profile of Pt Nanoparticle", fontsize=14)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    plt.tight_layout()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    compute_radial_stress_profile(
        trajectory_file="computed.xyz",
        bin_width=0.01,
        max_radius=10.0,
    )
