#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize relative force projection along bonds at a single frame.
User selects observation plane based on crystal lattice vectors (ab, bc, ca).
The force is projected onto the plane before being projected onto the bond direction.
Plots atoms as circles and force arrows along bonds, with unit cell outline correctly displayed.
"""

import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import textwrap
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
try:
    from parse_movement import add_parser_args, MovementParser
except ImportError:
    raise ImportError("parse_movement.py not found. Please ensure it is in the same directory.")

class MyFormatter(ap.RawDescriptionHelpFormatter, 
                  ap.ArgumentDefaultsHelpFormatter):
    pass

# ======================== Argument Parsing ========================
def parse_arguments():
    parser = ap.ArgumentParser(
        formatter_class=MyFormatter,
        description=textwrap.dedent("""
        Visualize relative force projection along bonds at a single frame.
        Atom indices are provided in pairs. The observation plane is defined by
        crystal lattice vectors: ab (a and b axes), bc (b and c), ca (c and a).
        Force is projected onto the plane before being projected onto the bond direction.
        Works correctly for non-orthogonal lattices (e.g., hexagonal).

        Example:
          python bond_force_vis.py -f MOVEMENT -id 25 19 25 11 -st 200 -plane ab -o force_plot
        """)
    )
    add_parser_args(parser)

    parser.add_argument(
        "-id", "--indices", 
        type=str, nargs='+', required=True,
        help="Atom indices (0-based) in pairs, e.g., '0 1 2 3' or with ranges '0-5 10-15'. "
                "Indices can repeat.",
        )
    parser.add_argument(
        "-plane", "--plane", 
        type=str, choices=['ab', 'bc', 'ca'], 
        default='ab',
        help="Observation plane defined by crystal lattice vectors: "
                "ab (a and b axes), bc (b and c), ca (c and a).",
        )
    parser.add_argument(
        "--scale", 
        type=float, default=1.0,
        help="Global scaling factor for arrow length (force in eV/Å). "
                "Larger values produce longer arrows.",
        )
    parser.add_argument(
        "-o", "--output", 
        type=Path, 
        default=Path("force_projection"),
        help="Output base name (without extension). Creates .png and .csv files.",
        )
    parser.add_argument(
        "-p", "--plot", 
        action="store_true",
        help="Show plot window interactively (plot is always saved).",
        )
    return parser.parse_args()

# ======================== Index Parsing ========================
def parse_indices_list_keep_order(indices_str_list):
    """Parse a list of strings containing integer indices and ranges, preserving order."""
    indices = []
    for token in indices_str_list:
        if '-' in token:
            start, end = map(int, token.split('-'))
            if start > end:
                raise ValueError(f"Invalid range: {token}")
            indices.extend(range(start, end+1))
        else:
            indices.append(int(token))
    return indices

# ======================== Geometry Utilities ========================
def get_pbc_vector(pos1_frac, pos2_frac, lattice):
    """Compute Cartesian vector from atom1 to atom2 with minimum image convention."""
    diff_frac = pos2_frac - pos1_frac
    diff_frac -= np.round(diff_frac)
    vec_cart = diff_frac @ lattice.T
    return vec_cart

def project_vector_onto_plane(v, v1, v2):
    """
    Project a 3D vector v onto the plane spanned by v1 and v2.
    Returns the projected vector (in Cartesian coordinates).
    """
    normal = np.cross(v1, v2)
    norm_n = np.linalg.norm(normal)
    if norm_n < 1e-6:
        return v
    normal = normal / norm_n
    v_proj = v - np.dot(v, normal) * normal
    return v_proj

def compute_relative_force_projection_in_plane(force1, force2, direction_vec, v1, v2):
    """
    Compute (F2 - F1) projected onto the plane, then onto the unit direction vector
    (also projected onto the plane). Returns scalar: positive for stretching.
    """
    delta_f = force2 - force1
    delta_f_proj = project_vector_onto_plane(delta_f, v1, v2)
    dir_proj = project_vector_onto_plane(direction_vec, v1, v2)
    norm_dir = np.linalg.norm(dir_proj)
    if norm_dir < 1e-6:
        return 0.0
    unit_dir = dir_proj / norm_dir
    return np.dot(delta_f_proj, unit_dir)

def build_plane_basis(v1, v2):
    """
    Build orthonormal basis for the plane spanned by v1 and v2.
    Returns: basis1 (unit vector along v1), basis2 (unit vector perpendicular to basis1 within the plane),
             len1 (|v1|), comp_parallel (projection of v2 onto basis1), len2_perp (perpendicular component length).
    """
    len1 = np.linalg.norm(v1)
    if len1 < 1e-6:
        raise ValueError("Zero-length lattice vector")
    basis1 = v1 / len1
    comp_parallel = np.dot(v2, basis1)
    v2_perp = v2 - comp_parallel * basis1
    len2_perp = np.linalg.norm(v2_perp)
    if len2_perp < 1e-6:
        # v2 is parallel to v1; create an arbitrary perpendicular vector in the plane
        if abs(basis1[0]) < 0.9:
            dummy = np.array([1, 0, 0])
        else:
            dummy = np.array([0, 1, 0])
        basis2 = np.cross(basis1, dummy)
        basis2 = basis2 / np.linalg.norm(basis2)
        len2_perp = 0.0
    else:
        basis2 = v2_perp / len2_perp
    return basis1, basis2, len1, comp_parallel, len2_perp

def cart_to_plane(pos_cart, basis1, basis2):
    """Convert Cartesian coordinates to 2D plane coordinates (u, v)."""
    u = np.dot(pos_cart, basis1)
    v = np.dot(pos_cart, basis2)
    return u, v

# ======================== Data Processing ========================
def load_single_frame(file_path, atom_indices, time):
    """Load a single frame from the trajectory at specified time."""
    data = MovementParser.parse(
        file_path=file_path,
        atom_indices=atom_indices,
        element_filter=None,
        start_time=time,
        end_time=time,
    )
    if data.n_frames == 0:
        raise RuntimeError(f"No frame found at time {time} fs.")
    print(f"Loaded frame at time {data.iter_time[0]:.2f} fs.")
    return data

def compute_pair_projections(frac_coords, forces, lattice, pairs, global_to_local, v1, v2):
    """
    For each pair, compute the relative force projection in the plane.
    Returns proj_values (list), direction_vecs (list of 3D vectors).
    """
    proj_values = []
    direction_vecs = []
    for idx1, idx2 in pairs:
        local1 = global_to_local[idx1]
        local2 = global_to_local[idx2]
        frac1 = frac_coords[local1]
        frac2 = frac_coords[local2]
        dir_vec = get_pbc_vector(frac1, frac2, lattice)
        f1 = forces[local1]
        f2 = forces[local2]
        proj = compute_relative_force_projection_in_plane(f1, f2, dir_vec, v1, v2)
        proj_values.append(proj)
        direction_vecs.append(dir_vec)
    return proj_values, direction_vecs

# ======================== Plotting ========================
def draw_unit_cell_and_atoms(ax, plane_coords, len1, comp_parallel, len2_perp):
    """Draw unit cell parallelogram and atoms."""
    corners = np.array([[0, 0],
                        [len1, 0],
                        [len1 + comp_parallel, len2_perp],
                        [comp_parallel, len2_perp],
                        [0, 0]])
    ax.plot(corners[:,0], corners[:,1], 'k--', linewidth=1, alpha=0.7, label='Unit cell')
    ax.scatter(plane_coords[:,0], plane_coords[:,1], s=50, c='k', marker='o', zorder=2, label='Atoms')

def draw_force_arrow(ax, start, end, f_rel, scale, bond_len):
    """Draw a single force arrow."""
    if f_rel > 0:
        arrow_start = start
        arrow_end = end
        color = 'r'
    else:
        arrow_start = end
        arrow_end = start
        color = 'b'
        f_rel = abs(f_rel)
    max_arrow_len = bond_len * 0.8
    visual_len = min(f_rel * scale, max_arrow_len)
    if visual_len < 0.01:
        return
    dx = arrow_end[0] - arrow_start[0]
    dy = arrow_end[1] - arrow_start[1]
    norm_dir = np.hypot(dx, dy)
    if norm_dir < 1e-6:
        return
    ux = dx / norm_dir
    uy = dy / norm_dir
    arrow_end_vis = (arrow_start[0] + ux * visual_len, arrow_start[1] + uy * visual_len)
    arrow = FancyArrowPatch(arrow_start, arrow_end_vis,
                            arrowstyle='->', mutation_scale=15,
                            color=color, linewidth=2, zorder=3)
    ax.add_patch(arrow)

def plot_force_projection(plane_coords, pairs, proj_values, global_to_plane,
                          len1, comp_parallel, len2_perp, plane, time, scale,
                          output_base, interactive):
    """
    Main plotting function: creates figure, draws all elements, saves to PNG,
    and optionally shows interactive window.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_unit_cell_and_atoms(ax, plane_coords, len1, comp_parallel, len2_perp)
    
    # Draw force arrows for each pair
    for (idx1, idx2), f_rel in zip(pairs, proj_values):
        start = global_to_plane[idx1]
        end = global_to_plane[idx2]
        bond_len = np.hypot(end[0]-start[0], end[1]-start[1])
        draw_force_arrow(ax, start, end, f_rel, scale, bond_len)
    
    ax.set_aspect('equal')
    ax.set_xlabel(f'Along {plane[0]}-axis direction (Å)')
    ax.set_ylabel(f'Perpendicular component in plane (Å)')
    ax.set_title(f'Force projection (in-plane) at t = {time:.2f} fs\nPlane: {plane.upper()}')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    
    # Save plot
    png_path = output_base.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {png_path}")
    
    if interactive:
        plt.show()
    else:
        plt.close(fig)

# ======================== Main ========================
def main():
    args = parse_arguments()
    
    # Validate time
    if args.start_time is None:
        raise ValueError("Please specify a single frame using -st (e.g., -st 200).")
    if args.end_time is not None and args.end_time != args.start_time:
        print(f"Warning: -st {args.start_time} != -et {args.end_time}. Using only frame at time {args.start_time}.")
    
    # Parse atom indices
    raw_indices = parse_indices_list_keep_order(args.indices)
    if len(raw_indices) % 2 != 0:
        raise ValueError(f"Number of indices ({len(raw_indices)}) is not even. Cannot form pairs.")
    pairs = [(raw_indices[i], raw_indices[i+1]) for i in range(0, len(raw_indices), 2)]
    print(f"Atom pairs: {pairs}")
    unique_indices = sorted(set(raw_indices))
    print(f"Unique atom indices: {unique_indices}")
    
    # Load trajectory data for the unique atoms at the specified time
    data = load_single_frame(args.file, unique_indices, args.start_time)
    frac_coords = data.position[0]
    lattice = data.lattice[0]
    forces = data.force[0]
    cart_coords = data.coordinate[0]
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(data.selected_indices)}
    
    # Get lattice vectors for the chosen plane
    if args.plane == 'ab':
        v1, v2 = lattice[0], lattice[1]
    elif args.plane == 'bc':
        v1, v2 = lattice[1], lattice[2]
    else:  # ca
        v1, v2 = lattice[2], lattice[0]
    
    # Build plane basis
    basis1, basis2, len1, comp_parallel, len2_perp = build_plane_basis(v1, v2)
    
    # Convert all unique atom Cartesian coordinates to plane coordinates
    plane_coords = np.array([cart_to_plane(cart_coords[i], basis1, basis2) for i in range(len(unique_indices))])
    
    # Compute force projections for each pair
    proj_values, direction_vecs = compute_pair_projections(
        frac_coords, forces, lattice, pairs, global_to_local, v1, v2
    )
    
    # Save CSV
    output_base = args.output
    if output_base.suffix:
        output_base = output_base.with_suffix('')
    csv_path = output_base.with_suffix('.csv')
    df = pd.DataFrame({
        'pair': [f"{p[0]}-{p[1]}" for p in pairs],
        'force_projection_eV_per_Ang': proj_values,
        'dx_Ang': [v[0] for v in direction_vecs],
        'dy_Ang': [v[1] for v in direction_vecs],
        'dz_Ang': [v[2] for v in direction_vecs]
    })
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    # Create mapping from global index to plane coordinates
    global_to_plane = {global_idx: plane_coords[i] for i, global_idx in enumerate(unique_indices)}
    
    # Plot everything
    plot_force_projection(
        plane_coords=plane_coords,
        pairs=pairs,
        proj_values=proj_values,
        global_to_plane=global_to_plane,
        len1=len1,
        comp_parallel=comp_parallel,
        len2_perp=len2_perp,
        plane=args.plane,
        time=data.iter_time[0],
        scale=args.scale,
        output_base=output_base,
        interactive=args.plot
    )


if __name__ == "__main__":
    main()