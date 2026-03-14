import os
import argparse
from typing import Tuple, Dict

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates

def extract_surface_geometry(mask_data: np.ndarray, smoothing_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    inner_core = binary_erosion(mask_data)
    boundary_mask = mask_data.astype(bool) ^ inner_core
    surface_indices = np.argwhere(boundary_mask)

    field_blur = gaussian_filter(mask_data.astype(float), sigma=smoothing_factor)
    dz, dy, dx = np.gradient(field_blur)
    
    normal_map = np.stack([-dz, -dy, -dx], axis=-1)
    raw_normals = normal_map[surface_indices[:, 0], surface_indices[:, 1], surface_indices[:, 2]]
    
    magnitudes = np.linalg.norm(raw_normals, axis=1, keepdims=True)
    unit_normals = np.divide(raw_normals, magnitudes, out=np.zeros_like(raw_normals), where=magnitudes != 0)
            
    return surface_indices, unit_normals


def trace_intensity_probes(
    image_volume: np.ndarray, 
    origin_pts: np.ndarray, 
    direction_vecs: np.ndarray, 
    intensity_cutoff: float, 
    voxel_dims: tuple
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = np.array(voxel_dims)
    depth_range_mm = np.arange(0, 30, 0.5)
    
    projection_step = direction_vecs / spacing
    
    sample_grid = origin_pts[:, None, :] + projection_step[:, None, :] * depth_range_mm[None, :, None]
    
    flat_grid = sample_grid.reshape(-1, 3).T
    signal_samples = map_coordinates(image_volume, flat_grid, order=1, mode='nearest')
    intensity_profiles = signal_samples.reshape(len(origin_pts), len(depth_range_mm))
    
    peak_intensities = np.max(intensity_profiles, axis=1)
    peak_locations = np.argmax(intensity_profiles, axis=1)
    active_probes = peak_intensities > intensity_cutoff
    
    return intensity_profiles, active_probes, peak_locations


def detect_fluid_gap(
    profiles: np.ndarray, 
    skull_peaks: np.ndarray, 
    valid_rays: np.ndarray, 
    low_percentile: float, 
    high_percentile: float
) -> np.ndarray:
    gap_indices = np.full(len(profiles), -1, dtype=int)
    linear_dist = np.arange(0, 30, 0.5)
    
    if not np.any(valid_rays): 
        return gap_indices
    
    dynamic_range = max(high_percentile - low_percentile, 1.0)
    
    dist_penalty = linear_dist[None, :] / 30.0
    intensity_penalty = profiles[valid_rays] / dynamic_range
    cost_matrix = dist_penalty - intensity_penalty
    
    temporal_mask = np.arange(profiles.shape[1])[None, :] < skull_peaks[valid_rays, None]
    cost_matrix[~temporal_mask] = -np.inf
    
    gap_indices[valid_rays] = np.argmax(cost_matrix, axis=1)
    return gap_indices


def localize_bone_threshold(
    profiles: np.ndarray, 
    gap_idx: np.ndarray, 
    peak_idx: np.ndarray, 
    is_valid: np.ndarray
) -> np.ndarray:
    edge_indices = np.full(len(profiles), -1, dtype=int)
    
    eligible = is_valid & (gap_idx != -1) & (gap_idx < peak_idx)
    
    if not np.any(eligible): 
        return edge_indices
    
    signal_diff = np.diff(profiles[eligible], axis=1)
    lb, ub = gap_idx[eligible], peak_idx[eligible]
    
    steps = np.arange(signal_diff.shape[1])
    window = (steps[None, :] >= lb[:, None]) & (steps[None, :] < ub[:, None])
    
    signal_diff[~window] = -np.inf
    
    best_grad_idx = np.argmax(signal_diff, axis=1)
    positive_slope = np.max(signal_diff, axis=1) > 0
    
    final_indices = np.zeros(len(profiles), dtype=bool)
    final_indices[eligible] = positive_slope
    
    edge_indices[final_indices] = best_grad_idx[positive_slope] + 1
    return edge_indices


def reconstruct_skull_volume(
    target_shape: tuple, 
    affine_mtx: np.ndarray, 
    seeds: np.ndarray, 
    vectors: np.ndarray, 
    hit_steps: np.ndarray, 
    voxel_size: tuple
) -> nib.Nifti1Image:

    valid_hits = hit_steps != -1
    p_start = seeds[valid_hits]
    v_unit = vectors[valid_hits]
    distance = hit_steps[valid_hits] * 0.5
    
    scaling = np.array(voxel_size)
    offset = (v_unit / scaling) * distance[:, None]
    world_coords = p_start + offset
    
    binary_vol = np.zeros(target_shape, dtype=np.uint8)
    discrete_coords = np.rint(world_coords).astype(int)
    
    in_bounds = (discrete_coords[:,0] >= 0) & (discrete_coords[:,0] < target_shape[0]) & \
                (discrete_coords[:,1] >= 0) & (discrete_coords[:,1] < target_shape[1]) & \
                (discrete_coords[:,2] >= 0) & (discrete_coords[:,2] < target_shape[2])
    
    idx = discrete_coords[in_bounds]
    binary_vol[idx[:,0], idx[:,1], idx[:,2]] = 1
    
    return nib.Nifti1Image(binary_vol, affine_mtx)


def skull_segmentation_engine(t1_img: nib.Nifti1Image, brain_mask: nib.Nifti1Image) -> nib.Nifti1Image:
    raw_data = t1_img.get_fdata()
    mask_data = brain_mask.get_fdata()
    pix_dims = t1_img.header.get_zooms()[:3]
    
    roi_signal = raw_data[mask_data > 0]
    if roi_signal.size == 0:
        raise RuntimeError("Error: No valid voxels found in the brain mask.")

    p2, p98 = np.percentile(roi_signal, [2, 98])
    cutoff = p2 + 0.1 * (p98 - p2)
    
    points, normals = extract_surface_geometry(mask_data)
    
    profiles, active, peaks = trace_intensity_probes(raw_data, points, normals, cutoff, pix_dims)
    
    gaps = detect_fluid_gap(profiles, peaks, active, p2, p98)
    
    edges = localize_bone_threshold(profiles, gaps, peaks, active)
    
    return reconstruct_skull_volume(raw_data.shape, t1_img.affine, points, normals, edges, pix_dims)

def load_mri(mri_path):

    mri_d = nib.load(mri_path)

    return mri_d
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mri', type=str, help="The path to T1 MRI.")
    parser.add_argument('--input_mask', type=str, help="The path to brain mask.")
    parser.add_argument('--output_path', type=str, default="./", help="The path where the segmetation will be saved, if don't work give the absolute path.")
   
    args = parser.parse_args()
    mri_path = args.input_mri
    mask_path = args.input_mask
    output_path = args.output_path
    
    mri = load_mri(mri_path)
    mask = load_mri(mask_path)

    skull = skull_segmentation_engine(mri, mask)
    nib.save(skull, os.path.join(output_path, f"{os.path.basename(mri_path).replace('.nii.gz', '_brain_skull.nii.gz')}"))