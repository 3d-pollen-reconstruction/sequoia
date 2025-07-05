import pickle
import numpy as np
import os
import glob
from pathlib import Path
import trimesh
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import networkx as nx
from tqdm import tqdm

def load_stl_files(directory):
    """Load all STL files from directory"""
    stl_files = glob.glob(os.path.join(directory, "*.stl"))
    meshes = []
    
    print(f"Found {len(stl_files)} STL files")
    
    for stl_file in stl_files:
        try:
            mesh = trimesh.load(stl_file)
            if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                meshes.append(mesh)
                print(f"Loaded: {os.path.basename(stl_file)} - {len(mesh.vertices)} vertices")
        except Exception as e:
            print(f"Error loading {stl_file}: {e}")
    
    return meshes

def normalize_to_unit_sphere(vertices):
    """
    Normalize mesh vertices to unit sphere:
    1. Center at origin
    2. Scale so max distance from origin is 1.0
    """
    vertices = vertices.copy().astype(np.float64)
    
    # Step 1: Center at origin (zero mean)
    centroid = np.mean(vertices, axis=0)
    vertices_centered = vertices - centroid
    
    # Step 2: Scale to unit sphere (max distance = 1.0)
    distances_from_origin = np.linalg.norm(vertices_centered, axis=1)
    max_distance = np.max(distances_from_origin)
    
    if max_distance > 0:
        vertices_normalized = vertices_centered / max_distance
    else:
        vertices_normalized = vertices_centered
    
    # Verify normalization
    final_distances = np.linalg.norm(vertices_normalized, axis=1)
    print(f"    Normalized - Max distance: {np.max(final_distances):.3f}, "
          f"Mean distance: {np.mean(final_distances):.3f}")
    print(f"    Bounds: X[{vertices_normalized[:,0].min():.3f}, {vertices_normalized[:,0].max():.3f}], "
          f"Y[{vertices_normalized[:,1].min():.3f}, {vertices_normalized[:,1].max():.3f}], "
          f"Z[{vertices_normalized[:,2].min():.3f}, {vertices_normalized[:,2].max():.3f}]")
    
    return vertices_normalized

def furthest_point_sampling_fast(vertices, num_samples):
    """Fast furthest point sampling"""
    if len(vertices) <= num_samples:
        return vertices
    
    n_vertices = len(vertices)
    sampled_indices = [0]
    distances_to_sampled = np.full(n_vertices, np.inf)
    
    pbar = tqdm(range(num_samples - 1), desc="FPS", leave=False)
    
    for _ in pbar:
        last_sampled = sampled_indices[-1]
        new_distances = np.linalg.norm(vertices - vertices[last_sampled], axis=1)
        distances_to_sampled = np.minimum(distances_to_sampled, new_distances)
        distances_to_sampled[sampled_indices] = 0
        furthest_idx = np.argmax(distances_to_sampled)
        sampled_indices.append(furthest_idx)
    
    pbar.close()
    return vertices[sampled_indices]

def resample_mesh_uniformly(mesh, num_samples):
    """Resample mesh to uniform number of points"""
    vertices = mesh.vertices
    
    if len(vertices) >= num_samples:
        # Use furthest point sampling for downsampling
        return furthest_point_sampling_fast(vertices, num_samples)
    else:
        # For upsampling, try surface sampling if available
        if hasattr(mesh, 'sample') and mesh.is_volume:
            try:
                points, _ = mesh.sample(num_samples)
                return points
            except:
                pass
        
        # Fallback: duplicate vertices to reach target
        sampled = vertices.copy()
        while len(sampled) < num_samples:
            remaining = num_samples - len(sampled)
            to_add = min(remaining, len(vertices))
            sampled = np.vstack([sampled, vertices[:to_add]])
        
        return sampled[:num_samples]

def compute_mean_shape_unit_sphere(meshes, target_vertices=156):
    """
    Compute mean shape with proper unit sphere normalization
    """
    print(f"Computing mean shape with {target_vertices} vertices...")
    print("All meshes will be normalized to unit sphere before averaging")
    
    normalized_meshes = []
    
    print("\nStep 1: Normalize each mesh to unit sphere")
    for i, mesh in enumerate(tqdm(meshes, desc="Normalizing meshes")):
        print(f"\nProcessing mesh {i+1}/{len(meshes)}")
        
        # Resample to target number of vertices
        print(f"  Original vertices: {len(mesh.vertices)}")
        resampled_vertices = resample_mesh_uniformly(mesh, target_vertices)
        print(f"  Resampled vertices: {len(resampled_vertices)}")
        
        # Normalize to unit sphere
        print(f"  Normalizing to unit sphere...")
        normalized_vertices = normalize_to_unit_sphere(resampled_vertices)
        
        normalized_meshes.append(normalized_vertices)
    
    print(f"\nStep 2: Computing mean across {len(normalized_meshes)} normalized meshes")
    
    # Stack all normalized meshes
    all_normalized = np.array(normalized_meshes)  # Shape: (n_meshes, target_vertices, 3)
    
    # Compute mean across meshes
    mean_vertices = np.mean(all_normalized, axis=0)  # Shape: (target_vertices, 3)
    
    print(f"\nStep 3: Final normalization of mean shape")
    # Normalize the mean shape to unit sphere as well
    mean_vertices_normalized = normalize_to_unit_sphere(mean_vertices)
    
    # Final verification
    print(f"\nFinal mean shape statistics:")
    distances = np.linalg.norm(mean_vertices_normalized, axis=1)
    print(f"  Number of vertices: {len(mean_vertices_normalized)}")
    print(f"  Distance from origin - Min: {np.min(distances):.3f}, "
          f"Max: {np.max(distances):.3f}, Mean: {np.mean(distances):.3f}")
    
    ranges = mean_vertices_normalized.max(axis=0) - mean_vertices_normalized.min(axis=0)
    print(f"  Coordinate ranges: X={ranges[0]:.3f}, Y={ranges[1]:.3f}, Z={ranges[2]:.3f}")
    
    # Check if shape is properly 3D
    min_range = np.min(ranges)
    if min_range < 0.1:
        print(f"  ⚠️  WARNING: Shape might be flat (min range: {min_range:.3f})")
    else:
        print(f"  ✅ Shape appears to be properly 3D")
    
    return mean_vertices_normalized

def align_mesh_preserve_shape(vertices):
    """
    Center at origin and scale by bounding-box diagonal.
    This avoids forcing a spherical shape, preserving morphology.
    """
    v = vertices.copy().astype(np.float64)
    
    # 1) Center at origin
    centroid = np.mean(v, axis=0)
    v_centered = v - centroid
    
    # 2) Scale to a consistent bounding-box diagonal
    bbox_min = np.min(v_centered, axis=0)
    bbox_max = np.max(v_centered, axis=0)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    if diagonal > 1e-8:
        v_aligned = v_centered / diagonal
    else:
        v_aligned = v_centered
    
    return v_aligned

def compute_mean_shape_preserving_pollen(meshes, target_vertices=156):
    """
    Compute a mean pollen shape that preserves overall morphology (not forcing a unit sphere).
    """
    print(f"Computing mean pollen shape with {target_vertices} vertices...")
    print("Center + bounding-box alignment (preserves shape ratios)")
    
    aligned_meshes = []
    
    for i, mesh in enumerate(tqdm(meshes, desc="Aligning pollen meshes")):
        print(f"  Mesh {i+1}/{len(meshes)}:")
        
        # Resample each mesh
        vertices = resample_mesh_uniformly(mesh, num_samples=target_vertices)
        print(f"    Resampled vertices: {len(vertices)}")
        
        # Align but do NOT force the shape to be a sphere
        aligned_vertices = align_mesh_preserve_shape(vertices)
        aligned_meshes.append(aligned_vertices)
    
    print(f"\nAveraging {len(aligned_meshes)} aligned pollen shapes...")
    # Stack and mean
    stacked = np.stack(aligned_meshes, axis=0)  # shape (N, V, 3)
    mean_vertices = np.mean(stacked, axis=0)    # shape (V, 3)
    
    # Optional: scale final shape so its largest dimension is 1
    # (But keep shape ratios intact)
    final_bbox_min = mean_vertices.min(axis=0)
    final_bbox_max = mean_vertices.max(axis=0)
    final_diagonal = np.linalg.norm(final_bbox_max - final_bbox_min)
    if final_diagonal > 1e-8:
        # Set largest dimension = 1
        mean_vertices = mean_vertices / final_diagonal
    
    # Print stats
    bounds_min = mean_vertices.min(axis=0)
    bounds_max = mean_vertices.max(axis=0)
    print(f"\nFinal shape stats (preserving morphology):")
    print(f"  Bounds in X: [{bounds_min[0]:.4f}, {bounds_max[0]:.4f}]")
    print(f"  Bounds in Y: [{bounds_min[1]:.4f}, {bounds_max[1]:.4f}]")
    print(f"  Bounds in Z: [{bounds_min[2]:.4f}, {bounds_max[2]:.4f}]")
    print(f"  Diagonal: {np.linalg.norm(bounds_max - bounds_min):.4f}")
    
    return mean_vertices

def create_mesh_connectivity(vertices, k_neighbors=8):
    """Create mesh connectivity using k-nearest neighbors"""
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(vertices)
    distances, indices = nbrs.kneighbors(vertices)
    
    faces = []
    for i, neighbors in enumerate(indices):
        for j in range(1, len(neighbors)-1):
            for k in range(j+1, len(neighbors)):
                face = [i, neighbors[j], neighbors[k], neighbors[1]]
                faces.append(face)
    
    return faces

def create_stage_data(vertices, faces):
    """Create stage data matching original structure"""
    n_vertices = len(vertices)
    row_indices = []
    col_indices = []
    
    for face in faces:
        for i in range(len(face)):
            for j in range(i+1, len(face)):
                row_indices.append(face[i])
                col_indices.append(face[j])
    
    return (np.array(row_indices), np.array(col_indices))

def create_pool_indices(n_vertices_source, n_vertices_target):
    """Create pooling indices matching original structure"""
    edges = []
    
    for i in range(n_vertices_source):
        for j in range(min(3, n_vertices_source-i-1)):
            if i + j + 1 < n_vertices_source:
                edges.append([i, i + j + 1])
    
    while len(edges) < 462:
        i = np.random.randint(0, n_vertices_source)
        j = np.random.randint(0, n_vertices_source)
        if i != j and [i, j] not in edges and [j, i] not in edges:
            edges.append([i, j])
    
    return np.array(edges[:462])

def create_pixel2mesh_data(mean_vertices):
    """Create complete Pixel2Mesh++ data structure"""
    
    print("Creating mesh connectivity...")
    faces = create_mesh_connectivity(mean_vertices)
    
    print("Creating stage data...")
    stage1_data = [create_stage_data(mean_vertices, faces), create_stage_data(mean_vertices, faces)]
    stage2_data = [create_stage_data(mean_vertices, faces), create_stage_data(mean_vertices, faces)]  
    stage3_data = [create_stage_data(mean_vertices, faces), create_stage_data(mean_vertices, faces)]
    
    print("Creating pooling indices...")
    pool_idx = [
        create_pool_indices(156, 156),
        create_pool_indices(618, 156)
    ]
    
    print("Creating face arrays...")
    faces_array = np.array(faces[:462])
    if faces_array.shape[1] != 4:
        if faces_array.shape[1] < 4:
            padding = np.zeros((faces_array.shape[0], 4 - faces_array.shape[1]), dtype=int)
            faces_array = np.hstack([faces_array, padding])
        else:
            faces_array = faces_array[:, :4]
    
    faces_list = [faces_array, faces_array[:200], faces_array[:100]]
    
    print("Creating sample coordinates...")
    sample_coord = furthest_point_sampling_fast(mean_vertices, 43)
    
    print("Creating Laplacian and Chebyshev data...")
    lape_idx = [
        (np.arange(156), np.arange(156)),
        (np.arange(100), np.arange(100)),
        (np.arange(50), np.arange(50))
    ]
    
    sample_cheb = [np.eye(43), np.eye(43) * 0.5]
    
    data = {
        'coord': mean_vertices,
        'stage1': stage1_data,
        'stage2': stage2_data,
        'stage3': stage3_data,
        'pool_idx': pool_idx,
        'faces': faces_list,
        'lape_idx': lape_idx,
        'sample_coord': sample_coord,
        'sample_cheb': sample_cheb,
        'sample_cheb_dense': sample_cheb,
        'sample_cheb_block_adj': [np.ones((43, 43), dtype=int), np.ones((43, 43), dtype=int)],
        'faces_triangle': faces_list
    }
    
    return data

def align_mesh_with_pca(vertices):
    """Aligns a mesh to its principal axes using PCA."""
    # 1. Center the vertices at the origin
    mean = np.mean(vertices, axis=0)
    centered_vertices = vertices - mean

    # 2. Compute the covariance matrix of the centered vertices
    covariance_matrix = np.cov(centered_vertices, rowvar=False)

    # 3. Find the eigenvectors (principal axes) of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # 4. Sort the eigenvectors by their corresponding eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_axes = eigenvectors[:, sorted_indices]

    # 5. Ensure a right-handed coordinate system to prevent mirrored shapes
    if np.linalg.det(principal_axes) < 0:
        principal_axes[:, 2] *= -1

    # 6. Rotate the vertices to align with the standard XYZ axes
    # This is the key step: it puts every mesh in the same orientation
    aligned_vertices = centered_vertices @ principal_axes

    return aligned_vertices

def compute_mean_shape_with_pca_alignment(meshes, target_vertices=156):
    """
    Computes the mean shape by first aligning each mesh to its principal axes.
    This is the most robust way to preserve features during averaging.
    """
    print("Computing mean shape with PCA alignment...")
    
    aligned_meshes = []
    for i, mesh in enumerate(tqdm(meshes, desc="PCA Aligning Meshes")):
        # 1. Resample to a consistent number of vertices
        vertices = resample_mesh_uniformly(mesh, target_vertices)
        
        # 2. Align the mesh to a standard orientation using PCA
        pca_aligned_vertices = align_mesh_with_pca(vertices)
        
        # 3. Normalize the size after alignment for consistency
        rms_distance = np.sqrt(np.mean(np.sum(pca_aligned_vertices**2, axis=1)))
        if rms_distance > 1e-8:
            pca_aligned_vertices /= rms_distance
            
        aligned_meshes.append(pca_aligned_vertices)
        
    # 4. Average the now-aligned meshes to get the true mean shape
    print("\nAveraging all PCA-aligned meshes...")
    stacked_meshes = np.stack(aligned_meshes, axis=0)
    mean_shape = np.mean(stacked_meshes, axis=0)
    
    # 5. Final scaling of the mean shape to fit in a unit box (preserves shape)
    max_range = np.max(mean_shape.max(axis=0) - mean_shape.min(axis=0))
    if max_range > 1e-8:
        mean_shape /= max_range
        
    print("\nFinal Mean Shape Stats (PCA Aligned):")
    ranges = mean_shape.max(axis=0) - mean_shape.min(axis=0)
    print(f"  Coordinate ranges: X={ranges[0]:.3f}, Y={ranges[1]:.3f}, Z={ranges[2]:.3f}")
    
    return mean_shape

def main():
    stl_directory = r"C:\Users\super\Documents\GitHub\sequoia\data\processed\meshes"
    
    print("Loading STL files...")
    meshes = load_stl_files(stl_directory)
    
    if len(meshes) == 0:
        print("No STL files found!")
        return
    
    print(f"\n{'='*60}")
    print(f"CREATING MEAN POLLEN SHAPE (PCA ALIGNED)")
    print(f"{'='*60}")
    
    try:
        # Call the new, correct function
        mean_vertices = compute_mean_shape_with_pca_alignment(meshes, target_vertices=156)
        
        print(f"\n{'='*60}")
        print("Creating Pixel2Mesh++ data structure...")
        pixel2mesh_data = create_pixel2mesh_data(mean_vertices)
        
        output_file = r"C:\Users\super\Documents\GitHub\sequoia\Pixel2MeshPlusPlus\data\pollen_mean_shape_prior.dat"
        
        with open(output_file, 'wb') as f:
            pickle.dump(pixel2mesh_data, f)
        
        print(f"✅ SUCCESS: Saved PCA-aligned mean pollen shape to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()