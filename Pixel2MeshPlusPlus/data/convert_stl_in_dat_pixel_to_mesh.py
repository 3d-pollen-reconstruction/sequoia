import pickle
import numpy as np
import os
import trimesh
from sklearn.neighbors import NearestNeighbors
import argparse

def normalize_to_unit_sphere(vertices):
    """Normalize mesh vertices to fit in a unit sphere."""
    # Center at origin
    centroid = np.mean(vertices, axis=0)
    vertices_centered = vertices - centroid
    
    # Scale to unit sphere
    max_distance = np.max(np.linalg.norm(vertices_centered, axis=1))
    if max_distance > 0:
        vertices_normalized = vertices_centered / max_distance
    else:
        vertices_normalized = vertices_centered
        
    return vertices_normalized

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

def furthest_point_sampling(vertices, num_samples):
    """Fast furthest point sampling for sample coordinates"""
    if len(vertices) <= num_samples:
        return vertices
    
    n_vertices = len(vertices)
    sampled_indices = [0]
    distances_to_sampled = np.full(n_vertices, np.inf)
    
    for _ in range(num_samples - 1):
        last_sampled = sampled_indices[-1]
        new_distances = np.linalg.norm(vertices - vertices[last_sampled], axis=1)
        distances_to_sampled = np.minimum(distances_to_sampled, new_distances)
        distances_to_sampled[sampled_indices] = 0
        furthest_idx = np.argmax(distances_to_sampled)
        sampled_indices.append(furthest_idx)
    
    return vertices[sampled_indices]

def create_pixel2mesh_data(vertices):
    """Create complete Pixel2Mesh++ data structure from vertices"""
    print("Creating data structure for Pixel2Mesh++...")
    
    # Ensure we have exactly 156 vertices
    if len(vertices) != 156:
        print(f"Warning: Expected 156 vertices, found {len(vertices)}.")
    
    print("Creating mesh connectivity...")
    faces = create_mesh_connectivity(vertices)
    
    print("Creating stage data...")
    stage1_data = [create_stage_data(vertices, faces), create_stage_data(vertices, faces)]
    stage2_data = [create_stage_data(vertices, faces), create_stage_data(vertices, faces)]  
    stage3_data = [create_stage_data(vertices, faces), create_stage_data(vertices, faces)]
    
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
    sample_coord = furthest_point_sampling(vertices, 43)
    
    print("Creating Laplacian and Chebyshev data...")
    lape_idx = [
        (np.arange(156), np.arange(156)),
        (np.arange(100), np.arange(100)),
        (np.arange(50), np.arange(50))
    ]
    
    sample_cheb = [np.eye(43), np.eye(43) * 0.5]
    
    data = {
        'coord': vertices,
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

def convert_stl_to_prior(stl_file, output_file):
    """Convert STL file to Pixel2Mesh++ prior format"""
    print(f"Loading STL file: {stl_file}")
    try:
        mesh = trimesh.load(stl_file)
        vertices = np.array(mesh.vertices)
        print(f"Loaded {len(vertices)} vertices")
        
        # Normalize vertices to unit sphere if desired
        # Uncomment if you need to normalize
        # vertices = normalize_to_unit_sphere(vertices)
        # print("Normalized vertices to unit sphere")
        
        # Create Pixel2Mesh++ data structure
        pixel2mesh_data = create_pixel2mesh_data(vertices)
        
        # Save to .dat file
        with open(output_file, 'wb') as f:
            pickle.dump(pixel2mesh_data, f)
            
        print(f"✅ SUCCESS: Saved prior to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert STL file to Pixel2Mesh++ prior format')
    parser.add_argument('--output', default='./mean_shape_prior.dat', help='Output .dat file name')
    stl_file = r"C:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\data\mean_pollen_shape.stl"
    
        
    convert_stl_to_prior(stl_file=stl_file, output_file=parser.parse_args().output)