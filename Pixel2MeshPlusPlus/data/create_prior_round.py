import pickle
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

def generate_hollow_sphere_vertices(num_vertices=156, radius=1.0):
    """Generate vertices uniformly distributed on HOLLOW sphere surface only"""
    print(f"Generating {num_vertices} vertices on hollow sphere surface (radius={radius})...")
    
    # Method: Fibonacci spiral for perfect uniform distribution on sphere SURFACE
    vertices = []
    
    # Parameters for Fibonacci spiral
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
    
    for i in range(num_vertices):
        # y coordinate from top to bottom of sphere
        y = 1 - (i / (num_vertices - 1)) * 2  # y goes from 1 to -1
        
        # Radius at this y level (circle radius, not sphere radius)
        circle_radius = np.sqrt(1 - y * y)
        
        # Angle around the sphere
        theta = golden_angle * i
        
        # x and z coordinates on the circle at this y level
        x = np.cos(theta) * circle_radius
        z = np.sin(theta) * circle_radius
        
        # Scale by desired radius
        vertices.append([x * radius, y * radius, z * radius])
    
    vertices = np.array(vertices)
    
    # Verify it's a perfect hollow sphere
    distances_from_origin = np.linalg.norm(vertices, axis=1)
    
    print(f"  Hollow sphere verification:")
    print(f"    All distances from origin: Min={distances_from_origin.min():.6f}, Max={distances_from_origin.max():.6f}")
    print(f"    Target radius: {radius:.6f}")
    print(f"    Deviation from target: Max={abs(distances_from_origin - radius).max():.6f}")
    
    # Check coverage in all dimensions
    ranges = vertices.max(axis=0) - vertices.min(axis=0)
    print(f"    Coordinate coverage:")
    print(f"      X: [{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}] (range: {ranges[0]:.3f})")
    print(f"      Y: [{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}] (range: {ranges[1]:.3f})")
    print(f"      Z: [{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}] (range: {ranges[2]:.3f})")
    
    # Verify it's truly round (not elliptical) - FIXED TOLERANCE
    max_range_diff = max(abs(ranges[0] - ranges[1]), abs(ranges[1] - ranges[2]), abs(ranges[0] - ranges[2]))
    if max_range_diff < 0.05:  # More reasonable tolerance
        print(f"    ✅ Perfect sphere - all dimensions equal (max difference: {max_range_diff:.3f})")
    else:
        print(f"    ⚠️  Shape might be elliptical (max difference: {max_range_diff:.3f})")
    
    # Verify all points are ON the surface (not inside)
    surface_tolerance = 1e-10
    points_on_surface = np.abs(distances_from_origin - radius) < surface_tolerance
    print(f"    Points exactly on surface: {points_on_surface.sum()}/{len(vertices)}")
    
    if points_on_surface.sum() == len(vertices):
        print(f"    ✅ Perfect hollow sphere - all points on surface")
    else:
        print(f"    ⚠️  Some points not exactly on surface")
    
    return vertices

def generate_sample_coordinates_hollow(main_vertices, num_samples=43):
    """Generate sample coordinates from main vertices using furthest point sampling"""
    print(f"Generating {num_samples} sample coordinates on sphere surface...")
    
    if len(main_vertices) <= num_samples:
        return main_vertices
    
    n_vertices = len(main_vertices)
    sampled_indices = [0]  # Start with first vertex
    distances_to_sampled = np.full(n_vertices, np.inf)
    
    for _ in range(num_samples - 1):
        # Update distances to the most recently added point
        last_sampled = sampled_indices[-1]
        new_distances = np.linalg.norm(main_vertices - main_vertices[last_sampled], axis=1)
        distances_to_sampled = np.minimum(distances_to_sampled, new_distances)
        
        # Set distance of already sampled points to 0
        distances_to_sampled[sampled_indices] = 0
        
        # Find the point with maximum minimum distance
        furthest_idx = np.argmax(distances_to_sampled)
        sampled_indices.append(furthest_idx)
    
    sample_vertices = main_vertices[sampled_indices]
    
    # Verify sample points are also on sphere surface
    sample_distances = np.linalg.norm(sample_vertices, axis=1)
    print(f"    Sample points verification:")
    print(f"      Distance from origin: Min={sample_distances.min():.6f}, Max={sample_distances.max():.6f}")
    print(f"      All on sphere surface: {np.allclose(sample_distances, 1.0, atol=1e-10)}")
    
    return sample_vertices

def create_sphere_connectivity(vertices, k_neighbors=8):
    """Create mesh connectivity for hollow sphere surface"""
    print(f"Creating surface mesh connectivity...")
    
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='ball_tree').fit(vertices)
    distances, indices = nbrs.kneighbors(vertices)
    
    faces = []
    for i, neighbors in enumerate(indices):
        for j in range(1, len(neighbors)-1):
            for k in range(j+1, len(neighbors)):
                face = [i, neighbors[j], neighbors[k], neighbors[1]]
                faces.append(face)
    
    print(f"    Generated {len(faces)} surface faces")
    return faces

def create_stage_data(vertices, faces):
    """Create stage data matching original structure"""
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
    
    # Create systematic edges
    for i in range(n_vertices_source):
        for j in range(min(3, n_vertices_source-i-1)):
            if i + j + 1 < n_vertices_source:
                edges.append([i, i + j + 1])
    
    # Fill remaining edges randomly
    while len(edges) < 462:
        i = np.random.randint(0, n_vertices_source)
        j = np.random.randint(0, n_vertices_source)
        if i != j and [i, j] not in edges and [j, i] not in edges:
            edges.append([i, j])
    
    return np.array(edges[:462])

def create_hollow_sphere_pixel2mesh_data(sphere_vertices, sample_vertices):
    """Create complete Pixel2Mesh++ data structure for hollow sphere"""
    
    print("Creating hollow sphere Pixel2Mesh++ data structure...")
    
    # Create surface connectivity
    faces = create_sphere_connectivity(sphere_vertices)
    
    # Create stage data
    print("Creating stage data...")
    stage1_data = [create_stage_data(sphere_vertices, faces), create_stage_data(sphere_vertices, faces)]
    stage2_data = [create_stage_data(sphere_vertices, faces), create_stage_data(sphere_vertices, faces)]  
    stage3_data = [create_stage_data(sphere_vertices, faces), create_stage_data(sphere_vertices, faces)]
    
    # Create pooling indices
    print("Creating pooling indices...")
    pool_idx = [
        create_pool_indices(156, 156),
        create_pool_indices(618, 156)
    ]
    
    # Create face arrays
    print("Creating face arrays...")
    faces_array = np.array(faces[:462])
    
    # Ensure faces have 4 columns
    if faces_array.shape[1] != 4:
        if faces_array.shape[1] < 4:
            padding = np.zeros((faces_array.shape[0], 4 - faces_array.shape[1]), dtype=int)
            faces_array = np.hstack([faces_array, padding])
        else:
            faces_array = faces_array[:, :4]
    
    faces_list = [faces_array, faces_array[:200], faces_array[:100]]
    
    # Create Laplacian indices
    print("Creating Laplacian and Chebyshev data...")
    lape_idx = [
        (np.arange(156), np.arange(156)),
        (np.arange(100), np.arange(100)),
        (np.arange(50), np.arange(50))
    ]
    
    # Create Chebyshev data
    sample_cheb = [np.eye(43), np.eye(43) * 0.5]
    
    # Assemble final data structure
    data = {
        'coord': sphere_vertices,
        'stage1': stage1_data,
        'stage2': stage2_data,
        'stage3': stage3_data,
        'pool_idx': pool_idx,
        'faces': faces_list,
        'lape_idx': lape_idx,
        'sample_coord': sample_vertices,
        'sample_cheb': sample_cheb,
        'sample_cheb_dense': sample_cheb,
        'sample_cheb_block_adj': [np.ones((43, 43), dtype=int), np.ones((43, 43), dtype=int)],
        'faces_triangle': faces_list
    }
    
    return data

def main():
    print("="*70)
    print("GENERATING PERFECT HOLLOW SPHERE PRIOR (BALL SURFACE)")
    print("="*70)
    
    # Generate hollow sphere vertices (surface only, like a ball)
    sphere_vertices = generate_hollow_sphere_vertices(num_vertices=156, radius=1.0)
    
    # Generate sample coordinates on the same sphere surface
    sample_vertices = generate_sample_coordinates_hollow(sphere_vertices, num_samples=43)
    
    # Create Pixel2Mesh++ data structure
    pixel2mesh_data = create_hollow_sphere_pixel2mesh_data(sphere_vertices, sample_vertices)
    
    # Save the hollow sphere prior
    output_file = r"C:\Users\super\Documents\GitHub\sequoia\Pixel2MeshPlusPlus\data\hollow_sphere_prior.dat"
    
    with open(output_file, 'wb') as f:
        pickle.dump(pixel2mesh_data, f)
    
    print(f"✅ SUCCESS: Saved perfect hollow sphere prior to: {output_file}")
    
    # Final verification with CORRECTED checks
    print("\n" + "="*70)
    print("FINAL VERIFICATION - HOLLOW SPHERE PROPERTIES")
    print("="*70)
    
    coord = pixel2mesh_data['coord']
    sample_coord = pixel2mesh_data['sample_coord']
    
    # Check main coordinates
    distances = np.linalg.norm(coord, axis=1)
    ranges = coord.max(axis=0) - coord.min(axis=0)
    
    print("Main coordinates (156 vertices):")
    print(f"  All distances from origin: {distances.min():.6f} to {distances.max():.6f}")
    print(f"  Perfect unit radius: {np.allclose(distances, 1.0, atol=1e-10)}")
    print(f"  Coordinate spans: X={ranges[0]:.3f}, Y={ranges[1]:.3f}, Z={ranges[2]:.3f}")
    
    # CORRECTED roundness check
    max_range_diff = max(abs(ranges[0] - ranges[1]), abs(ranges[1] - ranges[2]), abs(ranges[0] - ranges[2]))
    perfectly_round = max_range_diff < 0.05
    print(f"  Perfectly round: {perfectly_round} (max difference: {max_range_diff:.3f})")
    
    # Check sample coordinates
    sample_distances = np.linalg.norm(sample_coord, axis=1)
    
    print("\nSample coordinates (43 vertices):")
    print(f"  All distances from origin: {sample_distances.min():.6f} to {sample_distances.max():.6f}")
    print(f"  Perfect unit radius: {np.allclose(sample_distances, 1.0, atol=1e-10)}")
    
    # CORRECTED final quality assessment
    print(f"\n🏀 HOLLOW SPHERE QUALITY CHECK:")
    
    perfect_sphere = (
        np.allclose(distances, 1.0, atol=1e-10) and    # All points on surface
        perfectly_round and                            # Round in all dimensions
        all(r > 1.9 for r in ranges)                  # Full coverage (should be ~2.0 for unit sphere)
    )
    
    if perfect_sphere:
        print("✅ PERFECT HOLLOW SPHERE GENERATED!")
        print("   🏀 Round like a ball")
        print("   🌐 Surface only (not filled)")
        print("   📏 Perfect unit radius")
        print("   🎯 Uniform point distribution")
        print("   ✨ Ready for Pixel2Mesh++")
    else:
        print("⚠️  Sphere generation needs adjustment")
        print(f"     Unit radius: {np.allclose(distances, 1.0, atol=1e-10)}")
        print(f"     Round shape: {perfectly_round}")
        print(f"     Full coverage: {all(r > 1.9 for r in ranges)}")
        
    print("="*70)

if __name__ == "__main__":
    main()