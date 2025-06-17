import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_dat_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def visualize_mesh_data(data):
    """Visualize the mesh data structure"""
    
    print("=== Mesh Data Analysis ===")
    
    # Main coordinates
    coord = data['coord']
    print(f"Main coordinates: {coord.shape} vertices")
    print(f"Coordinate range: X[{coord[:,0].min():.3f}, {coord[:,0].max():.3f}], "
          f"Y[{coord[:,1].min():.3f}, {coord[:,1].max():.3f}], "
          f"Z[{coord[:,2].min():.3f}, {coord[:,2].max():.3f}]")
    
    # Sample coordinates  
    sample_coord = data['sample_coord']
    print(f"Sample coordinates: {sample_coord.shape} vertices")
    
    # Analyze stages
    for i, stage_key in enumerate(['stage1', 'stage2', 'stage3'], 1):
        stage_data = data[stage_key]
        print(f"Stage {i}: {len(stage_data)} elements")
        if len(stage_data) > 0:
            print(f"  First element type: {type(stage_data[0])}")
            if hasattr(stage_data[0], 'shape'):
                print(f"  First element shape: {stage_data[0].shape}")
    
    # Faces information
    faces = data['faces']
    faces_triangle = data['faces_triangle']
    print(f"Faces: {len(faces)} elements")
    print(f"Triangle faces: {len(faces_triangle)} elements")
    
    # Pool indices
    pool_idx = data['pool_idx']
    print(f"Pool indices: {len(pool_idx)} levels")
    
    return coord, sample_coord, faces, faces_triangle

def plot_3d_points(coord, sample_coord, title="Mesh Visualization"):
    """Plot 3D coordinates"""
    fig = plt.figure(figsize=(12, 5))
    
    # Plot main coordinates
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(coord[:, 0], coord[:, 1], coord[:, 2], c='blue', s=20, alpha=0.6)
    ax1.set_title(f'Main Coordinates ({coord.shape[0]} points)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot sample coordinates
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(sample_coord[:, 0], sample_coord[:, 1], sample_coord[:, 2], c='red', s=30, alpha=0.7)
    ax2.set_title(f'Sample Coordinates ({sample_coord.shape[0]} points)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def analyze_connectivity(data):
    """Analyze mesh connectivity and structure"""
    print("\n=== Connectivity Analysis ===")
    
    # Analyze faces
    faces = data['faces']
    if len(faces) > 0 and len(faces[0]) > 0:
        face_array = np.array(faces[0]) if isinstance(faces[0], list) else faces[0]
        print(f"First face set shape: {face_array.shape}")
        print(f"Face indices range: {face_array.min()} to {face_array.max()}")
    
    # Analyze Laplacian indices
    lape_idx = data['lape_idx']
    print(f"Laplacian indices: {len(lape_idx)} levels")
    
    # Analyze Chebyshev data
    sample_cheb = data['sample_cheb']
    print(f"Sample Chebyshev: {len(sample_cheb)} elements")
    
    sample_cheb_dense = data['sample_cheb_dense']
    print(f"Sample Chebyshev dense: {len(sample_cheb_dense)} elements")

def main():
    file_path = r"C:\Users\super\Documents\GitHub\sequoia\Pixel2MeshPlusPlus\data\custom_prior.dat"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        data = read_dat_file(file_path)
        
        # Analyze the data structure
        coord, sample_coord, faces, faces_triangle = visualize_mesh_data(data)
        
        # Analyze connectivity
        analyze_connectivity(data)
        
        # Plot 3D visualization
        plot_3d_points(coord, sample_coord)
        
        # Save a summary
        print("\n=== Data Summary ===")
        print("This appears to be a Pixel2Mesh++ template file containing:")
        print("- Hierarchical mesh stages (stage1, stage2, stage3)")
        print("- Main mesh coordinates and sample points")
        print("- Face connectivity information")
        print("- Pooling indices for mesh coarsening")
        print("- Laplacian and Chebyshev polynomial data for graph convolution")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
