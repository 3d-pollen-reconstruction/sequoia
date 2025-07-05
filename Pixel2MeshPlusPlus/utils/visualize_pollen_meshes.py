# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.

import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def load_obj_file(obj_path):
    """
    Load vertices and faces from an OBJ file.
    
    Returns:
        vertices (np.array): Nx3 array of vertex coordinates
        faces (np.array): Mx3 array of face indices (0-based)
    """
    vertices = []
    faces = []
    
    with open(obj_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '):
                # Vertex line: v x y z
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith('f '):
                # Face line: f v1 v2 v3 (1-based indices)
                parts = line.split()
                # Convert to 0-based indices
                face = [int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1]
                faces.append(face)
    
    return np.array(vertices), np.array(faces)

def compare_meshes(obj_files, titles=None, save_path=None):
    """
    Compare multiple mesh files side by side.
    
    Args:
        obj_files: List of OBJ file paths
        titles: List of titles for each subplot
        save_path: Path to save the comparison plot
    """
    n_meshes = len(obj_files)
    
    if titles is None:
        titles = [f"Mesh {i+1}" for i in range(n_meshes)]
    
    # Create subplots
    fig = plt.figure(figsize=(6 * n_meshes, 5))
    
    mesh_data = []
    
    for i, obj_file in enumerate(obj_files):
        try:
            vertices, faces = load_obj_file(obj_file)
            mesh_data.append((vertices, faces))
            
            ax = fig.add_subplot(1, n_meshes, i + 1, projection='3d')
            
            # Get face coordinates
            face_vertices = vertices[faces]
            
            # Create mesh
            mesh = Poly3DCollection(face_vertices, alpha=0.7)
            mesh.set_facecolor('lightblue')
            mesh.set_edgecolor('black')
            ax.add_collection3d(mesh)
            
            # Show vertices
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                      c='red', s=8, alpha=0.8)
            
            # Set equal aspect ratio
            X, Y, Z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
            mid_x, mid_y, mid_z = (X.max() + X.min()) * 0.5, (Y.max() + Y.min()) * 0.5, (Z.max() + Z.min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{titles[i]}\\nV: {len(vertices)}, F: {len(faces)}')
            
            print(f"{titles[i]}:")
            print(f"  Vertices: {len(vertices)}")
            print(f"  Faces: {len(faces)}")
            print(f"  File: {os.path.basename(obj_file)}")
            print()
            
        except Exception as e:
            print(f"Error loading {obj_file}: {str(e)}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    return fig, mesh_data

def analyze_mesh_resolution(vertices, faces):
    """
    Analyze mesh resolution and provide improvement suggestions.
    """
    n_vertices = len(vertices)
    n_faces = len(faces)
    
    # Calculate edge lengths
    edge_lengths = []
    for face in faces:
        v1, v2, v3 = vertices[face]
        edge_lengths.extend([
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v3 - v2),
            np.linalg.norm(v1 - v3)
        ])
    
    edge_lengths = np.array(edge_lengths)
    
    # Calculate surface area (approximate)
    total_area = 0
    for face in faces:
        v1, v2, v3 = vertices[face]
        # Cross product for triangle area
        cross = np.cross(v2 - v1, v3 - v1)
        area = 0.5 * np.linalg.norm(cross)
        total_area += area
    
    print("=== MESH RESOLUTION ANALYSIS ===")
    print(f"Vertices: {n_vertices}")
    print(f"Faces: {n_faces}")
    print(f"Vertex density: {n_vertices / total_area:.2f} vertices per unit area")
    print(f"Average edge length: {np.mean(edge_lengths):.6f}")
    print(f"Min edge length: {np.min(edge_lengths):.6f}")
    print(f"Max edge length: {np.max(edge_lengths):.6f}")
    print(f"Edge length std: {np.std(edge_lengths):.6f}")
    print(f"Total surface area: {total_area:.6f}")
    print()
    
    # Suggestions for improvement
    print("=== IMPROVEMENT SUGGESTIONS ===")
    if n_vertices < 1000:
        print("• LOW RESOLUTION: Consider subdivision or remeshing to increase vertex count")
        print("• Target: 2000-5000 vertices for better detail")
    elif n_vertices < 2000:
        print("• MEDIUM RESOLUTION: Could benefit from selective refinement")
        print("• Target: 3000-8000 vertices for high detail")
    else:
        print("• HIGH RESOLUTION: Good vertex count for detailed modeling")
    
    if np.std(edge_lengths) / np.mean(edge_lengths) > 0.5:
        print("• IRREGULAR MESH: High variation in edge lengths")
        print("• Consider remeshing for more uniform triangulation")
    
    print("• For Pixel2Mesh++, ensure:")
    print("  - Uniform triangle distribution")
    print("  - Smooth vertex normals")
    print("  - Consistent topology")
    print("  - Higher resolution in areas of detail")
    print()

def main():
    """
    Main function to visualize and analyze the pollen grain meshes.
    """
    # File paths
    obj_files = [
        r"c:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\utils\pollen_17814_Marigold_Calendula_officinalis_pollen_grain_00_predict_1.obj",
        r"c:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\utils\pollen_17814_Marigold_Calendula_officinalis_pollen_grain_00_predict_2.obj",
        r"c:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\utils\pollen_17814_Marigold_Calendula_officinalis_pollen_grain_00_predict.obj"
    ]
    
    titles = ["Prediction 1 (face1)", "Prediction 2 (face2)", "Prediction (face3)"]
    
    # Check if files exist
    existing_files = []
    existing_titles = []
    for i, obj_file in enumerate(obj_files):
        if os.path.exists(obj_file):
            existing_files.append(obj_file)
            existing_titles.append(titles[i])
        else:
            print(f"Warning: File not found: {obj_file}")
    
    if not existing_files:
        print("No OBJ files found!")
        return
    
    print(f"Found {len(existing_files)} OBJ files to visualize")
    print()
    
    # Create comparison plot
    save_path = r"c:\Users\super\Documents\Github\sequoia\Pixel2MeshPlusPlus\utils\pollen_mesh_comparison.png"
    fig, mesh_data = compare_meshes(existing_files, existing_titles, save_path)
    
    # Analyze each mesh
    for i, (vertices, faces) in enumerate(mesh_data):
        print(f"ANALYSIS FOR {existing_titles[i]}:")
        analyze_mesh_resolution(vertices, faces)
        print("-" * 50)
    
    # Show the plot
    plt.show()
    
    return fig, mesh_data

if __name__ == "__main__":
    main()
