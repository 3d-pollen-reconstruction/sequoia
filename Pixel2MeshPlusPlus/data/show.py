import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_sphere_data(file_path):
    """Load the sphere prior data"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_true_shape(coord):
    """Analyze the actual shape without any scaling tricks"""
    print("="*60)
    print("TRUE SHAPE ANALYSIS (NO SCALING TRICKS)")
    print("="*60)
    
    # Raw coordinate statistics
    print("Raw coordinate statistics:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        values = coord[:, i]
        print(f"  {axis}: min={values.min():.6f}, max={values.max():.6f}, range={values.max()-values.min():.6f}")
    
    # Distance analysis
    distances = np.linalg.norm(coord, axis=1)
    print(f"\nDistance from origin:")
    print(f"  Min: {distances.min():.8f}")
    print(f"  Max: {distances.max():.8f}")
    print(f"  Std deviation: {distances.std():.8f}")
    
    # Shape analysis
    ranges = coord.max(axis=0) - coord.min(axis=0)
    print(f"\nShape analysis:")
    print(f"  X range: {ranges[0]:.6f}")
    print(f"  Y range: {ranges[1]:.6f}")  
    print(f"  Z range: {ranges[2]:.6f}")
    
    # Calculate ratios to check for distortion
    x_to_y_ratio = ranges[0] / ranges[1]
    y_to_z_ratio = ranges[1] / ranges[2]
    x_to_z_ratio = ranges[0] / ranges[2]
    
    print(f"\nAxis ratios (should be ~1.0 for perfect sphere):")
    print(f"  X/Y ratio: {x_to_y_ratio:.6f}")
    print(f"  Y/Z ratio: {y_to_z_ratio:.6f}")
    print(f"  X/Z ratio: {x_to_z_ratio:.6f}")
    
    # Determine shape
    max_ratio_deviation = max(abs(x_to_y_ratio - 1.0), abs(y_to_z_ratio - 1.0), abs(x_to_z_ratio - 1.0))
    
    print(f"\nSHAPE VERDICT:")
    if max_ratio_deviation < 0.01:
        print(f"  ✅ PERFECT SPHERE (max deviation: {max_ratio_deviation:.6f})")
        shape_type = "PERFECT SPHERE"
    elif max_ratio_deviation < 0.05:
        print(f"  ✅ VERY ROUND (max deviation: {max_ratio_deviation:.6f})")
        shape_type = "VERY ROUND"
    elif max_ratio_deviation < 0.1:
        print(f"  ⚠️  SLIGHTLY ELONGATED (max deviation: {max_ratio_deviation:.6f})")
        shape_type = "SLIGHTLY ELONGATED"
    else:
        print(f"  ❌ SIGNIFICANTLY DISTORTED (max deviation: {max_ratio_deviation:.6f})")
        shape_type = "DISTORTED"
    
    return shape_type, ranges, max_ratio_deviation, distances

def plot_true_shape_no_tricks(coord, sample_coord, distances):
    """Plot with NO forced aspect ratios - shows true shape"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: 3D view with NATURAL scaling (no forced aspect ratio)
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(coord[:, 0], coord[:, 1], coord[:, 2], 
               c='blue', s=20, alpha=0.7, label='Main points')
    ax1.scatter(sample_coord[:, 0], sample_coord[:, 1], sample_coord[:, 2], 
               c='red', s=40, alpha=0.9, label='Sample points')
    ax1.set_title('3D View (Natural Scaling)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    # NO set_box_aspect() - let matplotlib scale naturally
    
    # Plot 2: X-Y projection 
    ax2 = fig.add_subplot(232)
    ax2.scatter(coord[:, 0], coord[:, 1], c='blue', s=15, alpha=0.7, label='Main')
    ax2.scatter(sample_coord[:, 0], sample_coord[:, 1], c='red', s=25, alpha=0.9, label='Sample')
    ax2.set_title('X-Y View (Top)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.legend()
    ax2.grid(True)
    # NO set_aspect('equal') - show true proportions
    
    # Plot 3: X-Z projection
    ax3 = fig.add_subplot(233)
    ax3.scatter(coord[:, 0], coord[:, 2], c='blue', s=15, alpha=0.7, label='Main')
    ax3.scatter(sample_coord[:, 0], sample_coord[:, 2], c='red', s=25, alpha=0.9, label='Sample')
    ax3.set_title('X-Z View (Front)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Y-Z projection
    ax4 = fig.add_subplot(234)
    ax4.scatter(coord[:, 1], coord[:, 2], c='blue', s=15, alpha=0.7, label='Main')
    ax4.scatter(sample_coord[:, 1], sample_coord[:, 2], c='red', s=25, alpha=0.9, label='Sample')
    ax4.set_title('Y-Z View (Side)')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.legend()
    ax4.grid(True)
    
    # Plot 5: Distance distribution - FIXED
    ax5 = fig.add_subplot(235)
    
    # Check if all distances are the same (perfect sphere)
    distance_range = distances.max() - distances.min()
    
    if distance_range < 1e-10:  # All distances identical
        # Show single bar for perfect sphere
        ax5.bar(['Perfect Sphere'], [len(distances)], color='green', alpha=0.7)
        ax5.set_title(f'Distance Distribution\n(All exactly {distances[0]:.6f})')
        ax5.set_ylabel('Count')
        ax5.text(0, len(distances)/2, f'All {len(distances)} points\nat distance {distances[0]:.6f}', 
                ha='center', va='center', fontweight='bold')
    else:
        # Normal histogram if there's variation
        n_bins = min(20, max(5, int(len(distances)/5)))  # Adaptive bins
        ax5.hist(distances, bins=n_bins, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect unit radius')
        ax5.set_title('Distance Distribution')
        ax5.set_xlabel('Distance from Origin')
        ax5.set_ylabel('Count')
        ax5.legend()
    
    ax5.grid(True)
    
    # Plot 6: Coordinate ranges comparison
    ax6 = fig.add_subplot(236)
    ranges = coord.max(axis=0) - coord.min(axis=0)
    bars = ax6.bar(['X', 'Y', 'Z'], ranges, color=['red', 'green', 'blue'], alpha=0.7)
    ax6.axhline(2.0, color='black', linestyle='--', label='Perfect sphere range (2.0)')
    ax6.set_title('Coordinate Ranges')
    ax6.set_ylabel('Range')
    ax6.legend()
    ax6.grid(True)
    
    # Add range values on bars
    for bar, range_val in zip(bars, ranges):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{range_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_reference_comparison(coord):
    """Compare with perfect mathematical sphere"""
    
    # Generate perfect sphere with same number of points
    n_points = len(coord)
    perfect_sphere = []
    
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1 - (i / (n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        perfect_sphere.append([x, y, z])
    
    perfect_sphere = np.array(perfect_sphere)
    
    # Compare shapes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Perfect sphere projections
    axes[0,0].scatter(perfect_sphere[:, 0], perfect_sphere[:, 1], c='green', s=10, alpha=0.7)
    axes[0,0].set_title('Perfect Sphere - X-Y')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    axes[0,0].grid(True)
    
    axes[0,1].scatter(perfect_sphere[:, 0], perfect_sphere[:, 2], c='green', s=10, alpha=0.7)
    axes[0,1].set_title('Perfect Sphere - X-Z')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Z')
    axes[0,1].grid(True)
    
    axes[0,2].scatter(perfect_sphere[:, 1], perfect_sphere[:, 2], c='green', s=10, alpha=0.7)
    axes[0,2].set_title('Perfect Sphere - Y-Z')
    axes[0,2].set_xlabel('Y')
    axes[0,2].set_ylabel('Z')
    axes[0,2].grid(True)
    
    # Your sphere projections
    axes[1,0].scatter(coord[:, 0], coord[:, 1], c='blue', s=10, alpha=0.7)
    axes[1,0].set_title('Your Sphere - X-Y')
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('Y')
    axes[1,0].grid(True)
    
    axes[1,1].scatter(coord[:, 0], coord[:, 2], c='blue', s=10, alpha=0.7)
    axes[1,1].set_title('Your Sphere - X-Z')
    axes[1,1].set_xlabel('X')
    axes[1,1].set_ylabel('Z')
    axes[1,1].grid(True)
    
    axes[1,2].scatter(coord[:, 1], coord[:, 2], c='blue', s=10, alpha=0.7)
    axes[1,2].set_title('Your Sphere - Y-Z')
    axes[1,2].set_xlabel('Y')
    axes[1,2].set_ylabel('Z')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical comparison
    perfect_ranges = perfect_sphere.max(axis=0) - perfect_sphere.min(axis=0)
    your_ranges = coord.max(axis=0) - coord.min(axis=0)
    
    print("\nCOMPARISON WITH PERFECT SPHERE:")
    print(f"Perfect sphere ranges: X={perfect_ranges[0]:.6f}, Y={perfect_ranges[1]:.6f}, Z={perfect_ranges[2]:.6f}")
    print(f"Your sphere ranges:    X={your_ranges[0]:.6f}, Y={your_ranges[1]:.6f}, Z={your_ranges[2]:.6f}")
    print(f"Differences:           X={abs(perfect_ranges[0]-your_ranges[0]):.6f}, Y={abs(perfect_ranges[1]-your_ranges[1]):.6f}, Z={abs(perfect_ranges[2]-your_ranges[2]):.6f}")

def sample_from_mesh_surface(mesh_vertices, mesh_faces, n_samples=200):
    """Sample points uniformly from mesh surface"""
    # Calculate face areas
    face_areas = calculate_face_areas(mesh_vertices, mesh_faces)
    
    # Sample faces proportional to their area
    face_probs = face_areas / np.sum(face_areas)
    selected_faces = np.random.choice(len(mesh_faces), n_samples, p=face_probs)
    
    # Sample points on selected faces
    surface_points = []
    for face_idx in selected_faces:
        # Get random barycentric coordinates
        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        r3 = 1 - r1 - r2
        
        # Get face vertices
        v1, v2, v3 = mesh_vertices[mesh_faces[face_idx]]
        
        # Calculate point on face
        point = r1 * v1 + r2 * v2 + r3 * v3
        surface_points.append(point)
    
    return np.array(surface_points)

def main():
    # Load the sphere data
    sphere_file = r"C:\Users\super\Documents\GitHub\sequoia\Pixel2MeshPlusPlus\data\mean_shape_prior.dat"
    
    print("Loading sphere data...")
    data = load_sphere_data(sphere_file)
    
    coord = data['coord']
    sample_coord = data['sample_coord']
    
    # Analyze true shape
    shape_type, ranges, deviation, distances = analyze_true_shape(coord)
    
    # Show plots with NO scaling tricks
    print("\nShowing TRUE shape (no forced scaling)...")
    plot_true_shape_no_tricks(coord, sample_coord, distances)
    
    # Compare with perfect reference
    print("Comparing with perfect mathematical sphere...")
    create_reference_comparison(coord)
    
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {shape_type}")
    if shape_type == "PERFECT SPHERE":
        print("🏀 Your sphere is mathematically perfect!")
        print("   If it looks distorted in plots, that's just matplotlib's default scaling.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
