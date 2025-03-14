import os
import trimesh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import manifold3d
import scipy
from pyntcloud import PyntCloud

class MeshCleaner:
    def __init__(
        self,
        data_dir="../data/raw/",
        output_dir="../data/processed/models",
        fill_holes=True,
        remove_duplicates=True,
        scale=True,
        center=True,
        remove_self_intersections=False,
        fix_normals=False,
        optimize_quality=False,
        smooth_surface=False,
        simplify_large_meshes=False,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.fill_holes = fill_holes  # Enables overall hole filling
        self.remove_duplicates = remove_duplicates
        self.scale = scale
        self.center = center
        self.remove_self_intersections = remove_self_intersections
        self.fix_normals = fix_normals
        self.optimize_quality = optimize_quality
        self.smooth_surface = smooth_surface
        self.simplify_large_meshes = simplify_large_meshes

    def analyze_mesh(self, mesh_path):
        """Loads and analyzes an STL mesh with extended metrics."""
        try:
            mesh = trimesh.load_mesh(mesh_path)

            # Basic geometry
            stats = {
                "file_name": os.path.basename(mesh_path),
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces),
                "bounding_box": mesh.bounds,
                "extents": mesh.extents.tolist(),
                "surface_area": mesh.area,
                "is_watertight": mesh.is_watertight,
                "euler_number": mesh.euler_number,  # Topology indicator (V-E+F)
            }

            # Volume only for watertight models
            try:
                if mesh.is_watertight:
                    stats["volume"] = mesh.volume
                    # Convexity (ratio of volume to convex hull)
                    try:
                        hull = mesh.convex_hull
                        if hull.volume > 0:
                            stats["convexity_ratio"] = mesh.volume / hull.volume
                    except Exception:
                        pass
            except Exception:
                pass

            # Edge statistics
            try:
                edges = mesh.edges_unique
                if len(edges) > 0:
                    edge_lengths = [
                        np.linalg.norm(mesh.vertices[e[0]] - mesh.vertices[e[1]])
                        for e in edges
                    ]
                    stats["mean_edge_length"] = np.mean(edge_lengths)
                    stats["min_edge_length"] = np.min(edge_lengths)
                    stats["max_edge_length"] = np.max(edge_lengths)
                    stats["edge_length_std"] = np.std(edge_lengths)
            except Exception:
                pass

            # Face orientation
            try:
                if len(mesh.faces) > 0 and hasattr(mesh, "face_normals"):
                    consistent = mesh.face_normals.dot(mesh.face_normals[0]) > 0
                    stats["normal_consistency"] = np.mean(consistent)
            except Exception:
                pass

            # Point density and aspect ratio
            try:
                if np.prod(mesh.extents) > 0:
                    stats["point_density"] = len(mesh.vertices) / np.prod(mesh.extents)
                if min(mesh.extents) > 0:
                    stats["aspect_ratio"] = max(mesh.extents) / min(mesh.extents)
            except Exception:
                pass

            return stats
        except Exception as e:
            print(f"Error analyzing {mesh_path}: {str(e)}")
            return {"file_name": os.path.basename(mesh_path), "error": str(e)}

    def plot_mesh(self, mesh_path):
        """Visualizes an STL model."""
        try:
            mesh = trimesh.load_mesh(mesh_path)
            mesh.show()
        except Exception as e:
            print(f"Error visualizing: {str(e)}")

    def fill_holes_advanced(self, mesh):
        """Fills holes using different methods depending on complexity."""
        original_watertight = mesh.is_watertight
        hole_fill_method = "None"

        # Stage 1: Try trimesh for small holes
        if not mesh.is_watertight and hasattr(mesh, "fill_holes"):
            try:
                mesh.fill_holes()
                if mesh.is_watertight:
                    hole_fill_method = "trimesh"
            except Exception as e:
                print(f"  Trimesh fill_holes failed: {str(e)}")

        # Stage 2: Try VTK for medium-sized holes
        if not mesh.is_watertight:
            try:
                import vtk
                from vtk.util import numpy_support

                # Convert trimesh to VTK PolyData
                points = vtk.vtkPoints()
                for v in mesh.vertices:
                    points.InsertNextPoint(v)

                triangles = vtk.vtkCellArray()
                for f in mesh.faces:
                    triangle = vtk.vtkTriangle()
                    for i in range(3):
                        triangle.GetPointIds().SetId(i, f[i])
                    triangles.InsertNextCell(triangle)

                poly_data = vtk.vtkPolyData()
                poly_data.SetPoints(points)
                poly_data.SetPolys(triangles)

                # Apply hole fill filter
                fill_filter = vtk.vtkFillHolesFilter()
                fill_filter.SetInputData(poly_data)
                fill_filter.SetHoleSize(mesh.scale * 0.01)  # Adjust based on model size
                fill_filter.Update()

                # Extract filled mesh
                filled_poly = fill_filter.GetOutput()

                # Convert back to Trimesh
                vertices = []
                for i in range(filled_poly.GetNumberOfPoints()):
                    vertices.append(filled_poly.GetPoint(i))
                vertices = np.array(vertices)

                faces = []
                for i in range(filled_poly.GetNumberOfCells()):
                    cell = filled_poly.GetCell(i)
                    if cell.GetNumberOfPoints() == 3:  # Only triangles
                        face = [cell.GetPointId(j) for j in range(3)]
                        faces.append(face)
                faces = np.array(faces)

                # Create new Trimesh with filled holes
                filled_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

                if filled_mesh.is_watertight:
                    mesh = filled_mesh
                    hole_fill_method = "vtk"
            except Exception as e:
                print(f"  VTK fill_holes failed: {str(e)}")

        # Stage 3: Try manifold3d for complex holes
        if not mesh.is_watertight:
            # at the time not working need to be fixed
            return mesh, hole_fill_method, original_watertight
            try:
                # Convert to float32 and uint32 for manifold3d
                vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
                faces = np.ascontiguousarray(mesh.faces, dtype=np.uint32)
                
                # Create a Manifold object
                manifold_mesh = manifold3d.Manifold()
                
                # Use available API methods to add mesh data
                try:
                    manifold_mesh.append_mesh(vertices, faces)
                except AttributeError:
                    try:
                        manifold_mesh.add_triangle_mesh(vertices, faces)
                    except AttributeError:
                        try:
                            manifold_mesh.from_mesh(vertices, faces)
                        except AttributeError:
                            print("Manifold3d fix failed: No available method to import mesh")
                            raise Exception("Manifold3d fix failed: No available method to import mesh")
                
                # Attempt to repair the mesh
                try:
                    manifold_mesh.set_tolerance(1e-5)
                    manifold_mesh = manifold_mesh.fix_manifold()
                except (AttributeError, TypeError):
                    pass  # Ignore if the method does not exist
                
                # Extract the repaired mesh
                try:
                    result = manifold_mesh.extract_mesh()
                    result_vertices, result_faces = result
                except (AttributeError, ValueError):
                    try:
                        result = manifold_mesh.get_mesh()
                        if hasattr(result, 'v') and hasattr(result, 'f'):
                            result_vertices = np.array(result.v)
                            result_faces = np.array(result.f)
                        else:
                            result_vertices = vertices
                            result_faces = faces
                    except AttributeError:
                        if hasattr(manifold_mesh, 'vertices') and hasattr(manifold_mesh, 'faces'):
                            result_vertices = manifold_mesh.vertices
                            result_faces = manifold_mesh.faces
                        else:
                            result_vertices = vertices
                            result_faces = faces
                
                # Create a new Trimesh with the repaired data
                fixed_mesh = trimesh.Trimesh(
                    vertices=np.array(result_vertices, dtype=np.float64),
                    faces=np.array(result_faces, dtype=np.int64)
                )
                
                if fixed_mesh.is_watertight:
                    mesh = fixed_mesh
                    hole_fill_method = "manifold3d"
                
            except Exception as e:
                print(f"  Manifold3d fix failed: {str(e)}")
                
            # If manifold3d failed, try PyMesh as a last resort
            if not mesh.is_watertight and hole_fill_method == "None":
                try:
                    mesh = self.advanced_hole_filling(mesh)
                    if mesh.is_watertight:
                        hole_fill_method = "pymesh"
                except Exception as e:
                    print(f"  PyMesh fix failed: {str(e)}")

        return mesh, hole_fill_method, original_watertight

    def clean_mesh(self, mesh_path, output_path):
        """Loads, cleans, and saves an STL mesh with enhanced hole filling."""
        try:
            mesh = trimesh.load_mesh(mesh_path)
            file_name = os.path.basename(mesh_path)
            original_stats = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_watertight": mesh.is_watertight,
            }

            print(f"Processing {file_name}: Watertight={mesh.is_watertight}")

            # Remove unused vertices and faces
            mesh.remove_unreferenced_vertices()

            if self.fill_holes:
                mesh, method_used, was_watertight = self.fill_holes_advanced(mesh)
                if mesh.is_watertight and not was_watertight:
                    print(f"  ✓ Holes successfully filled with {method_used}")
                elif not mesh.is_watertight:
                    print("  ✗ Mesh is still not watertight after repair")

            if self.remove_duplicates:
                # Updated method to remove duplicate faces
                mesh.update_faces(mesh.unique_faces())

            if self.scale:
                max_dim = max(mesh.extents)
                if max_dim > 0:
                    mesh.apply_scale(1.0 / max_dim)

            # Remove self-intersections
            if self.remove_self_intersections:
                mesh = self.remove_self_intersections(mesh)

            # Fix normals
            if self.fix_normals:
                mesh = self.fix_normals(mesh)

            # Additional hole filling (improved method)
            if self.fill_holes:
                mesh, method_used, was_watertight = self.fill_holes_advanced(mesh)

            # Smooth surface if desired
            if self.smooth_surface:
                mesh = self.smooth_mesh(mesh, method="taubin", iterations=3)

            # Simplify mesh if necessary
            if self.simplify_large_meshes and len(mesh.faces) > 100000:
                mesh = self.simplify_mesh(mesh, target_percent=0.5)

            # Optimize facet quality
            if self.optimize_quality:
                mesh = self.optimize_mesh_quality(mesh)

            if self.center:
                centroid = mesh.centroid
                mesh.apply_translation(-centroid)

            mesh.export(output_path)

            cleaned_stats = {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_watertight": mesh.is_watertight,
                "hole_fill_method": method_used if "method_used" in locals() else "None",
            }

            status = "✓" if mesh.is_watertight else "✗"
            print(
                f"{status} Saved: {os.path.basename(output_path)} "
                + f"(Vertices: {original_stats['vertices']} → {cleaned_stats['vertices']}, "
                + f"Faces: {original_stats['faces']} → {cleaned_stats['faces']}, "
                + f"Watertight: {original_stats['is_watertight']} → {cleaned_stats['is_watertight']})"
            )

            return cleaned_stats
        except Exception as e:
            print(f"Error cleaning mesh {mesh_path}: {str(e)}")
            return {"file_name": os.path.basename(mesh_path), "error": str(e)}

    def verify_models(self, models_dir):
        """Checks all models for watertightness and creates a report."""
        print("\nVerifying watertightness of cleaned models...")
        watertight_models = 0
        non_watertight_models = []

        all_models = [f for f in os.listdir(models_dir) if f.endswith(".stl")]
        for model_file in tqdm(all_models, desc="Checking models"):
            model_path = os.path.join(models_dir, model_file)
            try:
                mesh = trimesh.load_mesh(model_path)
                if mesh.is_watertight:
                    watertight_models += 1
                else:
                    non_watertight_models.append(model_file)
            except Exception as e:
                print(f"Error loading {model_file}: {str(e)}")
                non_watertight_models.append(f"{model_file} (Loading error)")

        # Create report
        print(
            f"\nWatertight models: {watertight_models}/{len(all_models)} ({watertight_models / len(all_models) * 100:.1f}%)"
        )
        if non_watertight_models:
            print(
                f"The following {len(non_watertight_models)} models are not watertight:"
            )
            for i, model in enumerate(non_watertight_models[:10], 1):  # Show only the first 10
                print(f"  {i}. {model}")
            if len(non_watertight_models) > 10:
                print(f"  ... and {len(non_watertight_models) - 10} more")

            # Save problematic models to a file
            with open("non_watertight_models.txt", "w") as f:
                for model in non_watertight_models:
                    f.write(f"{model}\n")

            print(
                "List of all problematic models saved to 'non_watertight_models.txt'."
            )

        return watertight_models, non_watertight_models

    def analyze_pointcloud(self, mesh_path):
        """Analyzes the mesh as a point cloud for basic statistics."""
        try:
            # Check if PyntCloud is available
            if "PyntCloud" not in globals():
                return {"file_name": os.path.basename(mesh_path)}

            mesh = trimesh.load_mesh(mesh_path)
            # Use a sample for large meshes
            points = mesh.sample(2000) if len(mesh.vertices) > 2000 else mesh.vertices

            if len(points) == 0:
                return {"file_name": os.path.basename(mesh_path)}

            # Simple geometric statistics without problematic eigenvalue computation
            points_df = pd.DataFrame(points, columns=["x", "y", "z"])

            # Calculate basic statistical measures
            stats = {
                "file_name": os.path.basename(mesh_path),
                "point_mean_x": np.mean(points_df["x"]),
                "point_mean_y": np.mean(points_df["y"]),
                "point_mean_z": np.mean(points_df["z"]),
                "point_std_x": np.std(points_df["x"]),
                "point_std_y": np.std(points_df["y"]),
                "point_std_z": np.std(points_df["z"]),
            }

            # Add measures for the distribution
            for col in ["x", "y", "z"]:
                stats[f"point_{col}_skew"] = 0  # Placeholder for skewness
                stats[f"point_{col}_kurtosis"] = 0  # Placeholder for kurtosis

                # Try to calculate skewness and kurtosis if scipy is available
                try:
                    from scipy import stats as scipystats

                    stats[f"point_{col}_skew"] = scipystats.skew(points_df[col])
                    stats[f"point_{col}_kurtosis"] = scipystats.kurtosis(points_df[col])
                except Exception:
                    pass

            return stats

        except Exception as e:
            print(f"Error during point cloud analysis: {str(e)}")
            return {"file_name": os.path.basename(mesh_path)}

    def visualize_results(self, df):
        """Creates visualizations of the analysis results."""
        if len(df) == 0:
            return

        # Remove non-numeric columns for correlation analysis
        try:
            # 1. Correlation matrix for numeric values
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) >= 2:
                plt.figure(figsize=(12, 10))
                corr_matrix = df[num_cols].corr()
                plt.matshow(corr_matrix, fignum=1, cmap="coolwarm")
                plt.colorbar()
                plt.xticks(range(len(num_cols)), num_cols, rotation=90)
                plt.yticks(range(len(num_cols)), num_cols)
                plt.title("Correlation between Mesh Properties")
                plt.savefig("mesh_correlation.png")
                plt.close()
        except Exception as e:
            print(f"Error during correlation analysis: {str(e)}")

        try:
            # 2. Scatterplot for key properties
            key_metrics = ["num_vertices", "num_faces"]
            if all(metric in df.columns for metric in key_metrics):
                plt.figure(figsize=(10, 8))
                plt.scatter(
                    df["num_vertices"],
                    df["num_faces"],
                    c=df["is_watertight"] if "is_watertight" in df.columns else None,
                    alpha=0.7,
                    s=50,
                )

                # Labels for only a few data points
                if len(df) <= 30:
                    for i, txt in enumerate(df["file_name"]):
                        plt.annotate(
                            txt,
                            (df["num_vertices"].iloc[i], df["num_faces"].iloc[i]),
                            fontsize=8,
                        )

                plt.xlabel("Number of Vertices")
                plt.ylabel("Number of Faces")
                plt.title("Mesh Complexity")
                plt.grid(True, alpha=0.3)
                plt.savefig("mesh_complexity.png")
                plt.close()

            # 3. Histogram for watertightness
            if "is_watertight" in df.columns:
                plt.figure(figsize=(8, 6))
                watertight_count = df["is_watertight"].sum()
                counts = [watertight_count, len(df) - watertight_count]
                plt.bar(
                    ["Watertight", "Not watertight"], counts, color=["green", "red"]
                )
                plt.title("Watertightness of Meshes")
                plt.savefig("mesh_watertight.png")
                plt.close()

        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")

    def smooth_mesh(self, mesh, method="laplacian", iterations=5, preserve_volume=True):
        """Smooths a mesh to reduce noise and artifacts."""
        try:
            if method == "laplacian":
                # Simple Laplacian Smoothing
                for _ in range(iterations):
                    # Calculate new vertex positions as average of neighbors
                    adjacency = mesh.vertex_adjacency_graph
                    new_vertices = mesh.vertices.copy()

                    for i in range(len(mesh.vertices)):
                        if i in adjacency:
                            neighbors = list(adjacency[i])
                            if neighbors:
                                neighbor_verts = mesh.vertices[neighbors]
                                new_vertices[i] = np.mean(neighbor_verts, axis=0)

                    # Apply new positions
                    original_volume = None
                    if preserve_volume:
                        original_volume = mesh.volume if mesh.is_watertight else None

                    mesh.vertices = new_vertices

                    # Volume preservation
                    if (
                        preserve_volume
                        and original_volume
                        and mesh.is_watertight
                        and mesh.volume > 0
                    ):
                        scale_factor = (original_volume / mesh.volume) ** (1 / 3)
                        mesh.apply_scale(scale_factor)

            elif method == "taubin":
                # Taubin Smoothing (reduces shrinkage)
                lambda_factor = 0.5
                mu_factor = -0.53  # Typically slightly larger than -lambda

                for _ in range(iterations):
                    # Lambda step (forward)
                    adjacency = mesh.vertex_adjacency_graph
                    new_vertices = mesh.vertices.copy()

                    for i in range(len(mesh.vertices)):
                        if i in adjacency:
                            neighbors = list(adjacency[i])
                            if neighbors:
                                neighbor_verts = mesh.vertices[neighbors]
                                delta = np.mean(neighbor_verts, axis=0) - mesh.vertices[i]
                                new_vertices[i] = mesh.vertices[i] + lambda_factor * delta

                    mesh.vertices = new_vertices

                    # Mu step (backward)
                    new_vertices = mesh.vertices.copy()
                    for i in range(len(mesh.vertices)):
                        if i in adjacency:
                            neighbors = list(adjacency[i])
                            if neighbors:
                                neighbor_verts = mesh.vertices[neighbors]
                                delta = np.mean(neighbor_verts, axis=0) - mesh.vertices[i]
                                new_vertices[i] = mesh.vertices[i] + mu_factor * delta

                    mesh.vertices = new_vertices

            return mesh
        except Exception as e:
            print(f"Smoothing failed: {str(e)}")
            return mesh

    def fix_normals(self, mesh):
        """Ensures consistent orientation of face normals."""
        try:
            # Attempt to fix normals so they all point outwards
            mesh.fix_normals()
            return mesh
        except Exception as e:
            print(f"Normal correction failed: {str(e)}")
            return mesh

    def simplify_mesh(self, mesh, target_faces=None, target_percent=None, preserve_topology=True):
        """Simplifies a mesh by reducing faces."""
        try:
            if target_faces is None and target_percent is not None:
                target_faces = int(len(mesh.faces) * target_percent)

            if target_faces is None:
                # Automatic simplification based on complexity
                if len(mesh.faces) > 500000:
                    target_faces = 200000
                elif len(mesh.faces) > 100000:
                    target_faces = 50000
                else:
                    # No simplification for small meshes
                    return mesh

            # Prevent excessive simplification
            target_faces = max(target_faces, 1000)

            # Only simplify if the mesh is significantly more complex
            if len(mesh.faces) > target_faces * 1.1:
                simplified = mesh.simplify_quadric_decimation(
                    target_faces, preserve_topology=preserve_topology
                )
                if simplified is not None:
                    return simplified
        except Exception as e:
            print(f"Mesh simplification failed: {str(e)}")

        return mesh

    def remove_self_intersections(self, mesh):
        """Attempts to remove self-intersections in the mesh."""
        try:
            # Check for self-intersections
            intersections = trimesh.repair.broken_faces(mesh)
            if len(intersections) > 0:
                # For many self-intersections, voxel_repair can help
                if len(intersections) > 100:
                    # Use voxelization for repair
                    voxelized = mesh.voxelized(pitch=mesh.scale / 100)
                    repaired = voxelized.as_trimesh()
                    if repaired is not None and len(repaired.vertices) > 0:
                        return repaired
                else:
                    # Remove problematic faces
                    mask = np.ones(len(mesh.faces), dtype=bool)
                    mask[intersections] = False
                    mesh.update_faces(mask)
                    mesh.remove_unreferenced_vertices()
        except Exception as e:
            print(f"Self-intersection removal failed: {str(e)}")

        return mesh

    def subdivide_mesh(self, mesh, iterations=1, algorithm="loop"):
        """Increases mesh resolution for smoother surfaces."""
        try:
            if algorithm == "loop":
                for _ in range(iterations):
                    mesh = mesh.subdivide_loop()
            elif algorithm == "midpoint":
                for _ in range(iterations):
                    mesh = mesh.subdivide()
            return mesh
        except Exception as e:
            print(f"Subdivision failed: {str(e)}")
            return mesh

    def optimize_mesh_quality(self, mesh):
        """Improves the quality of triangles/facets through edge flips."""
        try:
            # Identify bad triangles (with very acute angles)
            faces = mesh.faces
            vertices = mesh.vertices

            # Calculate quality metrics for each facet
            face_normals = mesh.face_normals
            face_quality = []

            for i, face in enumerate(faces):
                # Calculate edges of the triangle
                v0, v1, v2 = vertices[face]
                a = np.linalg.norm(v1 - v0)
                b = np.linalg.norm(v2 - v1)
                c = np.linalg.norm(v0 - v2)

                # Quality metric: ratio between perimeter and area
                s = (a + b + c) / 2  # Half-perimeter
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Area using Heron's formula
                quality = (a * b * c) / (8 * area) if area > 0 else float("inf")
                face_quality.append(quality)

            # Identify bad triangles for potential edge flips
            # (Edge flip logic would need to be implemented here)

            return mesh
        except Exception as e:
            print(f"Mesh quality optimization failed: {str(e)}")
            return mesh

    def advanced_hole_filling(self, mesh):
        """Uses Poisson reconstruction for complex cases."""
        try:
            if not mesh.is_watertight:
                # If PyMesh is available or via VTK
                try:
                    import pymesh

                    # Generate dense point cloud
                    points = mesh.sample(int(mesh.area * 10))
                    normals = mesh.face_normals[mesh.nearest.on_surface(points)[2]]

                    # Perform Poisson reconstruction
                    reconstructed = pymesh.poisson_reconstruction(points, normals, depth=8)

                    # Check if reconstruction was successful
                    if reconstructed.is_manifold and reconstructed.is_watertight:
                        return reconstructed
                except ImportError:
                    print("PyMesh not available - Poisson reconstruction skipped")
        except Exception as e:
            print(f"Advanced hole filling failed: {str(e)}")

        return mesh

    def align_mesh(self, mesh):
        """Aligns the mesh along its principal axes."""
        try:
            # Calculate principal axes via PCA
            matrix = mesh.principal_inertia_transform
            mesh.apply_transform(matrix)

            # Optional: Align so that the longest extent is along the Y-axis
            extents = mesh.bounding_box.extents
            if not np.allclose(extents, extents[0]):  # If not cubic
                order = np.argsort(extents)
                # Create rotation matrix for axis adjustment
                if order[2] != 1:  # If Y is not the longest
                    R = np.eye(4)
                    # Rotation logic would be implemented here
                    mesh.apply_transform(R)
        except Exception as e:
            print(f"Mesh alignment failed: {str(e)}")

        return mesh

    def process_all(self):
        """Processes all STL files in the folder with extended analysis."""
        mesh_results = []
        pointcloud_results = []

        stl_files = [f for f in os.listdir(self.data_dir) if f.endswith(".stl")]
        for file in tqdm(stl_files, desc="Processing Meshes"):
            file_path = os.path.join(self.data_dir, file)
            output_path = os.path.join(self.output_dir, file)

            # Standard mesh analysis
            mesh_info = self.analyze_mesh(file_path)
            mesh_results.append(mesh_info)

            # Point cloud based analysis
            cloud_info = self.analyze_pointcloud(file_path)
            pointcloud_results.append(cloud_info)

            # Clean and save mesh
            self.clean_mesh(file_path, output_path)

        # Check all models after processing
        watertight_count, non_watertight = self.verify_models(self.output_dir)

        # Merge and save results
        mesh_df = pd.DataFrame(mesh_results)
        cloud_df = pd.DataFrame(pointcloud_results)

        # Combine both analyses
        results_df = pd.merge(mesh_df, cloud_df, on="file_name", how="outer")

        # Remove problematic columns for CSV storage
        for col in results_df.columns:
            if isinstance(results_df[col].iloc[0] if not results_df.empty else None, (list, np.ndarray)):
                results_df[col] = results_df[col].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)

        results_df.to_csv("mesh_analysis.csv", index=False)

        # Visualize the data
        self.visualize_results(results_df)

        # Show statistics for watertight models
        if "is_watertight" in results_df.columns:
            watertight_count = results_df["is_watertight"].sum()
            print(
                f"\nWatertight models: {watertight_count}/{len(results_df)} ({watertight_count / len(results_df) * 100:.1f}%)"
            )

        try:
            import ace_tools as tools

            tools.display_dataframe_to_user(
                name="Mesh Analysis Results", dataframe=results_df
            )
        except Exception:
            print("\nAnalysis Summary:")
            print(results_df.describe())
            print("\nFile for detailed analysis: mesh_analysis.csv")

        return results_df


if __name__ == "__main__":
    # Print required libraries for optimal functionality
    print("Required libraries for optimal functionality:")
    print("pip install scipy trimesh pyntcloud pandas numpy matplotlib tqdm vtk manifold3d")

    cleaner = MeshCleaner()
    cleaner.process_all()
