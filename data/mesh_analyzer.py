
import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pyglet

class MeshAnalyzer:
    def __init__(self, data_dir="../data/raw/"):
        self.data_dir = data_dir
        os.makedirs("mesh_analysis", exist_ok=True)
    
    def analyze_watertight_status(self):
        """Analyzes the watertight status of all models."""
        stl_files = [f for f in os.listdir(self.data_dir) if f.endswith(".stl")]
        results = []
        
        for file in tqdm(stl_files, desc="Analyzing models"):
            file_path = os.path.join(self.data_dir, file)
            try:
                mesh = trimesh.load_mesh(file_path)
                
                # Collect basic properties
                stats = {
                    "file_name": file,
                    "num_vertices": len(mesh.vertices),
                    "num_faces": len(mesh.faces),
                    "is_watertight": mesh.is_watertight,
                    "euler_number": mesh.euler_number,
                    "surface_area": mesh.area,
                    "bounds": mesh.bounds,
                }
                
                # Volume only for watertight models
                if mesh.is_watertight:
                    stats["volume"] = mesh.volume
                else:
                    stats["volume"] = None
                
                # Additional topological properties
                try:
                    # Calculate number of open edges (indication of holes)
                    # Using an alternative method since edges_face_count is not available
                    boundaries = self.count_boundaries(mesh)
                    stats["boundary_count"] = boundaries
                except:
                    stats["boundary_count"] = None
                
                results.append(stats)
                
            except Exception as e:
                print(f"Error with {file}: {str(e)}")
                results.append({"file_name": file, "error": str(e)})
        
        # Create DataFrame
        df = pd.DataFrame(results)
        df.to_csv("mesh_analysis/watertight_status.csv", index=False)
        
        return df
    
    def count_boundaries(self, mesh):
        """Alternative method to count boundaries/holes without edges_face_count."""
        # Try to estimate using Euler characteristic and other properties
        # V - E + F = 2 - 2g - b (g=genus/holes through the body, b=boundary components)
        # We can't calculate this exactly without edges_face_count, but we can approximate
        
        if mesh.is_watertight:
            return 0
        
        # An approximation: We can assume non-watertight models have boundaries
        return 1
    
    def visualize_watertightness(self, df):
        """Visualizes the watertightness of the models."""
        # 1. Proportion of watertight vs non-watertight models
        plt.figure(figsize=(10, 6))
        watertight_count = df["is_watertight"].sum()
        non_watertight_count = len(df) - watertight_count
        
        labels = ["Watertight", "Non-watertight"]
        sizes = [watertight_count, non_watertight_count]
        colors = ["#66b3ff", "#ff9999"]
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Watertightness of 3D Models")
        plt.savefig("mesh_analysis/watertight_pie.png")
        plt.close()
        
        # 2. Relationship between complexity and watertightness
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df, 
            x="num_vertices", 
            y="num_faces", 
            hue="is_watertight",
            palette={True: "green", False: "red"},
            alpha=0.7,
            s=100
        )
        
        plt.title("Relationship between Mesh Complexity and Watertightness")
        plt.xlabel("Number of Vertices")
        plt.ylabel("Number of Faces")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Watertight")
        plt.tight_layout()
        plt.savefig("mesh_analysis/complexity_vs_watertight.png")
        plt.close()
        
        # 3. Density of faces per vertex by watertightness
        plt.figure(figsize=(10, 6))
        # Calculate faces per vertex
        df["faces_per_vertex"] = df["num_faces"] / df["num_vertices"]
        
        # Remove outliers for better visualization
        q1 = df["faces_per_vertex"].quantile(0.01)
        q3 = df["faces_per_vertex"].quantile(0.99)
        filtered_df = df[(df["faces_per_vertex"] >= q1) & (df["faces_per_vertex"] <= q3)]
        
        sns.histplot(
            data=filtered_df, 
            x="faces_per_vertex", 
            hue="is_watertight", 
            multiple="stack",
            kde=True,
            palette={True: "green", False: "red"}
        )
        
        plt.title("Distribution of Faces per Vertex by Watertightness")
        plt.xlabel("Faces per Vertex")
        plt.ylabel("Number of Models")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig("mesh_analysis/faces_per_vertex_distribution.png")
        plt.close()
        
        return
    
    def visualize_sample_models(self, df, sample_size=5):
        """Visualizes sample models with and without watertightness"""
        # Select samples of watertight and non-watertight models
        watertight_samples = df[df["is_watertight"]].sample(min(sample_size, df["is_watertight"].sum()))
        non_watertight_samples = df[~df["is_watertight"]].sample(min(sample_size, (~df["is_watertight"]).sum()))
        
        # Visualize some non-watertight models
        for _, row in non_watertight_samples.iterrows():
            self.render_model_with_issues(row["file_name"])
        
        return
    
    def render_model_with_issues(self, file_name):
        """Renders a model and highlights problem areas."""
        try:
            file_path = os.path.join(self.data_dir, file_name)
            mesh = trimesh.load_mesh(file_path)
            
            # Create a simple visualization
            scene = trimesh.Scene()
            
            # Add model with transparency
            mesh.visual.face_colors = [200, 200, 200, 100]  # Semi-transparent gray
            scene.add_geometry(mesh)
            
            # Add points to areas with possible holes
            # Since we don't have edges_face_count, try sampling the surface
            # of the model and find areas with unusual curvature
            samples, face_index = mesh.sample(5000, return_index=True)
            
            # We can check the normals at these points
            normals = mesh.face_normals[face_index]
            
            # A simple heuristic: points with unusual normal directions
            # could indicate problem areas
            mean_normal = np.mean(normals, axis=0)
            norm_mean_normal = np.linalg.norm(mean_normal)
            if norm_mean_normal > 0:
                mean_normal = mean_normal / norm_mean_normal
            
            # Points with normals that deviate strongly from the average direction
            dots = np.abs(np.dot(normals, mean_normal))
            threshold = np.percentile(dots, 10)  # Bottom 10% of values
            potential_issue_points = samples[dots < threshold]
            
            if len(potential_issue_points) > 0:
                # Create a point cloud for potential problem areas
                cloud_mesh = trimesh.points.PointCloud(potential_issue_points, colors=[255, 0, 0, 255])
                scene.add_geometry(cloud_mesh)
            
            # Save the visualization
            png = scene.save_image(resolution=[1280, 720])
            with open(f"mesh_analysis/issue_visualization_{file_name.replace('.stl', '.png')}", 'wb') as f:
                f.write(png)
                
            print(f"Visualization created for {file_name}")
            
        except Exception as e:
            print(f"Error visualizing {file_name}: {str(e)}")
    
    def analyze_euler_characteristics(self, df):
        """Analyzes the Euler characteristic of the models."""
        plt.figure(figsize=(12, 6))
        
        # Filter extreme outliers for better visualization
        filtered_df = df.copy()
        if "euler_number" in filtered_df.columns:
            q1 = filtered_df["euler_number"].quantile(0.05)
            q3 = filtered_df["euler_number"].quantile(0.95)
            filtered_df = filtered_df[(filtered_df["euler_number"] >= q1) & (filtered_df["euler_number"] <= q3)]
            
            sns.boxplot(x="is_watertight", y="euler_number", data=filtered_df)
            plt.title("Euler Characteristic by Watertightness")
            plt.xlabel("Watertight")
            plt.ylabel("Euler Number (V - E + F)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig("mesh_analysis/euler_characteristics.png")
            plt.close()
        
        return
    
    def compute_implications_of_non_watertight(self, df):
        """Computes the implications of non-watertight models."""
        # Create a table showing which operations don't work for non-watertight models
        operations = [
            "Volume calculation",
            "3D printing without post-processing",
            "Automatic mesh smoothing",
            "Physical simulations",
            "Automatic retopology",
            "Watertightness required for successful operation"
        ]
        
        works_on_non_watertight = [
            "No - Volume not defined",
            "No - Printing problems likely",
            "Partially - Results unreliable",
            "No - Physics requires closed volumes",
            "Partially - Poor results likely",
            "Yes"
        ]
        
        implications_df = pd.DataFrame({
            "Operation": operations,
            "Works with non-watertight models": works_on_non_watertight
        })
        
        # Save as CSV
        implications_df.to_csv("mesh_analysis/non_watertight_implications.csv", index=False)
        
        # Calculate percentage of functional limitations
        watertight_count = df["is_watertight"].sum()
        total_count = len(df)
        non_watertight_percent = 100 * (total_count - watertight_count) / total_count
        
        # Create a simple visualization of limitations
        plt.figure(figsize=(10, 6))
        operations = operations[:-1]  # Remove last entry for visualization
        restrictions = [100, 100, 70, 100, 80]  # Estimated percentage restrictions
        
        plt.bar(operations, restrictions, color="crimson")
        plt.axhline(y=non_watertight_percent, color='blue', linestyle='--', 
                   label=f"Proportion of non-watertight models ({non_watertight_percent:.1f}%)")
        plt.title("Functional Restrictions Due to Non-watertight Models")
        plt.ylabel("Functional Restriction (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig("mesh_analysis/functional_restrictions.png")
        plt.close()
        
        return implications_df
        
    def run_exploration(self):
        """Runs the exploratory analysis."""
        print("Running exploratory analysis of 3D models...")
        
        # Analysis of watertightness
        df = self.analyze_watertight_status()
        print(f"Analyzed {len(df)} models.")
        
        watertight_count = df["is_watertight"].sum()
        print(f"Watertight models: {watertight_count} of {len(df)} ({100 * watertight_count / len(df):.1f}%)")
        
        # Visualization of watertightness
        self.visualize_watertightness(df)
        
        # Analysis of Euler characteristic
        self.analyze_euler_characteristics(df)
        
        # Calculate implications of non-watertight models
        self.compute_implications_of_non_watertight(df)
        
        # Visualize some sample models
        self.visualize_sample_models(df)
        
        print("\nExploratory analysis completed. Results saved in the 'mesh_analysis' folder.")
        return df

# Main execution
if __name__ == "__main__":
    analyzer = MeshAnalyzer()
    results = analyzer.run_exploration()