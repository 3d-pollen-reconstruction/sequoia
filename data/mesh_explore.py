import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings

class MeshExplorer:
    def __init__(self, data_dir="../data/raw/"):
        self.data_dir = data_dir
        os.makedirs("mesh_analysis", exist_ok=True)
    
    def analyze_advanced_properties(self):
        """Analyzes additional geometric and topological properties of 3D models."""
        stl_files = [f for f in os.listdir(self.data_dir) if f.endswith(".stl")]
        results = []
        
        for file in tqdm(stl_files, desc="Analyzing 3D models"):
            file_path = os.path.join(self.data_dir, file)
            try:
                mesh = trimesh.load_mesh(file_path)
                stats = {
                    "file_name": file,
                    "bounding_box_volume": np.prod(mesh.extents),
                    "aspect_ratio": max(mesh.extents) / min(mesh.extents),
                    "convex_hull_volume": ConvexHull(mesh.vertices).volume,
                    "sphericity": self.compute_sphericity(mesh),
                    "symmetry_deviation": self.compute_symmetry_deviation(mesh),
                }
                results.append(stats)
            except Exception as e:
                print(f"Error with {file}: {str(e)}")
                results.append({"file_name": file, "error": str(e)})
        
        df = pd.DataFrame(results)
        df.to_csv("mesh_analysis/advanced_mesh_properties.csv", index=False)
        return df

    def compute_sphericity(self, mesh):
        """Computes sphericity of a model."""
        try:
            if mesh.volume is None or mesh.area == 0:
                return None
            return (np.pi ** (1/3) * (6 * mesh.volume) ** (2/3)) / mesh.area
        except:
            return None

    def compute_symmetry_deviation(self, mesh):
        """Computes symmetry deviation by comparing mirrored vertices."""
        mirrored_vertices = mesh.vertices.copy()
        mirrored_vertices[:, 0] *= -1  # Flip along X-axis
        deviation = np.mean(np.abs(mesh.vertices - mirrored_vertices))
        return deviation

    def visualize_advanced_properties(self, df):
        """Creates visualizations for shape complexity and geometric properties."""
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x="bounding_box_volume", y="convex_hull_volume")
        plt.title("Bounding Box Volume vs. Convex Hull Volume")
        plt.xlabel("Bounding Box Volume")
        plt.ylabel("Convex Hull Volume")
        plt.grid(True)
        plt.savefig("mesh_analysis/bounding_box_vs_hull_volume.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.histplot(df, x="sphericity", kde=True)
        plt.title("Sphericity Distribution")
        plt.savefig("mesh_analysis/sphericity_distribution.png")
        plt.close()
        
        return

    def perform_dimensionality_reduction(self, df):
        """Performs PCA and t-SNE on shape-related features."""
        feature_cols = ["bounding_box_volume", "aspect_ratio", "convex_hull_volume", "sphericity"]
        df_features = df.dropna(subset=feature_cols)[feature_cols]
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_features)
        df["pca_1"], df["pca_2"] = pca_result[:, 0], pca_result[:, 1]
        
        # Verwende explizit n_jobs=1, um die Parallelisierungswarnung zu vermeiden
        tsne = TSNE(n_components=2, perplexity=10, random_state=42, n_jobs=1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne_result = tsne.fit_transform(df_features)
        
        df["tsne_1"], df["tsne_2"] = tsne_result[:, 0], tsne_result[:, 1]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df["pca_1"], y=df["pca_2"])
        plt.title("PCA of Mesh Properties")
        plt.savefig("mesh_analysis/pca_analysis.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df["tsne_1"], y=df["tsne_2"])
        plt.title("t-SNE of Mesh Properties")
        plt.savefig("mesh_analysis/tsne_analysis.png")
        plt.close()
        
        return
    
    def run_advanced_analysis(self):
        """Executes the extended analysis workflow."""
        print("Running advanced 3D model analysis...")
        df = self.analyze_advanced_properties()
        self.visualize_advanced_properties(df)
        self.perform_dimensionality_reduction(df)
        print("Advanced analysis completed.")
        return df

if __name__ == "__main__":
    adv_analyzer = MeshExplorer()
    adv_results = adv_analyzer.run_advanced_analysis()