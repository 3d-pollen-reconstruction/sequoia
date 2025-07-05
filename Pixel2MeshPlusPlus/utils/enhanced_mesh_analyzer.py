import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

def load_and_visualize_original_mesh(obj_file, output_file):
    """
    Lädt ein .obj Mesh, verwendet die ursprünglichen Farben (falls vorhanden),
    fügt schwarze Wireframes hinzu und speichert als transparentes PNG.
    """
    try:
        # Lade das Mesh mit trimesh
        mesh = trimesh.load(obj_file)
        
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            print(f"Fehler: {obj_file} ist kein gültiges Mesh")
            return
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Bestimme ursprüngliche Farben
        mesh_color = None
        
        if hasattr(mesh, 'visual'):
            # Versuche vertex colors
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                if len(mesh.visual.vertex_colors) > 0:
                    vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0  # RGB nur, normalisiert
                    # Berechne Face-Farben aus Vertex-Farben
                    mesh_color = vertex_colors[faces].mean(axis=1)
                    print(f"Verwende ursprüngliche Vertex-Farben für {obj_file}")
            
            # Versuche face colors falls vertex colors nicht verfügbar
            elif hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
                if len(mesh.visual.face_colors) > 0:
                    mesh_color = mesh.visual.face_colors[:, :3] / 255.0  # RGB nur, normalisiert
                    print(f"Verwende ursprüngliche Face-Farben für {obj_file}")
            
            # Versuche material color
            elif hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                if hasattr(mesh.visual.material, 'diffuse'):
                    base_color = np.array(mesh.visual.material.diffuse[:3]) / 255.0
                    mesh_color = np.tile(base_color, (len(faces), 1))
                    print(f"Verwende Material-Farbe für {obj_file}")
        
        # Fallback: Standardfarbe
        if mesh_color is None:
            mesh_color = np.full((len(faces), 3), [0.7, 0.7, 0.7])  # Hellgrau
            print(f"Keine ursprünglichen Farben gefunden für {obj_file}, verwende Standard-Grau")
        
        # Definiere verschiedene Blickwinkel
        views = [
            ("front", 0, 0),      # Von vorne
            ("top", 90, 0),       # Von oben
            ("bottom", -90, 0)    # Von unten
        ]
        
        for view_name, elev, azim in views:
            # Erstelle zwei Versionen: mit und ohne Wireframes (nur für front)
            wireframe_modes = [True]  # Standard: mit Wireframes
            if view_name == "front":
                wireframe_modes = [True, False]  # Für front: mit und ohne Wireframes
            
            for wireframe_enabled in wireframe_modes:
            # Erstelle Figure mit transparentem Hintergrund
            fig = plt.figure(figsize=(12, 12))
            fig.patch.set_facecolor('none')  # Transparenter Hintergrund
            ax = fig.add_subplot(111, projection='3d')
            
            # Verstecke alle Axes-Elemente komplett
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            ax.grid(False)
            
            # Entferne Axes-Panes und Linien (robuste Methode für verschiedene matplotlib-Versionen)
            try:
                # Versuche axes panes zu entfernen
                if hasattr(ax, 'xaxis') and hasattr(ax.xaxis, 'pane'):
                    ax.xaxis.pane.fill = False
                    ax.xaxis.pane.set_edgecolor('none')
                if hasattr(ax, 'yaxis') and hasattr(ax.yaxis, 'pane'):
                    ax.yaxis.pane.fill = False
                    ax.yaxis.pane.set_edgecolor('none')
                if hasattr(ax, 'zaxis') and hasattr(ax.zaxis, 'pane'):
                    ax.zaxis.pane.fill = False
                    ax.zaxis.pane.set_edgecolor('none')
            except:
                pass  # Fallback für verschiedene matplotlib-Versionen
            
            try:
                # Entferne Achsen-Linien
                if hasattr(ax, 'xaxis') and hasattr(ax.xaxis, 'line'):
                    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                if hasattr(ax, 'yaxis') and hasattr(ax.yaxis, 'line'):
                    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
                if hasattr(ax, 'zaxis') and hasattr(ax.zaxis, 'line'):
                    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            except:
                pass
            
            # Erstelle Triangles für das Mesh
            triangles = vertices[faces]
            
            # Zeichne das Mesh mit ursprünglichen Farben (im Hintergrund)
            mesh_collection = Poly3DCollection(triangles, 
                                             facecolors=mesh_color, 
                                             alpha=0.85, 
                                             linewidths=0,
                                             edgecolors='none',
                                             zorder=1)  # Hintergrund
            ax.add_collection3d(mesh_collection)
            
            # Füge dicke schwarze Wireframes hinzu (im Vordergrund)
            for face in faces:
                triangle = vertices[face]
                # Schließe das Dreieck
                triangle_closed = np.vstack([triangle, triangle[0]])
                ax.plot(triangle_closed[:, 0], 
                       triangle_closed[:, 1], 
                       triangle_closed[:, 2], 
                       color='black', linewidth=1.2, alpha=1.0, zorder=10)  # Vordergrund, fett schwarz
            
            # Setze gleiche Skalierung für alle Achsen
            max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                                 vertices[:, 1].max() - vertices[:, 1].min(),
                                 vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
            
            mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
            mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
            mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Setze Blickwinkel
            ax.view_init(elev=elev, azim=azim)
            
            # Erstelle Dateinamen für verschiedene Ansichten
            view_output = output_file.replace('.png', f'_{view_name}.png')
            
            # Speichere als PNG mit transparentem Hintergrund
            plt.tight_layout()
            plt.savefig(view_output, dpi=300, bbox_inches='tight', 
                       facecolor='none', edgecolor='none', transparent=True)
            plt.close()
            
            print(f"Gespeichert: {view_output}")
        
    except Exception as e:
        print(f"Fehler beim Verarbeiten von {obj_file}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Hauptfunktion zum Visualisieren aller .obj Dateien"""
    
    # Finde alle .obj Dateien im aktuellen Verzeichnis
    obj_files = [f for f in os.listdir('.') if f.endswith('.obj')]
    
    if not obj_files:
        print("Keine .obj Dateien gefunden!")
        return
    
    print(f"Gefundene .obj Dateien: {obj_files}")
    
    for obj_file in obj_files:
        output_file = obj_file.replace('.obj', '_original_with_wireframe.png')
        print(f"\nVerarbeite {obj_file}...")
        load_and_visualize_original_mesh(obj_file, output_file)
    
    print(f"\nFertig! {len(obj_files)} Meshes wurden visualisiert.")

if __name__ == "__main__":
    main()