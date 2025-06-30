import os
import trimesh

def convert_stl_dir_to_obj(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mesh_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.stl')]
    
    for fname in mesh_files:
        stl_path = os.path.join(input_dir, fname)
        obj_name = os.path.splitext(fname)[0] + '.obj'
        obj_path = os.path.join(output_dir, obj_name)

        try:
            mesh = trimesh.load_mesh(stl_path)
            mesh.export(obj_path)
            print(f"[✓] Converted: {fname} → {obj_name}")
        except Exception as e:
            print(f"[✗] Failed: {fname} ({e})")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert all STL files in a directory to OBJ format.")
    parser.add_argument("--input_dir", required=True, help="Directory containing .stl files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save .obj files.")
    args = parser.parse_args()

    convert_stl_dir_to_obj(args.input_dir, args.output_dir)
