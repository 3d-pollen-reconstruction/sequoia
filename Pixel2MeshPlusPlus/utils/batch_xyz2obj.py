# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.

import sys
import numpy as np
import os
import glob

def get_face_path_and_subfolder(xyz_filename):
    """
    Determine which face file to use and which subfolder based on the xyz filename.
    
    Rules:
    - Files ending with '_1.xyz' -> face1.obj, subfolder '1'
    - Files ending with '_2.xyz' -> face2.obj, subfolder '2'  
    - Files ending with '.xyz' (no number) -> face3.obj, subfolder '3'
    """
    basename = os.path.basename(xyz_filename)
    
    if basename.endswith('_1.xyz'):
        return '../data/face1.obj', '1'
    elif basename.endswith('_2.xyz'):
        return '../data/face2.obj', '2'
    else:  # Files ending with just .xyz (no number)
        return '../data/face3.obj', '3'

def convert_xyz_to_obj(xyz_path, output_dir):
    """
    Convert a single xyz file to obj format.
    """
    try:
        # Load xyz data
        xyzf = np.loadtxt(xyz_path)
        
        # Get face path and subfolder based on filename
        face_path, subfolder = get_face_path_and_subfolder(xyz_path)
        
        # Create output directory structure
        xyz_dir_name = os.path.basename(os.path.dirname(xyz_path))
        final_output_dir = os.path.join(output_dir, xyz_dir_name, subfolder)
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Generate output filename
        xyz_basename = os.path.basename(xyz_path)
        obj_basename = xyz_basename.replace('.xyz', '.obj')
        obj_path = os.path.join(final_output_dir, obj_basename)
        
        # Load face data
        if not os.path.exists(face_path):
            print(f"Warning: Face file {face_path} not found. Skipping {xyz_path}")
            return False
            
        face = np.loadtxt(face_path, dtype='|S32')
        
        # Create vertex entries
        v = np.full((xyzf.shape[0], 1), 'v')
        
        # Combine vertices and faces
        out = np.vstack((np.hstack((v, xyzf)), face))
        
        # Save to obj file
        np.savetxt(obj_path, out, fmt='%s', delimiter=' ')
        
        print(f"Converted: {xyz_path} -> {obj_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {xyz_path}: {str(e)}")
        return False

def process_directories(input_dirs, output_base_dir):
    """
    Process multiple directories recursively to find and convert xyz files.
    """
    total_converted = 0
    total_failed = 0
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"Warning: Directory {input_dir} does not exist. Skipping.")
            continue
            
        print(f"Processing directory: {input_dir}")
        
        # Find all xyz files recursively
        xyz_files = glob.glob(os.path.join(input_dir, "**", "*.xyz"), recursive=True)
        
        print(f"Found {len(xyz_files)} xyz files in {input_dir}")
        
        for xyz_file in xyz_files:
            if convert_xyz_to_obj(xyz_file, output_base_dir):
                total_converted += 1
            else:
                total_failed += 1
    
    print("\nConversion complete!")
    print(f"Successfully converted: {total_converted} files")
    print(f"Failed to convert: {total_failed} files")

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_xyz2obj.py <input_dir1> [input_dir2] [input_dir3] ...")
        print("Example: python batch_xyz2obj.py ./data ./other_data")
        sys.exit(1)
    
    # Get input directories from command line arguments
    input_dirs = sys.argv[1:]
    
    # Output base directory
    output_base_dir = "pixel2meshplusplus/predictions"
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Input directories: {input_dirs}")
    print(f"Output base directory: {output_base_dir}")
    print("Face files will be selected automatically based on filename patterns:")
    print("  - Files ending with '_1.xyz' -> face1.obj -> subfolder '1'")
    print("  - Files ending with '_2.xyz' -> face2.obj -> subfolder '2'")
    print("  - Files ending with '.xyz' -> face3.obj -> subfolder '3'")
    print()
    
    # Process all directories
    process_directories(input_dirs, output_base_dir)

if __name__ == '__main__':
    main()
