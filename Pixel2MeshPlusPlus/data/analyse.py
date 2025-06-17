import pickle
import numpy as np

def analyze_original_p2mpp():
    """Analyze the original iccv_p2mpp.dat file structure"""
    file_path = r"C:\Users\super\Documents\GitHub\sequoia\Pixel2MeshPlusPlus\data\iccv_p2mpp.dat"
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print("=== Original Pixel2Mesh++ Data Analysis ===")
        print(f"Keys: {list(data.keys())}")
        print()
        
        # Main coordinates
        coord = data['coord']
        print(f"Main coordinates shape: {coord.shape}")
        
        # Sample coordinates
        sample_coord = data['sample_coord']
        print(f"Sample coordinates shape: {sample_coord.shape}")
        
        # Analyze stages
        for i, stage_key in enumerate(['stage1', 'stage2', 'stage3'], 1):
            stage_data = data[stage_key]
            print(f"\nStage {i}:")
            print(f"  Type: {type(stage_data)}")
            print(f"  Length: {len(stage_data)}")
            
            if len(stage_data) > 0:
                first_element = stage_data[0]
                print(f"  First element type: {type(first_element)}")
                
                if isinstance(first_element, np.ndarray):
                    print(f"  First element shape: {first_element.shape}")
                elif isinstance(first_element, list):
                    print(f"  First element length: {len(first_element)}")
                    if len(first_element) > 0:
                        print(f"  First sub-element type: {type(first_element[0])}")
                        if hasattr(first_element[0], 'shape'):
                            print(f"  First sub-element shape: {first_element[0].shape}")
        
        # Pool indices
        pool_idx = data['pool_idx']
        print(f"\nPool indices:")
        print(f"  Type: {type(pool_idx)}")
        print(f"  Length: {len(pool_idx)}")
        for i, pool in enumerate(pool_idx):
            if isinstance(pool, np.ndarray):
                print(f"  Pool {i}: shape {pool.shape}, unique values: {len(np.unique(pool))}")
            elif isinstance(pool, list):
                print(f"  Pool {i}: length {len(pool)}")
        
        # Faces
        faces = data['faces']
        print(f"\nFaces:")
        print(f"  Type: {type(faces)}")
        print(f"  Length: {len(faces)}")
        if len(faces) > 0:
            first_face = faces[0]
            print(f"  First face set type: {type(first_face)}")
            if isinstance(first_face, np.ndarray):
                print(f"  First face set shape: {first_face.shape}")
            elif isinstance(first_face, list):
                print(f"  First face set length: {len(first_face)}")
        
        # Laplacian indices
        lape_idx = data['lape_idx']
        print(f"\nLaplacian indices:")
        print(f"  Type: {type(lape_idx)}")
        print(f"  Length: {len(lape_idx)}")
        
        # Sample Chebyshev data
        sample_cheb = data['sample_cheb']
        print(f"\nSample Chebyshev:")
        print(f"  Type: {type(sample_cheb)}")
        print(f"  Length: {len(sample_cheb)}")
        
        # Determine actual hierarchical sizes
        print("\n=== Inferred Hierarchical Structure ===")
        print(f"Full resolution (coord): {coord.shape[0]} vertices")
        print(f"Sample resolution (sample_coord): {sample_coord.shape[0]} vertices")
        
        # Try to infer stage sizes from pool indices
        if len(pool_idx) > 0:
            current_size = coord.shape[0]
            for i, pool in enumerate(pool_idx):
                if isinstance(pool, np.ndarray):
                    next_size = len(np.unique(pool))
                    print(f"Stage {i+1} -> Stage {i+2}: {current_size} -> {next_size} vertices")
                    current_size = next_size
                elif isinstance(pool, list) and len(pool) > 0:
                    if isinstance(pool[0], np.ndarray):
                        next_size = len(np.unique(pool[0]))
                        print(f"Stage {i+1} -> Stage {i+2}: {current_size} -> {next_size} vertices")
                        current_size = next_size
        
        return data
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

if __name__ == "__main__":
    analyze_original_p2mpp()