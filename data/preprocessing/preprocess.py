from dotenv import load_dotenv

from pipeline import Pipeline

load_dotenv()

if __name__ == "__main__":
    pipeline = Pipeline(raw_mesh_dir='raw', output_dir='processed')
    pipeline.run()