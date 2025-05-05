from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv(""))
api.upload_large_folder(
    folder_path="./pollen/",
    repo_id="Etiiir/Pollen",
    repo_type="dataset",
)