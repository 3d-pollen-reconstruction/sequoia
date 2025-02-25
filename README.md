# 3d reconstruction dataset

This repository downloads 3D models, renders two orthogonal views and saves them as PNG images.

## Requirements

```
pip install -r requirements.txt
```

## Usage

1. Download the models from [here](https://fhnw365-my.sharepoint.com/:u:/g/personal/florin_barbisch_fhnw_ch/EcDq5jeX2tNPlVmVedqwwVUBo-9qi0_qJDaCAFzpkTK0fQ?e=oEiQp7) and put the data in the [./data](./data) folder. Alternatively you can also run `download_models.ipynb` to download the models from the source.
2. Run `random_render.py` to render a random model. Or import the `render_random_pollen()` function to use it in your own project.



## Output

- `data/models/` contains the downloaded models, each model's name starts with its ID (corresponds to the `id` column in `data/3d_pollen_library.csv`).
- `data/3d_pollen_library.csv` contains the metadata of the models. Here the paldat ([https://www.paldat.org](https://www.paldat.org)) and global pollen project ((https://globalpollenproject.org)[https://globalpollenproject.org]) links in the `description` might be of interest.
They have more structured information about the pollen species that could help in the analysis of the reconstruction (group reconstructions be a certain information (size, number of apertures, etc.)).
- `random_render.py` contains a function `render_random_pollen()` that return two orthogonal views/projections of a random pollen 3d model and the model's file name and rotation (of the pollen model). The two images are grayscale and have a width and height of 1024 pixels. Running just the file will save the screenshot of a random model in the project root folder to test the rendering function.
