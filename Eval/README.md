# Eval Pipeline Project

## Overview
The Eval Pipeline Project is designed to facilitate the evaluation of 3D reconstruction models. It provides a structured approach to load model checkpoints, perform evaluations, and generate metrics related to the reconstructed meshes.

## Project Structure
```
eval_pipeline_project
├── src
│   ├── eval_pipeline.py      # Contains the EvalPipeline class for orchestrating evaluations
│   ├── eval_runner.py        # Includes the EvalRunner class for executing evaluations
│   ├── mesh_utils.py         # Provides utility functions for mesh processing
│   └── __init__.py           # Marks the src directory as a Python package
├── checkpoints                # Directory for storing model checkpoint files
├── data                       # Directory for datasets and other data files
├── requirements.txt           # Lists project dependencies
└── README.md                  # Documentation for the project
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
1. Place your model checkpoints in the `checkpoints` directory.
2. Prepare your datasets and place them in the `data` directory.
3. Run the evaluation using the `EvalRunner` class from `src/eval_runner.py`.

## Example
```python
from src.eval_runner import EvalRunner

runner = EvalRunner()
runner.run_evaluation()
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.