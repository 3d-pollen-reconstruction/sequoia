<img width="1003" height="583" alt="bat_preprocessing" alt="Pipeline Overview" src="https://github.com/user-attachments/assets/032f7d71-304f-4faa-a22a-4e4da7d81605" />

# Sequoia: A Unified Multi‑Model Framework for 3D Reconstruction of Pollen Morphologies

This repository provides an integrated, reproducible and extensible framework for reconstructing 3D pollen geometry from multi‑view photographic and holographic inputs. It consolidates classical geometric methods (Visual Hull) with modern neural reconstruction paradigms (Pix2Vox, PixelNeRF, Pixel2Mesh++, diffusion / generative methods via Hunyuan3D‑2) and offers scalable Blender / VTK based preprocessing pipelines. Experiment management is orchestrated through Hydra; training and evaluation leverage PyTorch Lightning with consistent metric reporting.

## Table of Contents
1. Objective and Scope
2. Core Contributions
3. Directory Overview and Rationale
4. Model Families and Training Interfaces (Visual Hull | Pix2Vox (Hydra) | PixelNeRF | Pixel2Mesh++ | Hunyuan3D‑2 | Optional External Methods)
5. Data Architecture and Preprocessing Pipelines
6. Hydra Configuration & Experiment Design
7. Training Modalities (Containers / SLURM Recommended; Local Development Optional)
8. Evaluation and Metrics Infrastructure
9. Reproducibility, Logging and Experiment Tracking
10. Curated Script Index
11. Planned Extensions
12. Minimal Local Quickstart (Fallback Only)
13. FAQ
14. Citation and Attribution
15. Contact

---
## 1. Objective and Scope
Sequoia aims to establish a rigorous comparative substrate for heterogeneous 3D reconstruction techniques applied to pollen morphology. Emphasis is placed on methodological comparability, dataset modularity (synthetic multi‑view renders, holographic acquisitions) and transparent experiment lifecycle management. The design priorities are: (i) structural clarity, (ii) deterministic preprocessing, (iii) orthogonal configuration overrides, and (iv) separation of concerns between data generation, model definition, and evaluation.

## 2. Core Contributions
* Unified training abstraction (Lightning + Hydra) spanning volumetric, implicit, voxel‑grid, mesh refinement and generative paradigms.
* Visual Hull baseline (deterministic boolean volume carving) for geometric reference.
* Pix2Vox integration with granular control over pretrained loading, staged module activation (merger/refiner kick‑in), and selective freezing.
* PixelNeRF integration (multi‑view conditional radiance field) via dedicated SLURM scripts and container isolation.
* Pixel2Mesh++ multi‑stage mesh refinement pipeline backed by bespoke preprocessing conforming to its data contract.
* Hunyuan3D‑2 accelerated inference suite (octree / step budgets) for generative comparison.
* Two complementary augmentation frameworks (broad vs. compact deformation sets) enabling structural and morphological perturbations.
* Cohesive evaluation layer sharing metric mixins across models for directly comparable quantitative outputs.

## 3. Directory Overview and Rationale
```
configs/               Hydra configurations (data, model, experiment, trainer, callbacks)
core/                  Training orchestration, metric mixins, model registry
data/                  Dataset classes, augmentation entry points, utilities
data/preprocessing/    Blender & VTK pipelines (ShapeNet style renders, Pixel2Mesh converters, augmentation variants)
Pixel_Nerf/            Upstream / adapted PixelNeRF implementation and scripts
Pixel2MeshPlusPlus/    Pixel2Mesh++ codebase (cfgs, modified templates)
Hunyuan3D-2/           Hunyuan3D‑2 inference utilities and examples
notebooks/             Exploratory analysis, ablation studies, qualitative comparisons
scripts/               SLURM / submission / sweep scripts grouped by model family
evaluation_pipeline/   Programmatic runners for standardized evaluation (incl. holography)
evaluation_results/    Persisted quantitative and qualitative outputs
appendix/              Auxiliary external methods (InstantMesh, SparseFusion, SplatterImage)
docker/                Container build assets
gen_images/            Diagrams and illustrative assets
```
Key artefacts:
* `configs/train.yaml` – primary Hydra entrypoint (default composition; model switched via override).
* `configs/model/*.yaml` – model‑level hyperparameters and pretrained weight references.
* `configs/experiment/*.yaml` – composable scenario definitions (e.g. `vh_2img.yaml`, `pix2vox_aug_4img.yaml`).
* `core/train.py` – canonical training pipeline (freezing logic, pretrained weight injection, metrics bootstrap, W&B logging).
* `core/models/visual_hull.py` – reference implementation using structured back‑projection carving.
* `core/models/pix2vox/` – modular encoder / decoder / merger / refiner components.
* `data/augmentation.py` & `data/preprocessing/create_augmentations/augmentation.py` – broad vs. compact augmentation frameworks.
* `data/preprocessing/pixel2mesh/*.py` – conversion, normalization and multi‑view rendering for Pixel2Mesh++ datasets.
* `data/preprocessing/blender_pipeline/Shape_Net_Pipeline/` – ShapeNet‑style camera sphere rendering (PixelNeRF preparation).
* `notebooks/animated_mesh_comparison.ipynb` – consolidated qualitative benchmarking across model families.
* `scripts/*/*.sbatch` – curated SLURM launchers isolating variant hyperparameters.
* `Hunyuan3D-2/fast_shape_gen_pollen_orthogonal_*.py` – inference acceleration footprints (varying step count and octree depth).

## 4. Model Families and Training Interfaces
### 4.1 Visual Hull
Deterministic baseline producing a boolean occupancy volume through multi‑view silhouette intersection. Configured via `configs/model/visual_hull.yaml` and experiment overrides `configs/experiment/vh_*img.yaml` (1–6 views). Example:
```bash
python -m core.train experiment=vh_2img model=visual_hull data.default n_images=2
```
No optimizer state is persisted (pure geometric operator); only metrics are logged.

### 4.2 Pix2Vox (Hydra Integration)
Configuration at `configs/model/pix2vox.yaml` supports: learning rate, pretrained weights, staged activation thresholds (`merger_kickin`, `refiner_kickin`), dropout and explicit module freezing. Experiments differentiate augmentation usage and view cardinality (e.g. `pix2vox_aug_1img.yaml` … `pix2vox_aug_6img.yaml`, holography transfer variants). Example:
```bash
python -m core.train experiment=pix2vox_aug_4img model=pix2vox data.default n_images=4 data.include_augmentations=true
```
On‑the‑fly freezing: `model.frozen="[encoder,decoder]"`.

### 4.3 PixelNeRF
Resides under `Pixel_Nerf/`. Launch is containerized to stabilize dependency stacks and CUDA alignment. SLURM submissions (`scripts/slurm_sbatch_experiments/train_pixelnerf*.sbatch` and `scripts/pixelnerf/*.sbatch`) expose variations in encoder depth, view counts, fine/coarse sampling and loss choices. Example invocation segment:
```bash
singularity exec --nv \
  --bind /path/checkpoints:/container/checkpoints \
  --bind /path/sequoia/Pixel_Nerf/:/code \
  --pwd /code pixelnerf_new.sif \
  python3 train/org_train.py -n pollen_256_4_4 -c conf/exp/pollen.conf -D /code/pollen --nviews 4 --resume
```

### 4.4 Pixel2Mesh++
Hosted in `Pixel2MeshPlusPlus/` with modified configuration templates (`cfgs/`). Preprocessing scripts (`to_pixel2mesh_dir_original.py`, `to_pixel2mesh_dir_original_augments.py`) produce canonical multi‑view inputs (8 fixed azimuths at 30° elevation) and normalized meshes. SLURM scripts in `scripts/pixel2mesh/` encode a matrix of (view count × prior type × training regime), including mean/special/spherical priors, freeze vs. fine‑tune.

### 4.5 Hunyuan3D‑2
Inference‑centric generative pipeline (no training loop here). Scripts parameterize octree resolution and sampling steps (`fast_shape_gen_pollen_orthogonal_{5,10,50}steps.py`, `_octree32.py`, `_octree128.py`). Provides comparative generative baselines and rapid prototyping for diffusion‑based reconstructions.

### 4.6 Optional External Methods
Supplementary approaches located in `appendix/` (e.g. SparseFusion, InstantMesh, SplatterImage). These are included for qualitative or contextual benchmarking and may maintain independent requirement sets. Containerization is strongly advised to prevent cross‑contamination of dependencies.

## 5. Data Architecture and Preprocessing Pipelines
### 5.1 End‑to‑End Flow
Raw STL Mesh → (Cleaning / Repair via `data/mesh_cleaner.py`, `mesh_analyzer.py`) → Normalization (centering + scale invariance) → Multi‑view rendering (Blender) producing RGB + silhouettes → (Optional) augmentation (geometric + morphological) → Model‑specific dataset assembly (voxel grids, multi‑image tuples, mesh refinement inputs) → Training & evaluation artifacts.

### 5.2 Blender ShapeNet‑Style Pipeline (PixelNeRF)
Location: `data/preprocessing/blender_pipeline/Shape_Net_Pipeline/` and variant under `data/preprocessing/pixelnerf/Shape_Net_Pipeline/`. Principal scripts: `shapenet_spherical_renderer.py`, `parallel.py`, `augmentation.py`, `blender_interface.py`. Example (Windows, legacy Blender 2.7):
```bash
"C:\\Program Files\\Blender2.7\\blender.exe" --background --python shapenet_spherical_renderer.py --addons io_mesh_stl -- \
  --mesh_dir C:/path/meshes_obj/ --output_dir C:/out/shapenet_views --num_observations 128
```
Outputs consistent camera pose enumerations consumed by PixelNeRF (image–pose pairing).

### 5.3 Visual Hull and Pix2Vox Silhouette Alignment
Silhouettes are derived from the same multi‑view rendering stage. Visual Hull uses fixed azimuth angles embedded in `VisualHull.angles_deg`; ensure renderer azimuth parity. Pix2Vox leverages identical view subsets controlled through `n_images` in the data configuration or experiment overrides.

### 5.4 Pix2Vox Datasets
Augmented vs. base variants differentiated through experiments (`pix2vox_aug_*.yaml`) toggling `data.include_augmentations=true`. Pretrained weights (`Pix2Vox-A-ShapeNet.pth`) specified in `configs/model/pix2vox.yaml`.

### 5.5 Pixel2Mesh++ Conversion
Scripts:
* Original: `data/preprocessing/pixel2mesh/to_pixel2mesh_dir_original.py`
* Augmented: `data/preprocessing/pixel2mesh/to_pixel2mesh_dir_original_augments.py`
Processing stages: VTK normalization, mesh simplification (quadric decimation), canonical orientation, multi‑view rendering (8 fixed viewpoints) and persistence of mesh + normals + auxiliary attributes. Output directory structure mirrors instance granularity (`pixel2mesh_original/`, `pixel2mesh_original_augmented/`).

### 5.6 Augmentation Pipelines
1. Comprehensive (root: `data/augmentation.py`) – Deformations: swelling, shriveling, twisting, stretching, spikify, groove, wrinkle, asymmetry, full_combo. Progress recorded via `progress.json` enabling resumable execution.
2. Compact ( `data/preprocessing/create_augmentations/augmentation.py`) – Deformations: twisting, stretching, groove, asymmetry, full_combo, radical_reshape, irregular. Designed for rapid, orthogonally composable perturbations.
Example:
```bash
"C:\\Program Files\\Blender2.7\\blender.exe" --background --python data/augmentation.py --addons io_mesh_stl -- \
  --mesh_dir data/processed/meshes_repaired --output_dir data/processed/augmentation --num_augmentations 5
```
Augmented outputs feed downstream into Pixel2Mesh++ or Pix2Vox (if augmentation inclusion is activated via Hydra).

### 5.7 Holography Dataset Integration
Holographic preprocessing and domain adaptation support are configured via `configs/data/holo.yaml` (includes ripple removal transform sequence). Experiments such as `pix2vox_aug_holo_test.yaml` and `vh_2img_holo_test.yaml` facilitate zero‑shot or adaptation studies.

## 6. Hydra Configuration & Experiment Design
Layered composition:
* Data layer (`configs/data/*.yaml`): `PollenDataModule`, `HolographicPolenoDataModule` with parameters (`n_images`, augmentation toggles, batch size, transforms).
* Model layer (`configs/model/*.yaml`): hyperparameters, pretrained asset paths, module scheduling.
* Experiment layer (`configs/experiment/*.yaml`): curated override bundles (e.g. model swap, number of images, holography variants).
Execution example:
```bash
python -m core.train experiment=pix2vox_aug_3img model.pix2vox.lr=5e-5 seed=1234
```
Hyperparameter sweeps employ Optuna through `scripts/submit_optuna_sweep.sh` invoking multi‑run Hydra mode (`-m +sweep=pix2vox_optuna`).

## 7. Training Modalities
### 7.1 Recommended: Containerized Execution
Authoritative images: https://hub.docker.com/repositories/etiir
Justification: deterministic dependency graphs, GPU driver abstraction, trivial SLURM integration, avoidance of cross‑library ABI mismatches (Torch / VTK / Open3D / Blender).

### 7.2 Illustrative SLURM Patterns
Pix2Vox (4 views augmented):
```bash
sbatch --job-name=pix2vox_aug_4img <<'EOF'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
export WANDB_API_KEY=YOUR_KEY
singularity exec --nv pix2vox.sif \
  python -m core.train experiment=pix2vox_aug_4img
EOF
```
Visual Hull (lightweight CPU/GPU optional):
```bash
python -m core.train experiment=vh_4img data.default n_images=4
```
PixelNeRF, Pixel2Mesh++ and Hunyuan3D‑2 rely on their respective SLURM scripts under `scripts/` for consistent binding of code, checkpoints and environment.

### 7.3 Local Development (Non‑Benchmark Use Only)
Potential friction points: legacy Blender 2.7 API, VTK wheel platform specificity, CUDA / Torch ABI alignment. Containers remain the canonical path for reproducible results.

## 8. Evaluation and Metrics Infrastructure
Unified metric registration occurs via `core/metrics.py` and `MetricsMixin`, automatically invoked in Lightning `training_step` / `validation_step` / `test` flows. Analytical notebooks (`animated_mesh_comparison.ipynb`, `exp_5_number_of_views.ipynb`, `holo_explorer.py`) provide qualitative triangulation. Programmatic evaluation orchestrators: `evaluation_pipeline/runner.py`, `runner_holo.py`. Outputs are archived in `evaluation_results/` segregated by domain context.

## 9. Reproducibility, Logging and Experiment Tracking
* WandB integration (`WandbLogger`), project namespace `reconstruction`.
* Full Hydra configuration (resolved) persisted with each run for auditability.
* Determinism seeded via `pl.seed_everything(cfg.seed, workers=True)` if `seed` is defined.
* Checkpoints stored under `checkpoints/` (override capable).

## 10. Curated Script Index (Selected)
| Purpose | Path |
|---------|------|
| Visual Hull configurations | `configs/experiment/vh_*img.yaml` |
| Pix2Vox view count variants | `configs/experiment/pix2vox_aug_{1,3,4,5,6}img.yaml` |
| Pix2Vox frozen enc/dec | `configs/experiment/pix2vox_aug_frozen_encdec.yaml` |
| Holography transfer tests | `pix2vox_aug_holo_test.yaml`, `vh_2img_holo_test.yaml` |
| Pixel2Mesh++ preprocessing (original) | `data/preprocessing/pixel2mesh/to_pixel2mesh_dir_original.py` |
| Pixel2Mesh++ preprocessing (augmented) | `.../to_pixel2mesh_dir_original_augments.py` |
| Augmentation (comprehensive) | `data/augmentation.py` |
| Augmentation (compact) | `data/preprocessing/create_augmentations/augmentation.py` |
| ShapeNet rendering | `data/preprocessing/blender_pipeline/Shape_Net_Pipeline/shapenet_spherical_renderer.py` |
| Pix2Vox model modules | `core/models/pix2vox/` |
| Visual Hull model | `core/models/visual_hull.py` |
| Hydra training entry | `core/train.py` |
| Optuna sweep submission | `scripts/submit_optuna_sweep.sh` |

## 11. Planned Extensions
* Unified CLI façade spanning all model families.
* Automated camera pose JSON export collocated with rendered images.
* Cross‑model evaluation CLI for batch metric synthesis across checkpoints.
* Mesh quality metrics (watertightness, genus, self‑intersection) integrated into training loop.
* Preprocessing unit tests (normalization invariants, face count thresholds, scaling correctness).

## 12. Minimal Local Quickstart (Fallback Only)
Containers and provided SLURM scripts remain the authoritative path; local setup increases variance and failure surface.

### 12.1 Prerequisites
* Python 3.11 (root project) / Python 3.12 (Hunyuan3D‑2 sub‑environment)
* CUDA‑capable GPU with compatible drivers
* (Legacy augmentation) Blender 2.7x; consider migration plan to ≥3.x

### 12.2 Installation via uv
Root environment:
```bash
uv sync
```
Activation (PowerShell):
```powershell
./.venv/Scripts/Activate.ps1
```
Hunyuan3D‑2 (separate lock + Python 3.12):
```bash
cd Hunyuan3D-2
uv sync
```
PixelNeRF (isolated virtual environment example):
```bash
cd Pixel_Nerf
python -m venv .venv
source .venv/bin/activate  # or .venv/Scripts/Activate.ps1 on Windows
pip install -r requirements.txt
```
Pixel2Mesh++ follows the same isolation principle.

### 12.3 Sanity Test (Visual Hull)
```bash
python -m core.train experiment=vh_1img
```
Expected: metrics logged (WandB optional), rapid termination without GPU memory pressure.

### 12.4 Data Path Verification
Ensure `data/processed/` contains requisite mesh / augmentation outputs. Absent structures should trigger reruns of the relevant preprocessing scripts above.

### 12.5 Common Failure Modes
| Issue | Cause | Mitigation |
|-------|-------|-----------|
| Missing Blender Python modules | Incorrect binary path | Supply absolute Blender path, validate `--python` target |
| Torch / torchvision ABI mismatch | Divergent CUDA builds | Prefer container; else install matched wheel set |
| VTK ImportError | Platform wheel unavailability | Favor container execution |
| Headless Open3D rendering | No GUI backend | Utilize `pyglet<2`, fallback to off‑screen rendering or disable visualization |

### 12.6 Reinforced Recommendation
For benchmark‑quality results always rely on provided Docker / Singularity images and SLURM scripts; local variance undermines comparability.

## 13. FAQ
**Why containers?** Heterogeneous dependency surface (Blender 2.7 API, VTK, Open3D, PyTorch) produces brittle local stacks; containers encode stable, shareable environments.  
**Which view azimuths does Visual Hull assume?** `[0, 90, 180, 270, 45, 135, 225, 315]` degrees (see `VisualHull.angles_deg`).  
**How do I freeze Pix2Vox encoder / decoder?** Add `model.frozen="[encoder,decoder]"` to the Hydra command line.  
**Augmentation framework differences?** Root variant offers a broader morphological spectrum (swelling / spikify / wrinkle), compact variant emphasizes composable transformations for rapid diversification.  

## 14. Citation and Attribution
Please acknowledge upstream projects (PixelNeRF, Pix2Vox, Pixel2Mesh++, Hunyuan3D‑2, and others) in accordance with their respective licenses. This repository serves as an integrative orchestration layer and does not supersede original intellectual property claims.

## 15. Contact
For clarification, feature proposals, or issue reporting, file a ticket in the issue tracker or reach out to the maintainers listed in the project metadata.

---
Last Updated: 2025‑08‑13
