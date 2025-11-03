Efficient MedSAM2 — Student Model and Assessment Interface
==========================================================

Overview
--------

This repository provides an efficient, prompt-aware student model trained for medical image segmentation, together with tools to evaluate and compare it against the original MedSAM2 teacher model. The student models use a 4-channel input (RGB + soft box prior) and are optimized for reduced memory usage and faster inference while supporting prompt-driven segmentation via bounding boxes.

Repository contents
-------------------

- `streamlit_comparison_app.py` — Streamlit web application for side-by-side comparison of the student model and MedSAM2.
- `assessment_interface.py` — Alternate Streamlit demo interface.
- `cli_assessment.py` — Command-line tool for running inference and saving results.
- `prompt_based_learning.ipynb` — Training and experiment notebook.
- `requirements.txt` — Python dependencies required to run the tools.
- `*.pt` — Trained model checkpoints produced during experiments (student and teacher variants).

Installation
------------

1. Create and activate a virtual environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\activate

2. Install Python dependencies:

   pip install -r requirements.txt

Python 3.8 or newer is recommended. For GPU acceleration, install a CUDA-compatible PyTorch build.

Streamlit (interactive) usage
-----------------------------

1. Launch the comparison app:

   streamlit run streamlit_comparison_app.py

2. Open the local URL reported by Streamlit (typically `http://localhost:8501`).

3. In the sidebar: select a student checkpoint (for example `student_finetuned_full.pt` or `best_student_prompt_full.pt`), upload an input image, and draw or enter a bounding box to run prompt-based segmentation. Use the Reload Models button after adding or replacing checkpoint files in the working directory.

Command-line usage
------------------

Run the CLI tool to perform scripted inference and save results:

   python cli_assessment.py --image PATH_TO_IMAGE --bbox X1 Y1 X2 Y2 --output results/

Common options:
- `--image`: Path to an RGB input image.
- `--bbox`: Bounding box coordinates in pixels: `x1 y1 x2 y2`.
- `--output`: Output directory for masks and comparison images.
- `--device`: `cpu` or `cuda` (auto-detected by default).

Models and checkpoints
----------------------

Place pretrained checkpoints in the repository root or provide full paths when using the CLI. Typical checkpoint filenames used in experiments:

- `student_finetuned_full.pt`
- `best_student_prompt_full.pt`
- `student_finetuned_ema.pt`
- `best_student_kd_full*.pt`
- `MedSAM2_latest.pt` (teacher model — not included by default)

The Streamlit app will detect any available student checkpoints in the working directory and present them in the sidebar for selection.

Input and output formats
------------------------

- Input: RGB images. Inputs are resized to 320×320 for model inference by default; supply higher-resolution images when needed but be mindful of memory.
- Bounding boxes: `x1 y1 x2 y2` (pixel coordinates, top-left then bottom-right).
- Outputs: PNG masks, side-by-side comparison images, and a plain-text report containing timing and basic statistics.

Evaluation and metrics
----------------------

The tools report basic metrics useful for assessment:

- Inference time (ms)
- CPU / GPU memory usage
- Number of model parameters
- Segmentation statistics (positive pixel counts, coverage)

For formal segmentation evaluation (IoU, Dice, precision, recall), run the evaluation scripts or the training notebook with available ground-truth masks.

Troubleshooting
---------------

- Model loading failures: confirm checkpoint paths and PyTorch compatibility. The loader supports both full-model objects and state-dictionary checkpoints.
- MedSAM2 teacher load failures: ensure the `sam2` package, its configuration files, and the teacher checkpoint are available and accessible. Use the included diagnostic scripts if available.
- CUDA out-of-memory: switch to CPU inference (`--device cpu`) or use a machine with more GPU memory.

Development and contribution
----------------------------

Contributions are welcome under a standard GitHub workflow:

1. Fork the repository and create a feature branch.
2. Implement your changes, include tests or validation scripts when applicable.
3. Open a pull request with a clear description of the change and why it is needed.




