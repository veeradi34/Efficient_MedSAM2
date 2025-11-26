Efficient MedSAM2 — Student Model and Assessment Interface
==========================================================

Overview
--------

This repository provides an efficient, prompt-aware student model trained for medical image segmentation, together with tools to evaluate and compare it against the original MedSAM2 teacher model. The student models use a 4-channel input (RGB + soft box prior) and are optimized for reduced memory usage and faster inference while supporting prompt-driven segmentation via bounding boxes.

Repository contents
-------------------

### Core Files
- `streamlit_comparison_app.py` — Streamlit web application for side-by-side comparison of the student model and MedSAM2.
- `assessment_interface.py` — Alternate Streamlit demo interface.
- `cli_assessment.py` — Command-line tool for running inference and saving results.
- `prompt_based_learning.ipynb` — Training and experiment notebook.
- `requirements.txt` — Python dependencies required to run the tools.
- `*.pt` — Trained model checkpoints produced during experiments (student and teacher variants).

### Code Files
- `code files/` — Directory containing training notebooks and data extraction scripts.
  - `extract_images_and_masks.py` — Script for preprocessing medical images and masks.
  - `prompt-based-learning_finetuned.ipynb` — Jupyter notebook for prompt-based model training.
  - `student-medsam2-supervised-learning_followedbykd_2.ipynb` — Advanced training pipeline with knowledge distillation.

### Interface
- `Interface/` — Simple interface tools for model comparison.
  - `streamlit_comparison_app.py` — Basic Streamlit comparison interface.

### Models
- `models/` — Directory containing trained model checkpoints.
  - `best_student_kd_full_1.pt` — Best performing knowledge distillation model.
  - `best_student_prompt_full.pt` — Best prompt-based learning model.
  - `student_finetuned_ema.pt` — EMA (Exponential Moving Average) finetuned model.
  - `student_finetuned_full.pt` — Fully finetuned student model.

### Web Applications

#### Web-app/
Professional Streamlit-based medical segmentation platform with advanced features:
- **Authentication System**: Secure user registration, login, and session management
- **Model Management**: Multiple student model support with automatic discovery
- **Medical Image Processing**: Support for PNG, JPG, JPEG, BMP, TIFF, DICOM formats
- **Performance Monitoring**: Real-time inference tracking and system resource monitoring
- **Futuristic UI**: Dark cyber theme with glass morphism design elements
- **Docker Support**: Production-ready deployment with Docker and docker-compose

Key files:
- `main.py` — Main Streamlit application with authentication and model management
- `Dockerfile` & `docker-compose.yml` — Containerization for production deployment
- `launch.bat` — Windows batch script for easy local deployment
- `components/` — Modular UI components (auth, ui)
- `utils/` — Backend utilities (image processing, model management, performance monitoring)

#### Website/
Modern React + FastAPI web application for professional medical environments:
- **Professional Medical UI**: Clean, white background design for healthcare professionals
- **React Frontend**: Modern React 18 with Tailwind CSS and responsive design
- **FastAPI Backend**: High-performance API server with PyTorch integration
- **Real-time Analysis**: Fast image segmentation with comprehensive performance metrics
- **HIPAA-Ready Design**: Security-focused architecture for medical data handling
- **Model Analytics**: Advanced reporting and performance analytics dashboard

Frontend structure:
- `frontend/src/components/` — Reusable UI components (BoundingBoxDrawer, Header, Sidebar)
- `frontend/src/pages/` — Application pages (Dashboard, Segmentation, Analysis, Models, Login)
- `frontend/src/services/` — API integration and authentication services

Backend structure:
- `backend/main.py` — FastAPI server with medical image processing endpoints
- `backend/app/` — Modular backend architecture with API routes and models

Installation
------------

1. Create and activate a virtual environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\activate

Python 3.8 or newer is recommended. For GPU acceleration, install a CUDA-compatible PyTorch build.

Deployment Options
-----------------

### Option 1: Basic Streamlit Interface (Quick Start)

1. Launch the comparison app:

   streamlit run streamlit_comparison_app.py

2. Open the local URL reported by Streamlit (typically `http://localhost:8501`).

### Option 2: Professional Web Application (Web-app/)

For a full-featured medical segmentation platform with authentication and advanced UI:

**Local Development:**
```
cd Web-app
pip install -r requirements.txt
python main.py
```

**Docker Deployment:**
```
cd Web-app
docker-compose up -d
```

**Windows Quick Launch:**
```
cd Web-app
launch.bat
```

### Option 3: React + FastAPI Platform (Website/)

For a modern web application with separate frontend and backend:

**Backend Setup:**
```
cd Website/backend
pip install -r requirements.txt
python main.py
```

**Frontend Setup:**
```
cd Website/frontend
npm install
npm start
```

The React app will run on `http://localhost:3000` and the API on `http://localhost:8000`.

Usage Instructions
-----------------

### Basic Streamlit Interface
1. Select a student checkpoint (e.g., `student_finetuned_full.pt` or `best_student_prompt_full.pt`)
2. Upload an input image
3. Draw or enter a bounding box to run prompt-based segmentation
4. Use the Reload Models button after adding new checkpoint files

### Web-app Platform
1. Create an account or login with existing credentials
2. Select from available student models in the model management panel
3. Upload medical images (supports PNG, JPG, JPEG, BMP, TIFF, DICOM)
4. Draw bounding boxes for region-of-interest segmentation
5. View real-time performance metrics and segmentation results
6. Access historical analytics and performance reports

### React + FastAPI Website
1. Navigate to the Dashboard for system overview
2. Use the Segmentation page for interactive image analysis
3. Access the Models page for AI model management
4. View comprehensive analytics in the Analysis section
5. Professional UI optimized for healthcare environments

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




