@echo off
echo Starting Efficient MedSAM2 Web Application...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements
pip install -r requirements.txt

REM Create Streamlit config
python -c "from config import DeploymentUtils; DeploymentUtils.create_streamlit_config_dir()"

REM Launch application
streamlit run main.py --server.port 8501

pause
