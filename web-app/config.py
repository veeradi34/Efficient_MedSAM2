#!/usr/bin/env python3
"""
Deployment Configuration for Efficient MedSAM2 Web Application
=============================================================
Configuration file for production deployment settings.
"""

import os
import streamlit as st

class Config:
    """Application configuration"""
    
    # Application Settings
    APP_TITLE = "Efficient MedSAM2 - Medical AI Platform"
    APP_ICON = "🧠"
    APP_VERSION = "1.0.0"
    
    # Server Settings
    SERVER_PORT = int(os.getenv("PORT", 8501))
    SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "0.0.0.0")
    
    # Database Settings
    DATABASE_PATH = os.getenv("DATABASE_PATH", "users.db")
    
    # Security Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    SESSION_TIMEOUT_HOURS = int(os.getenv("SESSION_TIMEOUT_HOURS", 24))
    
    # Model Settings
    MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "../")
    DEFAULT_MODEL_SIZE = (320, 320)
    SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "bmp", "tiff", "dcm"]
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))
    
    # Performance Settings
    ENABLE_GPU = os.getenv("ENABLE_GPU", "True").lower() == "true"
    MAX_CONCURRENT_USERS = int(os.getenv("MAX_CONCURRENT_USERS", 10))
    PERFORMANCE_MONITORING = os.getenv("PERFORMANCE_MONITORING", "True").lower() == "true"
    
    # UI Settings
    THEME = os.getenv("THEME", "dark")
    ENABLE_ANIMATIONS = os.getenv("ENABLE_ANIMATIONS", "True").lower() == "true"
    
    # Logging Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")
    
    # Deployment Settings
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    @classmethod
    def is_production(cls):
        return cls.ENVIRONMENT == "production"
    
    @classmethod
    def is_development(cls):
        return cls.ENVIRONMENT == "development"

class StreamlitConfig:
    """Streamlit-specific configuration"""
    
    @staticmethod
    def configure_page():
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=Config.APP_TITLE,
            page_icon=Config.APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': f"# {Config.APP_TITLE}\nVersion {Config.APP_VERSION}\nAdvanced Medical Image Segmentation Platform"
            }
        )
    
    @staticmethod
    def configure_server():
        """Configure Streamlit server settings (for .streamlit/config.toml)"""
        return f"""
[server]
port = {Config.SERVER_PORT}
address = "{Config.SERVER_ADDRESS}"
maxUploadSize = {Config.MAX_FILE_SIZE_MB}
enableCORS = true
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#00f5ff"
backgroundColor = "#0a0a0a"
secondaryBackgroundColor = "#1a1a2e"
textColor = "#ffffff"

[logger]
level = "{Config.LOG_LEVEL}"

[runner]
magicEnabled = true
"""

class DeploymentUtils:
    """Utilities for deployment"""
    
    @staticmethod
    def create_streamlit_config_dir():
        """Create .streamlit directory with configuration"""
        import os
        
        config_dir = ".streamlit"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # Write config.toml
        config_path = os.path.join(config_dir, "config.toml")
        with open(config_path, "w") as f:
            f.write(StreamlitConfig.configure_server())
        
        return config_path
    
    @staticmethod
    def create_dockerfile():
        """Create Dockerfile for containerized deployment"""
        dockerfile_content = f"""# Efficient MedSAM2 Web Application Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models logs data

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV PORT={Config.SERVER_PORT}

# Expose port
EXPOSE {Config.SERVER_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{Config.SERVER_PORT}/_stcore/health

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port={Config.SERVER_PORT}", "--server.address=0.0.0.0"]
"""
        
        return dockerfile_content
    
    @staticmethod
    def create_docker_compose():
        """Create docker-compose.yml for local deployment"""
        compose_content = f"""version: '3.8'

services:
  efficient-medsam2:
    build: .
    ports:
      - "{Config.SERVER_PORT}:{Config.SERVER_PORT}"
    environment:
      - ENVIRONMENT=production
      - DATABASE_PATH=/app/data/users.db
      - MODEL_BASE_PATH=/app/models
      - SECRET_KEY=your-production-secret-key-here
      - MAX_FILE_SIZE_MB={Config.MAX_FILE_SIZE_MB}
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{Config.SERVER_PORT}/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  app_data:
  app_logs:
"""
        
        return compose_content
    
    @staticmethod
    def create_launch_script():
        """Create launch script for easy deployment"""
        if os.name == 'nt':  # Windows
            script_content = f"""@echo off
echo Starting Efficient MedSAM2 Web Application...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\\Scripts\\activate

REM Install requirements
pip install -r requirements.txt

REM Create Streamlit config
python -c "from config import DeploymentUtils; DeploymentUtils.create_streamlit_config_dir()"

REM Launch application
streamlit run main.py --server.port {Config.SERVER_PORT}

pause
"""
            filename = "launch.bat"
        else:  # Unix/Linux/Mac
            script_content = f"""#!/bin/bash
echo "Starting Efficient MedSAM2 Web Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create Streamlit config
python -c "from config import DeploymentUtils; DeploymentUtils.create_streamlit_config_dir()"

# Launch application
streamlit run main.py --server.port {Config.SERVER_PORT}
"""
            filename = "launch.sh"
        
        return script_content, filename

def setup_logging():
    """Setup application logging"""
    import logging
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_environment():
    """Validate deployment environment"""
    errors = []
    
    # Check required environment variables in production
    if Config.is_production():
        required_vars = ['SECRET_KEY', 'DATABASE_PATH']
        for var in required_vars:
            if not os.getenv(var):
                errors.append(f"Missing required environment variable: {var}")
    
    # Check model directory
    if not os.path.exists(Config.MODEL_BASE_PATH):
        errors.append(f"Model base path does not exist: {Config.MODEL_BASE_PATH}")
    
    return errors

if __name__ == "__main__":
    # Create deployment files
    utils = DeploymentUtils()
    
    print("Creating deployment configuration files...")
    
    # Create Streamlit config
    config_path = utils.create_streamlit_config_dir()
    print(f"Created Streamlit config: {config_path}")
    
    # Create Dockerfile
    dockerfile_content = utils.create_dockerfile()
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    print("Created Dockerfile")
    
    # Create docker-compose.yml
    compose_content = utils.create_docker_compose()
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    print("Created docker-compose.yml")
    
    # Create launch script
    script_content, filename = utils.create_launch_script()
    with open(filename, "w") as f:
        f.write(script_content)
    print(f"Created launch script: {filename}")
    
    # Make launch script executable on Unix systems
    if not filename.endswith('.bat'):
        import stat
        os.chmod(filename, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    
    print("\nDeployment configuration complete!")
    print("\nTo run the application:")
    print(f"1. Run the launch script: {filename}")
    print("2. Or use Docker: docker-compose up --build")
    print("3. Or run directly: streamlit run main.py")