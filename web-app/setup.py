#!/usr/bin/env python3
"""
Quick Setup Script for Efficient MedSAM2 Web Application
========================================================
Automated setup and validation for the medical segmentation platform.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class QuickSetup:
    """Automated setup for the web application"""
    
    def __init__(self):
        self.platform = platform.system()
        self.python_cmd = "python" if self.platform == "Windows" else "python3"
        self.current_dir = Path(__file__).parent
        self.parent_dir = self.current_dir.parent
        
    def print_banner(self):
        """Print setup banner"""
        print("=" * 60)
        print("🧠 EFFICIENT MEDSAM2 WEB APPLICATION SETUP")
        print("=" * 60)
        print(f"Platform: {self.platform}")
        print(f"Python: {sys.version}")
        print(f"Directory: {self.current_dir}")
        print("=" * 60)
    
    def check_python_version(self):
        """Check Python version compatibility"""
        print("🐍 Checking Python version...")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 9:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
            return True
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.9+")
            return False
    
    def check_dependencies(self):
        """Check if all dependencies can be imported"""
        print("📦 Checking dependencies...")
        
        dependencies = [
            ("streamlit", "Streamlit web framework"),
            ("torch", "PyTorch deep learning"),
            ("numpy", "Numerical computing"),
            ("PIL", "Image processing"),
            ("cv2", "Computer vision"),
            ("matplotlib", "Data visualization"),
            ("plotly", "Interactive plots"),
            ("sqlite3", "Database")
        ]
        
        missing = []
        
        for dep, description in dependencies:
            try:
                if dep == "cv2":
                    import cv2
                elif dep == "PIL":
                    from PIL import Image
                else:
                    __import__(dep)
                print(f"✅ {dep} - {description}")
            except ImportError:
                print(f"❌ {dep} - {description} (MISSING)")
                missing.append(dep)
        
        return len(missing) == 0, missing
    
    def install_dependencies(self):
        """Install missing dependencies"""
        print("📥 Installing dependencies...")
        
        try:
            subprocess.check_call([
                self.python_cmd, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def check_model_files(self):
        """Check for available model files"""
        print("🤖 Checking for model files...")
        
        model_patterns = [
            "student_finetuned_full.pt",
            "student_finetuned_ema.pt",
            "best_student*.pt",
            "inference_model.pt"
        ]
        
        found_models = []
        search_paths = [
            self.parent_dir,
            self.parent_dir / "Github" / "models",
            self.current_dir
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for pattern in model_patterns:
                    if "*" in pattern:
                        # Handle wildcard patterns
                        base_pattern = pattern.replace("*", "")
                        models = list(search_path.glob(pattern))
                        found_models.extend(models)
                    else:
                        model_path = search_path / pattern
                        if model_path.exists():
                            found_models.append(model_path)
        
        # Remove duplicates
        found_models = list(set(found_models))
        
        if found_models:
            print(f"✅ Found {len(found_models)} model file(s):")
            for model in found_models[:5]:  # Show first 5
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"   📁 {model.name} ({size_mb:.1f} MB)")
            if len(found_models) > 5:
                print(f"   ... and {len(found_models) - 5} more")
            return True
        else:
            print("⚠️ No model files found")
            print("   Please ensure your trained models are in the parent directory")
            return False
    
    def create_config_files(self):
        """Create necessary configuration files"""
        print("⚙️ Creating configuration files...")
        
        try:
            from config import DeploymentUtils
            
            # Create Streamlit config directory
            config_path = DeploymentUtils.create_streamlit_config_dir()
            print(f"✅ Created Streamlit config: {config_path}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to create config files: {e}")
            return False
    
    def test_application(self):
        """Test if the application can start"""
        print("🧪 Testing application startup...")
        
        try:
            # Try importing main modules
            sys.path.insert(0, str(self.current_dir))
            
            from components.auth import AuthManager
            from utils.model_manager import ModelManager
            from utils.image_processor import ImageProcessor
            from utils.performance_monitor import PerformanceMonitor
            
            # Test initialization
            auth_manager = AuthManager()
            model_manager = ModelManager()
            image_processor = ImageProcessor()
            performance_monitor = PerformanceMonitor()
            
            print("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Application test failed: {e}")
            return False
    
    def create_sample_user(self):
        """Create a sample user account"""
        print("👤 Creating sample user account...")
        
        try:
            from components.auth import AuthManager
            auth_manager = AuthManager()
            
            # Create demo user
            success, message = auth_manager.register_user(
                username="demo",
                email="demo@medsam2.com",
                password="MedSAM2Demo123!",
                full_name="Demo User",
                institution="Medical AI Research"
            )
            
            if success:
                print("✅ Sample user created:")
                print("   Username: demo")
                print("   Password: MedSAM2Demo123!")
                print("   Email: demo@medsam2.com")
            else:
                print(f"ℹ️ Sample user: {message}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create sample user: {e}")
            return False
    
    def print_launch_instructions(self):
        """Print final launch instructions"""
        print("\n" + "=" * 60)
        print("🚀 SETUP COMPLETE - LAUNCH INSTRUCTIONS")
        print("=" * 60)
        
        print("\nQuick Launch Options:")
        print("1. 🎯 Launch script:")
        if self.platform == "Windows":
            print("   launch.bat")
        else:
            print("   ./launch.sh")
        
        print("\n2. 🐳 Docker (if available):")
        print("   docker-compose up --build")
        
        print("\n3. 📱 Direct launch:")
        print("   streamlit run main.py")
        
        print("\n4. 🌐 Access the application:")
        print("   http://localhost:8501")
        
        print("\nLogin Information:")
        print("   Username: demo")
        print("   Password: MedSAM2Demo123!")
        
        print("\nNext Steps:")
        print("• Upload a medical image (MRI, CT scan)")
        print("• Select a trained model")
        print("• Define region of interest")
        print("• Run segmentation analysis")
        
        print("\n" + "=" * 60)
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_banner()
        
        # Setup steps
        steps = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Model Files", self.check_model_files),
            ("Configuration", self.create_config_files),
            ("Application Test", self.test_application),
            ("Sample User", self.create_sample_user)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            print(f"\n🔧 {step_name}...")
            try:
                success = step_func()
                if step_name == "Dependencies" and not success[0]:
                    print("📥 Attempting to install missing dependencies...")
                    success = self.install_dependencies()
                    if success:
                        # Recheck dependencies
                        success, missing = self.check_dependencies()
                        success = success[0] if isinstance(success, tuple) else success
                
                if not success:
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                failed_steps.append(step_name)
        
        # Summary
        print(f"\n📊 Setup Summary:")
        print(f"✅ Completed: {len(steps) - len(failed_steps)}/{len(steps)} steps")
        
        if failed_steps:
            print(f"❌ Failed: {', '.join(failed_steps)}")
            print("\nℹ️ The application may still work with limited functionality")
        else:
            print("🎉 All setup steps completed successfully!")
        
        self.print_launch_instructions()

if __name__ == "__main__":
    setup = QuickSetup()
    setup.run_setup()