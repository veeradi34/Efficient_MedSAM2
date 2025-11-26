# Efficient MedSAM2 Web Application

## 🧠 Advanced Medical Image Segmentation Platform

A professional, deployment-ready web application for medical image segmentation using efficient student models derived from MedSAM2. This platform features a futuristic UI, user authentication, real-time inference, and comprehensive performance monitoring.

## ✨ Features

### 🔐 Authentication System
- Secure user registration and login
- Password strength validation
- Session management with timeout
- User activity tracking
- SQLite database backend

### 🤖 Model Management
- Multiple efficient student model support
- Automatic model discovery and loading
- Real-time model switching
- Performance benchmarking
- Memory-optimized inference

### 🔬 Medical Image Processing
- Support for multiple formats: PNG, JPG, JPEG, BMP, TIFF, DICOM
- Advanced image enhancement (CLAHE, contrast adjustment)
- Medical imaging window/level adjustment
- Aspect-ratio preserving resize
- Real-time image validation

### 📊 Performance Monitoring
- Real-time inference time tracking
- Memory usage monitoring
- System resource tracking
- Historical performance analytics
- Comprehensive reporting

### 🎨 Futuristic UI
- Dark cyber theme with neon accents
- Glass morphism design elements
- Animated loading states
- Interactive data visualizations
- Responsive design for all devices

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (optional, but recommended)
- At least 4GB RAM
- Trained student model files

### Installation

1. **Clone or extract the Web-app folder**
2. **Navigate to the Web-app directory**
   ```bash
   cd "Web-app"
   ```

3. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\\Scripts\\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Ensure your trained models are accessible**
   - Place model files in the parent directory (Intensive Assessment)
   - Or update `MODEL_BASE_PATH` in `config.py`

6. **Run the application**
   ```bash
   streamlit run main.py
   ```

### Quick Launch Scripts

**Windows:**
```bash
launch.bat
```

**Linux/Mac:**
```bash
chmod +x launch.sh
./launch.sh
```

## 📁 Project Structure

```
Web-app/
├── main.py                 # Main application entry point
├── config.py              # Configuration and deployment settings
├── requirements.txt       # Python dependencies
├── launch.bat             # Windows launch script
├── launch.sh              # Unix launch script
├── Dockerfile             # Docker deployment configuration
├── docker-compose.yml     # Docker Compose configuration
├── components/
│   ├── __init__.py
│   ├── auth.py           # Authentication management
│   └── ui.py             # UI components and styling
├── utils/
│   ├── __init__.py
│   ├── model_manager.py   # Model loading and inference
│   ├── image_processor.py # Image processing utilities
│   └── performance_monitor.py # Performance monitoring
├── assets/               # Static assets (images, styles)
└── .streamlit/          # Streamlit configuration (auto-generated)
    └── config.toml
```

## 🏥 Supported Models

The application automatically discovers and supports the following efficient student models:

- **Student Finetuned Models**: `student_finetuned_full.pt`, `student_finetuned_ema.pt`
- **Cross-Attention Models**: `student_crossattention_full.pt`, `student_crossattention_finetuned.pt`
- **Knowledge Distillation Models**: `best_student_kd_full.pt`, `best_student_kd_full_1.pt`
- **Optimized Models**: `inference_model.pt`, `best_memory_safe_fusion_model.pt`
- **Prompt-based Models**: `best_student_prompt_full.pt`
- **Best Performing Models**: `best_student.pt`, `best_student_full.pt`

## 🖼️ Supported Image Formats

- **Standard Formats**: PNG, JPG, JPEG, BMP, TIFF
- **Medical Formats**: DICOM (.dcm)
- **Maximum File Size**: 50MB (configurable)
- **Automatic Format Detection**: Based on file extension
- **Image Enhancement**: Contrast adjustment, CLAHE filtering

## 🎯 User Guide

### 1. Registration
- Navigate to the "Register" tab
- Fill in required information (username, email, password)
- Password must meet security requirements
- Optional: Add full name and institution

### 2. Login
- Use username or email with password
- Sessions remain active for 24 hours (configurable)
- Automatic logout on browser close

### 3. Model Selection
- Browse available trained models
- View model information (size, parameters)
- Select the desired model for inference

### 4. Image Segmentation
- Upload medical image (MRI, CT scan, etc.)
- Define region of interest using bounding box sliders
- Click "Run Segmentation" for analysis
- View results with performance metrics

### 5. Results Analysis
- Segmentation overlay visualization
- Probability maps and binary masks
- Performance metrics (time, memory, coverage)
- Download results as PNG

## ⚙️ Configuration

### Environment Variables
```bash
# Server Configuration
PORT=8501
SERVER_ADDRESS=0.0.0.0

# Security
SECRET_KEY=your-secret-key-here
SESSION_TIMEOUT_HOURS=24

# Model Settings
MODEL_BASE_PATH=../
MAX_FILE_SIZE_MB=50

# Performance
ENABLE_GPU=True
MAX_CONCURRENT_USERS=10
PERFORMANCE_MONITORING=True

# Environment
ENVIRONMENT=production  # development, staging, production
DEBUG=False
```

### Streamlit Configuration
Located in `.streamlit/config.toml`:
- Server settings (port, address)
- Theme configuration (colors, fonts)
- Upload limits
- Security settings

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the container
docker build -t efficient-medsam2 .

# Run with Docker Compose
docker-compose up --build

# Or run directly
docker run -p 8501:8501 efficient-medsam2
```

### Production Deployment
```bash
# Copy model files to models directory
cp ../best_student_*.pt ./models/

# Set production environment variables
export ENVIRONMENT=production
export SECRET_KEY=your-secure-production-key

# Deploy with Docker Compose
docker-compose up -d
```

## 🔧 Development

### Adding New Models
1. Place model file in the models directory
2. Update `MODEL_PATTERNS` in `utils/model_manager.py`
3. Ensure model follows the `EfficientStudentModel` architecture

### Customizing UI
- Modify CSS variables in `main.py` `inject_custom_css()`
- Update color schemes in `components/ui.py`
- Add new components to `UIComponents` class

### Database Schema
The SQLite database includes:
- `users` table: User accounts and profiles
- `sessions` table: Active user sessions
- `user_activity` table: Activity logging

## 📊 Performance Metrics

### Inference Performance
- **Typical Speed**: 20-50ms per inference
- **Memory Usage**: 30-100MB per inference
- **Model Size**: 1-20MB depending on model
- **Throughput**: 20-50 inferences per second

### System Requirements
- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB RAM, quad-core CPU, GPU
- **Optimal**: 16GB RAM, 8-core CPU, CUDA GPU

## 🔒 Security Features

- Password hashing with PBKDF2
- CSRF protection enabled
- Session token validation
- SQL injection prevention
- File upload validation
- Size and format restrictions

## 🐛 Troubleshooting

### Common Issues

**Model Loading Failed**
- Ensure model files are in the correct directory
- Check file permissions
- Verify model file integrity

**Authentication Issues**
- Check database permissions
- Ensure SQLite is installed
- Verify password complexity requirements

**Performance Issues**
- Monitor system resources
- Enable GPU acceleration
- Reduce image size if necessary

**Connection Issues**
- Check port availability (default: 8501)
- Verify firewall settings
- Ensure no conflicting applications

### Logs and Debugging
- Application logs: `app.log`
- Streamlit logs: Terminal output
- Database errors: Check SQLite file permissions
- Performance data: Available in analytics dashboard

## 🔄 Updates and Maintenance

### Regular Tasks
- Monitor user activity and performance
- Update models when available
- Backup user database regularly
- Review and rotate security keys

### Scaling Considerations
- Use load balancer for multiple instances
- Implement Redis for session storage
- Consider PostgreSQL for production database
- Add caching for model loading

## 📝 License

This application is part of the Efficient MedSAM2 research project. Please refer to the main project license for usage terms.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request with documentation

## 📧 Support

For technical support or questions:
- Create an issue in the project repository
- Review the troubleshooting section
- Check the performance monitoring dashboard

---

## 🎉 Getting Started Checklist

- [ ] Install Python 3.9+
- [ ] Clone/extract Web-app folder
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Place model files in accessible location
- [ ] Run application: `streamlit run main.py`
- [ ] Access at `http://localhost:8501`
- [ ] Register new user account
- [ ] Select a trained model
- [ ] Upload test medical image
- [ ] Run segmentation and view results

**🚀 You're ready to use Efficient MedSAM2 for medical image segmentation!**