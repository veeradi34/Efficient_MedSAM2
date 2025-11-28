# MedSeg Professional - Mobile PWA

This is the Progressive Web App (PWA) version of MedSeg Professional, a medical image segmentation platform.

## Features

- Installable on mobile devices and desktops
- Offline-capable with service worker caching
- Professional medical image segmentation
- Camera capture and file upload for images
- Responsive design for mobile screens

## Setup

1. Ensure the backend is running:
   ```bash
   cd backend
   python main.py
   ```

2. The frontend is already built. To serve it:
   ```bash
   cd frontend/build
   npx serve -s .
   ```

3. Open the URL in a modern browser (Chrome recommended)

4. Click "Install" or "Add to Home Screen" to install the PWA

## Mobile Features

- **Camera Capture**: Use the "Take Photo" button to capture images directly from camera
- **File Upload**: Drag & drop or browse files as before
- **Touch-friendly**: Optimized for mobile interaction

## Development

To modify the PWA:

1. Make changes in `frontend/src/`
2. Build: `cd frontend && npm run build`
3. Serve the build folder

## Backend API

The PWA connects to the FastAPI backend running on port 8000.

## PWA Features

- Web App Manifest for installation
- Service Worker for caching
- Responsive design
- Standalone display mode

The API server will run on `http://localhost:8000`

### Access the Application

1. Open `http://localhost:3000` in your browser
2. Login with any username and password (demo mode)
3. Navigate to "Image Segmentation" to upload medical images
4. Select an AI model and configure bounding box parameters
5. Run segmentation analysis and view results

## Professional Features

### Dashboard
- System performance overview
- Recent analysis history
- Model status monitoring
- Quick action buttons

### Image Segmentation
- Drag-and-drop image upload
- Multiple AI model selection
- Bounding box configuration
- Real-time processing with metrics
- Professional results visualization

### Model Management
- View available AI models
- Model performance comparison
- Deploy and configure models
- Download and backup capabilities

### Analysis & Reports
- Performance trend analysis
- Model comparison charts
- Historical analysis records
- Export functionality

## Security & Compliance

- Secure authentication system
- CORS protection for API endpoints
- Input validation and sanitization
- Error handling and logging
- Ready for HIPAA compliance implementation

## Model Integration

The platform supports multiple AI models:

1. **Efficient Student Model v2.1**: Fast, lightweight model optimized for clinical use
2. **MedSAM2 Original**: State-of-the-art baseline for maximum accuracy
3. **Cross-Attention Model**: Experimental model with attention mechanisms

## Development

### Adding New Models

1. Implement model class in `backend/app/models/`
2. Add model configuration to ModelManager
3. Update frontend model selection interface
4. Test integration and performance

### Customizing UI

1. Modify Tailwind configuration in `frontend/tailwind.config.js`
2. Update global styles in `frontend/src/index.css`
3. Customize components in `frontend/src/components/`

### API Extensions

1. Add new endpoints in `backend/main.py`
2. Define Pydantic models for request/response
3. Update frontend services in `frontend/src/services/`

## Deployment

### Production Deployment

1. **Frontend**: Build React app and deploy to static hosting
2. **Backend**: Deploy FastAPI server with proper HTTPS configuration
3. **Models**: Host AI models on secure, compliant infrastructure
4. **Database**: Configure production database for user management

### Docker Deployment

```dockerfile
# Frontend Dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## License

This project is designed for professional medical imaging applications. Please ensure compliance with local healthcare regulations and data protection laws.

## Support

For technical support or feature requests, please contact the development team.

---

**MedSeg Professional** - Advanced Medical Image Segmentation Platform