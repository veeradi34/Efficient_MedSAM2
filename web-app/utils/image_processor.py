"""
Image Processor for Efficient MedSAM2 Web Application
===================================================
Handles image processing, validation, and format conversion for medical images.
"""

import streamlit as st
import numpy as np
import torch
from PIL import Image
import cv2
import io
import os
from typing import Tuple, Optional, Union

# Try to import PIL modules with fallback
try:
    from PIL import ImageEnhance
    PIL_ENHANCE_AVAILABLE = True
except ImportError:
    PIL_ENHANCE_AVAILABLE = False
    st.warning("⚠️ PIL.ImageEnhance not available - image enhancement will be skipped")

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

class ImageProcessor:
    """Handles image processing for medical image segmentation"""
    
    def __init__(self, target_size=(320, 320)):
        """Initialize image processor"""
        self.target_size = target_size
        self.supported_formats = {
            'png': 'PNG Image',
            'jpg': 'JPEG Image', 
            'jpeg': 'JPEG Image',
            'bmp': 'Bitmap Image',
            'tiff': 'TIFF Image',
            'tif': 'TIFF Image',
            'dcm': 'DICOM Medical Image'
        }
    
    def validate_image(self, uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded image file"""
        try:
            # Check file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension not in self.supported_formats:
                return False, f"Unsupported file format. Supported: {', '.join(self.supported_formats.keys())}"
            
            # Check file size (max 50MB)
            if uploaded_file.size > 50 * 1024 * 1024:
                return False, "File too large. Maximum size: 50MB"
            
            return True, "Valid image file"
            
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    def process_dicom(self, uploaded_file) -> Optional[np.ndarray]:
        """Process DICOM medical image"""
        if not PYDICOM_AVAILABLE:
            st.error("❌ DICOM support not available. Please install pydicom: pip install pydicom")
            return None
            
        try:
            # Read DICOM file
            dicom_data = pydicom.dcmread(io.BytesIO(uploaded_file.read()))
            
            # Extract pixel array
            pixel_array = dicom_data.pixel_array
            
            # Handle different DICOM formats
            if len(pixel_array.shape) == 2:
                # Grayscale DICOM
                # Normalize to 0-255 range
                pixel_array = pixel_array.astype(np.float32)
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255
                pixel_array = pixel_array.astype(np.uint8)
                
                # Convert to RGB
                img_array = np.stack([pixel_array] * 3, axis=-1)
            elif len(pixel_array.shape) == 3:
                # Color DICOM
                img_array = pixel_array
                if img_array.shape[-1] > 3:
                    img_array = img_array[:, :, :3]
            else:
                return None
            
            return img_array
            
        except Exception as e:
            st.error(f"DICOM processing error: {str(e)}")
            return None
    
    def process_standard_image(self, uploaded_file) -> Optional[np.ndarray]:
        """Process standard image formats (PNG, JPG, etc.)"""
        try:
            # Open with PIL
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Handle transparency
                    background = Image.new('RGB', image.size, (0, 0, 0))  # Black background
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            return img_array
            
        except Exception as e:
            st.error(f"Image processing error: {str(e)}")
            return None
    
    def enhance_medical_image(self, img_array: np.ndarray, 
                             contrast_factor: float = 1.2, 
                             brightness_factor: float = 1.1) -> np.ndarray:
        """Enhance medical image for better visualization"""
        if not PIL_ENHANCE_AVAILABLE:
            # Return original if PIL enhancement is not available
            return img_array
            
        try:
            # Convert to PIL for enhancement
            img_pil = Image.fromarray(img_array)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(contrast_factor)
            
            # Enhance brightness
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(brightness_factor)
            
            # Convert back to numpy
            enhanced_array = np.array(img_pil)
            
            return enhanced_array
            
        except Exception as e:
            return img_array  # Return original if enhancement fails
    
    def apply_clahe(self, img_array: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)"""
        try:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to the L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            return img_array  # Return original if CLAHE fails
    
    def resize_image(self, img_array: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """Resize image to target size"""
        try:
            if maintain_aspect:
                # Calculate resize dimensions maintaining aspect ratio
                h, w = img_array.shape[:2]
                target_h, target_w = self.target_size
                
                # Calculate scale factor
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize image
                resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Create target size image with padding
                result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # Center the image
                start_y = (target_h - new_h) // 2
                start_x = (target_w - new_w) // 2
                result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
                
                return result
            else:
                # Direct resize without maintaining aspect ratio
                return cv2.resize(img_array, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
        except Exception as e:
            st.error(f"Resize error: {str(e)}")
            return img_array
    
    def normalize_image(self, img_array: np.ndarray) -> torch.Tensor:
        """Convert image to normalized tensor"""
        try:
            # Convert to float and normalize to [0, 1]
            img_float = img_array.astype(np.float32) / 255.0
            
            # Convert to PyTorch tensor (C, H, W)
            img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)
            
            return img_tensor
            
        except Exception as e:
            st.error(f"Normalization error: {str(e)}")
            return None
    
    def process_image(self, uploaded_file, enhance: bool = True, apply_clahe_filter: bool = False):
        """Main image processing pipeline"""
        # Validate image
        is_valid, message = self.validate_image(uploaded_file)
        if not is_valid:
            st.error(message)
            return None, None, None
        
        # Determine file type and process accordingly
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'dcm':
            img_array = self.process_dicom(uploaded_file)
        else:
            img_array = self.process_standard_image(uploaded_file)
        
        if img_array is None:
            return None, None, None
        
        # Store original for display
        img_original = Image.fromarray(img_array)
        
        # Apply enhancements if requested
        if enhance:
            img_array = self.enhance_medical_image(img_array)
        
        if apply_clahe_filter:
            img_array = self.apply_clahe(img_array)
        
        # Resize for model input
        img_resized = self.resize_image(img_array, maintain_aspect=True)
        
        # Convert to tensor
        img_tensor = self.normalize_image(img_resized)
        
        if img_tensor is None:
            return None, None, None
        
        return img_tensor, img_resized, img_original
    
    def get_image_info(self, img_array: np.ndarray) -> dict:
        """Get comprehensive image information"""
        try:
            info = {
                'shape': img_array.shape,
                'dtype': str(img_array.dtype),
                'min_value': float(img_array.min()),
                'max_value': float(img_array.max()),
                'mean_value': float(img_array.mean()),
                'std_value': float(img_array.std()),
                'size_mb': img_array.nbytes / (1024 * 1024)
            }
            
            # Calculate histogram information
            if len(img_array.shape) == 3:
                # Color image
                info['channels'] = img_array.shape[2]
                info['is_color'] = True
                
                # Per-channel statistics
                for i, channel in enumerate(['Red', 'Green', 'Blue']):
                    channel_data = img_array[:, :, i]
                    info[f'{channel.lower()}_mean'] = float(channel_data.mean())
                    info[f'{channel.lower()}_std'] = float(channel_data.std())
            else:
                # Grayscale image
                info['channels'] = 1
                info['is_color'] = False
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def create_thumbnail(self, img_array: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """Create thumbnail of the image"""
        try:
            thumbnail = cv2.resize(img_array, size, interpolation=cv2.INTER_AREA)
            return thumbnail
        except Exception as e:
            return img_array
    
    def apply_medical_window(self, img_array: np.ndarray, 
                           window_center: float, window_width: float) -> np.ndarray:
        """Apply medical imaging window/level adjustment"""
        try:
            # Convert to float for processing
            img_float = img_array.astype(np.float32)
            
            # Calculate window bounds
            min_bound = window_center - window_width / 2
            max_bound = window_center + window_width / 2
            
            # Apply windowing
            windowed = np.clip(img_float, min_bound, max_bound)
            
            # Normalize to 0-255
            if max_bound > min_bound:
                windowed = (windowed - min_bound) / (max_bound - min_bound) * 255
            
            return windowed.astype(np.uint8)
            
        except Exception as e:
            return img_array