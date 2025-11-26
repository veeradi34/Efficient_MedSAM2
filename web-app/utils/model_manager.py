"""
Model Manager for Efficient MedSAM2 Web Application
==================================================
Handles model loading, inference, and management for efficient MedSAM2 models.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import tracemalloc
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EfficientStudentModel(nn.Module):
    """Efficient Student Model Architecture"""
    
    def __init__(self, input_channels=4, output_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            self._sep(16, 32, 2),
            self._sep(32, 64, 2),
            self._sep(64, 128, 2),
            self._sep(128, 256, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            nn.Conv2d(16, output_channels, 1),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _sep(self, in_ch, out_ch, stride):
        """Separable convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch), nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        f = self.encoder(x)
        y = self.decoder(f)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return y

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ModelManager:
    """Manages model loading, inference, and performance monitoring"""
    
    def __init__(self):
        """Initialize model manager"""
        self.device = self._setup_device()
        self.loaded_models = {}  # Cache for loaded models
        self.model_base_paths = self._get_model_base_paths()
    
    def _setup_device(self):
        """Setup compute device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    
    def _get_model_base_paths(self):
        """Get base paths to search for models"""
        current_dir = Path(__file__).parent.parent
        base_paths = [
            current_dir.parent,  # Parent directory (Intensive Assessment)
            current_dir.parent / "Github" / "models",  # Github models directory
            current_dir,  # Current directory
        ]
        return [p for p in base_paths if p.exists()]
    
    def get_available_models(self):
        """Get list of available student model files"""
        # Define model patterns with descriptions
        model_patterns = {
            # Latest trained models from novelty-assessment notebook
            "student_finetuned_full.pt": "Student Finetuned (Full Training)",
            "student_finetuned_ema.pt": "Student Finetuned (EMA)",
            "student_crossattention_full.pt": "Student Cross-Attention (Full)",
            "student_crossattention_finetuned.pt": "Student Cross-Attention (Finetuned)",
            "inference_model.pt": "Inference Model (Optimized)",
            "best_memory_safe_fusion_model.pt": "Memory-Safe Fusion Model",
            
            # Best model variations
            "best_student_prompt_full.pt": "Best Student (Prompt-based)",
            "best_student_kd_full_1.pt": "Best Student (Knowledge Distillation v1)",
            "best_student_kd_full.pt": "Best Student (Knowledge Distillation)",
            "best_student_full_1.pt": "Best Student (Full v1)",
            "best_student_full.pt": "Best Student (Full)",
            "best_student_1.pt": "Best Student (v1)",
            "best_student.pt": "Best Student",
            
            # Additional models
            "fusion_model_info.pt": "Fusion Model (Info)",
        }
        
        available_models = []
        
        # Search in all base paths
        for base_path in self.model_base_paths:
            for model_filename, description in model_patterns.items():
                model_path = base_path / model_filename
                if model_path.exists() and model_path.is_file():
                    try:
                        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
                        available_models.append({
                            'path': str(model_path),
                            'filename': model_filename,
                            'description': description,
                            'size_mb': file_size,
                            'display': f"{description} ({file_size:.1f} MB)",
                            'base_path': str(base_path)
                        })
                    except Exception as e:
                        continue  # Skip files that can't be accessed
        
        # Remove duplicates (prefer models in earlier base paths)
        seen_filenames = set()
        unique_models = []
        for model in available_models:
            if model['filename'] not in seen_filenames:
                unique_models.append(model)
                seen_filenames.add(model['filename'])
        
        # Sort by size (larger models first, likely better trained)
        unique_models.sort(key=lambda x: x['size_mb'], reverse=True)
        
        return unique_models
    
    def load_model(self, model_path: str):
        """Load a student model from the specified path"""
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]
        
        # Add EfficientStudentModel to global namespace for torch.load
        import __main__
        __main__.EfficientStudentModel = EfficientStudentModel
        
        # Load with torch.load
        model_data = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle different model storage formats
        if hasattr(model_data, 'eval'):
            # Model object directly
            model = model_data.to(self.device)
        elif isinstance(model_data, dict):
            # State dict or model info
            model = EfficientStudentModel(input_channels=4).to(self.device)
            
            if 'model_state_dict' in model_data:
                model.load_state_dict(model_data['model_state_dict'])
            elif 'state_dict' in model_data:
                model.load_state_dict(model_data['state_dict'])
            else:
                # Assume the dict is the state dict itself
                model.load_state_dict(model_data)
        else:
            # Create new model and load state
            model = EfficientStudentModel(input_channels=4).to(self.device)
            if hasattr(model_data, 'state_dict'):
                model.load_state_dict(model_data.state_dict())
        
        model.eval()
        
        # Cache the loaded model
        self.loaded_models[model_path] = model
        
        return model
    
    def make_soft_box_prior(self, h, w, box, pad=2, blur=5):
        """Create soft bounding box prior"""
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1-pad)
        y1 = max(0, y1-pad)
        x2 = min(w-1, x2+pad)
        y2 = min(h-1, y2+pad)
        
        prior = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        prior[y1:y2+1, x1:x2+1] = 1.0
        
        if blur and blur > 0:
            k = blur if blur % 2 == 1 else blur + 1
            prior = F.avg_pool2d(
                prior.unsqueeze(0).unsqueeze(0), 
                kernel_size=k, stride=1, padding=k//2
            ).squeeze()
        
        return prior.clamp(0, 1)
    
    def run_inference(self, model, img_tensor, bbox_coords):
        """Run model inference with performance monitoring"""
        try:
            # Start performance monitoring
            tracemalloc.start()
            start_time = time.time()
            
            # Prepare input
            H, W = img_tensor.shape[-2:]
            x1, y1, x2, y2 = bbox_coords
            
            # Convert to pixel coordinates
            x1_px, y1_px = int(x1 * W), int(y1 * H)
            x2_px, y2_px = int(x2 * W), int(y2 * H)
            
            # Create soft box prior
            soft_prior = self.make_soft_box_prior(H, W, (x1_px, y1_px, x2_px, y2_px))
            
            # Create 4-channel input (RGB + prior)
            img_4ch = torch.cat([img_tensor.to(self.device), soft_prior.unsqueeze(0)], dim=0)
            
            # Run inference
            with torch.no_grad():
                output = torch.sigmoid(model(img_4ch.unsqueeze(0)))[0, 0]
            
            # End performance monitoring
            end_time = time.time()
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
            memory_used = peak_memory / (1024 * 1024)  # Convert to MB
            
            # Convert output to numpy
            mask = output.cpu().numpy()
            
            return mask, (x1_px, y1_px, x2_px, y2_px), inference_time, memory_used
            
        except Exception as e:
            st.error(f"❌ Inference error: {str(e)}")
            # Return dummy results
            H, W = img_tensor.shape[-2:]
            dummy_mask = np.zeros((H, W))
            return dummy_mask, (0, 0, W, H), 0.0, 0.0
    
    def get_model_info(self, model_path: str):
        """Get detailed information about a model"""
        try:
            model = self.load_model(model_path)
            
            # Get parameter count
            param_count = model.count_trainable_parameters()
            
            # Get model size
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            # Get model architecture info
            architecture_info = {
                'total_parameters': param_count,
                'file_size_mb': file_size,
                'input_channels': 4,
                'output_channels': 1,
                'architecture': 'EfficientStudentModel',
                'encoder_layers': 5,
                'decoder_layers': 5
            }
            
            return architecture_info
            
        except Exception as e:
            return None
    
    def benchmark_model(self, model_path: str, num_runs: int = 10):
        """Benchmark model performance"""
        try:
            model = self.load_model(model_path)
            
            # Create dummy input
            dummy_img = torch.randn(3, 320, 320)
            dummy_bbox = (0.2, 0.2, 0.8, 0.8)
            
            inference_times = []
            memory_usages = []
            
            # Warm up
            for _ in range(3):
                _, _, _, _ = self.run_inference(model, dummy_img, dummy_bbox)
            
            # Benchmark runs
            for _ in range(num_runs):
                _, _, inf_time, mem_usage = self.run_inference(model, dummy_img, dummy_bbox)
                inference_times.append(inf_time)
                memory_usages.append(mem_usage)
            
            return {
                'mean_inference_time': np.mean(inference_times),
                'std_inference_time': np.std(inference_times),
                'min_inference_time': np.min(inference_times),
                'max_inference_time': np.max(inference_times),
                'mean_memory_usage': np.mean(memory_usages),
                'std_memory_usage': np.std(memory_usages)
            }
            
        except Exception as e:
            return None
    
    def clear_cache(self):
        """Clear loaded model cache"""
        self.loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def get_device_info(self):
        """Get device information"""
        device_info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            device_info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
                'gpu_memory_allocated': torch.cuda.memory_allocated(0) / (1024**3),  # GB
                'gpu_memory_reserved': torch.cuda.memory_reserved(0) / (1024**3),  # GB
            })
        
        return device_info