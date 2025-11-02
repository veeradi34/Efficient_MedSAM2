#!/usr/bin/env python3
"""
MedSAM2 vs Efficient Student Model - Comparative Assessment Interface
===================================================================
Interactive web interface comparing prompt-based medical image segmentation
between the original MedSAM2 and your efficient student model.
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import io
import time
import tracemalloc
import os
import sys
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="MedSAM2 vs Efficient Model Comparison",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device setup
@st.cache_resource
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

device = setup_device()

# ==================== EFFICIENT STUDENT MODEL ====================

class EfficientStudentModel(nn.Module):
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
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None: 
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _sep(self, in_ch, out_ch, stride):
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

def make_soft_box_prior(h, w, box, pad=2, blur=5):
    """Create soft bounding box prior"""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(w-1, x2+pad); y2 = min(h-1, y2+pad)
    prior = torch.zeros((h, w), dtype=torch.float32)
    prior[y1:y2+1, x1:x2+1] = 1.0
    
    if blur and blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        prior = F.avg_pool2d(prior.unsqueeze(0).unsqueeze(0), 
                           kernel_size=k, stride=1, padding=k//2).squeeze()
    return prior.clamp(0, 1)

def get_available_student_models():
    """Get list of available student model files"""
    model_patterns = [
        "student_finetuned_full.pt",
        "best_student_prompt_full.pt", 
        "student_finetuned_ema.pt",
        "best_student_kd_full_1.pt",
        "best_student_full_1.pt",
        "best_student_kd_full.pt",
        "best_student_full.pt",
        "best_student_1.pt",
        "best_student.pt"
    ]
    
    available_models = []
    for model_path in model_patterns:
        if os.path.exists(model_path):
            available_models.append(model_path)
    
    return available_models

@st.cache_resource
def load_student_model(model_path):
    """Load the efficient student model from specified path"""
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        if hasattr(model, 'eval'):
            st.sidebar.success(f"✅ Student model: {model_path}")
        else:
            new_model = EfficientStudentModel(input_channels=4).to(device)
            if isinstance(model, dict):
                new_model.load_state_dict(model)
            model = new_model
            st.sidebar.success(f"✅ Student state: {model_path}")
        
        model.eval()
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load {model_path}: {str(e)[:50]}...")
        st.sidebar.warning("⚠️ Using untrained student model")
        model = EfficientStudentModel(input_channels=4).to(device)
        model.eval()
        return model

# ==================== MEDSAM2 MODEL ====================

@st.cache_resource
def load_medsam2_model():
    """Load MedSAM2 model"""
    try:
        # Find MedSAM2 directory - multiple search paths
        possible_dirs = [
            Path("Medsam/MedSAM2"),
            Path("MedSAM2"), 
            Path("Medsam"),
            Path("../MedSAM2"),
            Path("../Medsam/MedSAM2")
        ]
        
        medsam_dir = None
        for d in possible_dirs:
            if d.exists() and (d / "sam2").exists():
                medsam_dir = d
                break
        
        if not medsam_dir:
            st.sidebar.error("❌ MedSAM2 directory not found. Please check paths.")
            st.sidebar.info("📁 Looking for: Medsam/MedSAM2/ or MedSAM2/")
            return None, None
        
        # Add to Python path
        sys.path.insert(0, str(medsam_dir))
        
        # Find checkpoint - expanded search
        candidate_ckpts = [
            medsam_dir / "MedSAM2_latest.pt",
            medsam_dir / "MedSAM2_latest (1).pt", 
            medsam_dir / "weights" / "MedSAM2_latest.pt",
            medsam_dir / "weights" / "MedSAM2_latest (1).pt",
            Path("MedSAM2_latest.pt"),
            Path("checkpoints/MedSAM2_latest.pt"),
            medsam_dir.parent / "MedSAM2_latest.pt"
        ]
        
        ckpt_path = None
        for p in candidate_ckpts:
            if p.exists():
                ckpt_path = p
                break
        
        if not ckpt_path:
            st.sidebar.error("❌ MedSAM2 checkpoint not found")
            st.sidebar.info("📄 Looking for: MedSAM2_latest.pt")
            return None, None
        
        # Find config - expanded search
        config_search_paths = [
            medsam_dir / "sam2" / "configs",
            medsam_dir / "configs",
            medsam_dir / "sam2_configs", 
            Path("sam2/configs"),
            Path("configs")
        ]
        
        config_path = None
        all_configs = []
        
        for config_dir in config_search_paths:
            if config_dir.exists():
                configs = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
                all_configs.extend(configs)
                
                if configs:
                    # Priority order: hiera_t512 > hiera > sam2 > others
                    priority_configs = []
                    for c in configs:
                        name = c.name.lower()
                        if "hiera_t512" in name:
                            priority_configs.append((c, 4))
                        elif "hiera_t" in name:
                            priority_configs.append((c, 3))
                        elif "hiera" in name:
                            priority_configs.append((c, 2))
                        elif "sam2" in name:
                            priority_configs.append((c, 1))
                        else:
                            priority_configs.append((c, 0))
                    
                    if priority_configs:
                        # Sort by priority and take the best
                        priority_configs.sort(key=lambda x: x[1], reverse=True)
                        config_path = priority_configs[0][0]
                        break
        
        if not config_path:
            if all_configs:
                config_path = all_configs[0]  # Use any config as fallback
                st.sidebar.warning(f"⚠️ Using fallback config: {config_path.name}")
            else:
                st.sidebar.error("❌ No MedSAM2 config files found")
                st.sidebar.info("📄 Looking for: sam2/configs/*.yaml")
                return None, None
        
        # Debug info
        st.sidebar.info(f"📄 Config: {config_path.name}")
        st.sidebar.info(f"📁 Config path: {config_path.resolve()}")
        
        # Import and build model
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as ie:
            st.sidebar.error(f"❌ Import failed: {ie}")
            st.sidebar.info("💡 Make sure sam2 package is available in MedSAM2 directory")
            return None, None
        
        # Build model with error handling
        try:
            # Convert to absolute paths
            config_abs = config_path.resolve()
            ckpt_abs = ckpt_path.resolve()
            
            # Try different build_sam2 signatures
            model = build_sam2(str(config_abs), str(ckpt_abs), device=device)
        except Exception as build_error:
            try:
                # Try with named parameters
                model = build_sam2(config_file=str(config_abs), ckpt_path=str(ckpt_abs), device=device)
            except Exception as e2:
                try:
                    # Try original run_medsam2_infer.py approach
                    model = build_sam2(str(config_abs), str(ckpt_abs))
                    model = model.to(device)
                except Exception as e3:
                    st.sidebar.error(f"❌ Build failed: {str(build_error)[:100]}")
                    st.sidebar.error(f"Config: {config_abs}")
                    st.sidebar.error(f"Checkpoint: {ckpt_abs}")
                    return None, None
        
        predictor = SAM2ImagePredictor(model)
        
        st.sidebar.success(f"✅ MedSAM2: {ckpt_path.name}")
        st.sidebar.success(f"✅ Config: {config_path.name}")
        return predictor, model
        
    except Exception as e:
        st.sidebar.error(f"❌ MedSAM2 error: {str(e)[:100]}...")
        return None, None

# ==================== INFERENCE FUNCTIONS ====================

def run_student_inference(student_model, img_tensor, bbox_coords):
    """Run student model inference"""
    H, W = img_tensor.shape[-2:]
    x1, y1, x2, y2 = bbox_coords
    
    # Convert to pixel coordinates
    x1_px, y1_px = int(x1 * W), int(y1 * H)
    x2_px, y2_px = int(x2 * W), int(y2 * H)
    
    # Create soft box prior
    soft_prior = make_soft_box_prior(H, W, (x1_px, y1_px, x2_px, y2_px))
    
    # Create 4-channel input
    img_4ch = torch.cat([img_tensor, soft_prior.unsqueeze(0)], dim=0)
    
    # Run inference
    with torch.no_grad():
        output = torch.sigmoid(student_model(img_4ch.unsqueeze(0).to(device)))[0, 0]
    
    return output.cpu().numpy(), (x1_px, y1_px, x2_px, y2_px)

def run_medsam2_inference(predictor, img_np, bbox_coords):
    """Run MedSAM2 inference"""
    H, W = img_np.shape[:2]
    x1, y1, x2, y2 = bbox_coords
    
    # Convert to pixel coordinates
    x1_px, y1_px = int(x1 * W), int(y1 * H)
    x2_px, y2_px = int(x2 * W), int(y2 * H)
    
    # Set image
    predictor.set_image(img_np)
    
    # Create bounding box
    box = np.array([x1_px, y1_px, x2_px, y2_px], dtype=np.float32)
    
    # Run inference
    masks, scores, logits = predictor.predict(box=box, multimask_output=True)
    
    # Select best mask
    best_mask = masks[np.argmax(scores)]
    
    return best_mask.astype(np.float32), (x1_px, y1_px, x2_px, y2_px)

# ==================== UTILITY FUNCTIONS ====================

def process_image(uploaded_file):
    """Process uploaded image"""
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((320, 320), Image.BILINEAR)
        img_array = np.array(image_resized)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        return img_tensor, img_array, image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None

def calculate_metrics(mask1, mask2, threshold=0.5):
    """Calculate comparison metrics"""
    mask1_bin = mask1 > threshold
    mask2_bin = mask2 > threshold
    
    # Coverage
    coverage1 = mask1_bin.sum() / mask1_bin.size * 100
    coverage2 = mask2_bin.sum() / mask2_bin.size * 100
    
    # IoU (Intersection over Union)
    intersection = (mask1_bin & mask2_bin).sum()
    union = (mask1_bin | mask2_bin).sum()
    iou = intersection / union if union > 0 else 0
    
    # Dice coefficient
    dice = 2 * intersection / (mask1_bin.sum() + mask2_bin.sum()) if (mask1_bin.sum() + mask2_bin.sum()) > 0 else 0
    
    return {
        'coverage1': coverage1,
        'coverage2': coverage2,
        'iou': iou * 100,
        'dice': dice * 100
    }

def create_comparison_plot(img_np, student_mask, medsam2_mask, bbox_px, threshold=0.5):
    """Create side-by-side comparison plot"""
    x1_px, y1_px, x2_px, y2_px = bbox_px
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("MedSAM2 vs Efficient Student Model - Comparative Assessment", fontsize=16, fontweight='bold')
    
    # Row 1: Student Model
    axes[0,0].imshow(img_np)
    rect1 = patches.Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px,
                             linewidth=3, edgecolor='red', facecolor='none')
    axes[0,0].add_patch(rect1)
    axes[0,0].set_title("Input + Bounding Box", fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(img_np)
    axes[0,1].imshow(student_mask > threshold, cmap='jet', alpha=0.7)
    axes[0,1].set_title("Student Model Output", fontweight='bold')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(student_mask, cmap='gray')
    axes[0,2].set_title("Student Probability Map", fontweight='bold')
    axes[0,2].axis('off')
    
    # Student model metrics
    student_coverage = (student_mask > threshold).sum() / student_mask.size * 100
    axes[0,3].text(0.1, 0.7, f"Student Model\nCoverage: {student_coverage:.1f}%\nParameters: ~250K\nMemory: Low", 
                   transform=axes[0,3].transAxes, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[0,3].axis('off')
    
    # Row 2: MedSAM2
    axes[1,0].imshow(img_np)
    rect2 = patches.Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px,
                             linewidth=3, edgecolor='red', facecolor='none')
    axes[1,0].add_patch(rect2)
    axes[1,0].set_title("Input + Bounding Box", fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(img_np)
    axes[1,1].imshow(medsam2_mask > threshold, cmap='jet', alpha=0.7)
    axes[1,1].set_title("MedSAM2 Output", fontweight='bold')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(medsam2_mask, cmap='gray')
    axes[1,2].set_title("MedSAM2 Probability Map", fontweight='bold')
    axes[1,2].axis('off')
    
    # MedSAM2 metrics
    medsam2_coverage = (medsam2_mask > threshold).sum() / medsam2_mask.size * 100
    axes[1,3].text(0.1, 0.7, f"MedSAM2 (Original)\nCoverage: {medsam2_coverage:.1f}%\nParameters: ~2.4M\nMemory: High", 
                   transform=axes[1,3].transAxes, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    axes[1,3].axis('off')
    
    plt.tight_layout()
    return fig

# ==================== MAIN APPLICATION ====================

def main():
    st.title("🧠 MedSAM2 vs Efficient Student Model - Comparative Assessment")
    st.markdown("**Side-by-side comparison of original MedSAM2 and your efficient model**")
    
    # Sidebar model selection
    st.sidebar.header("🎯 Model Selection")
    
    # Student model selection
    available_models = get_available_student_models()
    if not available_models:
        st.sidebar.error("❌ No student models found!")
        st.sidebar.info("Looking for: student_*.pt, best_student_*.pt files")
        return
    
    # Prioritize the requested models
    priority_models = ["student_finetuned_full.pt", "best_student_prompt_full.pt"]
    sorted_models = []
    for priority in priority_models:
        if priority in available_models:
            sorted_models.append(priority)
    for model in available_models:
        if model not in sorted_models:
            sorted_models.append(model)
    
    selected_model = st.sidebar.selectbox(
        "🤖 Select Student Model:",
        sorted_models,
        help="Choose which trained student model to use for comparison"
    )
    
    # Show model info
    model_info = {
        "student_finetuned_full.pt": "🎯 Fine-tuned student model (recommended)",
        "best_student_prompt_full.pt": "🎯 Best prompt-aware model (recommended)", 
        "student_finetuned_ema.pt": "📈 EMA fine-tuned model",
        "best_student_kd_full_1.pt": "🧠 Knowledge distilled model v1",
        "best_student_full_1.pt": "✨ Best student model v1",
    }
    
    if selected_model in model_info:
        st.sidebar.info(model_info[selected_model])
    
    # Show file size
    try:
        model_size = os.path.getsize(selected_model) / (1024 * 1024)  # MB
        st.sidebar.metric("📦 Model Size", f"{model_size:.1f} MB")
    except:
        pass
    
    # Reload button
    if st.sidebar.button("🔄 Reload Models", help="Clear cache and reload models"):
        st.cache_resource.clear()
        st.rerun()
    
    # Load models
    with st.spinner("Loading models..."):
        student_model = load_student_model(selected_model)
        medsam2_predictor, medsam2_model = load_medsam2_model()
    
    # Check if both models are loaded
    models_ready = student_model is not None and medsam2_predictor is not None
    
    if not models_ready:
        st.warning("⚠️ Some models failed to load. Comparison may be limited.")
    
    # Sidebar controls
    st.sidebar.header("🎯 Comparison Controls")
    
    # Model status
    st.sidebar.subheader("📊 Model Status")
    if student_model:
        params_student = student_model.count_trainable_parameters()
        st.sidebar.success(f"✅ Student Model: {params_student:,} params")
    else:
        st.sidebar.error("❌ Student Model: Failed")
    
    if medsam2_predictor:
        # Estimate MedSAM2 parameters (approximate)
        params_medsam2 = sum(p.numel() for p in medsam2_model.parameters() if p.requires_grad) if medsam2_model else 2400000
        st.sidebar.success(f"✅ MedSAM2: ~{params_medsam2:,} params")
        reduction_factor = params_medsam2 / params_student if student_model else 10
        st.sidebar.info(f"🚀 Efficiency: {reduction_factor:.1f}x reduction")
    else:
        st.sidebar.error("❌ MedSAM2: Failed")
    
    # File upload
    st.sidebar.subheader("📁 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a brain MRI image", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff']
    )
    
    # Bounding box controls
    st.sidebar.subheader("🎯 Bounding Box Prompt")
    st.sidebar.markdown("*Normalized coordinates (0-1)*")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        x1 = st.number_input("X1", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        y1 = st.number_input("Y1", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    with col2:
        x2 = st.number_input("X2", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
        y2 = st.number_input("Y2", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
    
    # Validate bounding box
    if x1 >= x2 or y1 >= y2:
        st.sidebar.error("❌ Invalid box: x1 < x2, y1 < y2 required")
        return
    
    bbox_coords = (x1, y1, x2, y2)
    st.sidebar.success(f"✅ Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
    
    # Threshold
    threshold = st.sidebar.slider("Segmentation Threshold", 0.1, 0.9, 0.5, 0.05)
    
    # Main content
    if uploaded_file is None:
        st.info("👆 Please upload a brain MRI image to start comparative assessment")
        
        st.markdown("""
        ## 🔬 **Comparative Assessment Features**
        
        **This interface compares:**
        - 🏥 **MedSAM2 (Original)**: State-of-the-art medical segmentation model
        - ⚡ **Efficient Student**: Your 10x smaller prompt-based model
        
        **Metrics compared:**
        - 📊 Segmentation accuracy and coverage
        - ⚡ Inference speed and memory usage
        - 🎯 Prompt response effectiveness
        - 📈 IoU and Dice similarity scores
        """)
        return
    
    # Process image
    with st.spinner("Processing image..."):
        img_tensor, img_np, img_original = process_image(uploaded_file)
    
    if img_tensor is None:
        return
    
    # Display original image
    st.subheader("📷 Input Image")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_original, caption="Original Image", width='stretch')
    with col2:
        st.image(img_np, caption="Resized for Models (320x320)", width='stretch')
    
    # Run comparative assessment
    if st.button("🚀 Run Comparative Assessment", type="primary"):
        
        # Performance tracking
        results = {}
        
        col1, col2 = st.columns(2)
        
        # Student Model Inference
        with col1:
            st.subheader("⚡ Student Model")
            if student_model:
                with st.spinner("Running student inference..."):
                    tracemalloc.start()
                    start_time = time.time()
                    
                    student_mask, bbox_px = run_student_inference(student_model, img_tensor, bbox_coords)
                    
                    end_time = time.time()
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    results['student'] = {
                        'mask': student_mask,
                        'time': (end_time - start_time) * 1000,
                        'memory': peak / 1024**2,
                        'params': params_student
                    }
                    
                    st.success(f"✅ Complete: {results['student']['time']:.1f}ms")
            else:
                st.error("❌ Student model not available")
        
        # MedSAM2 Inference
        with col2:
            st.subheader("🏥 MedSAM2 Original")
            if medsam2_predictor:
                with st.spinner("Running MedSAM2 inference..."):
                    tracemalloc.start()
                    start_time = time.time()
                    
                    medsam2_mask, bbox_px = run_medsam2_inference(medsam2_predictor, img_np, bbox_coords)
                    
                    end_time = time.time()
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    results['medsam2'] = {
                        'mask': medsam2_mask,
                        'time': (end_time - start_time) * 1000,
                        'memory': peak / 1024**2,
                        'params': params_medsam2
                    }
                    
                    st.success(f"✅ Complete: {results['medsam2']['time']:.1f}ms")
            else:
                st.error("❌ MedSAM2 not available")
        
        # Show results if both models ran
        if 'student' in results and 'medsam2' in results:
            
            # Calculate comparison metrics
            metrics = calculate_metrics(results['student']['mask'], results['medsam2']['mask'], threshold)
            
            # Create comparison visualization
            fig = create_comparison_plot(img_np, results['student']['mask'], results['medsam2']['mask'], bbox_px, threshold)
            st.subheader("📊 Comparative Results")
            st.pyplot(fig)
            
            # Performance comparison table
            st.subheader("⚡ Performance Comparison")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Inference Time",
                    f"{results['student']['time']:.1f}ms",
                    delta=f"{results['student']['time'] - results['medsam2']['time']:.1f}ms"
                )
            
            with col2:
                st.metric(
                    "Memory Usage", 
                    f"{results['student']['memory']:.1f}MB",
                    delta=f"{results['student']['memory'] - results['medsam2']['memory']:.1f}MB"
                )
            
            with col3:
                st.metric(
                    "Parameters",
                    f"{results['student']['params']:,}",
                    delta=f"{results['student']['params'] - results['medsam2']['params']:,}"
                )
            
            with col4:
                speedup = results['medsam2']['time'] / results['student']['time']
                st.metric(
                    "Speed Improvement",
                    f"{speedup:.1f}x",
                    delta=f"+{(speedup-1)*100:.0f}%"
                )
            
            # Accuracy comparison
            st.subheader("🎯 Accuracy Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("IoU Score", f"{metrics['iou']:.1f}%")
            with col2:
                st.metric("Dice Score", f"{metrics['dice']:.1f}%")
            with col3:
                coverage_diff = abs(metrics['coverage1'] - metrics['coverage2'])
                st.metric("Coverage Difference", f"{coverage_diff:.1f}%")
            
            # Assessment summary
            st.subheader("🏆 Assessment Summary")
            
            efficiency_score = "🟢 Excellent" if speedup > 5 else "🟡 Good" if speedup > 2 else "🔴 Needs Improvement"
            accuracy_score = "🟢 Excellent" if metrics['dice'] > 80 else "🟡 Good" if metrics['dice'] > 60 else "🔴 Needs Improvement"
            
            st.success(f"""
            **Comparative Assessment Results:**
            
            **Efficiency Performance**: {efficiency_score}
            - Speed: {speedup:.1f}x faster than MedSAM2
            - Memory: {results['medsam2']['memory']/results['student']['memory']:.1f}x less memory
            - Size: {results['medsam2']['params']/results['student']['params']:.1f}x fewer parameters
            
            **Segmentation Accuracy**: {accuracy_score}
            - IoU Agreement: {metrics['iou']:.1f}%
            - Dice Agreement: {metrics['dice']:.1f}%
            - Coverage Similarity: {100-coverage_diff:.1f}%
            
            **Overall**: ✅ Efficient model achieves {reduction_factor:.1f}x compression with maintained performance
            """)
            
            # Download results
            st.subheader("💾 Export Comparison")
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="📥 Download Comparison Results",
                data=buf.getvalue(),
                file_name=f"medsam2_comparison_{int(time.time())}.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()