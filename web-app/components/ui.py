"""
UI Components for Efficient MedSAM2 Web Application
==================================================
Custom UI components and visualization functions for the medical segmentation platform.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from PIL import Image
import base64
import io

class UIComponents:
    """Custom UI components for the application"""
    
    def __init__(self):
        """Initialize UI components"""
        self.primary_color = "#00f5ff"
        self.secondary_color = "#0080ff"
        self.accent_color = "#ff00ff"
        self.success_color = "#00ff88"
        self.error_color = "#ff4444"
        self.warning_color = "#ffaa00"
    
    def render_login_form(self, auth_manager):
        """Render the login form"""
        st.markdown("### 🔐 User Login")
        
        with st.form("login_form"):
            username_or_email = st.text_input(
                "👤 Username or Email",
                placeholder="Enter your username or email"
            )
            password = st.text_input(
                "🔒 Password",
                type="password",
                placeholder="Enter your password"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                login_submitted = st.form_submit_button("🚀 Login", use_container_width=True)
            with col2:
                forgot_password = st.form_submit_button("❓ Forgot Password", use_container_width=True)
        
        if login_submitted:
            if username_or_email and password:
                with st.spinner("Authenticating..."):
                    success, message, user_info = auth_manager.login_user(username_or_email, password)
                
                if success:
                    # Set session state
                    st.session_state.authenticated = True
                    st.session_state.username = user_info['username']
                    st.session_state.user_id = user_info['id']
                    st.session_state.session_token = user_info['session_token']
                    
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
            else:
                st.error("❌ Please enter both username/email and password")
        
        if forgot_password:
            st.info("🔄 Password reset functionality will be available soon")
    
    def render_register_form(self, auth_manager):
        """Render the registration form"""
        st.markdown("### 📝 Create New Account")
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input(
                    "👤 Username",
                    placeholder="Choose a unique username"
                )
                email = st.text_input(
                    "📧 Email",
                    placeholder="your.email@example.com"
                )
                full_name = st.text_input(
                    "👨‍⚕️ Full Name",
                    placeholder="Dr. John Doe"
                )
            
            with col2:
                password = st.text_input(
                    "🔒 Password",
                    type="password",
                    placeholder="Create a strong password"
                )
                password_confirm = st.text_input(
                    "🔒 Confirm Password",
                    type="password",
                    placeholder="Re-enter your password"
                )
                institution = st.text_input(
                    "🏥 Institution",
                    placeholder="Medical Institution (Optional)"
                )
            
            # Password requirements info
            with st.expander("🔐 Password Requirements"):
                st.markdown("""
                - At least 8 characters long
                - At least one uppercase letter (A-Z)
                - At least one lowercase letter (a-z)
                - At least one number (0-9)
                - At least one special character (!@#$%^&*...)
                """)
            
            register_submitted = st.form_submit_button("🎯 Create Account", use_container_width=True)
        
        if register_submitted:
            # Validate form
            if not all([username, email, password, password_confirm]):
                st.error("❌ Please fill in all required fields")
            elif password != password_confirm:
                st.error("❌ Passwords do not match")
            else:
                with st.spinner("Creating account..."):
                    success, message = auth_manager.register_user(
                        username, email, password, full_name, institution
                    )
                
                if success:
                    st.success(f"✅ {message}")
                    st.info("🔑 Please login with your new credentials")
                else:
                    st.error(f"❌ {message}")
    
    def render_metric_card(self, title: str, value: str, delta: str = None, 
                          help_text: str = None, color: str = None):
        """Render a custom metric card"""
        color_class = color or self.primary_color
        
        card_html = f"""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1rem;
            text-align: center;
            transition: all 0.3s ease;
        ">
            <h3 style="
                color: {color_class};
                font-family: 'Orbitron', monospace;
                font-size: 2rem;
                margin: 0;
                text-shadow: 0 0 10px {color_class}50;
            ">{value}</h3>
            <p style="
                color: #b8b9ba;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin: 0.5rem 0 0 0;
            ">{title}</p>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        if help_text:
            st.caption(help_text)
    
    def create_segmentation_plot(self, img_np: np.ndarray, mask: np.ndarray, 
                                bbox_px: tuple, threshold: float = 0.5):
        """Create segmentation visualization plot"""
        x1_px, y1_px, x2_px, y2_px = bbox_px
        
        # Create figure with dark theme
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('#0a0a0a')
        fig.suptitle('🧠 Efficient MedSAM2 - Segmentation Results', 
                    fontsize=16, fontweight='bold', color='#00f5ff')
        
        # Original image with bounding box
        axes[0,0].imshow(img_np)
        rect = patches.Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px,
                               linewidth=3, edgecolor='#00f5ff', facecolor='none')
        axes[0,0].add_patch(rect)
        axes[0,0].set_title('📷 Input Image + ROI', color='#00f5ff', fontweight='bold')
        axes[0,0].axis('off')
        
        # Mask overlay
        axes[0,1].imshow(img_np)
        axes[0,1].imshow(mask > threshold, cmap='jet', alpha=0.6)
        axes[0,1].set_title('🎯 Segmentation Overlay', color='#00f5ff', fontweight='bold')
        axes[0,1].axis('off')
        
        # Probability map
        im1 = axes[0,2].imshow(mask, cmap='hot', vmin=0, vmax=1)
        axes[0,2].set_title('🌡️ Probability Map', color='#00f5ff', fontweight='bold')
        axes[0,2].axis('off')
        plt.colorbar(im1, ax=axes[0,2], fraction=0.046, pad=0.04)
        
        # Binary mask
        binary_mask = mask > threshold
        axes[1,0].imshow(binary_mask, cmap='gray')
        axes[1,0].set_title(f'⚫ Binary Mask (t={threshold})', color='#00f5ff', fontweight='bold')
        axes[1,0].axis('off')
        
        # Contours
        axes[1,1].imshow(img_np)
        if binary_mask.any():
            import cv2
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                contour = contour.squeeze()
                if len(contour) > 2:
                    axes[1,1].plot(contour[:, 0], contour[:, 1], 
                                  color='#00f5ff', linewidth=2)
        axes[1,1].set_title('🔍 Contour Detection', color='#00f5ff', fontweight='bold')
        axes[1,1].axis('off')
        
        # Statistics
        coverage = binary_mask.sum() / binary_mask.size * 100
        confidence = mask.max() * 100
        mean_prob = mask[binary_mask].mean() * 100 if binary_mask.any() else 0
        
        axes[1,2].text(0.1, 0.8, f'📊 Segmentation Statistics\n\n'
                              f'Coverage: {coverage:.1f}%\n'
                              f'Max Confidence: {confidence:.1f}%\n'
                              f'Mean Probability: {mean_prob:.1f}%\n'
                              f'Threshold: {threshold:.2f}\n'
                              f'Pixels Segmented: {binary_mask.sum():,}\n'
                              f'Image Size: {mask.shape[0]}×{mask.shape[1]}',
                      transform=axes[1,2].transAxes, fontsize=12,
                      verticalalignment='top', color='#ffffff',
                      bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor=(0/255, 245/255, 255/255, 0.1),
                               edgecolor="#00f5ff"))
        axes[1,2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_performance_chart(self, metrics: dict):
        """Create interactive performance chart"""
        # Create performance metrics visualization
        fig = go.Figure()
        
        # Add inference time
        fig.add_trace(go.Scatter(
            x=['Current'],
            y=[metrics.get('total_time_seconds', 0) * 1000],  # Convert to ms
            mode='markers+text',
            name='Inference Time (ms)',
            marker=dict(size=20, color=self.primary_color),
            text=[f"{metrics.get('total_time_seconds', 0) * 1000:.1f}ms"],
            textposition="middle right"
        ))
        
        fig.update_layout(
            title="🚀 Performance Metrics",
            xaxis_title="Measurement",
            yaxis_title="Time (ms)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        
        return fig
    
    def create_memory_usage_chart(self, memory_mb: float):
        """Create memory usage donut chart"""
        # Estimate total available memory (simplified)
        total_memory = 8000  # 8GB estimate, you could get this from system info
        used_memory = memory_mb
        free_memory = total_memory - used_memory
        
        fig = go.Figure(data=[go.Pie(
            labels=['Used Memory', 'Free Memory'],
            values=[used_memory, free_memory],
            hole=.3,
            marker_colors=[self.accent_color, '#333333']
        )])
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_color='white'
        )
        
        fig.update_layout(
            title="💾 Memory Usage",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff')
        )
        
        return fig
    
    def render_loading_animation(self, text: str = "Processing..."):
        """Render custom loading animation"""
        loading_html = f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            background: rgba(0, 245, 255, 0.05);
            border-radius: 15px;
            border: 1px solid rgba(0, 245, 255, 0.2);
        ">
            <div style="
                width: 50px;
                height: 50px;
                border: 3px solid rgba(0, 245, 255, 0.3);
                border-radius: 50%;
                border-top: 3px solid #00f5ff;
                animation: spin 1s linear infinite;
                margin-right: 1rem;
            "></div>
            <span style="
                color: #00f5ff;
                font-family: 'Exo 2', sans-serif;
                font-size: 1.2rem;
                font-weight: 600;
            ">{text}</span>
        </div>
        
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
        
        return st.markdown(loading_html, unsafe_allow_html=True)
    
    def render_success_message(self, title: str, message: str, details: dict = None):
        """Render styled success message"""
        success_html = f"""
        <div style="
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <h3 style="
                color: {self.success_color};
                margin: 0 0 0.5rem 0;
                font-family: 'Exo 2', sans-serif;
            ">✅ {title}</h3>
            <p style="
                color: #ffffff;
                margin: 0;
            ">{message}</p>
        </div>
        """
        
        st.markdown(success_html, unsafe_allow_html=True)
        
        if details:
            with st.expander("📋 Details"):
                for key, value in details.items():
                    st.text(f"{key}: {value}")
    
    def render_error_message(self, title: str, message: str, suggestion: str = None):
        """Render styled error message"""
        error_html = f"""
        <div style="
            background: rgba(255, 68, 68, 0.1);
            border: 1px solid rgba(255, 68, 68, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <h3 style="
                color: {self.error_color};
                margin: 0 0 0.5rem 0;
                font-family: 'Exo 2', sans-serif;
            ">❌ {title}</h3>
            <p style="
                color: #ffffff;
                margin: 0;
            ">{message}</p>
        </div>
        """
        
        st.markdown(error_html, unsafe_allow_html=True)
        
        if suggestion:
            st.info(f"💡 Suggestion: {suggestion}")
    
    def render_feature_card(self, icon: str, title: str, description: str, 
                          highlight: bool = False):
        """Render feature card"""
        border_color = self.primary_color if highlight else "rgba(255, 255, 255, 0.1)"
        background = "rgba(0, 245, 255, 0.1)" if highlight else "rgba(255, 255, 255, 0.05)"
        
        card_html = f"""
        <div style="
            background: {background};
            backdrop-filter: blur(20px);
            border: 1px solid {border_color};
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
        ">
            <div style="
                font-size: 3rem;
                margin-bottom: 1rem;
            ">{icon}</div>
            <h3 style="
                color: #ffffff;
                margin: 0 0 1rem 0;
                font-family: 'Exo 2', sans-serif;
            ">{title}</h3>
            <p style="
                color: #b8b9ba;
                margin: 0;
                line-height: 1.6;
            ">{description}</p>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    def create_model_comparison_table(self, models_info: list):
        """Create interactive model comparison table"""
        import pandas as pd
        
        # Convert model info to DataFrame
        df = pd.DataFrame(models_info)
        
        # Style the dataframe
        styled_df = df.style.background_gradient(
            subset=['size_mb'], 
            cmap='viridis'
        ).format({
            'size_mb': '{:.1f} MB'
        })
        
        return styled_df
    
    def render_progress_bar(self, progress: float, text: str = "", 
                          color: str = None):
        """Render custom progress bar"""
        color = color or self.primary_color
        
        progress_html = f"""
        <div style="
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 0.5rem;
            margin: 1rem 0;
        ">
            <div style="
                background: {color};
                width: {progress}%;
                height: 20px;
                border-radius: 5px;
                transition: width 0.3s ease;
                position: relative;
            ">
                <span style="
                    position: absolute;
                    right: 10px;
                    top: 50%;
                    transform: translateY(-50%);
                    color: white;
                    font-size: 0.8rem;
                    font-weight: bold;
                ">{progress:.1f}%</span>
            </div>
            <p style="
                color: #b8b9ba;
                margin: 0.5rem 0 0 0;
                font-size: 0.9rem;
            ">{text}</p>
        </div>
        """
        
        st.markdown(progress_html, unsafe_allow_html=True)