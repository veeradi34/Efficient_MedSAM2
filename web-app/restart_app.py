#!/usr/bin/env python3
"""
Restart Web Application with Cache Clear
=======================================
Simple script to restart the medical segmentation web app with fresh cache
"""

import streamlit as st
import os
import sys

def clear_streamlit_cache():
    """Clear Streamlit cache and restart"""
    try:
        # Clear cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        print("✅ Streamlit cache cleared successfully")
        
        # Force reload
        st.rerun()
        
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")

if __name__ == "__main__":
    print("🔄 Clearing Streamlit cache...")
    
    # Clear any existing cache files
    cache_dir = os.path.expanduser("~/.streamlit")
    if os.path.exists(cache_dir):
        import shutil
        try:
            shutil.rmtree(cache_dir)
            print("✅ Cleared cache directory")
        except:
            print("⚠️ Could not clear cache directory")
    
    print("✅ Ready to restart application")
    print("🚀 Please restart the web application now")