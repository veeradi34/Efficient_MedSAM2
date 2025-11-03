"""
Extract 2D image and mask slices from MSD dataset volumes
This script extracts both images and their corresponding segmentation masks
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import json

# Configuration
msd_root = r'c:\Users\Veeraditya\Desktop\Python coding Vs\ML\IE643\Project'
output_root = r'c:\Users\Veeraditya\Desktop\Python coding Vs\ML\IE643\Project\2d_images'

# MSD Tasks (with correct paths)
tasks = {
    'Task01_BrainTumour': r'Task01_BrainTumour\Task01_BrainTumour',
    'Task02_Heart': r'MSD Dataset\Task02_Heart', 
    'Task04_Hippocampus': r'MSD Dataset\Task04_Hippocampus',
    'Task05_Prostate': r'MSD Dataset\Task05_Prostate',
    'Task06_Lung': r'MSD Dataset\Task06_Lung',
    'Task07_Pancreas': r'MSD Dataset\Task07_Pancreas',
    'Task08_HepaticVessel': r'MSD Dataset\Task08_HepaticVessel',
    'Task09_Spleen': r'MSD Dataset\Task09_Spleen',
    'Task10_Colon': r'MSD Dataset\Task10_Colon'
}

def normalize_slice(slice_data):
    """Normalize a 2D slice to 0-255 range"""
    slice_min = slice_data.min()
    slice_max = slice_data.max()
    
    if slice_max - slice_min < 1e-6:
        return np.zeros_like(slice_data, dtype=np.uint8)
    
    normalized = (slice_data - slice_min) / (slice_max - slice_min)
    return (normalized * 255).astype(np.uint8)

def extract_images_and_masks_for_task(task_name, task_folder):
    """Extract both image and mask slices for a specific task"""
    
    task_path = os.path.join(msd_root, task_folder)
    
    # Find image directory
    possible_image_dirs = [
        os.path.join(task_path, 'imagesTr'),
        os.path.join(task_path, 'images'),
    ]
    
    image_dir = None
    for path in possible_image_dirs:
        if os.path.exists(path):
            image_dir = path
            break
    
    # Find label directory
    possible_label_dirs = [
        os.path.join(task_path, 'labelsTr'),
        os.path.join(task_path, 'labels'),
    ]
    
    label_dir = None
    for path in possible_label_dirs:
        if os.path.exists(path):
            label_dir = path
            break
    
    if not image_dir:
        print(f"❌ Could not find images directory for {task_name}")
        return 0, 0
    
    if not label_dir:
        print(f"❌ Could not find labels directory for {task_name}")
        return 0, 0
    
    # Get all image and label files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz') or f.endswith('.nii')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz') or f.endswith('.nii')])
    
    # Filter out hidden/temp files
    image_files = [f for f in image_files if not f.startswith('._')]
    label_files = [f for f in label_files if not f.startswith('._')]
    
    if len(image_files) == 0:
        print(f"⚠️  No image files found in {image_dir}")
        return 0, 0
    
    if len(label_files) == 0:
        print(f"⚠️  No label files found in {label_dir}")
        return 0, 0
    
    print(f"\n{'='*60}")
    print(f"Processing {task_name}")
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Found {len(image_files)} image files")
    print(f"Found {len(label_files)} label files")
    print(f"{'='*60}")
    
    output_dir = os.path.join(output_root, task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    total_images = 0
    total_masks = 0
    
    # Match image and label files
    for image_file in tqdm(image_files, desc=f"Extracting {task_name}"):
        try:
            # Get case name
            case_name = image_file.replace('.nii.gz', '').replace('.nii', '')
            
            # Find corresponding label file
            label_file = None
            for lf in label_files:
                if case_name in lf or lf.replace('.nii.gz', '').replace('.nii', '') == case_name:
                    label_file = lf
                    break
            
            if not label_file:
                print(f"⚠️  No matching label for {image_file}")
                continue
            
            # Load image and label volumes
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, label_file)
            
            image_nii = nib.load(image_path)
            label_nii = nib.load(label_path)
            
            image_data = image_nii.get_fdata()
            label_data = label_nii.get_fdata()
            
            # Handle multi-channel images (take first channel if exists)
            if len(image_data.shape) == 4:
                image_data = image_data[..., 0]
            
            # Ensure same shape
            if image_data.shape != label_data.shape:
                print(f"⚠️  Shape mismatch for {case_name}: image {image_data.shape} vs label {label_data.shape}")
                continue
            
            # Extract slices along the depth axis (typically axis 2)
            num_slices = image_data.shape[2]
            
            for slice_idx in range(num_slices):
                # Extract image and mask slice
                image_slice = image_data[:, :, slice_idx]
                mask_slice = label_data[:, :, slice_idx]
                
                # Skip slices where both image and mask are empty
                if image_slice.max() == 0 and mask_slice.max() == 0:
                    continue
                
                # Normalize image slice
                image_normalized = normalize_slice(image_slice)
                
                # Binarize mask (anything > 0 is foreground)
                mask_binary = (mask_slice > 0).astype(np.uint8) * 255
                
                # Save image
                image_filename = f"{case_name}_slice{slice_idx:04d}.png"
                image_output_path = os.path.join(output_dir, image_filename)
                image_img = Image.fromarray(image_normalized)
                image_img.save(image_output_path)
                total_images += 1
                
                # Save mask
                mask_filename = f"{case_name}_slice{slice_idx:04d}_mask.png"
                mask_output_path = os.path.join(output_dir, mask_filename)
                mask_img = Image.fromarray(mask_binary)
                mask_img.save(mask_output_path)
                total_masks += 1
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    print(f"✅ Extracted {total_images} images and {total_masks} masks for {task_name}")
    return total_images, total_masks

# Main execution
if __name__ == "__main__":
    print("🚀 Starting image and mask extraction from MSD dataset...")
    print(f"MSD root: {msd_root}")
    print(f"Output root: {output_root}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    total_images = 0
    total_masks = 0
    task_summary = {}
    
    for task_name, task_folder in tasks.items():
        num_images, num_masks = extract_images_and_masks_for_task(task_name, task_folder)
        task_summary[task_name] = {'images': num_images, 'masks': num_masks}
        total_images += num_images
        total_masks += num_masks
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 80)
    
    for task_name, counts in task_summary.items():
        print(f"{task_name:30s}: {counts['images']:5d} images, {counts['masks']:5d} masks")
    
    print("-" * 80)
    print(f"{'TOTAL':30s}: {total_images:5d} images, {total_masks:5d} masks")
    print("=" * 80)
    
    # Save summary
    summary_path = os.path.join(output_root, 'extraction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_images': total_images,
            'total_masks': total_masks,
            'tasks': task_summary
        }, f, indent=2)
    
    print(f"✅ Summary saved to {summary_path}")
    print("\n🎉 Extraction complete!")
    print(f"📁 Files saved in: {output_root}")
    print("\nℹ️  Note: Each mask file is named with '_mask.png' suffix to match the corresponding image file.")
