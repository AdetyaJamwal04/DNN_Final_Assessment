"""
Analyze class distribution in the face mask detection dataset
"""
import os
import sys
from collections import Counter

sys.path.append('src')

from data.xml_parser import VOCParser

def analyze_dataset(annotations_dir='data/raw/annotations'):
    """Analyze class distribution"""
    parser = VOCParser()
    annotations = parser.parse_directory(annotations_dir)
    
    print("="*80)
    print("FACE MASK DETECTION DATASET ANALYSIS")
    print("="*80)
    
    print(f"\nTotal images: {len(annotations)}")
    
    # Count total objects and per-class distribution
    all_labels = []
    objects_per_image = []
    
    for ann in annotations:
        labels = ann['labels']
        all_labels.extend(labels)
        objects_per_image.append(len(labels))
    
    # Class distribution
    class_counts = Counter(all_labels)
    
    print(f"Total objects detected: {len(all_labels)}")
    print(f"\nClass Distribution:")
    print(f"  with_mask (0):           {class_counts[0]:4d} ({class_counts[0]/len(all_labels)*100:5.1f}%)")
    print(f"  without_mask (1):        {class_counts[1]:4d} ({class_counts[1]/len(all_labels)*100:5.1f}%)")
    print(f"  mask_weared_incorrect (2): {class_counts[2]:4d} ({class_counts[2]/len(all_labels)*100:5.1f}%)")
    
    # Objects per image statistics
    print(f"\nObjects per Image:")
    print(f"  Min: {min(objects_per_image)}")
    print(f"  Max: {max(objects_per_image)}")
    print(f"  Average: {sum(objects_per_image)/len(objects_per_image):.2f}")
    
    # Sample images with different object counts
    print(f"\nSample Images:")
    single_obj = [ann for ann in annotations if len(ann['labels']) == 1]
    multi_obj = [ann for ann in annotations if len(ann['labels']) > 3]
    
    print(f"  Images with 1 object: {len(single_obj)}")
    print(f"  Images with >3 objects: {len(multi_obj)}")
    
    print("\n" + "="*80)
    
    return {
        'total_images': len(annotations),
        'total_objects': len(all_labels),
        'class_distribution': dict(class_counts),
        'avg_objects_per_image': sum(objects_per_image)/len(objects_per_image)
    }

if __name__ == "__main__":
    stats = analyze_dataset()
