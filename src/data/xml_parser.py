"""
Pascal VOC XML Parser for Face Mask Detection Dataset
Converts XML annotations to TensorFlow-compatible format
"""
import xml.etree.ElementTree as ET
import os
import numpy as np
from typing import Dict, List, Tuple

class VOCParser:
    """Parse Pascal VOC XML annotations"""
    
    def __init__(self):
        self.class_mapping = {
            'with_mask': 0,
            'without_mask': 1,
            'mask_weared_incorrect': 2
        }
        
    def parse_xml(self, xml_path: str) -> Dict:
        """
        Parse single XML file
        
        Args:
            xml_path: Path to XML annotation file
            
        Returns:
            Dictionary containing image info, bboxes, and labels
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract image information
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Extract all objects (bounding boxes and labels)
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            # Get class label
            class_name = obj.find('name').text
            if class_name not in self.class_mapping:
                print(f"Warning: Unknown class '{class_name}' in {xml_path}")
                continue
                
            label = self.class_mapping[class_name]
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Normalize coordinates to [0, 1]
            xmin_norm = xmin / width
            ymin_norm = ymin / height
            xmax_norm = xmax / width
            ymax_norm = ymax / height
            
            boxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
            labels.append(label)
        
        return {
            'filename': filename,
            'width': width,
            'height': height,
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int32)
        }
    
    def parse_directory(self, annotations_dir: str) -> List[Dict]:
        """
        Parse all XML files in directory
        
        Args:
            annotations_dir: Path to directory containing XML files
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
        
        for xml_file in xml_files:
            xml_path = os.path.join(annotations_dir, xml_file)
            try:
                annotation = self.parse_xml(xml_path)
                if len(annotation['boxes']) > 0:  # Only include images with annotations
                    annotations.append(annotation)
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
                
        return annotations
    
    def convert_to_center_format(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert bounding boxes from [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
        
        Args:
            boxes: Array of shape (N, 4) in [xmin, ymin, xmax, ymax] format
            
        Returns:
            Array of shape (N, 4) in [center_x, center_y, width, height] format
        """
        center_x = (boxes[:, 0] + boxes[:, 2]) / 2
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        
        return np.stack([center_x, center_y, width, height], axis=1)
    
    def get_class_name(self, label: int) -> str:
        """Get class name from label index"""
        for name, idx in self.class_mapping.items():
            if idx == label:
                return name
        return "unknown"

if __name__ == "__main__":
    # Test the parser
    parser = VOCParser()
    
    # Example usage
    annotations_dir = "data/raw/annotations"
    if os.path.exists(annotations_dir):
        annotations = parser.parse_directory(annotations_dir)
        print(f"Parsed {len(annotations)} annotations")
        
        if annotations:
            print("\nFirst annotation example:")
            ann = annotations[0]
            print(f"  Filename: {ann['filename']}")
            print(f"  Image size: {ann['width']}x{ann['height']}")
            print(f"  Number of objects: {len(ann['boxes'])}")
            print(f"  Boxes: {ann['boxes']}")
            print(f"  Labels: {ann['labels']}")
    else:
        print(f"Annotations directory not found: {annotations_dir}")
