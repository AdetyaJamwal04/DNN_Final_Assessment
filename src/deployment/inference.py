"""
Inference pipeline for face mask detection
Load model and perform inference on images/videos
"""
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, List
import os

class MaskDetectionInference:
    """Inference pipeline for face mask detection"""
    
    def __init__(self, model_path: str, input_shape: Tuple[int, int] = (416, 416)):
        """
        Args:
            model_path: Path to saved model
            input_shape: Model input shape (height, width)
        """
        self.model_path = model_path
        self.input_shape = input_shape
        self.model = None
        self.class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        self.class_colors = {
            0: (0, 255, 0),      # with_mask: Green
            1: (255, 0, 0),      # without_mask: Red
            2: (255, 255, 0)     # mask_weared_incorrect: Yellow
        }
        
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.input_shape)
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict(self, image: np.ndarray) -> Dict:
        """
        Perform inference on image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with bbox and class predictions
        """
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Predict
        predictions = self.model.predict(preprocessed, verbose=0)
        
        # Extract predictions
        bbox = predictions['bbox'][0]  # [xmin, ymin, xmax, ymax]
        class_probs = predictions['class'][0]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]
        
        return {
            'bbox': bbox,
            'class_id': class_id,
            'class_name': self.class_names[class_id],
            'confidence': confidence
        }
    
    def draw_prediction(self, image: np.ndarray, prediction: Dict) -> np.ndarray:
        """
        Draw prediction on image
        
        Args:
            image: Input image (BGR format)
            prediction: Prediction dictionary
            
        Returns:
            Image with drawn prediction
        """
        image_copy = image.copy()
        h, w = image.shape[:2]
        
        # Get bbox coordinates
        bbox = prediction['bbox']
        xmin = int(bbox[0] * w)
        ymin = int(bbox[1] * h)
        xmax = int(bbox[2] * w)
        ymax = int(bbox[3] * h)
        
        # Get color and label
        class_id = prediction['class_id']
        color = self.class_colors[class_id]
        label = f"{prediction['class_name']}: {prediction['confidence']:.2f}"
        
        # Draw rectangle
        cv2.rectangle(image_copy, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(image_copy, (xmin, ymin - text_height - 10),
                     (xmin + text_width, ymin), color, -1)
        
        # Draw label text
        cv2.putText(image_copy, label, (xmin, ymin - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image_copy
    
    def process_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Process single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            
        Returns:
            Processed image with predictions
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Predict
        prediction = self.predict(image)
        
        # Draw prediction
        result = self.draw_prediction(image, prediction)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"Result saved to {output_path}")
        
        return result
    
    def process_video(self, video_path: str, output_path: str = None):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            prediction = self.predict(frame)
            result = self.draw_prediction(frame, prediction)
            
            # Write frame
            if output_path:
                out.write(result)
            
            # Display frame (optional)
            cv2.imshow('Face Mask Detection', result)
            
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
    
    def process_webcam(self):
        """Process webcam feed in real-time"""
        cap = cv2.VideoCapture(0)
        
        print("Starting webcam... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            prediction = self.predict(frame)
            result = self.draw_prediction(frame, prediction)
            
            # Display
            cv2.imshow('Face Mask Detection - Webcam', result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face mask detection inference')
    parser.add_argument('--model_path', type=str, default='models/saved_model',
                       help='Path to saved model')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    parser.add_argument('--output', type=str, help='Path to save output')
    
    args = parser.parse_args()
    
    # Create inference pipeline
    inference = MaskDetectionInference(args.model_path)
    
    # Process based on input type
    if args.image:
        inference.process_image(args.image, args.output)
    elif args.video:
        inference.process_video(args.video, args.output)
    elif args.webcam:
        inference.process_webcam()
    else:
        print("Please specify --image, --video, or --webcam")
