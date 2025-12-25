"""
Complete evaluation script for face mask detection model
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.xml_parser import VOCParser
from data.preprocessor import ImagePreprocessor, create_tf_dataset
from evaluation.metrics import evaluate_model, calculate_iou
from evaluation.visualize import BBoxVisualizer, plot_confusion_matrix, plot_metrics

class ModelEvaluator:
    """Complete model evaluation pipeline"""
    
    def __init__(self, 
                 model_path: str,
                 data_dir: str,
                 output_dir: str = 'evaluation_results'):
        """
        Args:
            model_path: Path to trained model
            data_dir: Directory containing test data
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        
    def load_model(self):
        """Load trained model"""
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        
    def load_test_data(self):
        """Load and prepare test data"""
        print("Loading test data...")
        
        annotations_dir = os.path.join(self.data_dir, 'annotations')
        images_dir = os.path.join(self.data_dir, 'images')
        
        parser = VOCParser()
        annotations = parser.parse_directory(annotations_dir)
        
        # Create test split
        preprocessor = ImagePreprocessor()
        splits = preprocessor.create_dataset_splits(annotations)
        
        print(f"Test set size: {len(splits['test'])}")
        
        # Create dataset
        test_ds = create_tf_dataset(
            splits['test'],
            images_dir,
            batch_size=16,
            shuffle=False,
            preprocessor=preprocessor
        )
        
        # Prepare for model
        def prepare_for_model(image, bbox, label):
            return image, {'bbox': bbox, 'class': label}
        
        test_ds = test_ds.map(prepare_for_model)
        
        return test_ds, splits['test'], images_dir
    
    def evaluate(self):
        """Run complete evaluation"""
        # Load model and data
        self.load_model()
        test_ds, test_annotations, images_dir = self.load_test_data()
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = evaluate_model(self.model, test_ds, self.class_names)
        
        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nmAP@IoU=0.5: {metrics['mAP@0.5']:.4f}")
        print(f"Average IoU: {metrics['average_iou']:.4f}")
        
        print("\nPer-class metrics:")
        report = metrics['classification_report']
        for i, class_name in enumerate(self.class_names):
            class_metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-Score: {class_metrics['f1-score']:.4f}")
        
        print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
        
        # Save confusion matrix
        print("\nGenerating visualizations...")
        conf_matrix_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(
            metrics['confusion_matrix'],
            self.class_names,
            conf_matrix_path
        )
        
        # Save metrics plot
        metrics_plot_path = os.path.join(self.output_dir, 'metrics.png')
        plot_metrics(metrics, metrics_plot_path)
        
        # Visualize predictions on test images
        print("\nVisualizing predictions...")
        self.visualize_predictions(test_ds, test_annotations, images_dir)
        
        # Save metrics to file
        metrics_file = os.path.join(self.output_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write("FACE MASK DETECTION - EVALUATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"mAP@IoU=0.5: {metrics['mAP@0.5']:.4f}\n")
            f.write(f"Average IoU: {metrics['average_iou']:.4f}\n\n")
            f.write("Per-class Metrics:\n")
            for class_name in self.class_names:
                class_metrics = report[class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
        
        print(f"\n✅ Evaluation complete! Results saved to {self.output_dir}")
        
        return metrics
    
    def visualize_predictions(self, test_ds, test_annotations, images_dir, num_samples=10):
        """Visualize predictions on test images"""
        visualizer = BBoxVisualizer()
        
        # Collect predictions
        images = []
        true_boxes = []
        true_labels = []
        pred_boxes = []
        pred_labels = []
        pred_confidences = []
        
        preprocessor = ImagePreprocessor()
        
        count = 0
        for ann in test_annotations[:num_samples]:
            try:
                image, boxes, labels = preprocessor.preprocess_annotation(ann, images_dir)
                
                # Predict
                image_batch = np.expand_dims(image, axis=0)
                prediction = self.model.predict(image_batch, verbose=0)
                
                # Extract predictions
                pred_box = prediction['bbox'][0]
                pred_class = prediction['class'][0]
                pred_class_id = np.argmax(pred_class)
                pred_conf = pred_class[pred_class_id]
                
                # Store
                images.append(image)
                true_boxes.append(boxes[0])
                true_labels.append(np.argmax(labels[0]))
                pred_boxes.append(pred_box)
                pred_labels.append(pred_class_id)
                pred_confidences.append(pred_conf)
                
                count += 1
            except Exception as e:
                print(f"Error visualizing {ann['filename']}: {e}")
        
        # Save visualizations
        vis_dir = os.path.join(self.output_dir, 'predictions')
        visualizer.save_predictions(
            images, true_boxes, true_labels,
            pred_boxes, pred_labels, pred_confidences,
            vis_dir, num_images=count
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate face mask detection model')
    parser.add_argument('--model_path', type=str, default='models/saved_model',
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    evaluator.evaluate()
