# Face Mask Detection System

A comprehensive deep learning system for detecting face masks in images using CNN-based object detection with bounding box regression and multi-class classification.

## 🎯 Project Overview

This project implements a face mask detection system that identifies whether people are:
- ✅ **with_mask**: Wearing a mask correctly
- ❌ **without_mask**: Not wearing a mask
- ⚠️ **mask_weared_incorrect**: Wearing a mask incorrectly

## 📊 Dataset

**Source**: [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

- **Format**: Pascal VOC XML annotations
- **Classes**: 3 (with_mask, without_mask, mask_weared_incorrect)
- **Annotations**: Bounding boxes with class labels

## 🏗️ Architecture

**Model**: Transfer Learning with MobileNetV2
- **Base**: Pre-trained MobileNetV2 (ImageNet weights)
- **Dual-Head Output**:
  - Bounding Box Regression Head (4 outputs: x, y, w, h)
  - Classification Head (3 outputs: class probabilities)
- **Loss Function**: MSE (bbox) + Categorical Crossentropy (class)

## 📁 Project Structure

```
DNN_Final_Assessment/
├── data/
│   ├── raw/                    # Original downloaded data
│   ├── processed/              # Processed images and labels
│   └── splits/                 # Train/val/test splits
├── models/
│   ├── checkpoints/            # Training checkpoints
│   ├── saved_model/            # Final SavedModel format
│   └── tflite/                 # Quantized TFLite models
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_training.ipynb
├── src/
│   ├── data/                   # Data processing
│   ├── models/                 # Model architecture
│   ├── training/               # Training scripts
│   ├── evaluation/             # Metrics and visualization
│   └── deployment/             # Deployment code
├── requirements.txt
└── README.md
```

## 🚀 Installation

```bash
# Clone or navigate to project directory
cd DNN_Final_Assessment

# Install dependencies
pip install -r requirements.txt
```

## 📥 Dataset Download

### Option 1: Using Kaggle API (Recommended)

```bash
# Configure Kaggle API (place kaggle.json in ~/.kaggle/)
kaggle datasets download -d andrewmvd/face-mask-detection
unzip face-mask-detection.zip -d data/raw/
```

### Option 2: Manual Download

1. Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
2. Download the dataset
3. Extract to `data/raw/`

## 🎓 Training

```bash
python src/training/train.py --epochs 50 --batch_size 16 --learning_rate 0.001
```

## 📈 Evaluation

```bash
python src/evaluation/evaluate.py --model_path models/saved_model
```

## 🌐 Deployment

### Streamlit Web App

```bash
streamlit run src/deployment/app.py
```

### TFLite Conversion

```bash
python src/deployment/quantize.py --model_path models/saved_model --output models/tflite
```

## 📊 Performance Metrics

- **mAP@IoU=0.5**: Target ≥ 0.75
- **Per-class F1 Score**: Target ≥ 0.70
- **Inference Time**: Real-time (≥15 FPS)

## 🛠️ Technologies Used

- **TensorFlow 2.x**: Deep learning framework
- **OpenCV**: Image processing and visualization
- **Streamlit**: Web application framework
- **MobileNetV2**: Transfer learning base model
- **TFLite**: Model optimization and quantization

## 📝 License

This project is for educational purposes as part of a deep learning internship assessment.

## 👥 Author

Deep Learning Intern - Smart Surveillance Company

## 🙏 Acknowledgments

- Dataset: [Kaggle Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- Base Model: MobileNetV2 (TensorFlow/Keras Applications)
