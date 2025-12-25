# Face Mask Detection System - Quick Start Guide

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
cd DNN_Final_Assessment
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: Using Kaggle API** (Recommended)

```bash
# Make sure you have kaggle.json in C:\Users\<username>\.kaggle\
python download_dataset.py
```

**Option B: Manual Download**

1. Visit: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
2. Download the dataset
3. Extract to `data/raw/`

Expected structure:
```
data/raw/
├── annotations/  (XML files)
└── images/       (JPG files)
```

## 🎯 Usage

### Training the Model

```bash
# Basic training
python src/training/train.py --data_dir data/raw --epochs 50

# Custom parameters
python src/training/train.py --data_dir data/raw --epochs 100 --batch_size 32 --learning_rate 0.0005
```

### Evaluation

```bash
# Evaluate trained model
python src/evaluation/evaluate.py --model_path models/saved_model --data_dir data/raw
```

Results will be saved to `evaluation_results/`:
- `metrics.txt` - Detailed metrics
- `confusion_matrix.png` - Confusion matrix
- `metrics.png` - Per-class metrics plot
- `predictions/` - Prediction visualizations

### Deployment

#### 1. Streamlit Web App

```bash
streamlit run src/deployment/app.py
```

Then open http://localhost:8501 in your browser.

#### 2. Command-line Inference

**Single Image:**
```bash
python src/deployment/inference.py --model_path models/saved_model --image path/to/image.jpg --output result.jpg
```

**Video:**
```bash
python src/deployment/inference.py --model_path models/saved_model --video path/to/video.mp4 --output output.mp4
```

**Webcam:**
```bash
python src/deployment/inference.py --model_path models/saved_model --webcam
```

#### 3. TFLite Conversion

```bash
# Dynamic quantization
python src/deployment/quantize.py --model_path models/saved_model --output models/tflite/model.tflite --quantization dynamic

# Test converted model
python src/deployment/quantize.py --model_path models/saved_model --output models/tflite/model.tflite --test
```

## 📊 Model Architecture

```
Input (416x416x3)
      ↓
MobileNetV2 (pretrained)
      ↓
Global Average Pooling
      ↓
  ┌───────┴────────┐
  ↓                ↓
Bbox Head      Class Head
(regression)   (classification)
  ↓                ↓
Output (4)     Output (3)
```

## 🎯 Expected Performance

- **mAP@IoU=0.5**: ≥ 0.75
- **Per-class F1**: ≥ 0.70
- **Inference Speed**: ≥ 15 FPS

## 🔧 Troubleshooting

### Dataset Download Issues

If Kaggle API fails:
1. Check that `kaggle.json` is in the correct location
2. Verify your Kaggle credentials
3. Try manual download from the website

### Training Issues

**Out of Memory:**
- Reduce batch size: `--batch_size 8`
- Use smaller input size (edit in code)

**Slow Training:**
- Check GPU availability
- Reduce image size
- Use fewer epochs for testing

### Deployment Issues

**Model Not Found:**
- Ensure model is trained and saved to `models/saved_model`
- Check model path in commands

**Streamlit Port Already in Use:**
```bash
streamlit run src/deployment/app.py --server.port 8502
```

## 📁 Project Structure

```
DNN_Final_Assessment/
├── data/                    # Dataset
├── models/                  # Saved models
├── src/                     # Source code
│   ├── data/               # Data processing
│   ├── models/             # Model architecture
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation
│   └── deployment/         # Deployment code
├── requirements.txt
├── README.md
└── download_dataset.py
```

## 🌟 Features

- ✅ Transfer learning with MobileNetV2
- ✅ Dual-head architecture (bbox + class)
- ✅ Data augmentation with bbox transformation
- ✅ Comprehensive metrics (mAP, IoU, precision, recall, F1)
- ✅ Streamlit web interface
- ✅ Real-time webcam detection
- ✅ TFLite quantization for deployment
- ✅ Visualization tools

## 📝 Notes

- The model detects **one face** per image (primary face)
- For multiple faces, extend the architecture for multi-object detection
- Bounding boxes are in normalized coordinates [0,1]

## 🆘 Support

For issues or questions:
1. Check this guide
2. Review error messages
3. Verify dataset structure
4. Check model is trained properly
