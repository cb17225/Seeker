# Seeker
Model that detects AI-generated images.

## Overview

This detector uses an ensemble approach with three main components:

**CNN Model**: Transfer learning with ResNet50 for high-level feature extraction

**Random Forest**: Classification based on color, texture, and frequency features

**SVM**: Support Vector Machine with handcrafted features

**Ensemble**: Combines all models for improved accuracy

## Dataset

This project uses the **CIFAKE** dataset, which is based on the CIFAR-10 structure.

The dataset contains 120,000 images total: 60,000 real images and 60,000 AI-generated images. All images are 32x32 pixels (we upscale them to 224x224 during preprocessing). The AI-generated images were created using Stable Diffusion with various prompts designed to match the categories in CIFAR-10.

You can find the dataset here: [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

### Why CIFAKE?

I chose this dataset because it's well-balanced and has been extensively tested in research. It includes diverse image categories like animals, vehicles, and everyday objects. The dataset is large enough for robust training but not so large that it takes forever to experiment with. Most importantly, it represents realistic detection scenarios where the AI is trying to mimic real images.

## Getting Started

### Prerequisites

You'll need to install these packages:

```bash
pip install torch torchvision
pip install opencv-python pillow numpy pandas
pip install scikit-learn scikit-image
pip install tqdm joblib
pip install kaggle
```

### Setting Up the Dataset

First, you need to get your Kaggle API credentials. Go to kaggle.com/settings and click "Create New API Token". This downloads a file called kaggle.json. Put this file in your home directory under .kaggle/

Now you can download the CIFAKE dataset:

```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    'birdy654/cifake-real-and-ai-generated-synthetic-images',
    path='./cifake_data/',
    unzip=True
)
```

After downloading, you need to organize the images into the right folder structure. The code expects this layout:

```
dataset/
├── real/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── ai/
    ├── image1.png
    ├── image2.png
    └── ...
```

If the CIFAKE download has a different structure, here's how to reorganize it:

```python
import shutil
from pathlib import Path

source_dir = Path('./cifake_data/train')
target_dir = Path('./dataset')

(target_dir / 'real').mkdir(parents=True, exist_ok=True)
(target_dir / 'ai').mkdir(parents=True, exist_ok=True)

# Adjust these paths based on how CIFAKE actually organizes files
for img in (source_dir / 'REAL').glob('*.png'):
    shutil.copy(img, target_dir / 'real')

for img in (source_dir / 'FAKE').glob('*.png'):
    shutil.copy(img, target_dir / 'ai')
```

## Usage

### Training the Models

Start by updating the configuration in the main() function:

```python
config = Config(
    data_path="./dataset/",
    epochs=10,
    model_dir="trained_models",
    batch_size=32
)
```

Then uncomment the training section:

```python
logging.info("STARTING TRAINING")
trainer = Trainer(config)
trainer.run()
logging.info("Training complete!")
```

Run the script:

```bash
python detector.py
```

The training process will load your dataset and split it 80/20 for training and testing. It trains the CNN model first with data augmentation (random flips, rotations, color adjustments). Then it extracts handcrafted features and trains the Random Forest and SVM models. All models get saved to the trained_models/ directory. You'll see accuracy metrics and detailed classification reports as it runs.

### Making Predictions

Once you've trained the models, you can use them to classify new images:

```python
from detector import Predictor, Config

config = Config(model_dir="trained_models")
predictor = Predictor(config)

results = predictor.predict("path/to/image.jpg")

print(f"CNN Prediction: {results['cnn']}")
print(f"Random Forest: {results['rf']}")
print(f"SVM: {results['svm']}")
print(f"Ensemble: {results['ensemble']}")
```

The output looks like this:

```json
{
  "cnn": {
    "real": 0.15,
    "ai": 0.85
  },
  "rf": {
    "real": 0.12,
    "ai": 0.88
  },
  "svm": {
    "real": 0.18,
    "ai": 0.82
  },
  "ensemble": {
    "real": 0.14,
    "ai": 0.86
  }
}
```

## Architecture

### CNN Model

The CNN uses a ResNet50 backbone that's been pretrained on ImageNet. I replaced the final classification layer with a custom one that has:
- Linear layer (2048 to 512 features)
- ReLU activation with 50% dropout
- Final linear layer (512 to 2 classes)

I use the AdamW optimizer with weight decay to prevent overfitting, and CrossEntropyLoss for training.

### Feature Extraction

For the traditional ML models, I extract handcrafted features from each image:
- Color histograms from RGB and HSV color spaces (192 features)
- Local Binary Patterns for texture plus edge density (33 features)
- Total of 225 features per image

### Traditional Models

The Random Forest uses 200 trees with a maximum depth of 15. The SVM uses an RBF kernel and is set up to output probability estimates. The ensemble combines both models using soft voting, which means it averages their probability predictions.

## Expected Performance

With the CIFAKE dataset, here's what you can typically expect:

CNN Accuracy: around 95-98%
Random Forest: around 85-90%
SVM: around 85-90%
Ensemble: around 88-92%

Keep in mind that actual performance depends on several factors. More training epochs generally help, but you need to watch for overfitting. The type of data augmentation matters too. And importantly, if you test on images from AI generators that weren't in the training set, accuracy will likely drop.

## Customization

### Adjusting Hyperparameters

You can tune the training by changing these settings:

```python
config = Config(
    epochs=20,              # More training iterations
    learning_rate=5e-5,     # Lower = more stable but slower
    batch_size=64,          # Larger = faster but needs more memory
    image_size=(256, 256),  # Bigger images capture more detail
)
```

### Using Different Datasets

To use your own dataset instead of CIFAKE:

1. Put your images in folders called real/ and ai/
2. Update the data_path in the config
3. Adjust image_size if your images are much larger or smaller
4. The rest of the training workflow stays the same

### Adding New Backbones

If you want to try EfficientNet or another architecture:

```python
# In CNNDetector.__init__()
if backbone_name == "efficientnet":
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = self.backbone.classifier[1].in_features
    self.backbone.classifier = nn.Identity()
```

## Project Structure

```
.
├── detector.py           # Main implementation
├── README.md            # This file
├── dataset/             # Your organized dataset
│   ├── real/
│   └── ai/
├── trained_models/      # Saved models (created after training)
│   ├── cnn_model.pth
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── ensemble_model.pkl
│   └── scaler.pkl
└── test_image.jpg      # Sample test image
```

## Troubleshooting

**"No images found in dataset"**

Check that your data_path is correct. Make sure you have real/ and ai/ subdirectories with images that end in .jpg, .jpeg, or .png.

**CUDA out of memory**

Try reducing the batch_size to 16 or even 8. You can also reduce image_size to 128x128. The code will automatically use CPU if GPU isn't available, it'll just be slower.

**Low accuracy on your own images**

The model learns the specific characteristics of CIFAKE's AI generator (Stable Diffusion). If you test on images from different generators or domains, you might need to fine-tune the model on your target data.

**Models not found during prediction**

Make sure you've run the training step first. Check that model_dir points to the right location and that all the .pth and .pkl files were created successfully.

## How It Works

Here's the step-by-step process:

1. Images are loaded from the real/ and ai/ folders and split 80/20 for training and testing
2. During training, images get randomly flipped, rotated, and color-adjusted to help the model generalize
3. The CNN (ResNet50) is fine-tuned on the detection task
4. Handcrafted features like color histograms and texture patterns are extracted from all training images
5. Random Forest and SVM models are trained on these extracted features
6. An ensemble combines all models using soft voting
7. For new images, all models make predictions and return probability scores

## References

CIFAKE Dataset: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

ResNet Paper: Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

LBP Features: Multiresolution Gray-Scale and Rotation Invariant Texture Classification

## License

This project is open source and available for educational and research purposes.

## Contributing

Feel free to contribute by adding support for more CNN backbones, implementing additional feature extractors, testing on different datasets, improving ensemble strategies, or adding visualization tools.

## Limitations

There are some important limitations to keep in mind. The model is trained specifically on Stable Diffusion images from CIFAKE, so it might not work as well on images from newer or different AI generators. Performance also depends on image quality and compression artifacts. CIFAKE uses relatively small 32x32 images which can limit how much detail we can extract. If you want to detect AI images in a completely different domain (like medical images or satellite imagery), you'll probably need to retrain on domain-specific data.

## Future Improvements

Some ideas for making this better:

Add attention visualization to see what parts of images the model focuses on

Test how well models trained on one AI generator work on others

Build a web interface with Gradio or Streamlit

Add batch prediction for processing multiple images at once

Implement model interpretability with SHAP or LIME

Include frequency domain analysis using FFT

Experiment with Vision Transformers instead of CNNs

---

If you find this useful, feel free to share your results or contribute improvements!
