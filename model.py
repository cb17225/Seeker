"""
AI-Generated Image Detector
A multi-model approach combining deep learning and traditional computer vision
"""

import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import zipfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed
from dataclasses import dataclass, field

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet50_Weights

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skimage.feature import local_binary_pattern

# Set up logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DatasetDownloader:
    """Handle downloading and organizing the CIFAKE dataset"""
    
    def __init__(self, download_path: str = './cifake_data', target_path: str = './dataset'):
        self.download_path = Path(download_path)
        self.target_path = Path(target_path)
        
    def download_cifake(self) -> bool:
        """Download CIFAKE dataset from Kaggle"""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            logging.info("Downloading CIFAKE dataset from Kaggle...")
            logging.info("This might take a few minutes depending on your connection...")
            
            api = KaggleApi()
            api.authenticate()
            
            # Download the dataset
            api.dataset_download_files(
                'birdy654/cifake-real-and-ai-generated-synthetic-images',
                path=str(self.download_path),
                unzip=True
            )
            
            logging.info("Download complete!")
            return True
            
        except ImportError:
            logging.error("Kaggle API not found. Install it with: pip install kaggle")
            return False
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            logging.error("Make sure you have kaggle.json in ~/.kaggle/")
            return False
    
    def organize_dataset(self) -> bool:
        """Organize downloaded dataset into real/ and ai/ folders"""
        try:
            logging.info("Organizing dataset into proper structure...")
            
            # Create target directories
            (self.target_path / 'real').mkdir(parents=True, exist_ok=True)
            (self.target_path / 'ai').mkdir(parents=True, exist_ok=True)
            
            # CIFAKE typically has train/ and test/ folders with REAL/ and FAKE/ subdirectories
            # We'll combine both train and test for a larger dataset
            moved_count = {'real': 0, 'ai': 0}
            
            for split in ['train', 'test']:
                split_path = self.download_path / split
                
                if not split_path.exists():
                    # Try without split folders
                    split_path = self.download_path
                
                # Look for REAL images
                real_source = split_path / 'REAL'
                if real_source.exists():
                    for img_file in real_source.glob('*.png'):
                        target_file = self.target_path / 'real' / f"{split}_{img_file.name}"
                        shutil.copy2(img_file, target_file)
                        moved_count['real'] += 1
                
                # Look for FAKE images
                fake_source = split_path / 'FAKE'
                if fake_source.exists():
                    for img_file in fake_source.glob('*.png'):
                        target_file = self.target_path / 'ai' / f"{split}_{img_file.name}"
                        shutil.copy2(img_file, target_file)
                        moved_count['ai'] += 1
            
            if moved_count['real'] == 0 and moved_count['ai'] == 0:
                logging.error("Could not find REAL/ and FAKE/ folders in the download")
                logging.error(f"Please check the structure in {self.download_path}")
                return False
            
            logging.info(f"Organized {moved_count['real']} real images and {moved_count['ai']} AI images")
            return True
            
        except Exception as e:
            logging.error(f"Failed to organize dataset: {e}")
            return False
    
    def setup(self, force_download: bool = False) -> bool:
        """
        Complete setup: download and organize dataset
        
        Args:
            force_download: If True, download even if dataset already exists
        """
        # Check if dataset already exists
        if not force_download:
            real_dir = self.target_path / 'real'
            ai_dir = self.target_path / 'ai'
            
            if real_dir.exists() and ai_dir.exists():
                real_count = len(list(real_dir.glob('*.png')))
                ai_count = len(list(ai_dir.glob('*.png')))
                
                if real_count > 0 and ai_count > 0:
                    logging.info(f"Dataset already exists with {real_count} real and {ai_count} AI images")
                    return True
        
        # Download dataset
        if not self.download_cifake():
            return False
        
        # Organize it
        if not self.organize_dataset():
            return False
        
        logging.info("Dataset setup complete!")
        return True


@dataclass
class Config:
    """Store all the configuration settings in one place"""
    # Where to find our data and save models
    data_path: str = "path/to/your/dataset"
    model_dir: str = "models"
    class_names: List[str] = field(default_factory=lambda: ["real", "ai"])

    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    
    # Training parameters
    test_size: float = 0.2
    batch_size: int = 32
    num_workers: int = 4
    random_state: int = 42

    # CNN settings
    cnn_backbone: str = "resnet50"
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Traditional ML settings
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 15,
        'random_state': 42,
        'n_jobs': -1
    })
    svm_params: Dict[str, Any] = field(default_factory=lambda: {
        'kernel': 'rbf',
        'probability': True,
        'random_state': 42
    })
    
    # Feature extraction parameters
    LBP_POINTS: int = 24
    LBP_RADIUS: int = 3
    HIST_BINS: int = 32


class ImageDataset(Dataset):
    """
    Simple dataset class that loads images from disk.
    Handles corrupted images gracefully by skipping them.
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except (IOError, OSError) as e:
            logging.warning(f"Skipping corrupted image: {image_path}")
            # Just skip to the next valid image
            return self.__getitem__((idx + 1) % len(self))


class FeatureExtractor:
    """
    Extract handcrafted features from images.
    These complement the deep learning features.
    """
    def __init__(self, config: Config):
        self.config = config

    def extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Get color distribution features from RGB and HSV color spaces"""
        features = []
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Extract histograms from both color spaces
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [self.config.HIST_BINS], [0, 256])
            features.extend(hist.flatten())
            
            hist = cv2.calcHist([hsv], [i], None, [self.config.HIST_BINS], [0, 256])
            features.extend(hist.flatten())
        
        return np.array(features)

    def extract_texture(self, image: np.ndarray) -> np.ndarray:
        """Extract texture patterns and edge information"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern for texture
        lbp = local_binary_pattern(
            gray,
            P=self.config.LBP_POINTS,
            R=self.config.LBP_RADIUS,
            method='uniform'
        )
        lbp_hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.config.LBP_POINTS + 2,
            range=(0, self.config.LBP_POINTS + 1)
        )
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine everything
        return np.array([
            *lbp_hist,
            np.mean(gray),
            np.std(gray),
            edge_density
        ])

    def extract_all(self, image: np.ndarray) -> np.ndarray:
        """Combine all feature types into one vector"""
        color_feats = self.extract_color_histogram(image)
        texture_feats = self.extract_texture(image)
        return np.concatenate([color_feats, texture_feats])


class CNNDetector(nn.Module):
    """
    CNN model using a pretrained ResNet backbone.
    We freeze the early layers and train a custom classifier on top.
    """
    def __init__(self, backbone_name: str, num_classes: int):
        super().__init__()
        
        if backbone_name == "resnet50":
            # Load pretrained ResNet
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the original classifier
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not supported yet")

        # Add our own classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class Trainer:
    """Handles training for both CNN and traditional ML models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = FeatureExtractor(config)
        self.scaler = StandardScaler()
        
        # Data augmentation for training
        self.train_transform = T.Compose([
            T.Resize(config.image_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # No augmentation for validation/test
        self.test_transform = T.Compose([
            T.Resize(config.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_data(self) -> Tuple:
        """Load images from the dataset folder"""
        data_dir = Path(self.config.data_path)
        image_paths = []
        labels = []

        # Walk through class folders (real, ai)
        for label, class_name in enumerate(self.config.class_names):
            class_path = data_dir / class_name
            
            if not class_path.exists():
                logging.warning(f"Can't find folder: {class_path}")
                continue
            
            # Grab all image files
            for img_file in class_path.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append(str(img_file))
                    labels.append(label)
        
        if not image_paths:
            raise FileNotFoundError(
                f"No images found in {data_dir}. "
                "Make sure you have 'real' and 'ai' subfolders with images."
            )

        # Split into train and test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths,
            labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels
        )

        # Create data loaders
        train_loader = DataLoader(
            ImageDataset(train_paths, train_labels, self.train_transform),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            ImageDataset(test_paths, test_labels, self.test_transform),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader, train_paths, train_labels, test_paths, test_labels

    def _train_cnn(self, train_loader: DataLoader, test_loader: DataLoader):
        """Train the CNN model"""
        logging.info("Starting CNN training...")
        
        model = CNNDetector(
            self.config.cnn_backbone,
            len(self.config.class_names)
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        best_acc = 0.0
        model_path = Path(self.config.model_dir) / "cnn_model.pth"
        
        for epoch in range(self.config.epochs):
            model.train()
            
            # Training loop
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Check validation accuracy
            val_acc, _, _ = self._evaluate_cnn(model, test_loader)
            logging.info(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.2f}%")
            
            # Save if this is our best model so far
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)

        logging.info(f"Training done! Best accuracy: {best_acc:.2f}%")

    def _evaluate_cnn(self, model: nn.Module, loader: DataLoader) -> Tuple[float, List[int], List[int]]:
        """Evaluate the CNN model on a dataset"""
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100.0 * correct / total
        return accuracy, all_labels, all_preds

    def _train_traditional_models(self, train_paths: List[str], train_labels: List[int]):
        """Train Random Forest and SVM models on extracted features"""
        logging.info("Extracting features for traditional ML models...")
        
        # Extract features in parallel for speed
        def extract_features(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.config.image_size)
            return self.feature_extractor.extract_all(img)

        features = np.array(
            Parallel(n_jobs=-1)(
                delayed(extract_features)(p)
                for p in tqdm(train_paths, desc="Processing images")
            )
        )
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train multiple models
        rf_model = RandomForestClassifier(**self.config.rf_params)
        svm_model = SVC(**self.config.svm_params)
        ensemble = VotingClassifier(
            estimators=[('rf', rf_model), ('svm', svm_model)],
            voting='soft'
        )
        
        models = {
            "rf": rf_model,
            "svm": svm_model,
            "ensemble": ensemble
        }
        
        for name, model in models.items():
            logging.info(f"Training {name.upper()}...")
            model.fit(features_scaled, train_labels)
            joblib.dump(model, Path(self.config.model_dir) / f"{name}_model.pkl")

        # Save the scaler too
        joblib.dump(self.scaler, Path(self.config.model_dir) / 'scaler.pkl')
        logging.info("Traditional models trained and saved")

    def run(self):
        """Run the complete training pipeline"""
        # Make sure we have a place to save models
        Path(self.config.model_dir).mkdir(exist_ok=True)
        
        # Load the data
        train_loader, test_loader, train_paths, train_labels, test_paths, test_labels = self._load_data()

        # Train CNN
        self._train_cnn(train_loader, test_loader)
        
        # Evaluate final CNN performance
        cnn_model = CNNDetector(
            self.config.cnn_backbone,
            len(self.config.class_names)
        ).to(self.device)
        cnn_model.load_state_dict(
            torch.load(Path(self.config.model_dir) / "cnn_model.pth")
        )
        
        acc, labels, preds = self._evaluate_cnn(cnn_model, test_loader)
        logging.info(f"Final CNN Test Accuracy: {acc:.2f}%")
        print("\nCNN Classification Report:")
        print(classification_report(labels, preds, target_names=self.config.class_names))

        # Train traditional models
        self._train_traditional_models(train_paths, train_labels)


class Predictor:
    """Load trained models and make predictions on new images"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.feature_extractor = FeatureExtractor(config)
        
        # Image preprocessing (same as test set)
        self.transform = T.Compose([
            T.Resize(config.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_models()

    def _load_models(self):
        """Load all trained models from disk"""
        model_dir = Path(self.config.model_dir)
        
        try:
            # Load CNN
            self.models['cnn'] = CNNDetector(
                self.config.cnn_backbone,
                len(self.config.class_names)
            ).to(self.device)
            self.models['cnn'].load_state_dict(
                torch.load(model_dir / 'cnn_model.pth', map_location=self.device)
            )
            self.models['cnn'].eval()
            
            # Load traditional models
            for model_name in ['rf', 'svm', 'ensemble']:
                model_path = model_dir / f"{model_name}_model.pkl"
                self.models[model_name] = joblib.load(model_path)
            
            # Load the feature scaler
            self.models['scaler'] = joblib.load(model_dir / 'scaler.pkl')

            logging.info("All models loaded successfully")
            
        except FileNotFoundError as e:
            logging.error(f"Couldn't load models: {e}")
            logging.error("Have you trained the models yet?")
            raise

    def predict(self, image_path: str) -> Dict[str, Dict[str, float]]:
        """
        Predict whether an image is AI-generated or real.
        Returns probabilities from all models.
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except IOError:
            logging.error(f"Can't open image: {image_path}")
            return {}

        results = {}

        # CNN prediction
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.models['cnn'](tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        results['cnn'] = {
            self.config.class_names[i]: float(probs[i])
            for i in range(len(self.config.class_names))
        }
        
        # Traditional model predictions
        # First extract features
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_cv = cv2.resize(image_cv, self.config.image_size)
        features = self.feature_extractor.extract_all(
            cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        )
        features_scaled = self.models['scaler'].transform(features.reshape(1, -1))
        
        # Get predictions from each traditional model
        for model_name in ['rf', 'svm', 'ensemble']:
            probs = self.models[model_name].predict_proba(features_scaled)[0]
            results[model_name] = {
                self.config.class_names[i]: float(probs[i])
                for i in range(len(self.config.class_names))
            }
            
        return results


def main():
    """Main entry point"""
    
    # Set up configuration
    config = Config(
        data_path="./dataset/",  # Update this to your dataset path
        epochs=5,
        model_dir="trained_models"
    )

    # Uncomment to train models
    # logging.info("=" * 50)
    # logging.info("STARTING TRAINING")
    # logging.info("=" * 50)
    # trainer = Trainer(config)
    # trainer.run()
    # logging.info("Training complete!")

    # Run prediction example
    logging.info("\n" + "=" * 50)
    logging.info("RUNNING PREDICTION EXAMPLE")
    logging.info("=" * 50)
    
    try:
        predictor = Predictor(config)
        
        # Create a dummy test image if needed
        test_image = "test_image.jpg"
        if not Path(test_image).exists():
            Image.new('RGB', (100, 100)).save(test_image)
            logging.info(f"Created dummy test image: {test_image}")
        
        # Make prediction
        predictions = predictor.predict(test_image)
        
        # Print results nicely
        import json
        print("\nPredictions:")
        print(json.dumps(predictions, indent=2))
        
    except FileNotFoundError:
        logging.error("Models not found. Please train the models first!")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
