"""
Training Script for Advanced Plant Classification Model
Trains the comprehensive plant classification system with global coverage
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import argparse

# Import our modules
from advanced_model import AdvancedPlantClassifier
from advanced_preprocessing import AdvancedImagePreprocessor
from growth_stage_detector import GrowthStageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantClassificationTrainer:
    def __init__(self, data_path: str, model_config: dict):
        """
        Initialize trainer
        
        Args:
            data_path: Path to training data
            model_config: Model configuration dictionary
        """
        self.data_path = data_path
        self.model_config = model_config
        self.preprocessor = AdvancedImagePreprocessor()
        
        # Initialize models
        self.plant_classifier = AdvancedPlantClassifier(**model_config)
        self.growth_detector = GrowthStageDetector()
        
        # Training data
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.class_names = []
        
    def prepare_data(self):
        """Prepare training data from directory structure"""
        logger.info("Preparing training data...")
        
        images = []
        labels = []
        class_names = []
        
        # Load global plant database for class mapping
        with open('global_plant_database.json', 'r') as f:
            plant_db = json.load(f)
        
        # Create class mapping
        class_to_idx = {}
        for i, plant in enumerate(plant_db):
            scientific_name = plant['ScientificName']
            if scientific_name not in class_to_idx:
                class_to_idx[scientific_name] = len(class_names)
                class_names.append(scientific_name)
        
        self.class_names = class_names
        
        # Load images from data directory
        for class_name in os.listdir(self.data_path):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                continue
                
            class_idx = class_to_idx.get(class_name, -1)
            if class_idx == -1:
                logger.warning(f"Class {class_name} not found in database")
                continue
            
            # Load images from class directory
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        # Load and preprocess image
                        image = Image.open(img_path)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Apply preprocessing
                        enhanced_image, _ = self.preprocessor.adaptive_preprocessing(image)
                        processed_img = self.preprocessor.preprocess_for_model(
                            enhanced_image, self.model_config['model_type']
                        )
                        
                        images.append(processed_img[0])  # Remove batch dimension
                        labels.append(class_idx)
                        
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {e}")
        
        if not images:
            raise ValueError("No training images found")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Split data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert labels to categorical
        num_classes = len(self.class_names)
        self.y_train = keras.utils.to_categorical(self.y_train, num_classes)
        self.y_val = keras.utils.to_categorical(self.y_val, num_classes)
        
        logger.info(f"Training data prepared: {len(self.X_train)} train, {len(self.X_val)} val")
        logger.info(f"Number of classes: {num_classes}")
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        # Training generator with augmentation
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation generator (no augmentation)
        val_datagen = keras.preprocessing.image.ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            self.X_train, self.y_train,
            batch_size=32,
            shuffle=True
        )
        
        val_generator = val_datagen.flow(
            self.X_val, self.y_val,
            batch_size=32,
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def train_plant_classifier(self, epochs=50):
        """Train the main plant classification model"""
        logger.info("Training plant classification model...")
        
        # Update model with correct number of classes
        self.model_config['num_classes'] = len(self.class_names)
        self.plant_classifier = AdvancedPlantClassifier(**self.model_config)
        
        # Build and compile model
        model = self.plant_classifier.build_model()
        self.plant_classifier.compile_model()
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        
        # Create callbacks
        callbacks = self.plant_classifier.create_callbacks('best_plant_model.h5')
        
        # Add custom callbacks
        callbacks.extend([
            keras.callbacks.TensorBoard(
                log_dir=f'logs/plant_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                histogram_freq=1
            ),
            keras.callbacks.ProgbarLogger(count_mode='steps')
        ])
        
        # Train model
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.plant_classifier.save_model('final_plant_model.h5')
        
        # Save class names
        with open('plant_class_names.json', 'w') as f:
            json.dump(self.class_names, f)
        
        return history
    
    def train_growth_stage_detector(self, epochs=30):
        """Train growth stage detection model"""
        logger.info("Training growth stage detection model...")
        
        # For growth stage training, we would need labeled growth stage data
        # This is a simplified version
        
        growth_stages = [
            'seedling', 'juvenile', 'vegetative', 'budding',
            'flowering', 'fruiting', 'mature', 'senescent'
        ]
        
        # Build model
        model = self.growth_detector._build_growth_stage_model()
        
        # For demonstration, we'll create synthetic training data
        # In practice, you would load real growth stage labeled data
        X_growth = np.random.random((1000, 224, 224, 3))
        y_growth = keras.utils.to_categorical(
            np.random.randint(0, len(growth_stages), 1000),
            len(growth_stages)
        )
        
        X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(
            X_growth, y_growth, test_size=0.2, random_state=42
        )
        
        # Train model
        history = model.fit(
            X_train_g, y_train_g,
            validation_data=(X_val_g, y_val_g),
            epochs=epochs,
            batch_size=32,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5),
                keras.callbacks.ReduceLROnPlateau(patience=3)
            ]
        )
        
        # Save model
        self.growth_detector.save_model('growth_stage_model.h5')
        
        return history
    
    def fine_tune_models(self, fine_tune_epochs=20):
        """Fine-tune pre-trained models"""
        logger.info("Fine-tuning models...")
        
        # Fine-tune plant classifier
        self.plant_classifier.fine_tune_model(unfreeze_layers=50)
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators()
        
        # Fine-tune with lower learning rate
        history = self.plant_classifier.model.fit(
            train_gen,
            epochs=fine_tune_epochs,
            validation_data=val_gen,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5),
                keras.callbacks.ReduceLROnPlateau(patience=3)
            ]
        )
        
        # Save fine-tuned model
        self.plant_classifier.save_model('fine_tuned_plant_model.h5')
        
        return history
    
    def evaluate_models(self):
        """Evaluate trained models"""
        logger.info("Evaluating models...")
        
        if self.plant_classifier.model and self.X_val is not None:
            # Evaluate plant classifier
            val_loss, val_acc = self.plant_classifier.model.evaluate(
                self.X_val, self.y_val, verbose=0
            )
            
            logger.info(f"Plant Classifier - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Get predictions for detailed analysis
            predictions = self.plant_classifier.model.predict(self.X_val)
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(self.y_val, axis=1)
            
            # Calculate per-class accuracy
            class_accuracies = {}
            for i, class_name in enumerate(self.class_names):
                class_mask = true_classes == i
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(predicted_classes[class_mask] == i)
                    class_accuracies[class_name] = class_acc
            
            # Save evaluation results
            eval_results = {
                'overall_accuracy': float(val_acc),
                'overall_loss': float(val_loss),
                'class_accuracies': class_accuracies,
                'num_classes': len(self.class_names),
                'evaluation_date': datetime.now().isoformat()
            }
            
            with open('evaluation_results.json', 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            return eval_results
        
        return None
    
    def plot_training_history(self, history, save_path='training_plots.png'):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training & validation accuracy
        axes[0, 0].plot(history.history['accuracy'])
        axes[0, 0].plot(history.history['val_accuracy'])
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend(['Train', 'Validation'])
        
        # Plot training & validation loss
        axes[0, 1].plot(history.history['loss'])
        axes[0, 1].plot(history.history['val_loss'])
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend(['Train', 'Validation'])
        
        # Plot learning rate if available
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Training plots saved to {save_path}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Advanced Plant Classification Model')
    parser.add_argument('--data_path', type=str, default='training_data',
                       help='Path to training data directory')
    parser.add_argument('--model_type', type=str, default='efficientnet_b4',
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--fine_tune', action='store_true',
                       help='Perform fine-tuning after initial training')
    
    args = parser.parse_args()
    
    # Model configuration
    model_config = {
        'model_type': args.model_type,
        'num_classes': 1000,  # Will be updated based on data
        'input_shape': (224, 224, 3),
        'use_mixed_precision': True
    }
    
    # Initialize trainer
    trainer = PlantClassificationTrainer(args.data_path, model_config)
    
    try:
        # Prepare data
        trainer.prepare_data()
        
        # Train plant classifier
        history = trainer.train_plant_classifier(epochs=args.epochs)
        
        # Train growth stage detector
        growth_history = trainer.train_growth_stage_detector()
        
        # Fine-tune if requested
        if args.fine_tune:
            fine_tune_history = trainer.fine_tune_models()
        
        # Evaluate models
        eval_results = trainer.evaluate_models()
        
        # Plot training history
        trainer.plot_training_history(history)
        
        logger.info("Training completed successfully!")
        
        if eval_results:
            logger.info(f"Final accuracy: {eval_results['overall_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
