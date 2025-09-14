"""
Advanced Plant Classification Model Architecture
Supports multiple state-of-the-art models with transfer learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    EfficientNetB4, EfficientNetB7, ResNet152V2, 
    DenseNet201, InceptionResNetV2
)
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedPlantClassifier:
    def __init__(self, 
                 model_type='efficientnet_b4',
                 num_classes=1000,
                 input_shape=(224, 224, 3),
                 use_mixed_precision=True):
        """
        Initialize advanced plant classifier
        
        Args:
            model_type: Type of backbone model to use
            num_classes: Number of plant classes to classify
            input_shape: Input image shape
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.class_names = []
        self.growth_stage_model = None
        
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
    
    def build_model(self, include_growth_stage=True):
        """
        Build the complete model architecture
        """
        # Build main classification model
        self.model = self._build_backbone_model()
        
        # Build growth stage detection model if requested
        if include_growth_stage:
            self.growth_stage_model = self._build_growth_stage_model()
        
        return self.model
    
    def _build_backbone_model(self):
        """
        Build the main plant classification model
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape)
        
        # Data augmentation layer (applied during training)
        augmentation = self._create_augmentation_layer()
        x = augmentation(inputs)
        
        # Backbone model
        if self.model_type == 'efficientnet_b4':
            backbone = EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        elif self.model_type == 'efficientnet_b7':
            backbone = EfficientNetB7(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        elif self.model_type == 'resnet152':
            backbone = ResNet152V2(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        elif self.model_type == 'densenet201':
            backbone = DenseNet201(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        elif self.model_type == 'inception_resnet':
            backbone = InceptionResNetV2(
                weights='imagenet',
                include_top=False,
                input_tensor=x
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Freeze backbone layers initially
        backbone.trainable = False
        
        # Feature extraction
        x = backbone.output
        
        # Global pooling and feature processing
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Dense layers with dropout
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-task outputs
        # Main classification output
        main_output = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='plant_classification',
            dtype='float32'
        )(x)
        
        # Plant type output (crop vs weed)
        type_output = layers.Dense(
            2, 
            activation='softmax', 
            name='plant_type',
            dtype='float32'
        )(x)
        
        # Confidence estimation output
        confidence_output = layers.Dense(
            1, 
            activation='sigmoid', 
            name='confidence',
            dtype='float32'
        )(x)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[main_output, type_output, confidence_output],
            name=f'advanced_plant_classifier_{self.model_type}'
        )
        
        return model
    
    def _build_growth_stage_model(self):
        """
        Build growth stage detection model
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # Lightweight backbone for growth stage detection
        backbone = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs
        )
        backbone.trainable = False
        
        x = backbone.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Growth stage classification (7 common stages)
        growth_stages = [
            'seedling', 'juvenile', 'vegetative', 
            'flowering', 'fruiting', 'mature', 'senescent'
        ]
        
        stage_output = layers.Dense(
            len(growth_stages),
            activation='softmax',
            name='growth_stage',
            dtype='float32'
        )(x)
        
        model = Model(inputs=inputs, outputs=stage_output, name='growth_stage_classifier')
        return model
    
    def _create_augmentation_layer(self):
        """
        Create data augmentation layer
        """
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
        ], name="augmentation")
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compile the model with appropriate loss functions and metrics
        """
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5
        )
        
        # Loss functions for multi-task learning
        losses = {
            'plant_classification': 'categorical_crossentropy',
            'plant_type': 'categorical_crossentropy',
            'confidence': 'binary_crossentropy'
        }
        
        # Loss weights
        loss_weights = {
            'plant_classification': 1.0,
            'plant_type': 0.5,
            'confidence': 0.3
        }
        
        # Metrics
        metrics = {
            'plant_classification': ['accuracy', 'top_5_accuracy'],
            'plant_type': ['accuracy'],
            'confidence': ['mae']
        }
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        if self.growth_stage_model:
            self.growth_stage_model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def create_callbacks(self, model_save_path='best_model.h5'):
        """
        Create training callbacks
        """
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_plant_classification_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.CSVLogger('training_log.csv'),
        ]
        
        return callbacks
    
    def fine_tune_model(self, unfreeze_layers=50):
        """
        Fine-tune the model by unfreezing top layers
        """
        if self.model is None:
            raise ValueError("Model must be built before fine-tuning")
        
        # Unfreeze top layers of backbone
        backbone = self.model.layers[1]  # Assuming backbone is second layer
        backbone.trainable = True
        
        # Freeze bottom layers
        for layer in backbone.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=1e-5)
    
    def predict_with_confidence(self, image, return_all_outputs=False):
        """
        Make prediction with confidence estimation
        """
        if self.model is None:
            raise ValueError("Model must be built and loaded before prediction")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(image, verbose=0)
        main_pred, type_pred, confidence = predictions
        
        # Process main classification
        class_idx = np.argmax(main_pred[0])
        class_confidence = main_pred[0][class_idx]
        
        # Process plant type
        is_weed = np.argmax(type_pred[0]) == 1
        type_confidence = type_pred[0][np.argmax(type_pred[0])]
        
        # Overall confidence
        overall_confidence = confidence[0][0]
        
        result = {
            'predicted_class': class_idx,
            'class_name': self.class_names[class_idx] if self.class_names else f"Class_{class_idx}",
            'class_confidence': float(class_confidence),
            'is_weed': is_weed,
            'type_confidence': float(type_confidence),
            'overall_confidence': float(overall_confidence),
            'final_confidence': float(class_confidence * overall_confidence)
        }
        
        # Add growth stage prediction if available
        if self.growth_stage_model:
            stage_pred = self.growth_stage_model.predict(image, verbose=0)
            stage_idx = np.argmax(stage_pred[0])
            growth_stages = [
                'seedling', 'juvenile', 'vegetative', 
                'flowering', 'fruiting', 'mature', 'senescent'
            ]
            result['growth_stage'] = growth_stages[stage_idx]
            result['stage_confidence'] = float(stage_pred[0][stage_idx])
        
        if return_all_outputs:
            result['all_class_probabilities'] = main_pred[0].tolist()
            result['type_probabilities'] = type_pred[0].tolist()
        
        return result
    
    def load_class_names(self, class_names_path):
        """
        Load class names from file
        """
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                if class_names_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.class_names = list(data.keys())
                    else:
                        self.class_names = data
                else:
                    self.class_names = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: Class names file not found at {class_names_path}")
    
    def save_model(self, save_path):
        """
        Save the complete model
        """
        if self.model:
            self.model.save(save_path)
            print(f"Main model saved to {save_path}")
        
        if self.growth_stage_model:
            stage_path = save_path.replace('.h5', '_growth_stage.h5')
            self.growth_stage_model.save(stage_path)
            print(f"Growth stage model saved to {stage_path}")
    
    def load_model(self, model_path, growth_stage_path=None):
        """
        Load a saved model
        """
        self.model = keras.models.load_model(model_path)
        print(f"Main model loaded from {model_path}")
        
        if growth_stage_path and os.path.exists(growth_stage_path):
            self.growth_stage_model = keras.models.load_model(growth_stage_path)
            print(f"Growth stage model loaded from {growth_stage_path}")

# Utility functions
def create_ensemble_model(model_paths, weights=None):
    """
    Create an ensemble of multiple models
    """
    models = []
    for path in model_paths:
        model = keras.models.load_model(path)
        models.append(model)
    
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    def ensemble_predict(image):
        predictions = []
        for model in models:
            pred = model.predict(image, verbose=0)
            predictions.append(pred[0])  # Main classification output
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    return ensemble_predict

def calculate_model_uncertainty(predictions):
    """
    Calculate prediction uncertainty using entropy
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-8
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Calculate entropy
    entropy = -np.sum(predictions * np.log(predictions))
    
    # Normalize entropy (0 = certain, 1 = uncertain)
    max_entropy = np.log(len(predictions))
    normalized_uncertainty = entropy / max_entropy
    
    return normalized_uncertainty
