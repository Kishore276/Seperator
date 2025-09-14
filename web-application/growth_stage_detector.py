"""
Plant Growth Stage Detection System
Identifies plants at different growth stages for more accurate classification
"""

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

@dataclass
class GrowthStageInfo:
    """Information about a plant growth stage"""
    stage: str
    confidence: float
    characteristics: List[str]
    typical_features: Dict[str, str]
    identification_tips: List[str]

class GrowthStageDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize growth stage detector
        
        Args:
            model_path: Path to pre-trained growth stage model
        """
        self.model = None
        self.growth_stages = [
            'seedling', 'juvenile', 'vegetative', 'budding',
            'flowering', 'fruiting', 'mature', 'senescent'
        ]
        
        # Load growth stage characteristics
        self.stage_characteristics = self._load_stage_characteristics()
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = self._build_growth_stage_model()
    
    def _load_stage_characteristics(self) -> Dict:
        """Load characteristics for each growth stage"""
        return {
            'seedling': {
                'description': 'Early growth stage with first true leaves',
                'features': {
                    'size': 'Very small (1-5 cm)',
                    'leaves': 'Cotyledons and first true leaves',
                    'stem': 'Thin, tender stem',
                    'roots': 'Developing root system'
                },
                'visual_cues': [
                    'Small size relative to mature plant',
                    'Simple leaf structure',
                    'Tender, green appearance',
                    'Low to ground growth'
                ]
            },
            'juvenile': {
                'description': 'Young plant with developing characteristics',
                'features': {
                    'size': 'Small to medium (5-20 cm)',
                    'leaves': 'True leaves developing',
                    'stem': 'Strengthening stem',
                    'roots': 'Expanding root system'
                },
                'visual_cues': [
                    'Recognizable plant structure',
                    'Multiple true leaves',
                    'Increasing height',
                    'More robust appearance'
                ]
            },
            'vegetative': {
                'description': 'Active growth phase, no reproductive structures',
                'features': {
                    'size': 'Medium to large size',
                    'leaves': 'Full leaf development',
                    'stem': 'Strong, well-developed stem',
                    'roots': 'Extensive root system'
                },
                'visual_cues': [
                    'Rapid growth',
                    'Dense foliage',
                    'No flowers or buds',
                    'Healthy green color'
                ]
            },
            'budding': {
                'description': 'Formation of flower buds, pre-flowering',
                'features': {
                    'size': 'Near mature size',
                    'leaves': 'Mature leaves',
                    'buds': 'Visible flower buds',
                    'stem': 'Fully developed'
                },
                'visual_cues': [
                    'Flower buds visible',
                    'Buds may be swelling',
                    'Plant preparing to flower',
                    'Mature vegetative growth'
                ]
            },
            'flowering': {
                'description': 'Active flowering stage',
                'features': {
                    'size': 'Mature size',
                    'flowers': 'Open flowers present',
                    'leaves': 'Mature foliage',
                    'reproductive': 'Active pollination'
                },
                'visual_cues': [
                    'Visible flowers',
                    'Bright colors (often)',
                    'Possible pollinators present',
                    'Peak attractiveness'
                ]
            },
            'fruiting': {
                'description': 'Fruit/seed development stage',
                'features': {
                    'size': 'Mature size',
                    'fruits': 'Developing fruits/seeds',
                    'flowers': 'May have some flowers',
                    'energy': 'Focused on reproduction'
                },
                'visual_cues': [
                    'Visible fruits or seed pods',
                    'Flowers may be fading',
                    'Plant energy in reproduction',
                    'Fruits developing/maturing'
                ]
            },
            'mature': {
                'description': 'Fully developed plant at peak',
                'features': {
                    'size': 'Maximum size',
                    'structure': 'Complete development',
                    'reproduction': 'Capable of reproduction',
                    'stability': 'Stable growth'
                },
                'visual_cues': [
                    'Full size and structure',
                    'Well-established appearance',
                    'May have multiple growth cycles',
                    'Robust and healthy'
                ]
            },
            'senescent': {
                'description': 'Aging/declining stage',
                'features': {
                    'leaves': 'Yellowing or browning',
                    'structure': 'May be declining',
                    'energy': 'Reduced vigor',
                    'reproduction': 'End of cycle'
                },
                'visual_cues': [
                    'Yellowing leaves',
                    'Reduced vigor',
                    'Possible die-back',
                    'End of growing season'
                ]
            }
        }
    
    def _build_growth_stage_model(self):
        """Build a CNN model for growth stage classification"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(224, 224, 3)),
            
            # Data augmentation
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Feature extraction layers
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            
            # Classification layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.growth_stages), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def detect_growth_stage(self, image, return_all_stages=False) -> GrowthStageInfo:
        """
        Detect the growth stage of a plant in an image
        
        Args:
            image: Input image (PIL Image or numpy array)
            return_all_stages: Whether to return probabilities for all stages
            
        Returns:
            GrowthStageInfo object with detected stage and details
        """
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Get model prediction if available
        if self.model:
            predictions = self.model.predict(processed_image, verbose=0)
            stage_probs = predictions[0]
            predicted_stage_idx = np.argmax(stage_probs)
            confidence = stage_probs[predicted_stage_idx]
            predicted_stage = self.growth_stages[predicted_stage_idx]
        else:
            # Fallback to rule-based detection
            predicted_stage, confidence = self._rule_based_stage_detection(image)
        
        # Get stage characteristics
        stage_chars = self.stage_characteristics.get(predicted_stage, {})
        
        # Create result
        result = GrowthStageInfo(
            stage=predicted_stage,
            confidence=float(confidence),
            characteristics=stage_chars.get('visual_cues', []),
            typical_features=stage_chars.get('features', {}),
            identification_tips=self._get_identification_tips(predicted_stage)
        )
        
        if return_all_stages and self.model:
            result.all_stage_probabilities = {
                stage: float(prob) for stage, prob in zip(self.growth_stages, stage_probs)
            }
        
        return result
    
    def _preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize to model input size
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(image)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _rule_based_stage_detection(self, image) -> Tuple[str, float]:
        """
        Rule-based growth stage detection using image analysis
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Analyze image characteristics
        features = self._extract_visual_features(img_array, hsv)
        
        # Rule-based classification
        if features['has_flowers']:
            if features['has_fruits']:
                return 'fruiting', 0.8
            else:
                return 'flowering', 0.85
        elif features['has_buds']:
            return 'budding', 0.75
        elif features['is_very_small']:
            return 'seedling', 0.7
        elif features['is_yellowing']:
            return 'senescent', 0.7
        elif features['is_small']:
            return 'juvenile', 0.6
        elif features['is_dense_green']:
            return 'vegetative', 0.65
        else:
            return 'mature', 0.5
    
    def _extract_visual_features(self, img_rgb, img_hsv) -> Dict[str, bool]:
        """Extract visual features from image"""
        features = {}
        
        # Color analysis
        h, s, v = cv2.split(img_hsv)
        
        # Check for flowers (bright colors, high saturation)
        bright_mask = (s > 100) & (v > 100)
        flower_colors = ((h < 30) | (h > 150)) & bright_mask  # Red/pink/purple
        yellow_flowers = (h >= 15) & (h <= 35) & bright_mask  # Yellow
        features['has_flowers'] = np.sum(flower_colors | yellow_flowers) > (img_rgb.shape[0] * img_rgb.shape[1] * 0.05)
        
        # Check for fruits (round/oval shapes, various colors)
        # This is simplified - would need more sophisticated shape detection
        features['has_fruits'] = False  # Placeholder
        
        # Check for buds (small bright spots)
        features['has_buds'] = False  # Placeholder
        
        # Size estimation (relative to image)
        green_mask = (h >= 35) & (h <= 85) & (s > 50)
        plant_area = np.sum(green_mask)
        total_area = img_rgb.shape[0] * img_rgb.shape[1]
        plant_ratio = plant_area / total_area
        
        features['is_very_small'] = plant_ratio < 0.1
        features['is_small'] = plant_ratio < 0.3
        features['is_dense_green'] = plant_ratio > 0.6 and np.mean(s[green_mask]) > 80
        
        # Check for yellowing (senescence)
        yellow_mask = (h >= 15) & (h <= 35) & (s > 30)
        yellow_ratio = np.sum(yellow_mask) / total_area
        features['is_yellowing'] = yellow_ratio > 0.2
        
        return features
    
    def _get_identification_tips(self, stage: str) -> List[str]:
        """Get identification tips for a specific growth stage"""
        tips = {
            'seedling': [
                "Look for cotyledons (seed leaves)",
                "Very small size compared to mature plant",
                "Simple leaf structure",
                "Close to ground level"
            ],
            'juvenile': [
                "True leaves are developing",
                "Plant structure becoming recognizable",
                "Increasing in height and complexity",
                "No reproductive structures yet"
            ],
            'vegetative': [
                "Rapid growth and leaf development",
                "No flowers or buds visible",
                "Dense, healthy green foliage",
                "Focus on size increase"
            ],
            'budding': [
                "Small flower buds visible",
                "Buds may be swelling",
                "Plant near mature size",
                "Preparing for flowering"
            ],
            'flowering': [
                "Open flowers clearly visible",
                "Often bright colors",
                "May attract pollinators",
                "Peak visual appeal"
            ],
            'fruiting': [
                "Fruits or seed pods developing",
                "Some flowers may still be present",
                "Plant energy focused on reproduction",
                "Fruits in various stages of development"
            ],
            'mature': [
                "Full size and development",
                "Well-established structure",
                "May show multiple growth cycles",
                "Robust and stable appearance"
            ],
            'senescent': [
                "Yellowing or browning leaves",
                "Reduced vigor and growth",
                "End of growing season signs",
                "Natural aging process"
            ]
        }
        
        return tips.get(stage, [])
    
    def get_stage_progression(self, plant_type: str) -> List[str]:
        """Get typical growth stage progression for a plant type"""
        # This could be customized based on plant type
        # For now, return general progression
        return [
            'seedling', 'juvenile', 'vegetative', 
            'budding', 'flowering', 'fruiting', 'mature'
        ]
    
    def save_model(self, save_path: str):
        """Save the growth stage model"""
        if self.model:
            self.model.save(save_path)
            print(f"Growth stage model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a pre-trained growth stage model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Growth stage model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = self._build_growth_stage_model()

# Utility functions
def detect_plant_growth_stage(image_path: str) -> GrowthStageInfo:
    """Convenience function to detect growth stage from image file"""
    detector = GrowthStageDetector()
    image = Image.open(image_path)
    return detector.detect_growth_stage(image)

def analyze_growth_progression(image_paths: List[str]) -> List[GrowthStageInfo]:
    """Analyze growth progression from multiple images"""
    detector = GrowthStageDetector()
    results = []
    
    for path in image_paths:
        image = Image.open(path)
        stage_info = detector.detect_growth_stage(image)
        results.append(stage_info)
    
    return results
