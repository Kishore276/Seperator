"""
Comprehensive Testing and Validation Framework
Tests model performance across different plant species, image qualities, and growth stages
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from advanced_model import AdvancedPlantClassifier
from advanced_preprocessing import AdvancedImagePreprocessor
from growth_stage_detector import GrowthStageDetector
from ensemble_system import PlantClassificationEnsemble, PredictionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    detailed_results: Dict
    timestamp: str

class PlantClassificationTestFramework:
    def __init__(self, test_data_path: str = "test_data"):
        """
        Initialize testing framework
        
        Args:
            test_data_path: Path to test data directory
        """
        self.test_data_path = test_data_path
        self.results = []
        self.models = {}
        self.preprocessor = AdvancedImagePreprocessor()
        
        # Create test data structure if it doesn't exist
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Setup test environment and directories"""
        os.makedirs(self.test_data_path, exist_ok=True)
        os.makedirs(f"{self.test_data_path}/high_quality", exist_ok=True)
        os.makedirs(f"{self.test_data_path}/medium_quality", exist_ok=True)
        os.makedirs(f"{self.test_data_path}/low_quality", exist_ok=True)
        os.makedirs(f"{self.test_data_path}/growth_stages", exist_ok=True)
        os.makedirs("test_results", exist_ok=True)
    
    def load_models(self, model_configs: Dict):
        """Load models for testing"""
        for model_name, config in model_configs.items():
            try:
                if model_name == 'advanced_classifier':
                    model = AdvancedPlantClassifier(**config)
                    if os.path.exists(config.get('model_path', '')):
                        model.load_model(config['model_path'])
                    else:
                        model.build_model()
                        model.compile_model()
                    self.models[model_name] = model
                    
                elif model_name == 'growth_stage_detector':
                    model = GrowthStageDetector(config.get('model_path'))
                    self.models[model_name] = model
                    
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
    
    def test_image_quality_robustness(self, test_images: List[str], 
                                    true_labels: List[str]) -> TestResult:
        """Test model performance across different image qualities"""
        logger.info("Testing image quality robustness...")
        
        results = {
            'high_quality': {'predictions': [], 'confidences': []},
            'medium_quality': {'predictions': [], 'confidences': []},
            'low_quality': {'predictions': [], 'confidences': []}
        }
        
        for img_path, true_label in zip(test_images, true_labels):
            # Load original image
            original_img = Image.open(img_path)
            
            # Create different quality versions
            qualities = self._create_quality_variants(original_img)
            
            for quality_level, img in qualities.items():
                if 'advanced_classifier' in self.models:
                    # Preprocess and predict
                    processed_img = self.preprocessor.preprocess_for_model(img)
                    prediction = self.models['advanced_classifier'].predict_with_confidence(processed_img)
                    
                    results[quality_level]['predictions'].append(prediction['class_name'])
                    results[quality_level]['confidences'].append(prediction['final_confidence'])
        
        # Calculate metrics for each quality level
        quality_metrics = {}
        for quality_level in results.keys():
            if results[quality_level]['predictions']:
                accuracy = accuracy_score(true_labels, results[quality_level]['predictions'])
                avg_confidence = np.mean(results[quality_level]['confidences'])
                quality_metrics[quality_level] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'sample_count': len(results[quality_level]['predictions'])
                }
        
        return TestResult(
            test_name="Image Quality Robustness",
            accuracy=np.mean([m['accuracy'] for m in quality_metrics.values()]),
            precision=0.0,  # Would need more detailed calculation
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=np.array([]),
            detailed_results=quality_metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def test_growth_stage_detection(self, test_images: List[str], 
                                  true_stages: List[str]) -> TestResult:
        """Test growth stage detection accuracy"""
        logger.info("Testing growth stage detection...")
        
        if 'growth_stage_detector' not in self.models:
            logger.warning("Growth stage detector not loaded")
            return self._create_empty_result("Growth Stage Detection")
        
        predictions = []
        confidences = []
        
        for img_path, true_stage in zip(test_images, true_stages):
            try:
                image = Image.open(img_path)
                stage_info = self.models['growth_stage_detector'].detect_growth_stage(image)
                predictions.append(stage_info.stage)
                confidences.append(stage_info.confidence)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                predictions.append('unknown')
                confidences.append(0.0)
        
        # Calculate metrics
        accuracy = accuracy_score(true_stages, predictions)
        
        # Create confusion matrix
        unique_stages = list(set(true_stages + predictions))
        cm = confusion_matrix(true_stages, predictions, labels=unique_stages)
        
        return TestResult(
            test_name="Growth Stage Detection",
            accuracy=accuracy,
            precision=0.0,  # Would calculate from confusion matrix
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=cm,
            detailed_results={
                'avg_confidence': np.mean(confidences),
                'stage_distribution': dict(zip(*np.unique(predictions, return_counts=True))),
                'unique_stages': unique_stages
            },
            timestamp=datetime.now().isoformat()
        )
    
    def test_species_classification(self, test_images: List[str], 
                                  true_species: List[str]) -> TestResult:
        """Test species classification accuracy"""
        logger.info("Testing species classification...")
        
        if 'advanced_classifier' not in self.models:
            logger.warning("Advanced classifier not loaded")
            return self._create_empty_result("Species Classification")
        
        predictions = []
        confidences = []
        
        for img_path, true_species in zip(test_images, true_species):
            try:
                image = Image.open(img_path)
                processed_img = self.preprocessor.preprocess_for_model(image)
                prediction = self.models['advanced_classifier'].predict_with_confidence(processed_img)
                
                predictions.append(prediction['class_name'])
                confidences.append(prediction['final_confidence'])
                
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                predictions.append('unknown')
                confidences.append(0.0)
        
        # Calculate metrics
        accuracy = accuracy_score(true_species, predictions)
        
        # Create confusion matrix
        unique_species = list(set(true_species + predictions))
        cm = confusion_matrix(true_species, predictions, labels=unique_species)
        
        return TestResult(
            test_name="Species Classification",
            accuracy=accuracy,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=cm,
            detailed_results={
                'avg_confidence': np.mean(confidences),
                'species_distribution': dict(zip(*np.unique(predictions, return_counts=True))),
                'unique_species': unique_species
            },
            timestamp=datetime.now().isoformat()
        )
    
    def test_ensemble_performance(self, test_images: List[str], 
                                true_labels: List[str]) -> TestResult:
        """Test ensemble system performance"""
        logger.info("Testing ensemble performance...")
        
        ensemble = PlantClassificationEnsemble()
        predictions = []
        confidences = []
        consensus_scores = []
        
        for img_path, true_label in zip(test_images, true_labels):
            try:
                # Create mock predictions from different sources
                mock_predictions = self._create_mock_predictions(img_path)
                
                # Get ensemble result
                ensemble_result = ensemble.combine_predictions(mock_predictions)
                
                predictions.append(ensemble_result.scientific_name)
                confidences.append(ensemble_result.confidence)
                consensus_scores.append(ensemble_result.consensus_score)
                
            except Exception as e:
                logger.error(f"Error in ensemble prediction for {img_path}: {e}")
                predictions.append('unknown')
                confidences.append(0.0)
                consensus_scores.append(0.0)
        
        accuracy = accuracy_score(true_labels, predictions)
        
        return TestResult(
            test_name="Ensemble Performance",
            accuracy=accuracy,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=np.array([]),
            detailed_results={
                'avg_confidence': np.mean(confidences),
                'avg_consensus': np.mean(consensus_scores),
                'high_consensus_count': sum(1 for s in consensus_scores if s > 0.7)
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _create_quality_variants(self, image: Image.Image) -> Dict[str, Image.Image]:
        """Create different quality variants of an image"""
        variants = {}
        
        # High quality (original)
        variants['high_quality'] = image
        
        # Medium quality (slight compression and noise)
        medium = image.copy()
        medium_array = np.array(medium)
        # Add slight noise
        noise = np.random.normal(0, 10, medium_array.shape).astype(np.uint8)
        medium_array = np.clip(medium_array + noise, 0, 255)
        variants['medium_quality'] = Image.fromarray(medium_array)
        
        # Low quality (heavy compression, blur, noise)
        low = image.copy()
        low_array = np.array(low)
        # Add blur
        low_array = cv2.GaussianBlur(low_array, (5, 5), 0)
        # Add noise
        noise = np.random.normal(0, 25, low_array.shape).astype(np.uint8)
        low_array = np.clip(low_array + noise, 0, 255)
        # Reduce resolution
        h, w = low_array.shape[:2]
        low_array = cv2.resize(low_array, (w//2, h//2))
        low_array = cv2.resize(low_array, (w, h))
        variants['low_quality'] = Image.fromarray(low_array)
        
        return variants
    
    def _create_mock_predictions(self, img_path: str) -> List[PredictionResult]:
        """Create mock predictions for ensemble testing"""
        # This would normally come from actual model predictions
        mock_predictions = [
            PredictionResult(
                source='Plant.id',
                common_name='Test Plant',
                scientific_name='Testus plantus',
                confidence=0.8,
                is_weed=True,
                is_crop=False
            ),
            PredictionResult(
                source='advanced_cnn',
                common_name='Test Plant',
                scientific_name='Testus plantus',
                confidence=0.75,
                is_weed=True,
                is_crop=False
            )
        ]
        return mock_predictions
    
    def _create_empty_result(self, test_name: str) -> TestResult:
        """Create empty test result"""
        return TestResult(
            test_name=test_name,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=np.array([]),
            detailed_results={},
            timestamp=datetime.now().isoformat()
        )
    
    def generate_test_report(self, output_path: str = "test_results/test_report.html"):
        """Generate comprehensive test report"""
        if not self.results:
            logger.warning("No test results to report")
            return
        
        # Create HTML report
        html_content = self._create_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Test report saved to {output_path}")
    
    def _create_html_report(self) -> str:
        """Create HTML test report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Plant Classification Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .test-result { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Plant Classification Test Report</h1>
            <p>Generated on: {timestamp}</p>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for result in self.results:
            html += f"""
            <div class="test-result">
                <h2>{result.test_name}</h2>
                <div class="metric">Accuracy: {result.accuracy:.3f}</div>
                <div class="metric">Precision: {result.precision:.3f}</div>
                <div class="metric">Recall: {result.recall:.3f}</div>
                <div class="metric">F1-Score: {result.f1_score:.3f}</div>
                <h3>Detailed Results:</h3>
                <pre>{json.dumps(result.detailed_results, indent=2)}</pre>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html

# Utility functions for testing
def run_comprehensive_tests(test_data_path: str = "test_data") -> List[TestResult]:
    """Run all comprehensive tests"""
    framework = PlantClassificationTestFramework(test_data_path)
    
    # Load models
    model_configs = {
        'advanced_classifier': {
            'model_type': 'efficientnet_b4',
            'num_classes': 1000,
            'model_path': 'advanced_plant_model.h5'
        },
        'growth_stage_detector': {
            'model_path': 'growth_stage_model.h5'
        }
    }
    
    framework.load_models(model_configs)
    
    # Run tests (would need actual test data)
    test_images = []  # Load actual test images
    test_labels = []  # Load actual labels
    
    results = []
    
    if test_images and test_labels:
        # Run quality robustness test
        quality_result = framework.test_image_quality_robustness(test_images, test_labels)
        results.append(quality_result)
        
        # Run species classification test
        species_result = framework.test_species_classification(test_images, test_labels)
        results.append(species_result)
        
        # Run ensemble test
        ensemble_result = framework.test_ensemble_performance(test_images, test_labels)
        results.append(ensemble_result)
    
    framework.results = results
    framework.generate_test_report()
    
    return results
