"""
Ensemble System for Plant Classification
Combines multiple models and APIs with sophisticated confidence scoring
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Single prediction result from a model or API"""
    source: str
    common_name: str
    scientific_name: str
    confidence: float
    is_weed: bool
    is_crop: bool
    additional_info: Dict = None
    model_uncertainty: float = 0.0

@dataclass
class EnsembleResult:
    """Final ensemble prediction result"""
    common_name: str
    scientific_name: str
    confidence: float
    consensus_score: float
    uncertainty: float
    is_weed: bool
    is_crop: bool
    supporting_sources: List[str]
    alternative_predictions: List[Dict]
    quality_metrics: Dict

class PlantClassificationEnsemble:
    def __init__(self, confidence_threshold=0.3, consensus_threshold=0.6):
        """
        Initialize ensemble system
        
        Args:
            confidence_threshold: Minimum confidence to consider a prediction
            consensus_threshold: Minimum consensus score for high confidence result
        """
        self.confidence_threshold = confidence_threshold
        self.consensus_threshold = consensus_threshold
        
        # Source weights based on reliability
        self.source_weights = {
            'Plant.id': 0.25,
            'PlantNet': 0.20,
            'advanced_cnn': 0.30,
            'legacy_cnn': 0.15,
            'local_database': 0.10
        }
        
        # Uncertainty factors
        self.uncertainty_factors = {
            'low_confidence': 0.3,
            'conflicting_sources': 0.4,
            'single_source': 0.5,
            'image_quality': 0.2
        }
    
    def combine_predictions(self, predictions: List[PredictionResult], 
                          image_quality_score: float = 0.5) -> EnsembleResult:
        """
        Combine multiple predictions into ensemble result
        
        Args:
            predictions: List of prediction results from different sources
            image_quality_score: Quality score of input image (0-1)
            
        Returns:
            EnsembleResult with combined prediction and confidence metrics
        """
        if not predictions:
            return self._create_no_prediction_result()
        
        # Filter predictions by confidence threshold
        valid_predictions = [p for p in predictions if p.confidence >= self.confidence_threshold]
        
        if not valid_predictions:
            return self._create_low_confidence_result(predictions)
        
        # Group predictions by scientific name
        grouped_predictions = self._group_predictions_by_species(valid_predictions)
        
        # Calculate weighted consensus for each species
        species_scores = self._calculate_species_consensus(grouped_predictions)
        
        # Select best prediction
        best_species = max(species_scores.keys(), key=lambda x: species_scores[x]['weighted_score'])
        best_group = grouped_predictions[best_species]
        
        # Calculate ensemble metrics
        consensus_score = species_scores[best_species]['consensus']
        uncertainty = self._calculate_uncertainty(best_group, species_scores, image_quality_score)
        final_confidence = self._calculate_final_confidence(best_group, consensus_score, uncertainty)
        
        # Determine plant type (weed/crop)
        is_weed, is_crop = self._determine_plant_type(best_group)
        
        # Get representative prediction
        representative = self._get_representative_prediction(best_group)
        
        # Create alternative predictions
        alternatives = self._create_alternative_predictions(species_scores, best_species)
        
        return EnsembleResult(
            common_name=representative.common_name,
            scientific_name=representative.scientific_name,
            confidence=final_confidence,
            consensus_score=consensus_score,
            uncertainty=uncertainty,
            is_weed=is_weed,
            is_crop=is_crop,
            supporting_sources=[p.source for p in best_group],
            alternative_predictions=alternatives,
            quality_metrics={
                'total_sources': len(predictions),
                'valid_sources': len(valid_predictions),
                'consensus_sources': len(best_group),
                'image_quality': image_quality_score,
                'prediction_diversity': self._calculate_diversity(grouped_predictions)
            }
        )
    
    def _group_predictions_by_species(self, predictions: List[PredictionResult]) -> Dict[str, List[PredictionResult]]:
        """Group predictions by scientific name with fuzzy matching"""
        groups = defaultdict(list)
        
        for prediction in predictions:
            # Normalize scientific name
            normalized_name = self._normalize_scientific_name(prediction.scientific_name)
            
            # Find existing group or create new one
            matched_group = None
            for existing_name in groups.keys():
                if self._are_species_similar(normalized_name, existing_name):
                    matched_group = existing_name
                    break
            
            if matched_group:
                groups[matched_group].append(prediction)
            else:
                groups[normalized_name].append(prediction)
        
        return dict(groups)
    
    def _normalize_scientific_name(self, name: str) -> str:
        """Normalize scientific name for comparison"""
        return name.lower().strip().replace('_', ' ')
    
    def _are_species_similar(self, name1: str, name2: str) -> bool:
        """Check if two scientific names refer to the same species"""
        # Simple similarity check - could be enhanced with fuzzy matching
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        # Check if genus and species match
        if len(words1) >= 2 and len(words2) >= 2:
            return list(words1)[:2] == list(words2)[:2]
        
        # Fallback to exact match
        return name1 == name2
    
    def _calculate_species_consensus(self, grouped_predictions: Dict[str, List[PredictionResult]]) -> Dict[str, Dict]:
        """Calculate consensus scores for each species"""
        species_scores = {}
        
        for species, predictions in grouped_predictions.items():
            # Calculate weighted average confidence
            total_weight = 0
            weighted_confidence = 0
            
            for pred in predictions:
                weight = self.source_weights.get(pred.source, 0.1)
                weighted_confidence += pred.confidence * weight
                total_weight += weight
            
            avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
            
            # Calculate consensus score based on agreement
            consensus = min(1.0, len(predictions) / 3.0)  # More sources = higher consensus
            
            # Adjust for confidence variance
            confidences = [p.confidence for p in predictions]
            confidence_variance = np.var(confidences) if len(confidences) > 1 else 0
            consensus *= (1 - min(0.5, confidence_variance / 100))  # Penalize high variance
            
            species_scores[species] = {
                'weighted_score': avg_confidence * consensus,
                'consensus': consensus,
                'avg_confidence': avg_confidence,
                'source_count': len(predictions),
                'confidence_variance': confidence_variance
            }
        
        return species_scores
    
    def _calculate_uncertainty(self, best_group: List[PredictionResult], 
                             all_scores: Dict, image_quality: float) -> float:
        """Calculate prediction uncertainty"""
        uncertainty = 0.0
        
        # Factor 1: Low confidence predictions
        avg_confidence = np.mean([p.confidence for p in best_group])
        if avg_confidence < 0.7:
            uncertainty += self.uncertainty_factors['low_confidence']
        
        # Factor 2: Conflicting sources (multiple strong predictions)
        strong_alternatives = sum(1 for scores in all_scores.values() 
                                if scores['weighted_score'] > 0.5)
        if strong_alternatives > 1:
            uncertainty += self.uncertainty_factors['conflicting_sources']
        
        # Factor 3: Single source prediction
        if len(best_group) == 1:
            uncertainty += self.uncertainty_factors['single_source']
        
        # Factor 4: Poor image quality
        if image_quality < 0.5:
            uncertainty += self.uncertainty_factors['image_quality'] * (1 - image_quality)
        
        # Factor 5: Model-specific uncertainty
        model_uncertainties = [p.model_uncertainty for p in best_group if p.model_uncertainty > 0]
        if model_uncertainties:
            uncertainty += np.mean(model_uncertainties) * 0.3
        
        return min(1.0, uncertainty)
    
    def _calculate_final_confidence(self, best_group: List[PredictionResult], 
                                  consensus_score: float, uncertainty: float) -> float:
        """Calculate final confidence score"""
        # Base confidence from predictions
        base_confidence = np.mean([p.confidence for p in best_group]) / 100.0
        
        # Adjust for consensus
        consensus_adjusted = base_confidence * (0.5 + 0.5 * consensus_score)
        
        # Adjust for uncertainty
        final_confidence = consensus_adjusted * (1 - uncertainty)
        
        return min(1.0, max(0.0, final_confidence))
    
    def _determine_plant_type(self, predictions: List[PredictionResult]) -> Tuple[bool, bool]:
        """Determine if plant is weed or crop based on predictions"""
        weed_votes = sum(1 for p in predictions if p.is_weed)
        crop_votes = sum(1 for p in predictions if p.is_crop)
        
        # Use majority voting with confidence weighting
        weighted_weed = sum(p.confidence for p in predictions if p.is_weed)
        weighted_crop = sum(p.confidence for p in predictions if p.is_crop)
        
        is_weed = weighted_weed > weighted_crop and weed_votes > 0
        is_crop = weighted_crop > weighted_weed and crop_votes > 0
        
        return is_weed, is_crop
    
    def _get_representative_prediction(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Get the most representative prediction from a group"""
        # Return the prediction with highest confidence
        return max(predictions, key=lambda p: p.confidence)
    
    def _create_alternative_predictions(self, species_scores: Dict, best_species: str) -> List[Dict]:
        """Create list of alternative predictions"""
        alternatives = []
        
        for species, scores in species_scores.items():
            if species != best_species and scores['weighted_score'] > 0.2:
                alternatives.append({
                    'scientific_name': species,
                    'confidence': scores['avg_confidence'],
                    'consensus': scores['consensus'],
                    'source_count': scores['source_count']
                })
        
        # Sort by weighted score
        alternatives.sort(key=lambda x: x['confidence'], reverse=True)
        return alternatives[:3]  # Return top 3 alternatives
    
    def _calculate_diversity(self, grouped_predictions: Dict) -> float:
        """Calculate prediction diversity score"""
        if len(grouped_predictions) <= 1:
            return 0.0
        
        # Shannon diversity index
        total_predictions = sum(len(group) for group in grouped_predictions.values())
        diversity = 0.0
        
        for group in grouped_predictions.values():
            if len(group) > 0:
                proportion = len(group) / total_predictions
                diversity -= proportion * math.log2(proportion)
        
        # Normalize to 0-1 range
        max_diversity = math.log2(len(grouped_predictions))
        return diversity / max_diversity if max_diversity > 0 else 0.0
    
    def _create_no_prediction_result(self) -> EnsembleResult:
        """Create result when no predictions are available"""
        return EnsembleResult(
            common_name="Unknown",
            scientific_name="Unknown",
            confidence=0.0,
            consensus_score=0.0,
            uncertainty=1.0,
            is_weed=False,
            is_crop=False,
            supporting_sources=[],
            alternative_predictions=[],
            quality_metrics={'total_sources': 0, 'valid_sources': 0}
        )
    
    def _create_low_confidence_result(self, predictions: List[PredictionResult]) -> EnsembleResult:
        """Create result when all predictions are below confidence threshold"""
        best_pred = max(predictions, key=lambda p: p.confidence)
        
        return EnsembleResult(
            common_name=best_pred.common_name,
            scientific_name=best_pred.scientific_name,
            confidence=best_pred.confidence / 100.0,
            consensus_score=0.0,
            uncertainty=0.8,
            is_weed=best_pred.is_weed,
            is_crop=best_pred.is_crop,
            supporting_sources=[best_pred.source],
            alternative_predictions=[],
            quality_metrics={
                'total_sources': len(predictions),
                'valid_sources': 0,
                'note': 'All predictions below confidence threshold'
            }
        )

# Utility functions
def create_prediction_result(source: str, common_name: str, scientific_name: str,
                           confidence: float, is_weed: bool = False, is_crop: bool = False,
                           additional_info: Dict = None, uncertainty: float = 0.0) -> PredictionResult:
    """Helper function to create PredictionResult objects"""
    return PredictionResult(
        source=source,
        common_name=common_name,
        scientific_name=scientific_name,
        confidence=confidence,
        is_weed=is_weed,
        is_crop=is_crop,
        additional_info=additional_info or {},
        model_uncertainty=uncertainty
    )

def calculate_ensemble_confidence(predictions: List[PredictionResult], 
                                image_quality: float = 0.5) -> EnsembleResult:
    """Convenience function to calculate ensemble confidence"""
    ensemble = PlantClassificationEnsemble()
    return ensemble.combine_predictions(predictions, image_quality)
