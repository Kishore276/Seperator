"""
Multi-Source Plant Identification Integration
Combines multiple APIs and databases for comprehensive plant identification
"""

import requests
import json
import base64
import io
import time
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
from PIL import Image
import numpy as np
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlantIdentification:
    """Data class for plant identification results"""
    common_name: str
    scientific_name: str
    confidence: float
    source: str
    family: str = ""
    is_weed: bool = False
    is_crop: bool = False
    additional_info: Dict = None

class MultiSourcePlantIdentifier:
    def __init__(self):
        """Initialize multi-source plant identifier"""
        self.api_keys = self._load_api_keys()
        self.session = None
        
        # API endpoints
        self.endpoints = {
            'plantid': 'https://api.plant.id/v2/identify',
            'plantnet': 'https://my-api.plantnet.org/v2/identify',
            'inaturalist': 'https://api.inaturalist.org/v1/identifications',
            'gbif': 'https://api.gbif.org/v1/species/match',
            'tropicos': 'http://services.tropicos.org/Name/Search'
        }
        
        # Load local databases
        self.local_db = self._load_local_databases()
        
    def _load_api_keys(self):
        """Load API keys from environment or config file"""
        api_keys = {}
        
        # Try to load from environment variables
        api_keys['plantid'] = os.getenv('PLANT_ID_API_KEY', 'hemT6yPa28kZxSjltp9Lr9TZYXNkWnmndjws4ud9l8JmeHb8cS')
        api_keys['plantnet'] = os.getenv('PLANTNET_API_KEY', '')
        api_keys['tropicos'] = os.getenv('TROPICOS_API_KEY', '')
        
        # Try to load from config file
        try:
            with open('config/api_keys.json', 'r') as f:
                file_keys = json.load(f)
                api_keys.update(file_keys)
        except FileNotFoundError:
            logger.warning("API keys config file not found")
        
        return api_keys
    
    def _load_local_databases(self):
        """Load local plant databases"""
        databases = {}
        
        try:
            # Load global plant database
            with open('global_plant_database.json', 'r') as f:
                databases['global'] = json.load(f)
            
            # Load weed info database
            with open('Weed_info.json', 'r') as f:
                databases['weeds'] = json.load(f)
                
        except FileNotFoundError as e:
            logger.warning(f"Local database file not found: {e}")
        
        return databases
    
    async def identify_plant_comprehensive(self, image_path: str, 
                                         confidence_threshold: float = 0.1) -> List[PlantIdentification]:
        """
        Comprehensive plant identification using multiple sources
        """
        results = []
        
        # Create async session
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Run all identification methods concurrently
            tasks = [
                self._identify_plantid(image_path),
                self._identify_plantnet(image_path),
                self._identify_local_database(image_path),
                self._search_gbif_species(image_path),
            ]
            
            # Execute all tasks
            api_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in api_results:
                if isinstance(result, list):
                    results.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"API call failed: {result}")
        
        # Filter by confidence threshold
        filtered_results = [r for r in results if r.confidence >= confidence_threshold]
        
        # Merge and rank results
        merged_results = self._merge_and_rank_results(filtered_results)
        
        return merged_results
    
    async def _identify_plantid(self, image_path: str) -> List[PlantIdentification]:
        """Identify plant using Plant.id API"""
        if not self.api_keys.get('plantid'):
            return []
        
        try:
            # Prepare image
            with open(image_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # API request
            data = {
                'images': [img_data],
                'modifiers': ['crops_fast', 'similar_images'],
                'plant_details': ['common_names', 'url', 'name_authority', 'wiki_description']
            }
            
            headers = {
                'Api-Key': self.api_keys['plantid'],
                'Content-Type': 'application/json'
            }
            
            async with self.session.post(self.endpoints['plantid'], 
                                       json=data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._parse_plantid_response(result)
                else:
                    logger.error(f"Plant.id API error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Plant.id identification failed: {e}")
            return []
    
    async def _identify_plantnet(self, image_path: str) -> List[PlantIdentification]:
        """Identify plant using PlantNet API"""
        if not self.api_keys.get('plantnet'):
            return []
        
        try:
            # PlantNet requires specific project and organ parameters
            url = f"{self.endpoints['plantnet']}/weurope"
            
            with open(image_path, 'rb') as img_file:
                files = {'images': img_file}
                data = {'organs': 'leaf'}
                params = {'api-key': self.api_keys['plantnet']}
                
                async with self.session.post(url, data=data, 
                                           params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_plantnet_response(result)
                    else:
                        logger.error(f"PlantNet API error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"PlantNet identification failed: {e}")
            return []
    
    async def _identify_local_database(self, image_path: str) -> List[PlantIdentification]:
        """Search local databases for plant information"""
        # This would typically involve running local ML models
        # For now, return empty list as placeholder
        return []
    
    async def _search_gbif_species(self, image_path: str) -> List[PlantIdentification]:
        """Search GBIF database for species information"""
        # GBIF doesn't do image identification, but can provide species info
        # This is a placeholder for integration with GBIF taxonomy
        return []
    
    def _parse_plantid_response(self, response: Dict) -> List[PlantIdentification]:
        """Parse Plant.id API response"""
        results = []
        
        if 'suggestions' in response:
            for suggestion in response['suggestions']:
                plant_name = suggestion.get('plant_name', 'Unknown')
                probability = suggestion.get('probability', 0.0)
                
                # Extract additional information
                plant_details = suggestion.get('plant_details', {})
                common_names = plant_details.get('common_names', [])
                common_name = common_names[0] if common_names else plant_name
                
                # Determine if it's a weed or crop
                is_weed = self._is_weed(plant_name, common_name)
                is_crop = self._is_crop(plant_name, common_name)
                
                identification = PlantIdentification(
                    common_name=common_name,
                    scientific_name=plant_name,
                    confidence=probability,
                    source='Plant.id',
                    is_weed=is_weed,
                    is_crop=is_crop,
                    additional_info=plant_details
                )
                
                results.append(identification)
        
        return results
    
    def _parse_plantnet_response(self, response: Dict) -> List[PlantIdentification]:
        """Parse PlantNet API response"""
        results = []
        
        if 'results' in response:
            for result in response['results']:
                species = result.get('species', {})
                scientific_name = species.get('scientificNameWithoutAuthor', 'Unknown')
                common_names = species.get('commonNames', [])
                common_name = common_names[0] if common_names else scientific_name
                score = result.get('score', 0.0)
                
                # Determine if it's a weed or crop
                is_weed = self._is_weed(scientific_name, common_name)
                is_crop = self._is_crop(scientific_name, common_name)
                
                identification = PlantIdentification(
                    common_name=common_name,
                    scientific_name=scientific_name,
                    confidence=score,
                    source='PlantNet',
                    family=species.get('family', {}).get('scientificNameWithoutAuthor', ''),
                    is_weed=is_weed,
                    is_crop=is_crop,
                    additional_info=species
                )
                
                results.append(identification)
        
        return results
    
    def _is_weed(self, scientific_name: str, common_name: str) -> bool:
        """Determine if plant is a weed based on databases"""
        # Check against local weed database
        if 'weeds' in self.local_db:
            for weed in self.local_db['weeds']:
                if (scientific_name.lower() in weed.get('ScientificName', '').lower() or
                    common_name.lower() in weed.get('WeedType', '').lower()):
                    return True
        
        # Check against global database
        if 'global' in self.local_db:
            for plant in self.local_db['global']:
                if (plant.get('PlantType') == 'Weed' and
                    (scientific_name.lower() in plant.get('ScientificName', '').lower() or
                     common_name.lower() in plant.get('CommonName', '').lower())):
                    return True
        
        # Common weed keywords
        weed_keywords = [
            'weed', 'grass', 'dandelion', 'thistle', 'nettle', 'dock',
            'plantain', 'clover', 'chickweed', 'purslane', 'pigweed'
        ]
        
        name_lower = f"{scientific_name} {common_name}".lower()
        return any(keyword in name_lower for keyword in weed_keywords)
    
    def _is_crop(self, scientific_name: str, common_name: str) -> bool:
        """Determine if plant is a crop based on databases"""
        # Check against global database
        if 'global' in self.local_db:
            for plant in self.local_db['global']:
                if (plant.get('PlantType') == 'Crop' and
                    (scientific_name.lower() in plant.get('ScientificName', '').lower() or
                     common_name.lower() in plant.get('CommonName', '').lower())):
                    return True
        
        # Common crop keywords
        crop_keywords = [
            'wheat', 'rice', 'corn', 'maize', 'soybean', 'cotton', 'potato',
            'tomato', 'pepper', 'bean', 'pea', 'sunflower', 'barley', 'oat'
        ]
        
        name_lower = f"{scientific_name} {common_name}".lower()
        return any(keyword in name_lower for keyword in crop_keywords)
    
    def _merge_and_rank_results(self, results: List[PlantIdentification]) -> List[PlantIdentification]:
        """Merge similar results and rank by confidence"""
        if not results:
            return []
        
        # Group similar results
        grouped = {}
        for result in results:
            key = result.scientific_name.lower()
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Merge grouped results
        merged = []
        for group in grouped.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Merge multiple results for same species
                merged_result = self._merge_similar_results(group)
                merged.append(merged_result)
        
        # Sort by confidence
        merged.sort(key=lambda x: x.confidence, reverse=True)
        
        return merged
    
    def _merge_similar_results(self, similar_results: List[PlantIdentification]) -> PlantIdentification:
        """Merge multiple results for the same species"""
        # Use weighted average for confidence
        total_confidence = sum(r.confidence for r in similar_results)
        avg_confidence = total_confidence / len(similar_results)
        
        # Use the result with highest confidence as base
        best_result = max(similar_results, key=lambda x: x.confidence)
        
        # Combine sources
        sources = [r.source for r in similar_results]
        combined_source = ', '.join(set(sources))
        
        # Merge additional info
        combined_info = {}
        for result in similar_results:
            if result.additional_info:
                combined_info.update(result.additional_info)
        
        return PlantIdentification(
            common_name=best_result.common_name,
            scientific_name=best_result.scientific_name,
            confidence=avg_confidence,
            source=combined_source,
            family=best_result.family,
            is_weed=any(r.is_weed for r in similar_results),
            is_crop=any(r.is_crop for r in similar_results),
            additional_info=combined_info
        )
    
    def get_plant_details(self, scientific_name: str) -> Dict:
        """Get detailed information about a plant species"""
        details = {}
        
        # Search local databases
        if 'global' in self.local_db:
            for plant in self.local_db['global']:
                if scientific_name.lower() in plant.get('ScientificName', '').lower():
                    details.update(plant)
                    break
        
        if 'weeds' in self.local_db:
            for weed in self.local_db['weeds']:
                if scientific_name.lower() in weed.get('ScientificName', '').lower():
                    details.update(weed)
                    break
        
        return details

# Utility functions
async def identify_plant_from_file(image_path: str) -> List[PlantIdentification]:
    """Convenience function to identify plant from image file"""
    identifier = MultiSourcePlantIdentifier()
    return await identifier.identify_plant_comprehensive(image_path)

def run_identification(image_path: str) -> List[PlantIdentification]:
    """Synchronous wrapper for plant identification"""
    return asyncio.run(identify_plant_from_file(image_path))
