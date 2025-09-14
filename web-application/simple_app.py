"""
Simplified Plant Classification Web Application
Works without TensorFlow and advanced modules
"""

from flask import Flask, render_template, request, jsonify
import requests
import os
import json
import difflib
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import dotenv
import logging

# Initialize Flask app
dotenv.load_dotenv()
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key from environment variable
API_KEY = os.environ.get("PLANT_ID_API_KEY", "hemT6yPa28kZxSjltp9Lr9TZYXNkWnmndjws4ud9l8JmeHb8cS")
API_URL = "https://api.plant.id/v2/identify"

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load plant databases
def load_plant_databases():
    """Load plant databases"""
    global_db = []
    weed_db = []
    
    try:
        with open('global_plant_database.json', 'r') as f:
            global_db = json.load(f)
        logger.info(f"Loaded {len(global_db)} plants from global database")
    except FileNotFoundError:
        logger.warning("Global plant database not found")
    
    try:
        with open('Weed_info.json', 'r') as f:
            weed_db = json.load(f)
        logger.info(f"Loaded {len(weed_db)} weeds from legacy database")
    except FileNotFoundError:
        logger.warning("Legacy weed database not found")
    
    return global_db, weed_db

GLOBAL_PLANT_DB, WEED_DATA = load_plant_databases()

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_best_match(plant_name, database):
    """Find best matching plant in database with improved matching"""
    if not database:
        return None

    plant_name_lower = plant_name.lower()

    # First, try exact matches
    for item in database:
        # Check scientific name
        if plant_name_lower in item.get('ScientificName', '').lower():
            return item
        # Check common name
        if plant_name_lower in item.get('CommonName', '').lower():
            return item
        # Check alternative names
        alt_names = item.get('AlternativeNames', [])
        for alt_name in alt_names:
            if plant_name_lower in alt_name.lower():
                return item
        # Check weed type (for legacy database)
        if plant_name_lower in item.get('WeedType', '').lower():
            return item

    # If no exact match, try fuzzy matching
    plant_names = []
    for item in database:
        names_to_check = []
        if 'ScientificName' in item:
            names_to_check.append(item['ScientificName'])
        if 'CommonName' in item:
            names_to_check.append(item['CommonName'])
        if 'WeedType' in item:
            names_to_check.append(item['WeedType'])
        if 'AlternativeNames' in item:
            names_to_check.extend(item['AlternativeNames'])

        plant_names.extend(names_to_check)

    matches = difflib.get_close_matches(plant_name, plant_names, n=1, cutoff=0.7)  # Increased from 0.5 to 0.7 for stricter matching

    if matches:
        for item in database:
            if (matches[0] in [item.get('ScientificName', ''), item.get('CommonName', ''), item.get('WeedType', '')] or
                matches[0] in item.get('AlternativeNames', [])):
                return item

    return None

def enhanced_plant_search(plant_name, global_db, weed_db):
    """Enhanced plant search with keyword matching"""
    plant_name_lower = plant_name.lower()

    # First, check if this plant should NEVER be identified as cotton
    cotton_exclusions = [
        'dandelion', 'taraxacum', 'daisy', 'aster', 'sunflower', 'helianthus',
        'rose', 'rosa', 'tulip', 'lily', 'orchid', 'violet', 'pansy',
        'grass', 'poaceae', 'lawn', 'turf',
        'wheat', 'triticum', 'corn', 'maize', 'zea', 'rice', 'oryza',
        'tomato', 'solanum', 'potato', 'pepper', 'capsicum',
        'lettuce', 'lactuca', 'cabbage', 'brassica', 'carrot', 'daucus',
        'oak', 'quercus', 'pine', 'pinus', 'maple', 'acer', 'birch', 'betula',
        'cactus', 'opuntia', 'aloe', 'succulent', 'fern', 'moss',
        'plantain', 'plantago', 'clover', 'trifolium', 'chickweed', 'stellaria'
    ]

    # If it matches any exclusion, don't return cotton even if fuzzy matching suggests it
    for exclusion in cotton_exclusions:
        if exclusion in plant_name_lower:
            # Search in databases but exclude cotton results
            match = find_best_match(plant_name, global_db)
            if match and 'cotton' not in match.get('CommonName', '').lower():
                return match, match.get('PlantType', 'unknown').lower()

            match = find_best_match(plant_name, weed_db)
            if match:
                return match, 'weed'

            # If no good match found, return None to avoid cotton misidentification
            return None, None

    # Cotton-specific keywords (only if not excluded above)
    cotton_keywords = ['cotton', 'gossypium', 'kapas', 'upland', 'pima', 'fiber']
    if any(keyword in plant_name_lower for keyword in cotton_keywords):
        for plant in global_db:
            if plant.get('CommonName', '').lower() == 'cotton':
                return plant, 'crop'

    # Search in global database first
    match = find_best_match(plant_name, global_db)
    if match:
        return match, match.get('PlantType', 'unknown').lower()

    # Search in weed database
    match = find_best_match(plant_name, weed_db)
    if match:
        return match, 'weed'

    return None, None

def is_likely_non_plant(plant_name, confidence):
    """Check if the result is likely not a plant"""
    plant_name_lower = plant_name.lower()

    # Plant-related keywords that should NOT be rejected even with low confidence
    plant_keywords = [
        'plant', 'tree', 'flower', 'leaf', 'leaves', 'stem', 'root', 'seed',
        'weed', 'grass', 'herb', 'shrub', 'bush', 'vine', 'moss', 'fern',
        'crop', 'cotton', 'corn', 'wheat', 'rice', 'soybean', 'bean',
        'tomato', 'potato', 'pepper', 'lettuce', 'cabbage', 'carrot',
        'rose', 'daisy', 'sunflower', 'lily', 'orchid', 'tulip',
        'oak', 'pine', 'maple', 'birch', 'willow', 'palm',
        'cactus', 'succulent', 'bamboo', 'algae', 'lichen'
    ]

    # If it contains plant keywords, don't reject it even with low confidence
    for keyword in plant_keywords:
        if keyword in plant_name_lower:
            logger.info(f"Plant keyword '{keyword}' found in '{plant_name}' - not rejecting despite low confidence")
            return False

    # Common non-plant misidentifications
    non_plant_indicators = [
        'object', 'artifact', 'tool', 'machine', 'building', 'structure',
        'fabric', 'plastic', 'metal', 'glass', 'paper', 'wood',
        'animal', 'person', 'human', 'face', 'hand', 'body',
        'food', 'cooked', 'processed', 'bottle', 'container'
    ]

    # Check for explicit non-plant indicators
    for indicator in non_plant_indicators:
        if indicator in plant_name_lower:
            logger.info(f"Non-plant indicator '{indicator}' found in '{plant_name}'")
            return True

    # Very low confidence suggests uncertain identification, but only if no plant keywords
    if confidence < 20:  # Lowered threshold and only for very low confidence
        logger.info(f"Very low confidence ({confidence}%) for '{plant_name}' - rejecting as non-plant")
        return True

    # Check for non-plant keywords
    for indicator in non_plant_indicators:
        if indicator in plant_name_lower:
            return True

    return False

def is_likely_cotton_field(plant_name, confidence):
    """Enhanced cotton detection based on common misidentifications"""
    plant_name_lower = plant_name.lower()

    # First, explicitly exclude plants that should NEVER be identified as cotton
    cotton_exclusions = [
        'dandelion', 'taraxacum',  # Common weeds that are definitely not cotton
        'daisy', 'aster', 'sunflower', 'helianthus',
        'rose', 'rosa', 'tulip', 'lily', 'orchid', 'violet', 'pansy',
        'grass', 'poaceae', 'lawn', 'turf',
        'wheat', 'triticum', 'corn', 'maize', 'zea', 'rice', 'oryza',
        'tomato', 'solanum', 'potato', 'pepper', 'capsicum',
        'lettuce', 'lactuca', 'cabbage', 'brassica', 'carrot', 'daucus',
        'oak', 'quercus', 'pine', 'pinus', 'maple', 'acer', 'birch', 'betula',
        'cactus', 'opuntia', 'aloe', 'succulent', 'fern', 'moss',
        'plantain', 'plantago',  # Common weed, not cotton
        'clover', 'trifolium',  # Legume, not cotton
        'chickweed', 'stellaria',  # Common weed
        'purslane', 'portulaca',  # Succulent weed
        'lamb', 'lambsquarters', 'chenopodium'  # Common weeds
    ]

    # If it matches any exclusion, definitely not cotton
    for exclusion in cotton_exclusions:
        if exclusion in plant_name_lower:
            logger.info(f"Cotton exclusion detected: {plant_name} contains '{exclusion}' - not cotton")
            return False

    # Plants commonly confused with cotton in agricultural settings
    cotton_field_indicators = [
        'soybean', 'glycine max', 'soya',  # Very commonly confused with cotton
        'bean', 'legume', 'pulse',
        'okra', 'abelmoschus', 'lady finger',  # Same family as cotton
        'hibiscus', 'malva',  # Same family
        'siam weed', 'chromolaena',
        'ageratum', 'parthenium', 'lantana',
        'amaranth', 'pigweed',
        'morning glory', 'ipomoea',
        'castor', 'ricinus',
        # Additional indicators for young cotton plants
        'ceiba', 'bombax', 'kapok',  # Tree cotton family
        'abronia', 'sand verbena',  # Sometimes confused with young cotton
        'sweet potato', 'batatas',  # Young leaves can look similar
    ]

    # Special handling for very low confidence results that might be young cotton
    # BUT only if they don't match known non-cotton plants
    if confidence < 15:  # Much more restrictive threshold
        # First check if it's a known non-cotton plant
        known_non_cotton_plants = [
            'dandelion', 'taraxacum', 'daisy', 'aster', 'sunflower', 'helianthus',
            'rose', 'rosa', 'tulip', 'lily', 'orchid', 'violet', 'pansy',
            'grass', 'poaceae', 'wheat', 'triticum', 'corn', 'maize', 'zea',
            'rice', 'oryza', 'barley', 'hordeum', 'oat', 'avena',
            'tomato', 'solanum', 'potato', 'pepper', 'capsicum',
            'lettuce', 'lactuca', 'cabbage', 'brassica', 'carrot', 'daucus',
            'oak', 'quercus', 'pine', 'pinus', 'maple', 'acer', 'birch', 'betula',
            'cactus', 'opuntia', 'aloe', 'succulent', 'fern', 'moss'
        ]

        # Don't correct to cotton if it's a known non-cotton plant
        for non_cotton in known_non_cotton_plants:
            if non_cotton in plant_name_lower:
                logger.info(f"Known non-cotton plant detected: {plant_name} - not correcting to cotton")
                return False

        # Only for truly unidentified/uncertain results with very specific indicators
        very_uncertain_indicators = [
            'unknown plant', 'unidentified plant', 'uncertain plant',
            'plant species', 'unknown species', 'unidentified species'
        ]

        for indicator in very_uncertain_indicators:
            if indicator in plant_name_lower:
                logger.info(f"Possible young cotton detected: {plant_name} (very low confidence: {confidence}%)")
                return True

    # If confidence is low and it's a known cotton field indicator
    if confidence < 75:  # Increased threshold for cotton detection
        for indicator in cotton_field_indicators:
            if indicator in plant_name_lower:
                logger.info(f"Cotton field detected: {plant_name} -> likely cotton (confidence: {confidence}%)")
                return True

    return False

def is_likely_misidentified_weed(plant_name, confidence):
    """Check if the API result is likely a misidentified crop"""
    plant_name_lower = plant_name.lower()

    # Common misidentifications by Plant.id API
    weed_misidentifications = {
        'siam weed': ['cotton', 'gossypium'],
        'chromolaena odorata': ['cotton', 'gossypium'],
        'ageratum': ['cotton'],
        'parthenium': ['cotton'],
        'lantana': ['cotton']
    }

    # If it's a known misidentification, flag it (regardless of confidence for cotton)
    for weed, crops in weed_misidentifications.items():
        if weed in plant_name_lower:
            # For cotton misidentifications, always correct if confidence < 80
            if crops[0] == 'cotton' and confidence < 80:
                return True, crops[0]
            # For other crops, only correct if confidence is very low
            elif confidence < 60:
                return True, crops[0]

    return False, None

def smart_plant_identification(api_result, confidence, global_db, weed_db):
    """Smart identification that can override API results when needed"""

    # FIRST: Check if this might be a cotton field misidentified as something else
    # This must happen before non-plant detection to catch young cotton plants
    if is_likely_cotton_field(api_result, confidence):
        logger.info(f"Cotton field detected: {api_result} -> correcting to cotton")
        # Search for cotton in our database
        plant_info, plant_type = enhanced_plant_search('cotton', global_db, weed_db)
        if plant_info:
            return plant_info, plant_type, f"Agricultural field detected - corrected to cotton (was: {api_result})"

    # SECOND: Check if this is likely not a plant at all (only after cotton check)
    if is_likely_non_plant(api_result, confidence):
        logger.info(f"Non-plant detected: {api_result} (confidence: {confidence}%)")
        return None, None, f"This appears to be a non-plant object. Please upload a clear image of a plant."

    # Check if this might be a misidentification
    is_misidentified, likely_crop = is_likely_misidentified_weed(api_result, confidence)

    if is_misidentified:
        logger.info(f"Potential misidentification detected: {api_result} -> likely {likely_crop}")
        # Search for the likely correct crop
        plant_info, plant_type = enhanced_plant_search(likely_crop, global_db, weed_db)
        if plant_info:
            return plant_info, plant_type, f"Corrected identification (was: {api_result})"

    # Use the API result as-is
    plant_info, plant_type = enhanced_plant_search(api_result, global_db, weed_db)
    return plant_info, plant_type, None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and plant identification"""
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
    else:
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'})
        filename = secure_filename(file.filename)
    
    # Save the file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Call Plant.id API
        with open(filepath, 'rb') as img_file:
            response = requests.post(
                API_URL,
                headers={'Api-Key': API_KEY},
                files={'images': img_file},
                data={'organs': 'leaf'}
            )
            logger.info(f'API status: {response.status_code}')

        if response.status_code == 200:
            try:
                result = response.json()
                if 'suggestions' in result and len(result['suggestions']) > 0:
                    # Get the best suggestion
                    best_suggestion = result['suggestions'][0]
                    plant_name = best_suggestion['plant_name'].strip()
                    confidence = best_suggestion['probability'] * 100
                    
                    # Use smart identification that can correct API mistakes
                    plant_info, plant_type, correction_note = smart_plant_identification(
                        plant_name, confidence, GLOBAL_PLANT_DB, WEED_DATA
                    )

                    # Handle non-plant detection
                    if plant_info is None and plant_type is None and correction_note:
                        response_data = {
                            'error': correction_note,
                            'ImagePath': f"/static/images/{filename}",
                            'timestamp': datetime.now().isoformat()
                        }
                        return jsonify(response_data)

                    if plant_info and plant_type in ['crop', 'weed']:
                        # Found in database
                        response_data = {
                            'PlantName': plant_info.get('CommonName', plant_info.get('WeedType', plant_name)),
                            'ScientificName': plant_info.get('ScientificName', plant_name),
                            'Confidence': f"{confidence:.2f}%",
                            'Family': plant_info.get('Family', 'Unknown'),
                            'Origin': plant_info.get('Origin', 'Unknown'),
                            'isWeed': plant_type == 'weed',
                            'isCrop': plant_type == 'crop',
                            'ImagePath': f"/static/images/{filename}",
                            'timestamp': datetime.now().isoformat()
                        }

                        # Add correction note if identification was corrected
                        if correction_note:
                            response_data['CorrectionNote'] = correction_note
                        
                        if plant_type == 'weed':
                            response_data.update({
                                'Message': f"🌿 This is identified as a weed: {plant_info.get('CommonName', plant_info.get('WeedType', plant_name))}",
                                'ControlMeasure': plant_info.get('ControlMeasure', 'No specific control measures available'),
                                'Climate': plant_info.get('Climate', 'Unknown'),
                                'AdditionalInfo': plant_info.get('AdditionalInfo', ''),
                                'ToxicityLevel': plant_info.get('ToxicityLevel', 'Unknown'),
                                'EconomicImpact': plant_info.get('EconomicImpact', 'Unknown')
                            })
                        elif plant_type == 'crop':
                            message = f"🌾 This is identified as a crop: {plant_info.get('CommonName', plant_name)}"
                            if correction_note:
                                message += f" ⚠️ {correction_note}"

                            response_data.update({
                                'Message': message,
                                'CultivationInfo': plant_info.get('CultivationInfo', 'No specific cultivation information available'),
                                'Climate': plant_info.get('Climate', 'Unknown'),
                                'AdditionalInfo': plant_info.get('AdditionalInfo', ''),
                                'EconomicImpact': plant_info.get('EconomicImpact', 'Unknown'),
                                'GlobalDistribution': plant_info.get('GlobalDistribution', 'Unknown'),
                                'IdentificationTips': plant_info.get('IdentificationTips', 'No specific identification tips available')
                            })
                        
                        # Add growth stage info if available
                        if 'GrowthStages' in plant_info:
                            response_data['GrowthStages'] = plant_info['GrowthStages']
                        
                        return jsonify(response_data)
                    
                    else:
                        # Check in legacy weed database
                        matched_weed = find_best_match(plant_name, WEED_DATA)
                        if matched_weed:
                            response_data = {
                                'PlantName': matched_weed.get('WeedType', plant_name),
                                'ScientificName': matched_weed.get('ScientificName', plant_name),
                                'Confidence': f"{confidence:.2f}%",
                                'isWeed': True,
                                'Message': f"This is identified as a weed: {matched_weed.get('WeedType', plant_name)}",
                                'ControlMeasure': matched_weed.get('ControlMeasure', 'No specific control measures available'),
                                'Climate': matched_weed.get('Climate', 'Unknown'),
                                'AdditionalInfo': matched_weed.get('AdditionalInfo', ''),
                                'ImagePath': f"/static/images/{filename}",
                                'timestamp': datetime.now().isoformat()
                            }
                            return jsonify(response_data)
                        
                        else:
                            # Unknown plant
                            response_data = {
                                'PlantName': plant_name,
                                'ScientificName': plant_name,
                                'Confidence': f"{confidence:.2f}%",
                                'isWeed': False,
                                'Message': "Plant identified but not in our database. Please consult with local experts.",
                                'ImagePath': f"/static/images/{filename}",
                                'timestamp': datetime.now().isoformat()
                            }
                            return jsonify(response_data)
                
                else:
                    return jsonify({'error': 'No plant detected in the image'})
                    
            except ValueError:
                return jsonify({'error': 'Invalid response from API'})
        else:
            return jsonify({
                'error': 'API request failed',
                'status_code': response.status_code,
                'details': response.text
            })
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'An error occurred during plant identification',
            'details': str(e),
            'ImagePath': f"/static/images/{filename}"
        })

@app.route('/api/stats')
def get_system_stats():
    """Get system statistics and capabilities"""
    return jsonify({
        'database_info': {
            'global_plants': len(GLOBAL_PLANT_DB),
            'legacy_weeds': len(WEED_DATA),
            'api_sources': ['Plant.id', 'Local Database']
        },
        'capabilities': [
            'Global crop and weed identification',
            'Scientific and common names',
            'Control measures for weeds',
            'Cultivation info for crops',
            'Plant family and origin information'
        ],
        'status': 'Basic mode - TensorFlow features not available'
    })

@app.route('/plant_details/<scientific_name>')
def get_plant_details(scientific_name):
    """Get detailed information about a specific plant"""
    try:
        # Search in global database
        for plant in GLOBAL_PLANT_DB:
            if scientific_name.lower() in plant.get('ScientificName', '').lower():
                return jsonify(plant)
        
        # Search in legacy weed database
        for weed in WEED_DATA:
            if scientific_name.lower() in weed.get('ScientificName', '').lower():
                return jsonify(weed)
        
        return jsonify({'error': 'Plant not found in database'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    logger.info("🌱 Starting Simple Plant Classification System...")
    logger.info("🌐 Web interface available at: http://localhost:8080")
    logger.info("📊 Database loaded with {} global plants and {} legacy weeds".format(
        len(GLOBAL_PLANT_DB), len(WEED_DATA)))
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
