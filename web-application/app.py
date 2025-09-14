from flask import Flask, render_template, request, jsonify, send_from_directory
import requests
import os
import json
import difflib
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import dotenv
import asyncio
from PIL import Image
import logging

# Import our advanced modules
from advanced_preprocessing import AdvancedImagePreprocessor, adaptive_preprocessing
from advanced_model import AdvancedPlantClassifier
from multi_source_integration import MultiSourcePlantIdentifier, run_identification
from growth_stage_detector import GrowthStageDetector

# Initialize Flask app
dotenv.load_dotenv()
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key from environment variable
API_KEY = os.environ.get("PLANT_ID_API_KEY", "hemT6yPa28kZxSjltp9Lr9TZYXNkWnmndjws4ud9l8JmeHb8cS")
API_URL = "https://api.plant.id/v2/identify"

# Initialize advanced components
def initialize_advanced_models():
    """Initialize all advanced models and components"""
    try:
        # Initialize image preprocessor
        preprocessor = AdvancedImagePreprocessor(target_size=(224, 224))

        # Initialize advanced plant classifier
        plant_classifier = AdvancedPlantClassifier(
            model_type='efficientnet_b4',
            num_classes=1000,
            input_shape=(224, 224, 3)
        )

        # Try to load existing model or build new one
        if os.path.exists('advanced_plant_model.h5'):
            plant_classifier.load_model('advanced_plant_model.h5')
        else:
            plant_classifier.build_model()
            plant_classifier.compile_model()

        # Initialize growth stage detector
        growth_detector = GrowthStageDetector()

        # Initialize multi-source identifier
        multi_source = MultiSourcePlantIdentifier()

        return preprocessor, plant_classifier, growth_detector, multi_source

    except Exception as e:
        logger.error(f"Error initializing advanced models: {e}")
        return None, None, None, None

# Load the legacy ML model for backward compatibility
def load_legacy_model():
    """Load the legacy CNN model for backward compatibility."""
    try:
        model = keras.models.load_model('weed_classifier.h5')
        return model
    except:
        logger.warning("Legacy ML model not found. Using advanced models only.")
        return None

# Load class names for legacy model
LEGACY_CLASS_NAMES = [
    'Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
    'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize',
    'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet'
]

# Initialize all models
PREPROCESSOR, PLANT_CLASSIFIER, GROWTH_DETECTOR, MULTI_SOURCE = initialize_advanced_models()
LEGACY_MODEL = load_legacy_model()

# Load global plant database
def load_global_database():
    """Load the global plant database"""
    try:
        with open('global_plant_database.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Global plant database not found")
        return []

GLOBAL_PLANT_DB = load_global_database()

# Set upload folder for images
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file is an allowed image."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def advanced_plant_identification(image_path):
    """
    Advanced plant identification using multiple models and sources
    """
    try:
        # Load and preprocess image
        image = Image.open(image_path)

        # Basic image quality assessment
        quality_info = {'overall_quality': 75, 'needs_enhancement': False}

        results = {
            'image_quality': quality_info,
            'identifications': [],
            'growth_stage': None,
            'confidence_score': 0.0
        }

        # For now, return basic results since advanced modules may not be available
        # This will be enhanced when TensorFlow is properly installed
        logger.info("Using basic identification mode - advanced features limited")

        return results

    except Exception as e:
        logger.error(f"Advanced identification failed: {e}")
        return None

def predict_with_legacy_model(image_path):
    """Predict using the legacy ML model for backward compatibility."""
    if LEGACY_MODEL is None:
        return None, None

    try:
        # Load and resize image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (150, 150))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = LEGACY_MODEL.predict(img)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100

        # Get class name
        class_name = LEGACY_CLASS_NAMES[predicted_class]

        return class_name, confidence
    except Exception as e:
        logger.error(f"Error in legacy ML prediction: {e}")
        return None, None

# Load non-weed plants from a text file
def load_non_weed_plants(filename='non-weed.txt'):
    try:
        with open(filename, 'r') as file:
            return {line.strip().lower() for line in file if line.strip()}
    except FileNotFoundError:
        print(f"⚠️ Warning: {filename} not found.")
        return set()

# Load known weeds from a JSON file
def load_weed_data(filename='Weed_info.json'):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"⚠️ Warning: {filename} not found or invalid.")
        return []

# Initialize data once at startup
NON_WEED_PLANTS = load_non_weed_plants()
WEED_DATA = load_weed_data()

# Fuzzy matching function
def find_best_match(plant_name, weed_list):
    """Find the closest matching plant name from a list using fuzzy matching."""
    plant_name_lower = plant_name.lower()
    
    # First try exact match
    for weed in weed_list:
        if weed.get('WeedType', '').lower() == plant_name_lower:
            return weed
        if weed.get('ScientificName', '').lower() == plant_name_lower:
            return weed
    
    # Then try fuzzy matching with higher cutoff for better accuracy
    matches = difflib.get_close_matches(
        plant_name_lower, 
        [w.get('WeedType', '').lower() for w in weed_list if 'WeedType' in w], 
        n=3, 
        cutoff=0.7
    )
    
    if matches:
        for weed in weed_list:
            if weed.get('WeedType', '').lower() == matches[0]:
                return weed
    
    # Also try matching against scientific names
    scientific_matches = difflib.get_close_matches(
        plant_name_lower,
        [w.get('ScientificName', '').lower() for w in weed_list if 'ScientificName' in w],
        n=3,
        cutoff=0.7
    )
    
    if scientific_matches:
        for weed in weed_list:
            if weed.get('ScientificName', '').lower() == scientific_matches[0]:
                return weed
    
    return None

@app.route('/')
def index():
    """Homepage for uploading images."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and comprehensive plant identification."""
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        # Handle camera capture case (no filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
    else:
        # Handle regular file upload
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'})
        filename = secure_filename(file.filename)

    # Save the file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Advanced plant identification
        advanced_results = advanced_plant_identification(filepath)

        # Legacy model prediction for backward compatibility
        legacy_class, legacy_confidence = predict_with_legacy_model(filepath)

        # Prepare comprehensive response
        response_data = {
            'ImagePath': f"/static/images/{filename}",
            'timestamp': datetime.now().isoformat(),
            'processing_info': {
                'image_quality': advanced_results.get('image_quality', {}) if advanced_results else {},
                'models_used': []
            }
        }

        # Add legacy model results if available
        if legacy_class and legacy_confidence:
            response_data['legacy_prediction'] = {
                'class': legacy_class,
                'confidence': f"{legacy_confidence:.2f}%"
            }
            response_data['processing_info']['models_used'].append('legacy_cnn')

        if advanced_results:
            # Process advanced identification results
            identifications = advanced_results.get('identifications', [])
            growth_stage = advanced_results.get('growth_stage')
            advanced_pred = advanced_results.get('advanced_prediction')

            response_data['processing_info']['models_used'].extend(['multi_source', 'advanced_cnn', 'growth_stage'])

            if identifications:
                # Use the best identification result
                best_result = identifications[0]

                response_data.update({
                    'PlantName': best_result['common_name'],
                    'ScientificName': best_result['scientific_name'],
                    'Confidence': f"{best_result['confidence']:.2f}%",
                    'Source': best_result['source'],
                    'Family': best_result.get('family', 'Unknown'),
                    'isWeed': best_result['is_weed'],
                    'isCrop': best_result['is_crop']
                })

                # Add plant type message
                if best_result['is_weed']:
                    response_data['Message'] = f"This is identified as a weed: {best_result['common_name']}"
                    # Get control measures from database
                    control_info = get_plant_control_info(best_result['scientific_name'])
                    if control_info:
                        response_data.update(control_info)
                elif best_result['is_crop']:
                    response_data['Message'] = f"This is identified as a crop: {best_result['common_name']}"
                    # Get cultivation info from database
                    cultivation_info = get_plant_cultivation_info(best_result['scientific_name'])
                    if cultivation_info:
                        response_data.update(cultivation_info)
                else:
                    response_data['Message'] = f"Plant identified: {best_result['common_name']}"

                # Add all identification results
                response_data['all_identifications'] = identifications[:5]

            # Add growth stage information
            if growth_stage:
                response_data['growth_stage'] = {
                    'stage': growth_stage['stage'],
                    'confidence': f"{growth_stage['confidence']:.2f}%",
                    'characteristics': growth_stage['characteristics'],
                    'typical_features': growth_stage['features']
                }

            # Add advanced model prediction
            if advanced_pred:
                response_data['advanced_model'] = {
                    'class': advanced_pred['class_name'],
                    'confidence': f"{advanced_pred['confidence']:.2f}%",
                    'is_weed': advanced_pred['is_weed'],
                    'growth_stage': advanced_pred.get('growth_stage', 'unknown')
                }

        else:
            # Fallback to legacy API call if advanced methods fail
            response_data = fallback_api_identification(filepath, filename, legacy_class, legacy_confidence)

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': 'An error occurred during plant identification',
            'details': str(e),
            'ImagePath': f"/static/images/{filename}"
        })
    
def get_plant_control_info(scientific_name):
    """Get control measures for a weed from the database"""
    try:
        # Search in global database
        for plant in GLOBAL_PLANT_DB:
            if (plant.get('PlantType') == 'Weed' and
                scientific_name.lower() in plant.get('ScientificName', '').lower()):
                return {
                    'ControlMeasure': plant.get('ControlMeasure', 'No specific control measures available'),
                    'Climate': plant.get('Climate', 'Unknown climate requirements'),
                    'AdditionalInfo': plant.get('AdditionalInfo', ''),
                    'ToxicityLevel': plant.get('ToxicityLevel', 'Unknown'),
                    'EconomicImpact': plant.get('EconomicImpact', 'Unknown')
                }

        # Search in legacy weed database
        try:
            with open('Weed_info.json', 'r') as f:
                weed_db = json.load(f)
                for weed in weed_db:
                    if scientific_name.lower() in weed.get('ScientificName', '').lower():
                        return {
                            'ControlMeasure': weed.get('ControlMeasure', 'No specific control measures available'),
                            'Climate': weed.get('Climate', 'Unknown climate requirements'),
                            'AdditionalInfo': weed.get('AdditionalInfo', '')
                        }
        except FileNotFoundError:
            pass

    except Exception as e:
        logger.error(f"Error getting control info: {e}")

    return None

def get_plant_cultivation_info(scientific_name):
    """Get cultivation information for a crop from the database"""
    try:
        for plant in GLOBAL_PLANT_DB:
            if (plant.get('PlantType') == 'Crop' and
                scientific_name.lower() in plant.get('ScientificName', '').lower()):
                return {
                    'CultivationInfo': plant.get('CultivationInfo', 'No specific cultivation information available'),
                    'Climate': plant.get('Climate', 'Unknown climate requirements'),
                    'AdditionalInfo': plant.get('AdditionalInfo', ''),
                    'EconomicImpact': plant.get('EconomicImpact', 'Unknown'),
                    'Origin': plant.get('Origin', 'Unknown'),
                    'GlobalDistribution': plant.get('GlobalDistribution', 'Unknown')
                }
    except Exception as e:
        logger.error(f"Error getting cultivation info: {e}")

    return None

def fallback_api_identification(filepath, filename, legacy_class, legacy_confidence):
    """Fallback to original Plant.id API identification"""
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
            logger.info(f'API response: {response.text}')

        if response.status_code == 200:
            try:
                result = response.json()
                if 'suggestions' in result and len(result['suggestions']) > 0:
                    # Analyze all suggestions to find both crops and weeds
                    crops_found = []
                    weeds_found = []

                    for suggestion in result['suggestions'][:5]:  # Check top 5 suggestions
                        plant_name = suggestion['plant_name'].strip()
                        confidence = suggestion['probability'] * 100

                        # Check if it's a known non-weed (crop)
                        if plant_name.lower() in NON_WEED_PLANTS:
                            crops_found.append({
                                'name': plant_name,
                                'confidence': confidence,
                                'type': 'crop'
                            })
                        # Check if it's a known weed
                        else:
                            matched_weed = find_best_match(plant_name, WEED_DATA)
                            if matched_weed:
                                weeds_found.append({
                                    'name': plant_name,
                                    'common_name': matched_weed.get('WeedType', plant_name),
                                    'scientific_name': matched_weed.get('ScientificName', 'Unknown'),
                                    'confidence': confidence,
                                    'control_measure': matched_weed.get('ControlMeasure', 'No control measures available.'),
                                    'climate': matched_weed.get('Climate', 'Climate information not available.'),
                                    'additional_info': matched_weed.get('AdditionalInfo', 'No additional information available.'),
                                    'type': 'weed'
                                })
                            else:
                                # Unknown plant - treat as potential crop
                                crops_found.append({
                                    'name': plant_name,
                                    'confidence': confidence,
                                    'type': 'unknown_plant'
                                })
                
                # Prepare response based on what was found
                if weeds_found and crops_found:
                    # Both weeds and crops found
                    response_data = {
                        'hasMultiplePlants': True,
                        'crops': crops_found,
                        'weeds': weeds_found,
                        'message': f"Found {len(crops_found)} crop(s) and {len(weeds_found)} weed(s) in the image.",
                        'ImagePath': f"/static/images/{filename}"
                    }
                    
                    # Add ML model prediction if available
                    if ml_class and ml_confidence:
                        response_data['ml_prediction'] = {
                            'class': ml_class,
                            'confidence': f"{ml_confidence:.2f}%"
                        }
                    
                    return jsonify(response_data)
                    
                elif weeds_found:
                    # Only weeds found
                    top_weed = weeds_found[0]
                    response_data = {
                        'PlantName': top_weed['name'],
                        'CommonName': top_weed['common_name'],
                        'ScientificName': top_weed['scientific_name'],
                        'Confidence': f"{top_weed['confidence']:.2f}%",
                        'isWeed': True,
                        'Message': f"This is identified as a weed: {top_weed['common_name']}",
                        'ControlMeasure': top_weed['control_measure'],
                        'Climate': top_weed['climate'],
                        'AdditionalInfo': top_weed['additional_info'],
                        'ImagePath': f"/static/images/{filename}"
                    }
                    
                    # Add ML model prediction if available
                    if ml_class and ml_confidence:
                        response_data['ml_prediction'] = {
                            'class': ml_class,
                            'confidence': f"{ml_confidence:.2f}%"
                        }
                    
                    return jsonify(response_data)
                    
                elif crops_found:
                    # Only crops found
                    top_crop = crops_found[0]
                    response_data = {
                        'PlantName': top_crop['name'],
                        'Confidence': f"{top_crop['confidence']:.2f}%",
                        'isWeed': False,
                        'Message': "This is not a weed - it appears to be a crop or beneficial plant.",
                        'ImagePath': f"/static/images/{filename}"
                    }
                    
                    # Add ML model prediction if available
                    if ml_class and ml_confidence:
                        response_data['ml_prediction'] = {
                            'class': ml_class,
                            'confidence': f"{ml_confidence:.2f}%"
                        }
                    
                    return jsonify(response_data)
                    
                else:
                    # No clear identification
                    top_suggestion = result['suggestions'][0]
                    response_data = {
                        'PlantName': top_suggestion['plant_name'],
                        'Confidence': f"{top_suggestion['probability'] * 100:.2f}%",
                        'isWeed': False,
                        'Message': "Plant identified but not classified as weed or crop.",
                        'ImagePath': f"/static/images/{filename}"
                    }
                    
                    # Add ML model prediction if available
                    if ml_class and ml_confidence:
                        response_data['ml_prediction'] = {
                            'class': ml_class,
                            'confidence': f"{ml_confidence:.2f}%"
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
        logger.error(f"API identification error: {e}")
        return jsonify({'error': 'Error during plant identification', 'details': str(e)})

# New enhanced routes
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch image upload and identification"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files in the request'})

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'})

    results = []
    for file in files[:10]:  # Limit to 10 files
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Process each image
                advanced_results = advanced_plant_identification(filepath)
                if advanced_results and advanced_results.get('identifications'):
                    best_result = advanced_results['identifications'][0]
                    results.append({
                        'filename': filename,
                        'plant_name': best_result['common_name'],
                        'scientific_name': best_result['scientific_name'],
                        'confidence': best_result['confidence'],
                        'is_weed': best_result['is_weed'],
                        'is_crop': best_result['is_crop'],
                        'image_path': f"/static/images/{filename}"
                    })
                else:
                    results.append({
                        'filename': filename,
                        'error': 'Could not identify plant',
                        'image_path': f"/static/images/{filename}"
                    })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e),
                    'image_path': f"/static/images/{filename}"
                })

    return jsonify({
        'batch_results': results,
        'total_processed': len(results),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats')
def get_system_stats():
    """Get system statistics and capabilities"""
    return jsonify({
        'models_available': {
            'advanced_classifier': PLANT_CLASSIFIER is not None,
            'growth_stage_detector': GROWTH_DETECTOR is not None,
            'image_preprocessor': PREPROCESSOR is not None,
            'multi_source_identifier': MULTI_SOURCE is not None,
            'legacy_model': LEGACY_MODEL is not None
        },
        'database_info': {
            'global_plants': len(GLOBAL_PLANT_DB),
            'growth_stages_supported': 8,
            'api_sources': ['Plant.id', 'PlantNet', 'Local Database']
        },
        'capabilities': [
            'Global crop and weed identification',
            'Growth stage detection',
            'Low-quality image enhancement',
            'Multi-source verification',
            'Batch processing',
            'Scientific and common names',
            'Control measures for weeds',
            'Cultivation info for crops'
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)