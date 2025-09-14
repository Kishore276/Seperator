"""
Test API endpoint with simulated Siam weed correction
"""

import requests
import json
from PIL import Image
import numpy as np
import io

def create_test_image():
    """Create a simple test image"""
    # Create a simple green image (simulating a plant)
    img_array = np.zeros((300, 300, 3), dtype=np.uint8)
    img_array[:, :, 1] = 100  # Green channel
    img_array[50:250, 50:250, 1] = 200  # Brighter green center
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def test_api_correction():
    """Test the API correction functionality"""
    print("🧪 Testing API Correction for Cotton Misidentification")
    print("=" * 60)
    
    # Create test image
    test_image = create_test_image()
    
    # Test the API endpoint
    url = "http://localhost:8080/predict"
    
    try:
        # Send request
        files = {'file': ('test_cotton.jpg', test_image, 'image/jpeg')}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response received successfully!")
            print(f"📝 Plant Name: {result.get('PlantName', 'Unknown')}")
            print(f"🔬 Scientific Name: {result.get('ScientificName', 'Unknown')}")
            print(f"📊 Confidence: {result.get('Confidence', 'Unknown')}")
            print(f"🌾 Is Crop: {result.get('isCrop', False)}")
            print(f"🌿 Is Weed: {result.get('isWeed', False)}")
            
            if result.get('CorrectionNote'):
                print(f"⚠️  Correction Applied: {result['CorrectionNote']}")
            else:
                print("ℹ️  No correction applied")
            
            if result.get('Message'):
                print(f"💬 Message: {result['Message']}")
            
            # Display additional crop information if available
            if result.get('isCrop'):
                print("\n🌾 Crop Information:")
                if result.get('Family'):
                    print(f"   👨‍👩‍👧‍👦 Family: {result['Family']}")
                if result.get('Origin'):
                    print(f"   🌍 Origin: {result['Origin']}")
                if result.get('CultivationInfo'):
                    print(f"   🚜 Cultivation: {result['CultivationInfo'][:100]}...")
                if result.get('IdentificationTips'):
                    print(f"   🔍 ID Tips: {result['IdentificationTips'][:100]}...")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the server is running on localhost:8080")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_correction()
