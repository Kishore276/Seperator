"""
Test cotton field detection for common misidentifications
"""

import json
from simple_app import smart_plant_identification, load_plant_databases, is_likely_cotton_field

def test_cotton_field_detection():
    """Test cotton field detection with common misidentifications"""
    
    # Load databases
    global_db, weed_db = load_plant_databases()
    
    print("🌾 Testing Cotton Field Detection")
    print("=" * 50)
    
    # Test cases that should be corrected to cotton
    cotton_field_cases = [
        ("Soybean", 65),  # Very common misidentification
        ("Glycine max", 60),  # Scientific name for soybean
        ("soya bean", 45),
        ("Bean plant", 55),
        ("Okra", 70),  # Same family as cotton
        ("Abelmoschus esculentus", 65),  # Okra scientific name
        ("Hibiscus", 60),  # Same family
        ("Morning glory", 50),
        ("Ipomoea", 55),
        ("Castor plant", 60),
        ("Ricinus communis", 65),
        ("Amaranth", 70),
        ("Pigweed", 68),
    ]
    
    print(f"📊 Database loaded: {len(global_db)} global plants, {len(weed_db)} weeds")
    print()
    
    print("🔍 Testing Cotton Field Indicators:")
    print("-" * 40)
    
    for plant_name, confidence in cotton_field_cases:
        print(f"Testing: '{plant_name}' (confidence: {confidence}%)")
        
        # Test cotton field detection
        is_cotton_field = is_likely_cotton_field(plant_name, confidence)
        print(f"   🌾 Cotton field detected: {is_cotton_field}")
        
        # Test smart identification
        plant_info, plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        if plant_info and plant_type:
            print(f"   ✅ Corrected to: {plant_info.get('CommonName', 'Unknown')}")
            print(f"   📝 Scientific: {plant_info.get('ScientificName', 'Unknown')}")
            print(f"   🏷️  Type: {plant_type}")
            if correction_note:
                print(f"   📝 Note: {correction_note}")
        else:
            print(f"   ❌ No correction applied")
        print()
    
    # Test cases that should NOT be corrected (high confidence, different plants)
    print("🌿 Testing High Confidence Non-Cotton Plants:")
    print("-" * 50)
    
    non_cotton_cases = [
        ("Dandelion", 85),  # High confidence weed
        ("Rose", 90),  # High confidence flower
        ("Soybean", 85),  # High confidence soybean (should not correct)
        ("Tomato", 88),  # High confidence vegetable
    ]
    
    for plant_name, confidence in non_cotton_cases:
        print(f"Testing: '{plant_name}' (confidence: {confidence}%)")
        
        is_cotton_field = is_likely_cotton_field(plant_name, confidence)
        print(f"   🌾 Cotton field detected: {is_cotton_field}")
        
        plant_info, plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        if correction_note and 'cotton' in correction_note.lower():
            print(f"   ⚠️  Incorrectly corrected to cotton!")
        else:
            print(f"   ✅ Correctly not corrected to cotton")
        print()

if __name__ == "__main__":
    test_cotton_field_detection()
