"""
Test script to verify Siam weed misidentification correction
"""

import json
from simple_app import smart_plant_identification, load_plant_databases, is_likely_misidentified_weed

def test_siam_weed_correction():
    """Test that Siam weed is correctly identified as cotton when appropriate"""
    
    # Load databases
    global_db, weed_db = load_plant_databases()
    
    print("🧪 Testing Siam Weed Correction")
    print("=" * 50)
    
    # Test cases that should be corrected
    test_cases = [
        ("Siam weed", 65),  # Low confidence
        ("Chromolaena odorata", 60),  # Low confidence
        ("siam weed", 45),  # Very low confidence
        ("SIAM WEED", 70),  # Higher confidence - should still correct
    ]
    
    print(f"📊 Database loaded: {len(global_db)} global plants, {len(weed_db)} weeds")
    print()
    
    for plant_name, confidence in test_cases:
        print(f"🔍 Testing: '{plant_name}' (confidence: {confidence}%)")
        
        # Test misidentification detection
        is_misidentified, likely_crop = is_likely_misidentified_weed(plant_name, confidence)
        print(f"   🤔 Misidentified: {is_misidentified}")
        if is_misidentified:
            print(f"   🎯 Likely crop: {likely_crop}")
        
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
            print(f"   ❌ No correction found")
        print()
    
    # Test cases that should NOT be corrected (high confidence weeds)
    print("🌿 Testing High Confidence Weeds (should NOT be corrected)")
    print("-" * 60)
    
    high_confidence_cases = [
        ("Siam weed", 85),  # High confidence - should accept as weed
        ("Dandelion", 90),  # Different weed, high confidence
    ]
    
    for plant_name, confidence in high_confidence_cases:
        print(f"🔍 Testing: '{plant_name}' (confidence: {confidence}%)")
        
        is_misidentified, likely_crop = is_likely_misidentified_weed(plant_name, confidence)
        print(f"   🤔 Misidentified: {is_misidentified}")
        
        plant_info, plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        if plant_info:
            print(f"   📝 Result: {plant_info.get('CommonName', plant_info.get('WeedType', 'Unknown'))}")
            print(f"   🏷️  Type: {plant_type}")
            if correction_note:
                print(f"   📝 Note: {correction_note}")
            else:
                print(f"   ✅ No correction applied (as expected)")
        print()

if __name__ == "__main__":
    test_siam_weed_correction()
