"""
Test young cotton plant detection to ensure they're not rejected as non-plants
"""

import json
from simple_app import smart_plant_identification, load_plant_databases, is_likely_non_plant, is_likely_cotton_field

def test_young_cotton_detection():
    """Test detection of young cotton plants with very low confidence"""
    
    # Load databases
    global_db, weed_db = load_plant_databases()
    
    print("🌱 Testing Young Cotton Plant Detection")
    print("=" * 50)
    
    # Test cases that represent young cotton plants with very low confidence
    young_cotton_cases = [
        ("Ceiba pentandra", 2.64),  # Your actual case
        ("Abronia fragrans", 3.22),  # Another actual case
        ("Ipomoea batatas", 13.29),  # Sweet potato - young leaves similar to cotton
        ("Unknown plant", 5.0),  # Very low confidence unidentified
        ("Small green plant", 8.0),
        ("Young seedling", 12.0),
        ("Unidentified leaf", 7.5),
        ("Green sprout", 4.2),
        ("Plant species", 6.8),
        ("Bombax ceiba", 9.1),  # Cotton tree family
        ("Kapok tree", 11.5),  # Cotton family
        ("Sand verbena", 14.2),  # Sometimes confused with cotton
    ]
    
    print(f"📊 Database loaded: {len(global_db)} global plants, {len(weed_db)} weeds")
    print()
    
    print("🔍 Testing Young Cotton Cases (Should NOT be rejected):")
    print("-" * 55)
    
    for plant_name, confidence in young_cotton_cases:
        print(f"Testing: '{plant_name}' (confidence: {confidence}%)")
        
        # Test non-plant detection (should be False for these)
        is_non_plant = is_likely_non_plant(plant_name, confidence)
        print(f"   🚫 Non-plant detected: {is_non_plant}")
        
        # Test cotton field detection
        is_cotton_field = is_likely_cotton_field(plant_name, confidence)
        print(f"   🌾 Cotton field detected: {is_cotton_field}")
        
        # Test smart identification
        plant_info, plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        if plant_info is None and plant_type is None and correction_note:
            print(f"   ❌ INCORRECTLY REJECTED: {correction_note}")
        elif plant_info and 'cotton' in plant_info.get('CommonName', '').lower():
            print(f"   ✅ CORRECTLY IDENTIFIED AS COTTON")
            print(f"   📝 Result: {plant_info.get('CommonName', 'Unknown')}")
            if correction_note:
                print(f"   📝 Note: {correction_note}")
        else:
            print(f"   ⚠️  Processed but not identified as cotton")
            if plant_info:
                print(f"   📝 Identified as: {plant_info.get('CommonName', 'Unknown')}")
        print()
    
    # Test cases that SHOULD still be rejected (actual non-plants)
    print("🚫 Testing Actual Non-Plants (Should be rejected):")
    print("-" * 50)
    
    actual_non_plants = [
        ("Metal object", 15),
        ("Plastic bottle", 18),
        ("Building structure", 12),
        ("Human face", 22),
        ("Cooked food", 19),
        ("Glass container", 16),
    ]
    
    for item_name, confidence in actual_non_plants:
        print(f"Testing: '{item_name}' (confidence: {confidence}%)")
        
        is_non_plant = is_likely_non_plant(item_name, confidence)
        print(f"   🚫 Non-plant detected: {is_non_plant}")
        
        plant_info, plant_type, correction_note = smart_plant_identification(
            item_name, confidence, global_db, weed_db
        )
        
        if plant_info is None and plant_type is None and correction_note:
            print(f"   ✅ CORRECTLY REJECTED")
        else:
            print(f"   ❌ INCORRECTLY ACCEPTED")
        print()

if __name__ == "__main__":
    test_young_cotton_detection()
