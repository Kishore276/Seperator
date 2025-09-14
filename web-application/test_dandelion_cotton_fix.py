"""
Test to ensure dandelions and other legitimate plants are NOT identified as cotton
"""

import json
from simple_app import smart_plant_identification, load_plant_databases, is_likely_cotton_field

def test_dandelion_cotton_fix():
    """Test that dandelions and other plants are not incorrectly identified as cotton"""
    
    # Load databases
    global_db, weed_db = load_plant_databases()
    
    print("🌼 Testing Dandelion & Other Plants - Should NOT be Cotton")
    print("=" * 60)
    
    # Test cases that should NOT be identified as cotton
    non_cotton_cases = [
        # Dandelions (the main issue)
        ("Dandelion", 45),
        ("Taraxacum officinale", 50),
        ("Common dandelion", 40),
        
        # Other common weeds
        ("Daisy", 55),
        ("White clover", 60),
        ("Plantain", 48),
        ("Chickweed", 52),
        
        # Flowers
        ("Rose", 70),
        ("Sunflower", 80),
        ("Tulip", 65),
        ("Lily", 75),
        
        # Grasses
        ("Grass", 60),
        ("Lawn grass", 55),
        ("Bermuda grass", 58),
        
        # Crops (non-cotton)
        ("Tomato", 70),
        ("Corn", 75),
        ("Wheat", 68),
        ("Rice", 72),
        
        # Trees
        ("Oak tree", 80),
        ("Pine tree", 85),
        ("Maple", 78),
        
        # Vegetables
        ("Lettuce", 65),
        ("Cabbage", 70),
        ("Carrot", 68),
        
        # Succulents
        ("Cactus", 75),
        ("Aloe vera", 80),
        ("Succulent", 70),
    ]
    
    print(f"📊 Database loaded: {len(global_db)} global plants, {len(weed_db)} weeds")
    print()
    
    print("🔍 Testing Non-Cotton Plants (Should NOT be identified as cotton):")
    print("-" * 65)
    
    cotton_misidentifications = 0
    total_tests = len(non_cotton_cases)
    
    for plant_name, confidence in non_cotton_cases:
        print(f"Testing: '{plant_name}' (confidence: {confidence}%)")
        
        # Test cotton field detection
        is_cotton_field = is_likely_cotton_field(plant_name, confidence)
        print(f"   🌾 Cotton field detected: {is_cotton_field}")
        
        # Test smart identification
        plant_info, plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        # Check if it was incorrectly identified as cotton
        is_cotton_result = False
        if plant_info and 'cotton' in plant_info.get('CommonName', '').lower():
            is_cotton_result = True
        if correction_note and 'cotton' in correction_note.lower():
            is_cotton_result = True
            
        if is_cotton_result:
            print(f"   ❌ INCORRECTLY IDENTIFIED AS COTTON!")
            if plant_info:
                print(f"      Result: {plant_info.get('CommonName', 'Unknown')}")
            if correction_note:
                print(f"      Note: {correction_note}")
            cotton_misidentifications += 1
        else:
            print(f"   ✅ Correctly NOT identified as cotton")
            if plant_info:
                print(f"      Identified as: {plant_info.get('CommonName', 'Unknown')}")
        print()
    
    # Test legitimate cotton cases to ensure they still work
    print("🌾 Testing Legitimate Cotton Cases (Should be identified as cotton):")
    print("-" * 70)
    
    cotton_cases = [
        ("Soybean", 45),  # Should be corrected to cotton
        ("Glycine max", 50),
        ("Siam weed", 40),
        ("Chromolaena odorata", 35),
        ("Okra", 55),
        ("Hibiscus", 48),
        ("Ceiba pentandra", 3),  # Very low confidence young cotton
        ("Abronia fragrans", 5),
    ]
    
    cotton_correct_identifications = 0
    cotton_total_tests = len(cotton_cases)
    
    for plant_name, confidence in cotton_cases:
        print(f"Testing: '{plant_name}' (confidence: {confidence}%)")
        
        is_cotton_field = is_likely_cotton_field(plant_name, confidence)
        print(f"   🌾 Cotton field detected: {is_cotton_field}")
        
        plant_info, plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        is_cotton_result = False
        if plant_info and 'cotton' in plant_info.get('CommonName', '').lower():
            is_cotton_result = True
        if correction_note and 'cotton' in correction_note.lower():
            is_cotton_result = True
            
        if is_cotton_result:
            print(f"   ✅ Correctly identified as cotton")
            cotton_correct_identifications += 1
        else:
            print(f"   ❌ FAILED to identify as cotton")
        print()
    
    # Summary
    print("📊 SUMMARY:")
    print("=" * 40)
    print(f"Non-Cotton Plants: {total_tests - cotton_misidentifications}/{total_tests} correctly NOT identified as cotton")
    print(f"Cotton Plants: {cotton_correct_identifications}/{cotton_total_tests} correctly identified as cotton")
    print(f"Cotton Misidentifications: {cotton_misidentifications}")
    
    if cotton_misidentifications == 0:
        print("🎉 SUCCESS: No plants incorrectly identified as cotton!")
    else:
        print(f"⚠️  WARNING: {cotton_misidentifications} plants incorrectly identified as cotton")

if __name__ == "__main__":
    test_dandelion_cotton_fix()
