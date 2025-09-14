"""
Debug why Plantain and Carrot are being identified as cotton
"""

import json
from simple_app import (smart_plant_identification, load_plant_databases, 
                       is_likely_cotton_field, enhanced_plant_search, 
                       is_likely_misidentified_weed, find_best_match)

def debug_plantain_carrot():
    """Debug the identification process for Plantain and Carrot"""
    
    # Load databases
    global_db, weed_db = load_plant_databases()
    
    print("🔍 Debugging Plantain and Carrot Identification")
    print("=" * 50)
    
    test_cases = [
        ("Plantain", 48),
        ("Carrot", 68),
    ]
    
    for plant_name, confidence in test_cases:
        print(f"\n🌱 Debugging: '{plant_name}' (confidence: {confidence}%)")
        print("-" * 40)
        
        # Step 1: Cotton field detection
        is_cotton_field = is_likely_cotton_field(plant_name, confidence)
        print(f"1. Cotton field detected: {is_cotton_field}")
        
        # Step 2: Misidentified weed check
        is_misidentified, likely_crop = is_likely_misidentified_weed(plant_name, confidence)
        print(f"2. Misidentified weed: {is_misidentified}, likely crop: {likely_crop}")
        
        # Step 3: Enhanced plant search
        plant_info, plant_type = enhanced_plant_search(plant_name, global_db, weed_db)
        print(f"3. Enhanced search result: {plant_info.get('CommonName', 'None') if plant_info else 'None'}, type: {plant_type}")
        
        # Step 4: Direct database search
        global_match = find_best_match(plant_name, global_db)
        weed_match = find_best_match(plant_name, weed_db)
        print(f"4. Global DB match: {global_match.get('CommonName', 'None') if global_match else 'None'}")
        print(f"5. Weed DB match: {weed_match.get('CommonName', 'None') if weed_match else 'None'}")
        
        # Step 5: Full smart identification
        final_plant_info, final_plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        print(f"6. Final result: {final_plant_info.get('CommonName', 'None') if final_plant_info else 'None'}")
        print(f"7. Final type: {final_plant_type}")
        print(f"8. Correction note: {correction_note}")
        
        # Check if it's cotton
        is_cotton_result = False
        if final_plant_info and 'cotton' in final_plant_info.get('CommonName', '').lower():
            is_cotton_result = True
        if correction_note and 'cotton' in correction_note.lower():
            is_cotton_result = True
            
        print(f"9. Is identified as cotton: {is_cotton_result}")
        
        if is_cotton_result:
            print("❌ PROBLEM: This should NOT be identified as cotton!")
        else:
            print("✅ GOOD: Not identified as cotton")

if __name__ == "__main__":
    debug_plantain_carrot()
