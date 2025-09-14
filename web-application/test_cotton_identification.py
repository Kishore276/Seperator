"""
Test script to verify cotton identification works correctly
"""

import json
from simple_app import enhanced_plant_search, load_plant_databases

def test_cotton_identification():
    """Test cotton identification with various inputs"""
    
    # Load databases
    global_db, weed_db = load_plant_databases()
    
    print("🧪 Testing Cotton Identification")
    print("=" * 50)
    
    # Test cases for cotton
    test_cases = [
        "cotton",
        "Cotton",
        "COTTON",
        "Gossypium hirsutum",
        "gossypium",
        "upland cotton",
        "american cotton",
        "white cotton",
        "cotton plant",
        "kapas",
        "fiber crop"
    ]
    
    print(f"📊 Database loaded: {len(global_db)} global plants, {len(weed_db)} weeds")
    print()
    
    for test_input in test_cases:
        print(f"🔍 Testing: '{test_input}'")
        
        plant_info, plant_type = enhanced_plant_search(test_input, global_db, weed_db)
        
        if plant_info and plant_type:
            print(f"   ✅ Found: {plant_info.get('CommonName', 'Unknown')}")
            print(f"   📝 Scientific: {plant_info.get('ScientificName', 'Unknown')}")
            print(f"   🏷️  Type: {plant_type}")
            print(f"   🌍 Family: {plant_info.get('Family', 'Unknown')}")
        else:
            print(f"   ❌ Not found")
        print()
    
    # Test specific cotton varieties
    print("🌱 Testing Cotton Varieties")
    print("-" * 30)
    
    cotton_varieties = [
        "Gossypium hirsutum",
        "Gossypium barbadense", 
        "pima cotton",
        "sea island cotton"
    ]
    
    for variety in cotton_varieties:
        print(f"🔍 Testing: '{variety}'")
        plant_info, plant_type = enhanced_plant_search(variety, global_db, weed_db)
        
        if plant_info and plant_type:
            print(f"   ✅ Found: {plant_info.get('CommonName', 'Unknown')}")
            print(f"   📝 Scientific: {plant_info.get('ScientificName', 'Unknown')}")
            if 'AlternativeNames' in plant_info:
                print(f"   🔄 Alt Names: {', '.join(plant_info['AlternativeNames'])}")
        else:
            print(f"   ❌ Not found")
        print()

if __name__ == "__main__":
    test_cotton_identification()
