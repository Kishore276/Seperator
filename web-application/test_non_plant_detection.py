"""
Test non-plant detection functionality
"""

import json
from simple_app import smart_plant_identification, load_plant_databases, is_likely_non_plant

def test_non_plant_detection():
    """Test non-plant detection with various inputs"""
    
    # Load databases
    global_db, weed_db = load_plant_databases()
    
    print("🚫 Testing Non-Plant Detection")
    print("=" * 50)
    
    # Test cases that should be detected as non-plants
    non_plant_cases = [
        ("Metal object", 25),  # Very low confidence
        ("Plastic bottle", 30),
        ("Building structure", 20),
        ("Human face", 40),
        ("Animal", 35),
        ("Cooked food", 45),
        ("Fabric", 30),
        ("Glass", 25),
        ("Paper", 20),
        ("Unknown object", 15),  # Very low confidence
        ("Artifact", 28),
        ("Tool", 32),
    ]
    
    print(f"📊 Database loaded: {len(global_db)} global plants, {len(weed_db)} weeds")
    print()
    
    print("🔍 Testing Non-Plant Cases:")
    print("-" * 30)
    
    for item_name, confidence in non_plant_cases:
        print(f"Testing: '{item_name}' (confidence: {confidence}%)")
        
        # Test non-plant detection
        is_non_plant = is_likely_non_plant(item_name, confidence)
        print(f"   🚫 Non-plant detected: {is_non_plant}")
        
        # Test smart identification
        plant_info, plant_type, correction_note = smart_plant_identification(
            item_name, confidence, global_db, weed_db
        )
        
        if plant_info is None and plant_type is None and correction_note:
            print(f"   ✅ Correctly rejected as non-plant")
            print(f"   📝 Message: {correction_note}")
        else:
            print(f"   ❌ Incorrectly identified as plant")
        print()
    
    # Test cases that should NOT be detected as non-plants (legitimate plants)
    print("🌱 Testing Legitimate Plant Cases:")
    print("-" * 40)
    
    plant_cases = [
        ("Rose", 85),  # High confidence plant
        ("Dandelion", 80),
        ("Oak tree", 90),
        ("Grass", 75),
        ("Fern", 82),
        ("Moss", 70),
        ("Cactus", 88),
        ("Sunflower", 92),
    ]
    
    for plant_name, confidence in plant_cases:
        print(f"Testing: '{plant_name}' (confidence: {confidence}%)")
        
        is_non_plant = is_likely_non_plant(plant_name, confidence)
        print(f"   🚫 Non-plant detected: {is_non_plant}")
        
        plant_info, plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        if is_non_plant:
            print(f"   ❌ Incorrectly rejected as non-plant")
        else:
            print(f"   ✅ Correctly accepted as potential plant")
        print()
    
    # Test edge cases (low confidence but plant names)
    print("⚠️  Testing Edge Cases (Low Confidence Plants):")
    print("-" * 50)
    
    edge_cases = [
        ("Unknown plant", 25),  # Low confidence but mentions plant
        ("Some flower", 30),
        ("Tree species", 28),
        ("Weed type", 22),
        ("Leaf", 35),
    ]
    
    for plant_name, confidence in edge_cases:
        print(f"Testing: '{plant_name}' (confidence: {confidence}%)")
        
        is_non_plant = is_likely_non_plant(plant_name, confidence)
        print(f"   🚫 Non-plant detected: {is_non_plant}")
        
        plant_info, plant_type, correction_note = smart_plant_identification(
            plant_name, confidence, global_db, weed_db
        )
        
        if plant_info is None and plant_type is None and correction_note:
            print(f"   📝 Result: {correction_note}")
        else:
            print(f"   📝 Processed as potential plant")
        print()

if __name__ == "__main__":
    test_non_plant_detection()
