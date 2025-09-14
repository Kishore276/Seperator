"""
Debug what's in the database that's causing the matching issues
"""

import json
from simple_app import load_plant_databases

def debug_database_contents():
    """Check what's in the databases"""
    
    # Load databases
    global_db, weed_db = load_plant_databases()
    
    print("🗃️ Database Contents Analysis")
    print("=" * 40)
    
    print(f"\n📊 Global Database ({len(global_db)} plants):")
    print("-" * 30)
    for i, plant in enumerate(global_db):
        print(f"{i+1}. {plant.get('CommonName', 'Unknown')} ({plant.get('ScientificName', 'Unknown')})")
        alt_names = plant.get('AlternativeNames', [])
        if alt_names:
            print(f"   Alt names: {alt_names}")
    
    print(f"\n📊 Weed Database ({len(weed_db)} weeds):")
    print("-" * 30)
    for i, weed in enumerate(weed_db[:10]):  # Show first 10
        print(f"{i+1}. {weed.get('CommonName', 'Unknown')} ({weed.get('ScientificName', 'Unknown')})")
        weed_type = weed.get('WeedType', '')
        if weed_type:
            print(f"   Type: {weed_type}")
    
    print("\n🔍 Checking why 'Plantain' matches 'Cotton':")
    print("-" * 45)
    
    # Check each plant in global DB for potential matches with "Plantain"
    search_term = "plantain"
    for plant in global_db:
        common_name = plant.get('CommonName', '').lower()
        scientific_name = plant.get('ScientificName', '').lower()
        alt_names = [name.lower() for name in plant.get('AlternativeNames', [])]
        
        # Check if search term is IN any of these (current logic)
        if (search_term in common_name or 
            search_term in scientific_name or 
            any(search_term in alt for alt in alt_names)):
            print(f"MATCH: {plant.get('CommonName')} - '{search_term}' found in:")
            if search_term in common_name:
                print(f"  - Common name: '{common_name}'")
            if search_term in scientific_name:
                print(f"  - Scientific name: '{scientific_name}'")
            for alt in alt_names:
                if search_term in alt:
                    print(f"  - Alt name: '{alt}'")
    
    print("\n🔍 Checking why 'Carrot' matches 'Cotton':")
    print("-" * 42)
    
    # Check each plant in global DB for potential matches with "Carrot"
    search_term = "carrot"
    for plant in global_db:
        common_name = plant.get('CommonName', '').lower()
        scientific_name = plant.get('ScientificName', '').lower()
        alt_names = [name.lower() for name in plant.get('AlternativeNames', [])]
        
        # Check if search term is IN any of these (current logic)
        if (search_term in common_name or 
            search_term in scientific_name or 
            any(search_term in alt for alt in alt_names)):
            print(f"MATCH: {plant.get('CommonName')} - '{search_term}' found in:")
            if search_term in common_name:
                print(f"  - Common name: '{common_name}'")
            if search_term in scientific_name:
                print(f"  - Scientific name: '{scientific_name}'")
            for alt in alt_names:
                if search_term in alt:
                    print(f"  - Alt name: '{alt}'")

if __name__ == "__main__":
    debug_database_contents()
