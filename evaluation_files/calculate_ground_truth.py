import json
import psycopg2
import pandas as pd
from shapely import wkb
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database configuration
from config.database_config import POSTGRES_CONFIG

# Import geo_functions methods
from core.geo_functions import ids_of_type, geo_calculate, global_id_geo

# Database connections using config
conn = psycopg2.connect(**POSTGRES_CONFIG)
cur = conn.cursor()

def get_tables_for_entity(fclass=None, name=None):
    """Get tables containing the entity using SQL queries"""
    tables_found = []
    
    # Define table-column mappings
    table_checks = [
        ('landuse', 'fclass', 'area'),  # (actual_table, column, geo_functions_table)
        ('points', 'fclass', 'points'),
        ('lines', 'fclass', 'lines'),
        ('buildings', 'type', 'buildings'),  # buildings uses 'type' instead of 'fclass'
        ('soilcomplete', 'leg_text', 'soil')  # soil uses 'leg_text'
    ]
    
    for actual_table, fclass_col, geo_table in table_checks:
        try:
            # Build query based on what we're looking for
            if fclass and name:
                # Both fclass and name - check if both exist
                if actual_table == 'soilcomplete':
                    # Soil doesn't have name column
                    if fclass:
                        query = f"SELECT COUNT(*) FROM {actual_table} WHERE {fclass_col} = %s"
                        cur.execute(query, (fclass,))
                else:
                    query = f"SELECT COUNT(*) FROM {actual_table} WHERE {fclass_col} = %s AND name = %s"
                    cur.execute(query, (fclass, name))
            elif fclass:
                # Only fclass
                query = f"SELECT COUNT(*) FROM {actual_table} WHERE {fclass_col} = %s"
                cur.execute(query, (fclass,))
            elif name:
                # Only name
                if actual_table == 'soilcomplete':
                    continue  # Soil doesn't have name column
                query = f"SELECT COUNT(*) FROM {actual_table} WHERE name = %s"
                cur.execute(query, (name,))
            else:
                continue
                
            count = cur.fetchone()[0]
            if count > 0:
                tables_found.append(geo_table)
                
        except Exception as e:
            # Table or column might not exist
            continue
    
    return tables_found

def get_entity_data(entity):
    """Get data for an entity from all relevant tables"""
    entity_data = {}
    
    # Get tables for this entity
    tables = get_tables_for_entity(
        fclass=entity.get("fclass"),
        name=entity.get("name")
    )
    
    if not tables:
        return entity_data
    
    # Get data from each table
    for table in tables:
        type_dict = {
            'non_area_col': {},
            'area_num': None
        }
        
        # Add fclass if exists
        if entity.get("fclass"):
            type_dict['non_area_col']['fclass'] = {entity["fclass"]}
        
        # Add name if exists
        if entity.get("name"):
            type_dict['non_area_col']['name'] = {entity["name"]}
        
        # If no specific filter, get all
        if not type_dict['non_area_col']:
            type_dict['non_area_col']['fclass'] = {'all'}
        
        try:
            result = ids_of_type(table, type_dict, test_mode=True)
            if result and 'id_list' in result:
                entity_data.update(result['id_list'])
        except Exception as e:
            continue
    
    return entity_data

def process_single_query(query_data, query_index):
    """Process a single query from JSONL - handles multiple entities and relationships"""
    print(f"\n{'='*60}")
    print(f"Query {query_index}: {query_data['query_result']}")
    print(f"Words: {query_data['words']}")
    print(f"Relationships: {query_data['relationships']}")
    print(f"Expected num_result: {query_data['num_result']}")
    
    words = query_data["words"]
    relationships = query_data["relationships"]
    num_result = query_data["num_result"]
    
    # Get data for all entities
    entities_data = []
    for i, entity in enumerate(words):
        print(f"\nEntity {i+1}: {entity}")
        data = get_entity_data(entity)
        print(f"  Found {len(data)} items")
        entities_data.append(data)
    
    # Check if all entities have data
    if any(len(data) == 0 for data in entities_data):
        print(f"[ERROR] Some entities have no data")
        return []
    
    # Process relationships
    if len(relationships) == 1:
        # Simple case: 2 entities, 1 relationship
        results = calculate_spatial_relationship(
            entities_data[0], 
            entities_data[1], 
            relationships[0]
        )
    elif len(relationships) == 2:
        # 3 entities, 2 relationships (A rel1 B, A rel2 C)
        # First entity relates to second and third
        results1 = calculate_spatial_relationship(
            entities_data[0], 
            entities_data[1], 
            relationships[0]
        )
        
        results2 = calculate_spatial_relationship(
            entities_data[0], 
            entities_data[2], 
            relationships[1]
        )
        
        # Find intersection - entities from first set that satisfy both relationships
        if results1 and results2:
            # Create dictionaries keyed by full_id
            dict1 = {r['full_id']: r for r in results1}
            dict2 = {r['full_id']: r for r in results2}
            # Find common IDs
            common_ids = set(dict1.keys()).intersection(set(dict2.keys()))
            # Return the element info for common IDs
            results = [dict1[id] for id in common_ids]
        else:
            results = []
            
    elif len(relationships) == 3:
        # 4 entities, 3 relationships (A rel1 B, A rel2 C, A rel3 D)
        # First entity relates to all others
        results1 = calculate_spatial_relationship(
            entities_data[0], 
            entities_data[1], 
            relationships[0]
        )
        
        results2 = calculate_spatial_relationship(
            entities_data[0], 
            entities_data[2], 
            relationships[1]
        )
        
        results3 = calculate_spatial_relationship(
            entities_data[0], 
            entities_data[3], 
            relationships[2]
        )
        
        # Find intersection - entities that satisfy all three relationships
        if results1 and results2 and results3:
            dict1 = {r['full_id']: r for r in results1}
            dict2 = {r['full_id']: r for r in results2}
            dict3 = {r['full_id']: r for r in results3}
            # Find common IDs across all three
            common_ids = set(dict1.keys()).intersection(set(dict2.keys())).intersection(set(dict3.keys()))
            # Return the element info for common IDs
            results = [dict1[id] for id in common_ids]
        else:
            results = []
    else:
        print(f"[ERROR] Unsupported number of relationships: {len(relationships)}")
        results = []
    
    print(f"\nActual results count: {len(results)}")
    
    if len(results) == num_result:
        print(f"[MATCH] Count matches! ({len(results)} == {num_result})")
    else:
        print(f"[MISMATCH] Count mismatch! ({len(results)} != {num_result})")
    
    return results

def parse_element_id(element_id):
    """Parse element ID to extract table, fclass, name, osm_id"""
    # Format: table_fclass_name_osmid or table_name_fclass_osmid
    parts = element_id.split('_')
    
    if len(parts) >= 4:
        # Try different parsing patterns
        if parts[0] in ['area', 'points', 'lines', 'buildings', 'soil']:
            table = parts[0]
            # Remaining parts could be fclass_name_osmid or name_fclass_osmid
            # The last part is always osm_id
            osm_id = parts[-1]
            
            # Middle parts are fclass and/or name
            if len(parts) == 4:
                # table_fclass_name_osmid or table_name_fclass_osmid
                fclass = parts[1]
                name = parts[2]
            elif len(parts) == 3:
                # table_fclass_osmid
                fclass = parts[1]
                name = ""
            else:
                # More complex - rejoin middle parts
                fclass = parts[1]
                name = '_'.join(parts[2:-1])
        else:
            # Fallback
            table = "unknown"
            fclass = parts[0] if len(parts) > 0 else ""
            name = '_'.join(parts[1:-1]) if len(parts) > 2 else ""
            osm_id = parts[-1] if len(parts) > 0 else ""
    else:
        # Fallback for unexpected format
        table = "unknown"
        fclass = ""
        name = ""
        osm_id = element_id
    
    return {
        'table': table,
        'fclass': fclass,
        'name': name,
        'osm_id': osm_id,
        'full_id': element_id
    }

def calculate_spatial_relationship(first_data, second_data, relationship):
    """Calculate spatial relationship between two datasets"""
    results = []
    
    # Map relationships to modes
    relationship_mode_map = {
        "contains": "contains",
        "in": "in",
        "within": "in",
        "intersects": "intersects",
        "intersect": "intersects",
        "buffer": "buffer",
        "around": "buffer",
        "near": "buffer"
    }
    
    # Get the mode for the relationship
    mode = relationship_mode_map.get(relationship, relationship)
    
    # Calculate using geo_calculate
    try:
        result = geo_calculate(
            data_list1_original=first_data,  # 主语
            data_list2_original=second_data,  # 宾语
            mode=mode,
            buffer_number=500 if mode == "buffer" else 0,  # 500m buffer for buffer mode
            test_mode=True
        )
        
        # Extract results - for all relationships, use subject
        if result and 'subject' in result and 'id_list' in result['subject']:
            # Return detailed info for each result
            results = []
            for element_id in result['subject']['id_list'].keys():
                element_info = parse_element_id(element_id)
                results.append(element_info)
                        
    except Exception as e:
        print(f"Error calculating {relationship}: {e}")
    
    return results

def process_jsonl_file(input_file, start_line=0, end_line=None):
    """Process a JSONL file and return detailed results"""
    detailed_results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        
        if end_line is None:
            end_line = len(all_lines)
        
        lines = all_lines[start_line:min(end_line, len(all_lines))]
        
        print(f"\n{'='*60}")
        print(f"Processing {input_file}")
        print(f"Lines {start_line+1} to {start_line+len(lines)}")
        print(f"{'='*60}")
        
        for idx, line in enumerate(tqdm(lines, desc="Processing queries"), start_line+1):
            try:
                query_data = json.loads(line.strip())
                results = process_single_query(query_data, idx)
                
                # Store detailed results
                detailed_results.append({
                    'query_index': idx,
                    'query': query_data['query_result'],
                    'words': query_data['words'],
                    'relationships': query_data['relationships'],
                    'expected_count': query_data['num_result'],
                    'actual_count': len(results),
                    'match': len(results) == query_data['num_result'],
                    'results': results  # List of elements with table, fclass, name, osm_id
                })
                
            except Exception as e:
                print(f"Error processing line {idx}: {e}")
                detailed_results.append({
                    'query_index': idx,
                    'query': query_data.get('query_result', 'Unknown'),
                    'words': query_data.get('words', []),
                    'relationships': query_data.get('relationships', []),
                    'expected_count': query_data.get('num_result', 0),
                    'actual_count': 0,
                    'match': False,
                    'results': [],
                    'error': str(e)
                })
                continue
    
    return detailed_results

def main():
    # Get the evaluation_files directory path
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define all JSONL files to process
    jsonl_files = [
        ("evaluation_first_dataset.jsonl", "first"),
        ("evaluation_second_dataset.jsonl", "second"),
        ("evaluation_third_dataset.jsonl", "third"),
        ("evaluation_fourth_dataset.jsonl", "fourth")
    ]
    
    # Process each file
    all_results = {}
    
    for filename, label in jsonl_files:
        input_file = os.path.join(eval_dir, filename)
        
        # Check if file exists
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        # You can adjust these ranges for each file
        if label == "first":
            start_line, end_line = 0, 100  # Process first 100 lines
        else:
            start_line, end_line = 0, 100  # Process first 100 lines
        
        # Process the file
        detailed_results = process_jsonl_file(input_file, start_line, end_line)
        all_results[label] = detailed_results
        
        # Save results as JSONL in the same directory
        output_file = os.path.join(eval_dir, f"ground_truth_{label}_{start_line+1}_to_{start_line+len(detailed_results)}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in detailed_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        # Print summary for this file
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {label.upper()} FILE:")
        print(f"{'='*60}")
        
        # Group by relationship complexity
        complexity_stats = {}
        for result in detailed_results:
            num_entities = len(result['words'])
            num_relationships = len(result['relationships'])
            complexity = f"{num_entities}entities_{num_relationships}rels"
            if complexity not in complexity_stats:
                complexity_stats[complexity] = {'total': 0, 'matches': 0}
            complexity_stats[complexity]['total'] += 1
            if result['match']:
                complexity_stats[complexity]['matches'] += 1
        
        # Print stats
        for complexity, stats in complexity_stats.items():
            match_rate = stats['matches'] * 100 / stats['total'] if stats['total'] > 0 else 0
            print(f"{complexity}: {stats['matches']}/{stats['total']} ({match_rate:.1f}%)")
        
        # Overall stats for this file
        total_matches = sum(1 for result in detailed_results if result['match'])
        total_queries = len(detailed_results)
        overall_rate = total_matches * 100 / total_queries if total_queries > 0 else 0
        
        print(f"\nOverall for {label}: {total_matches}/{total_queries} ({overall_rate:.1f}%)")
        print(f"Results saved to: {output_file}")
    
    # Print overall summary across all files
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY ACROSS ALL FILES:")
    print(f"{'='*60}")
    
    grand_total_matches = 0
    grand_total_queries = 0
    
    for label, results in all_results.items():
        matches = sum(1 for r in results if r['match'])
        total = len(results)
        rate = matches * 100 / total if total > 0 else 0
        print(f"{label}: {matches}/{total} ({rate:.1f}%)")
        grand_total_matches += matches
        grand_total_queries += total
    
    if grand_total_queries > 0:
        grand_rate = grand_total_matches * 100 / grand_total_queries
        print(f"\nGrand Total: {grand_total_matches}/{grand_total_queries} ({grand_rate:.1f}%)")
    
    # Close connections
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()