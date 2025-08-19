import re
from services.chat_py import *
import json
import requests
from shapely.geometry import Polygon
from shapely.wkb import dumps


def parse_coordinates(input_string):
    """Parse coordinate string into a dictionary mapping place names to coordinates."""
    lines = input_string.strip().split('\n')
    coordinates_dict = {}
    
    for line in lines:
        # Use regex to capture coordinates and place names with spaces
        match = re.match(r'([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(.+)', line)
        if match:
            lon1, lat1, lon2, lat2, place = match.groups()
            # Create list with latitudes first, then longitudes
            coordinates = [lat1, lat2, lon1, lon2]
            coordinates_dict[place] = coordinates
    
    return coordinates_dict


def process_query(query, messages=None):
    """Process a location query using GPT to get bounding box coordinates."""
    if query is None:
        return None
    
    if messages is None:
        messages = []
    
    ask_prompt = """
    Give me the exact boundingbox according to the query 
    Example:
    User: Munich Moosach
    Return [48.165392, 48.205211, 11.465675, 11.541543]
    Return in json:
    ```json
    {
    "boundingbox":[]
    }
    ```
    """
    
    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(query)))
    result = chat_single(messages, 'json_few_shot', verbose=True)
    
    return result['boundingbox']


def find_boundbox(name):
    """
    Find bounding box coordinates for a given location name.
    
    Args:
        name: Location name to search for
        
    Returns:
        Tuple of (coordinates list, WKB hex string, 'bounding')
    """
    # Predefined locations dictionary
    predefined_locations = {
        'Munich': ['48.061625', '48.248098', '11.360777', '11.72291'],
        'Augsburg': ['48.2581', '48.4587', '10.7634', '10.9593'],
        'Munich Moosach': ['48.165392', '48.205211', '11.465675', '11.541543'],
        'Munich Maxvorstadt': ['48.139603', '48.157637', '11.538923', '11.588192'],
        'Munich Ismaning': ['48.201643', '48.278978', '11.643546', '11.759782'],
        'Freising': ['48.330606', '48.449032', '11.640451', '11.792508'],
        'Oberschleissheim': ['48.213303', '48.280737', '11.499759', '11.615142'],
        'Hadern': ['48.095394', '48.130837', '11.462969', '11.502689']
    }
    
    # Check if location is in predefined dictionary
    if name in predefined_locations:
        bounding_box = predefined_locations[name]
    else:
        bounding_box = process_query(name)
    
    # Convert to float coordinates
    coordinates = [float(coord) for coord in bounding_box]
    
    # Decompose coordinates into latitude and longitude
    lat1, lat2, lon1, lon2 = coordinates
    print(name, coordinates)
    
    # Build rectangle polygon from four corner points
    rectangle = Polygon([
        (lon1, lat1), (lon1, lat2), (lon2, lat2), 
        (lon2, lat1), (lon1, lat1)
    ])
    
    # Convert rectangle to WKB format
    wkb = dumps(rectangle, hex=True)
    
    return coordinates, wkb, 'bounding'