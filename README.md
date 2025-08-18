# GeoQA - Geographic Question Answering System

## Architecture

```
┌────────────────────────────────────────────────┐
│              Flask Application                 │
├────────────────────────────────────────────────┤
│                Core Services                   │
├──────────────┬──────────────┬──────────────────┤
│  Spatial DB  │    Neo4j     │    ChromaDB      │
│   (Spatial)  │   (Graph)    │   (Vectors)      │
└──────────────┴──────────────┴──────────────────┘
```

## Prerequisites

- Python 3.9+
- PostgreSQL with PostGIS extension
- Neo4j Database
- ChromaDB
- OpenAI API key

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd geo_qa
```

2. **Download the dataset**
Download the geographic dataset from Hugging Face:
```bash
# Download from: https://huggingface.co/datasets/boboIloveyou/Geo-QA-Dataset/tree/main/dataset_sql
# The dataset contains SQL dumps for PostgreSQL with spatial data mentioned in paper
```

3. **Deploy dataset to PostgreSQL**
Import the downloaded SQL dumps into your PostgreSQL database:
```bash
# Create database with PostGIS extension
createdb -U postgres osm_database
psql -U postgres -d osm_database -c "CREATE EXTENSION postgis;"

# Import the dataset SQL files
psql -U postgres -d osm_database < path/to/dataset_sql/[dataset_file].sql
```

4. **Update database configuration**
Edit `config/database_config.py` to match your database settings:
```python
POSTGRES_CONFIG = {
    "dbname": "osm_database",
    "user": "your_postgres_user",
    "host": "localhost",
    "password": "your_postgres_password"
}

NEO4J_CONFIG = {
    "uri": "neo4j://127.0.0.1:7687",
    "user": "neo4j",
    "password": "your_neo4j_password"
}
```

5. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

6. **Configure environment variables**
Create a `config/.env` file:
```env
OPENAI_API_KEY=your_openai_api_key
```

7. **Set up databases**
- Ensure PostgreSQL is running with the imported data
- Start Neo4j database
- Launch ChromaDB server:
```bash
chroma run --host localhost --port 8000
```

8. **Build Neo4j graph from PostgreSQL data**
```bash
python build_graph_for_database.py
```

9. **Initialize ChromaDB vector database**
```bash
python populate_chroma_vectors.py
```

## Usage

### Starting the Application

```bash
python app_changed_agent.py
```

The application will start on `http://localhost:5001`

### Example Queries

1. **Find entities by type**
   - "Show me all hospitals"
   - "Find buildings of type school"

2. **Spatial relationships**
   - "Which parks contain playgrounds?"
   - "Find restaurants within 100m of the main station"

3. **Complex queries**
   - "Show large commercial areas that intersect with residential zones"
   - "Find all schools not near any parks"

## Project Structure

```
geo_qa/
├── app_changed_agent.py      # Main Flask application
├── config/
│   ├── config.py             # Application configuration
│   └── database_config.py    # Database configurations
├── core/
│   ├── ask_functions_agent.py # Core agent logic
│   ├── geo_functions.py      # Spatial operations
│   ├── bounding_box.py       # Bounding box utilities
│   ├── neo4j_mapper.py       # Neo4j integration
│   └── agent_search_fast.py  # search functions
├── services/
│   ├── rag_service.py        # RAG and vector search
│   └── chat_py.py            # LLM integration
├── utils/
│   ├── optimization_utils.py # Performance optimizations
│   ├── security_utils.py     # Security utilities
│   └── levenshtein.py        # String similarity
├── static/                   # Static files and GeoJSON
├── templates/                # HTML templates
├── evaluation_files/         # Test datasets from paper
├── populate_chroma_vectors.py # ChromaDB initialization
└── build_graph_for_database.py # Neo4j graph builder
```

## Evaluation Datasets

The `evaluation_files/` directory contains four test datasets used in the paper, each in JSONL format.


### Dataset Descriptions

1. **evaluation_first_dataset.jsonl**: Simple containment queries
   - Tests basic spatial containment relationships
   - Example: "Which pitch contains a recycling glass?"

2. **evaluation_second_dataset.jsonl**: Complex spatial queries with names
   - Tests combinations of intersections and buffer operations
   - Includes specific entity names
   - Example: "River named Sempt that intersects with stream named Kleine Sempt and is within 500m of hotel named Gasthof Sempt"

3. **evaluation_third_dataset.jsonl**: Named containment queries
   - Tests containment with specific entity names
   - Example: "Park named Kurpark that contains a pitch named Freiluft-Schach"

4. **evaluation_fourth_dataset.jsonl**: Multi-relationship queries
   - Tests complex combinations of multiple spatial relationships
   - Example: "River that intersects with multiple streams and is within buffer distance of multiple hotels"

### Fields Explanation
- `query_result`: The natural language query to be processed
- `words`: Array of entities to search for, each with:
  - `fclass`: Feature classification (required)
  - `name`: Specific entity name (optional)
- `relationships`: Spatial operations to perform (contains, intersects, buffer)
- `num_result`: Expected number of results for validation

### Ground Truth Calculation
To calculate ground truth values for the evaluation datasets:

```bash
cd evaluation_files
python calculate_ground_truth.py
```

This script:
- Reads each evaluation dataset
- Processes queries using actual spatial database operations
- Generates ground truth results with detailed entity information
- Outputs results to `ground_truth_[dataset]_*.jsonl` files
- Provides accuracy statistics for each dataset
