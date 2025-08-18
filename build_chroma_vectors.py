#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Populate Chroma Vector Database Script

This script reads fclass and name values from PostgreSQL database tables
and stores their vector embeddings in ChromaDB collections.

Collections:
- fclass_vector: Stores all fclass values with their embeddings
- name_vector: Stores all name values with their embeddings
"""

import os
import sys
import logging
from typing import Dict, List, Set, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append('.')

from config.database_config import (
    POSTGRES_CONFIG,
    COL_NAME_MAPPING_DICT,
    get_actual_column_name,
    get_actual_table_name
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('config/.env')

# Initialize OpenAI client
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
if not os.environ['OPENAI_API_KEY']:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

client = OpenAI()


class ChromaVectorPopulator:
    """Class to handle populating Chroma vector database from PostgreSQL"""
    
    def __init__(self):
        """Initialize database connections and collections"""
        # PostgreSQL connection
        self.pg_conn = None
        self.connect_postgres()
        
        # ChromaDB client and collections
        self.chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Delete existing collections if they exist to start fresh
        try:
            self.chroma_client.delete_collection("fclass_vector")
            logger.info("Deleted existing fclass_vector collection")
        except:
            pass
        
        try:
            self.chroma_client.delete_collection("name_vector")
            logger.info("Deleted existing name_vector collection")
        except:
            pass
        
        # Create new collections with correct names
        self.fclass_collection = self.chroma_client.create_collection(
            name="fclass_vector",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-3-small"
            )
        )
        logger.info("Created fclass_vector collection")
        
        self.name_collection = self.chroma_client.create_collection(
            name="name_vector",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-3-small"
            )
        )
        logger.info("Created name_vector collection")
    
    def connect_postgres(self):
        """Establish PostgreSQL database connection"""
        try:
            self.pg_conn = psycopg2.connect(**POSTGRES_CONFIG)
            self.pg_conn.autocommit = True
            logger.info("Connected to PostgreSQL database")
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def close_connections(self):
        """Close all database connections"""
        if self.pg_conn:
            self.pg_conn.close()
            logger.info("PostgreSQL connection closed")
    
    def fetch_distinct_values(self, table_name: str, column_type: str) -> Set[str]:
        """
        Fetch distinct values from a table column
        
        Args:
            table_name: Logical table name (e.g., 'buildings', 'area')
            column_type: Either 'fclass' or 'name'
            
        Returns:
            Set of distinct non-null values
        """
        actual_table = get_actual_table_name(table_name)
        actual_column = get_actual_column_name(table_name, column_type)
        
        if not actual_column:
            return set()
        
        try:
            with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if column exists
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s 
                    AND column_name = %s
                    AND table_schema = 'public'
                """, (actual_table, actual_column))
                
                if not cur.fetchone():
                    logger.warning(f"Column {actual_column} not found in table {actual_table}")
                    return set()
                
                # Fetch distinct non-null values
                query = f"""
                    SELECT DISTINCT {actual_column} as value
                    FROM {actual_table}
                    WHERE {actual_column} IS NOT NULL
                    AND {actual_column} != ''
                    ORDER BY {actual_column}
                """
                
                cur.execute(query)
                results = cur.fetchall()
                
                values = {str(row['value']) for row in results if row['value']}
                logger.info(f"Fetched {len(values)} distinct {column_type} values from {table_name}")
                return values
                
        except psycopg2.Error as e:
            logger.error(f"Error fetching {column_type} from {table_name}: {e}")
            return set()
    
    def populate_fclass_vectors(self):
        """Populate fclass_vector collection with all fclass values"""
        logger.info("Starting to populate fclass_vector collection...")
        
        all_fclass_values = set()
        fclass_by_table = {}
        
        # Collect all fclass values from each table
        for table_name in COL_NAME_MAPPING_DICT.keys():
            fclass_values = self.fetch_distinct_values(table_name, 'fclass')
            if fclass_values:
                fclass_by_table[table_name] = fclass_values
                all_fclass_values.update(fclass_values)
        
        if not all_fclass_values:
            logger.warning("No fclass values found in any table")
            return
        
        logger.info(f"Total unique fclass values to process: {len(all_fclass_values)}")
        
        # Prepare data for batch insertion
        documents = []
        ids = []
        metadatas = []
        
        for fclass_value in tqdm(all_fclass_values, desc="Preparing fclass data"):
            # Find which tables contain this fclass value
            tables_containing = [
                table for table, values in fclass_by_table.items() 
                if fclass_value in values
            ]
            
            documents.append(fclass_value)
            # Create unique ID: fclass_<value_cleaned>
            clean_value = fclass_value.replace(' ', '_').replace('/', '_')[:50]
            ids.append(f"fclass_{clean_value}_{hash(fclass_value) % 100000}")
            metadatas.append({
                "value": fclass_value,
                "type": "fclass",
                "tables": ",".join(tables_containing)
            })
        
        # Add to ChromaDB in batches
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(documents), batch_size), 
                      desc="Inserting fclass vectors", 
                      total=total_batches):
            batch_end = min(i + batch_size, len(documents))
            
            try:
                self.fclass_collection.add(
                    documents=documents[i:batch_end],
                    ids=ids[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
            except Exception as e:
                logger.error(f"Error inserting fclass batch {i//batch_size}: {e}")
        
        logger.info(f"Successfully populated {len(documents)} fclass vectors")
    
    def populate_name_vectors(self):
        """Populate name_vector collection with all name values"""
        logger.info("Starting to populate name_vector collection...")
        
        all_name_values = set()
        name_by_table = {}
        
        # Collect all name values from each table (except soil which doesn't have names)
        for table_name in COL_NAME_MAPPING_DICT.keys():
            if table_name == 'soil':  # Skip soil table for names
                continue
                
            name_values = self.fetch_distinct_values(table_name, 'name')
            if name_values:
                name_by_table[table_name] = name_values
                all_name_values.update(name_values)
        
        if not all_name_values:
            logger.warning("No name values found in any table")
            return
        
        logger.info(f"Total unique name values to process: {len(all_name_values)}")
        
        # Prepare data for batch insertion
        documents = []
        ids = []
        metadatas = []
        
        for name_value in tqdm(all_name_values, desc="Preparing name data"):
            # Find which tables contain this name value
            tables_containing = [
                table for table, values in name_by_table.items() 
                if name_value in values
            ]
            
            documents.append(name_value)
            # Create unique ID: name_<value_cleaned>
            clean_value = name_value.replace(' ', '_').replace('/', '_')[:50]
            ids.append(f"name_{clean_value}_{hash(name_value) % 100000}")
            metadatas.append({
                "value": name_value,
                "type": "name",
                "tables": ",".join(tables_containing)
            })
        
        # Add to ChromaDB in batches
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(documents), batch_size), 
                      desc="Inserting name vectors", 
                      total=total_batches):
            batch_end = min(i + batch_size, len(documents))
            
            try:
                self.name_collection.add(
                    documents=documents[i:batch_end],
                    ids=ids[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )
            except Exception as e:
                logger.error(f"Error inserting name batch {i//batch_size}: {e}")
        
        logger.info(f"Successfully populated {len(documents)} name vectors")
    
    def verify_collections(self):
        """Verify that collections were populated correctly"""
        logger.info("\n=== Verification ===")
        
        # Check fclass collection
        fclass_count = self.fclass_collection.count()
        logger.info(f"fclass_vector collection contains {fclass_count} items")
        
        # Sample query for fclass
        if fclass_count > 0:
            sample_results = self.fclass_collection.query(
                query_texts=["road"],
                n_results=3
            )
            logger.info("Sample fclass query for 'road':")
            for doc, dist, meta in zip(
                sample_results['documents'][0], 
                sample_results['distances'][0],
                sample_results['metadatas'][0]
            ):
                logger.info(f"  - {doc} (distance: {dist:.3f}, tables: {meta['tables']})")
        
        # Check name collection
        name_count = self.name_collection.count()
        logger.info(f"name_vector collection contains {name_count} items")
        
        # Sample query for name
        if name_count > 0:
            sample_results = self.name_collection.query(
                query_texts=["main street"],
                n_results=3
            )
            logger.info("Sample name query for 'main street':")
            for doc, dist, meta in zip(
                sample_results['documents'][0], 
                sample_results['distances'][0],
                sample_results['metadatas'][0]
            ):
                logger.info(f"  - {doc} (distance: {dist:.3f}, tables: {meta['tables']})")
    
    def run(self):
        """Main execution method"""
        try:
            logger.info("Starting Chroma vector population process...")
            
            # Populate fclass vectors
            self.populate_fclass_vectors()
            
            # Populate name vectors
            self.populate_name_vectors()
            
            # Verify the collections
            self.verify_collections()
            
            logger.info("\n✅ Chroma vector population completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during population process: {e}")
            raise
        finally:
            self.close_connections()


def main():
    """Main entry point"""
    try:
        # Check if ChromaDB is running
        try:
            test_client = chromadb.HttpClient(host="localhost", port=8000)
            test_client.heartbeat()
        except Exception as e:
            logger.error("ChromaDB server is not running on localhost:8000")
            logger.error("Please start ChromaDB server first with: chroma run --host localhost --port 8000")
            sys.exit(1)
        
        # Run the populator
        populator = ChromaVectorPopulator()
        populator.run()
        
    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()