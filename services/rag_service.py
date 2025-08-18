# -*- coding: utf-8 -*-
"""
Unified RAG (Retrieval-Augmented Generation) Service Module

This module combines all RAG-related functionality including:
- OpenAI embeddings and similarity calculations
- ChromaDB vector search operations  
- Text processing and similarity utilities
"""

import os
import re
import json
import string
import ast
from typing import List, Union, Dict, Any, Optional, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd
import inflect
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dotenv import load_dotenv

from core.geo_functions import *
from services.chat_py import *

# Load environment variables
load_dotenv()
pd.set_option('display.max_rows', None)

# Initialize OpenAI client
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Initialize ChromaDB client and collections
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
name_collection = chroma_client.get_or_create_collection("buildings_name_vec")
fclass_collection = chroma_client.get_or_create_collection("fclass_vector")

# OpenAI embedding function for ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ['OPENAI_API_KEY'],
    model_name="text-embedding-3-small"
)

# Initialize inflect engine for plural/singular conversion
p = inflect.engine()


# ============================================================================
# OpenAI Embedding Functions (from rag_model_openai.py)
# ============================================================================

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Get OpenAI embeddings for the given text.
    
    Args:
        text: Input text to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of float embedding values
    """
    text = str(text).replace("\n", " ")
    try:
        embed = client.embeddings.create(input=[text], model=model).data[0].embedding
        return embed
    except Exception as e:
        raise Exception(f"Embedding error for text '{text}': {e}")


def cosine_similarity_openai(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_vector_store(words_origin: List[str], label: str) -> None:
    """
    Build vector store for a list of words and save to JSONL file.
    
    Args:
        words_origin: List of words to vectorize
        label: Label for the output file
    """
    words = [element for element in words_origin if element]
    with open(f'{label}_vectors.jsonl', 'a') as jsonl_file:
        for word in tqdm(words):
            vector = get_embedding(word)
            data = {word: list(vector)}
            jsonl_file.write(json.dumps(data) + '\n')


def calculate_similarity_openai(label: str, key_vector_template: str) -> List[str]:
    """
    Calculate similarity using OpenAI embeddings from CSV file.
    
    Args:
        label: Label to identify the CSV file
        key_vector_template: Query text to compare against
        
    Returns:
        List of similar labels sorted by similarity
    """
    key_vector = get_embedding(key_vector_template)
    df = pd.read_csv(f'{label}_vectors.csv')
    df['vector'] = df['vector'].apply(ast.literal_eval)
    
    df['cosine_similarity'] = df['vector'].apply(
        lambda v: cosine_similarity_openai(np.array(v), key_vector)
    )
    
    filtered_df = df[df['cosine_similarity'] > 0.5]
    sorted_df = filtered_df.sort_values(by='cosine_similarity', ascending=False)
    labels_list = sorted_df['label'].tolist()
    return labels_list


def format_list_string(input_str: str) -> str:
    """Format list string for JSON parsing."""
    match = re.search(r'\{\s*"[^"]+"\s*:\s*\[(.*?)\]\s*\}', input_str)
    if not match:
        return "Invalid input format"
    
    list_content = match.group(1)
    elements = [e.strip() for e in list_content.split(',')]
    
    formatted_elements = []
    for elem in elements:
        if not re.match(r'^([\'"])(.*)\1$', elem):
            elem = f'"{elem}"'
        formatted_elements.append(elem)
    
    return f'{{ "similar_words":[{", ".join(formatted_elements)}]}}'


def process_texts_openai(selected_words: List[str], key_word: str) -> List[str]:
    """
    Process and filter text using OpenAI based on similarity to keyword.
    
    Args:
        selected_words: List of words to filter
        key_word: Target keyword for similarity matching
        
    Returns:
        List of filtered words most similar to the keyword
    """
    if key_word in selected_words:
        return [key_word]
    
    for word in selected_words:
        if ':' in word:
            _, right_part = word.split(':', 1)
            if key_word == right_part.strip().lower():
                return [word]
    
    # Check if all elements contain ":"
    if all(":" in word for word in selected_words):
        index_in_list = True
        index_dict = {word.split(":")[0]: word for word in selected_words}
        
        ask_prompt = """
        Please find most similar words from the words list according to the key word.
        
        Return a list of corresponding index for most similar elements. 
        List content are different soil ingredients, you need to pick the corresponding soil for user's query.
        Before return me a json, please first short think about it.
        For example, for soil not good for construction, you need to think about Soil stability, bearing capacity and drainage.
        
        Return in json like:
        ```json
        {"similar_words": []}
        ```
        """
    else:
        index_in_list = False
        index_dict = {index: word for index, word in enumerate(selected_words)}
        
        ask_prompt = """
        Please find most similar words from the words list according to the key word.
        Return list of corresponding numerical index for most similar elements, index start from 0.
        Only filter out words which really not match. If the word do have some common with key word, then keep it.
        Sometimes I will give you germany words.
        If you see a word in words list same meaning with input word, then just return that one without others.
        
        If user ask about highway, please only return labels which is road. Please be strict!
        Please focus more on the major part of query.
        Before return me a json, please first short think about it.
        
        Return in json like:
        ```json
        {"similar_words": []}
        ```
        """
    
    messages = []
    user_prompt = f"selected_words={selected_words}\nkey_word={key_word}"
    
    messages.append(message_template('system', ask_prompt))
    messages.append(message_template('user', str(user_prompt)))
    result = chat_single(messages, temperature=1, verbose=True, mode='json_few_shot')
    
    print(result)
    
    agent_filter_result = result['similar_words']
    
    final_result = []
    for i in agent_filter_result:
        if i in index_dict:
            final_result.append(index_dict[i])
    
    agent_filter_result = final_result
    reduced_results = list(set(selected_words) - set(agent_filter_result))
    print("agent_filter_result", agent_filter_result)
    
    return agent_filter_result


# ============================================================================
# ChromaDB Vector Search Functions (from rag_chroma.py)
# ============================================================================

def get_top_100_similar(documents: List[str], distances: List[float]) -> List[str]:
    """
    Get the top 100 most similar documents based on distances.
    
    Args:
        documents: List of document data
        distances: List of distances corresponding to each document
        
    Returns:
        The top 100 most similar documents
    """
    sorted_docs = sorted(zip(distances, documents), key=lambda x: x[0])
    top_100_docs = [doc for _, doc in sorted_docs[:100]]
    return top_100_docs


def find_matching_words(word_list: List[Union[str, float]], word: str) -> List[str]:
    """
    Find words in list that match the input word (case-insensitive).
    
    Args:
        word_list: List of words to search
        word: Target word to match
        
    Returns:
        List containing the first matching word, or empty list if no match
    """
    lower_word = word.lower()
    for w in word_list:
        if isinstance(w, str) and w.lower() == lower_word:
            return [w]
    return []


def calculate_similarity_chroma(
    query: str,
    name_dict_4_similarity: Optional[Dict] = None,
    added_name_list: List[str] = [],
    results_num: int = 60,
    openai_filter: bool = False,
    mode: str = "name",
    give_list: List[str] = []
) -> Tuple[List[str], bool]:
    """
    Calculate similarity using ChromaDB vector search.
    
    Args:
        query: Search query text
        name_dict_4_similarity: Dictionary for name similarity (unused)
        added_name_list: Additional names to add to results
        results_num: Number of results to retrieve
        openai_filter: Whether to use OpenAI filtering
        mode: "name" or "fclass" to determine which collection to use
        give_list: If provided, filter results to only include these items
        
    Returns:
        Tuple of (similar items list, is_strong_match boolean)
    """
    openai_filter = False  # Override to always False
    
    collection = name_collection if mode == "name" else fclass_collection
    
    results = collection.query(
        query_embeddings=list(get_embedding(query)),
        n_results=results_num
    )
    
    distances_duplicate = results['distances'][0]
    documents_duplicate = results['documents'][0]
    
    # Remove duplicates
    unique_doc_dist = {}
    for doc, dist in zip(documents_duplicate, distances_duplicate):
        if doc not in unique_doc_dist:
            unique_doc_dist[doc] = dist
    
    documents = list(unique_doc_dist.keys())
    distances = list(unique_doc_dist.values())
    
    # Filter by give_list if provided
    if give_list:
        filtered_results = [
            (doc, dist) for doc, dist in zip(documents, distances) 
            if doc in give_list
        ]
        documents = [doc for doc, _ in filtered_results]
        distances = [dist for _, dist in filtered_results]
    
    top_texts = get_top_100_similar(documents, distances) + added_name_list
    total_match = find_matching_words(top_texts, query)
    
    if total_match:
        return total_match, True
        
    if openai_filter:
        result = process_texts_openai(top_texts, query)
        results = list(set(top_texts) - set(result))
        return result, False
    else:
        filtered_results = [
            documents[i] for i in range(len(distances)) 
            if distances[i] < 0.6
        ]
        return filtered_results, False


# ============================================================================
# Text Processing Utilities (from rag_model.py)
# ============================================================================

def convert_plural_singular(sentence: str) -> str:
    """
    Convert words in sentence between plural and singular forms.
    
    Args:
        sentence: Input sentence
        
    Returns:
        Sentence with converted words
    """
    try:
        words = sentence.split()
        
        def convert_word(word):
            stripped_word = re.sub(r'[^a-zA-Z]', '', word)
            if p.singular_noun(stripped_word):
                converted = p.singular_noun(stripped_word)
            else:
                converted = p.plural_noun(stripped_word)
            return word.replace(stripped_word, converted) if converted else word
        
        converted_words = [convert_word(word) for word in words]
        return ' '.join(converted_words)
    except:
        return sentence


def find_word_in_sentence(
    words_set: set,
    sentence: str,
    iteration: bool = True,
    judge_strong: bool = False
) -> Optional[Union[str, Tuple[str, bool]]]:
    """
    Find word from set that appears in sentence.
    
    Args:
        words_set: Set of words to search for
        sentence: Sentence to search in
        iteration: Whether to try plural/singular conversion
        judge_strong: Whether to return strong match indicator
        
    Returns:
        Matching word, optionally with strong match indicator
    """
    filtered_words = {word for word in words_set if word and word != ''}
    sorted_words = sorted(filtered_words, key=len, reverse=True)
    lower_sentence = sentence.lower()
    
    for word in sorted_words:
        lower_word = word.lower()
        if lower_word == sentence.strip().lower():
            if judge_strong:
                return word, True
            else:
                return word
                
        index = lower_sentence.find(lower_word)
        
        while index != -1:
            before = lower_sentence[index - 1] if index > 0 else ' '
            after = lower_sentence[index + len(lower_word)] if index + len(lower_word) < len(lower_sentence) else ' '
            
            if before in string.whitespace + string.punctuation and after in string.whitespace + string.punctuation:
                return word
            
            index = lower_sentence.find(lower_word, index + 1)
    
    if iteration:
        converted_sentence = convert_plural_singular(sentence)
        if converted_sentence != sentence:
            return find_word_in_sentence(words_set, converted_sentence, False)
    
    return None


def calculate_similarity(
    words: List[str],
    key_word: str,
    mode: Optional[str] = None,
    openai_filter: Optional[bool] = None
) -> Union[List[str], Tuple[List[str], bool]]:
    """
    Calculate similarity between keywords and word list.
    
    Note: This function has dependencies on a 'model' object that needs
    to be initialized with sentence transformers if using local embeddings.
    Currently falls back to OpenAI filtering for large word lists.
    
    Args:
        words: List of words to compare
        key_word: Target keyword
        mode: Optional mode ('judge_strong', 'print', etc.)
        openai_filter: Whether to use OpenAI filtering
        
    Returns:
        List of similar words, optionally with strong match indicator
    """
    strong_indicate = False
    words = list(words)
    
    # Direct match check
    if key_word in words:
        strong_indicate = True
        return ([key_word], strong_indicate) if mode == 'judge_strong' else [key_word]
    
    # Check for colon-separated matches
    for word in words:
        if ':' in word:
            _, right_part = word.split(':', 1)
            if key_word == right_part.strip().lower():
                strong_indicate = True
                return ([word], strong_indicate) if mode == 'judge_strong' else [word]
    
    # Check if word appears in sentence
    found_word = find_word_in_sentence(set(words), key_word)
    if found_word:
        return ([found_word], strong_indicate) if mode == 'judge_strong' else [found_word]
    
    # Use OpenAI filter for small word lists or when requested
    if len(words) <= 40 and openai_filter:
        filter_result = process_texts_openai(words, key_word)
        if mode == 'judge_strong':
            return filter_result, strong_indicate
        return filter_result
    
    # For larger lists or when local embeddings would be used,
    # fall back to OpenAI filtering as sentence transformers are commented out
    if openai_filter:
        # Select top 40 words for OpenAI processing
        # Since we don't have local embeddings, just take first 40
        top_words = words[:40]
        filter_result = process_texts_openai(top_words, key_word)
        
        if mode == 'judge_strong':
            return filter_result, strong_indicate
        return filter_result
    
    # Default return empty list if no processing method available
    return ([], strong_indicate) if mode == 'judge_strong' else []


# ============================================================================
# Exported functions for backward compatibility
# ============================================================================

__all__ = [
    # OpenAI embedding functions
    'get_embedding',
    'cosine_similarity_openai',
    'build_vector_store',
    'calculate_similarity_openai',
    'process_texts_openai',
    'format_list_string',
    
    # ChromaDB functions
    'calculate_similarity_chroma',
    'get_top_100_similar',
    'find_matching_words',
    
    # Text processing utilities
    'convert_plural_singular',
    'find_word_in_sentence',
    'calculate_similarity',
]