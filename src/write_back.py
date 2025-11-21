"""
Write predicted triples back to Stardog.
This module takes predicted edges and inserts them as RDF triples into Stardog.
"""

import stardog
from predict import predict_missing_links
from build_pyg_graph import load_from_csv
from model import LinkPredictionModel
import torch


def infer_relation_type(src_uri: str, dst_uri: str) -> str:
    """
    Infer the relationship type based on source and target URIs.
    Based on the Graph_Neural_Network ontology structure.
    
    All predicted relationships are prefixed with 'predicted' to distinguish
    them from existing relationships in the graph.
    
    Returns:
        Relationship name with 'predicted' prefix (e.g., 'predictedStoresFor', 
        'predictedTransportsTo', 'predictedLink', etc.)
    """
    src_type = src_uri.split(':')[-1].split(':')[0] if ':' in src_uri else ''
    dst_type = dst_uri.split(':')[-1].split(':')[0] if ':' in dst_uri else ''
    
    # Map based on source and target types
    # All predicted relationships get 'predicted' prefix
    if 'OilField' in src_uri and 'Pipeline' in dst_uri:
        return 'predictedIsSuppliedBy'
    elif 'Pipeline' in src_uri and 'Refinery' in dst_uri:
        return 'predictedTransportsTo'
    elif 'Refinery' in src_uri and 'StorageTerminal' in dst_uri:
        return 'predictedStoresAt'
    elif 'StorageTerminal' in src_uri and 'CustomerRegion' in dst_uri:
        return 'predictedStoresFor'
    else:
        # For unknown relationship types, use generic predicted link
        return 'predictedLink'


def write_predictions_to_stardog(
    conn: stardog.Connection,
    predictions: list,
    min_probability: float = 0.5
):
    """
    Insert predicted edges as RDF triples into Stardog.
    
    Args:
        conn: Stardog connection object
        predictions: List of tuples (source_uri, target_uri, probability)
        min_probability: Minimum probability threshold for inserting triples
    """
    # Filter predictions by probability threshold
    filtered = [(src, dst, prob) for src, dst, prob in predictions if prob >= min_probability]
    
    if not filtered:
        print(f"No predictions above threshold {min_probability}")
        return
    
    print(f"Inserting {len(filtered)} predicted triples into Stardog...")
    
    # Build SPARQL INSERT query
    # Infer relationship type based on source and target node types
    triples = []
    for src_uri, dst_uri, prob in filtered:
        relation = infer_relation_type(src_uri, dst_uri)
        triple = f"<{src_uri}> <urn:Graph_Neural_Network:{relation}> <{dst_uri}> ."
        triples.append(triple)
    
    # Combine into INSERT query
    insert_query = f"""
    PREFIX gnn: <urn:Graph_Neural_Network:>
    
    INSERT DATA {{
        {' '.join(triples)}
    }}
    """
    
    try:
        conn.begin()
        conn.update(insert_query)
        conn.commit()
        print(f"Successfully inserted {len(filtered)} triples!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting triples: {e}")
        raise


def write_with_relation_type(
    conn: stardog.Connection,
    predictions: list,
    relation_mapping: dict,
    min_probability: float = 0.5
):
    """
    Insert predicted edges with specific relation types using a custom mapping.
    
    Args:
        conn: Stardog connection object
        predictions: List of tuples (source_uri, target_uri, probability)
        relation_mapping: Dict mapping (src_type, dst_type) -> relation_name
                         e.g., {('OilField', 'Pipeline'): 'isSuppliedBy'}
        min_probability: Minimum probability threshold
    """
    filtered = [(src, dst, prob) for src, dst, prob in predictions if prob >= min_probability]
    
    if not filtered:
        print(f"No predictions above threshold {min_probability}")
        return
    
    print(f"Inserting {len(filtered)} predicted triples with relation types...")
    
    # Build triples with relation types from mapping
    triples = []
    for src_uri, dst_uri, prob in filtered:
        # Try to infer types from URIs
        src_type = None
        dst_type = None
        for key_type in ['OilField', 'Pipeline', 'Refinery', 'StorageTerminal', 'CustomerRegion']:
            if key_type in src_uri:
                src_type = key_type
            if key_type in dst_uri:
                dst_type = key_type
        
        # Look up relation in mapping, or use default inference
        if src_type and dst_type and (src_type, dst_type) in relation_mapping:
            relation = relation_mapping[(src_type, dst_type)]
        else:
            relation = infer_relation_type(src_uri, dst_uri)
        
        triple = f"<{src_uri}> <urn:Graph_Neural_Network:{relation}> <{dst_uri}> ."
        triples.append(triple)
    
    insert_query = f"""
    PREFIX gnn: <urn:Graph_Neural_Network:>
    
    INSERT DATA {{
        {' '.join(triples)}
    }}
    """
    
    try:
        conn.begin()
        conn.update(insert_query)
        conn.commit()
        print(f"Successfully inserted {len(filtered)} triples!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting triples: {e}")
        raise


if __name__ == '__main__':
    # Stardog Cloud connection details
    ENDPOINT = 'https://presales.stardog.cloud:5820'
    USERNAME = 'aramcodemo2'
    PASSWORD = 'stardog'
    DATABASE = 'JF-gnn'
    
    # Import connection function
    from extract_from_stardog import get_stardog_connection
    
    # Load model and data from Stardog-extracted CSV files
    print("Loading model and data from Stardog...")
    data = load_from_csv('../data/nodes_from_stardog.csv', '../data/edges_from_stardog.csv')
    
    input_dim = data.x.shape[1]
    model = LinkPredictionModel(
        input_dim=input_dim,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2
    )
    model.load_state_dict(torch.load('../data/model.pt', map_location='cpu'))
    
    # Get predictions
    print("Generating predictions...")
    predictions = predict_missing_links(model, data, top_k=20)
    
    # Connect to Stardog and write back
    print("\nConnecting to Stardog...")
    conn = get_stardog_connection(ENDPOINT, USERNAME, PASSWORD, DATABASE)
    
    print("Writing predictions to Stardog...")
    write_predictions_to_stardog(
        conn,
        predictions,
        min_probability=0.6  # Only insert high-confidence predictions
    )
    
    conn.close()
    print("Done!")

