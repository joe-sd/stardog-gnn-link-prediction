# Stardog + PyTorch Geometric Link Prediction Demo

This demo showcases a complete integration between **Stardog** (semantic graph database) and **PyTorch Geometric** (graph neural networks) for link prediction on RDF supply chain data.

## What This Demo Shows

1. Extract RDF graph data from Stardog using SPARQL queries
2. Convert it to PyTorch Geometric format
3. Train a Graph Neural Network (GraphSAGE) for link prediction
4. Write predicted triples back to Stardog in a named graph

## Use Case

This demo demonstrates how link prediction can reveal **business opportunities** (like market expansion) rather than just finding missing data. See `CUSTOMER_STORY.md` for a detailed use case scenario.

## Quick Start

See `Pre-demo setup.md` for step-by-step instructions on running and presenting this demo.

## Project Structure

```
project/
├── data/
│   ├── OilFields.csv          # Entity CSV files (map these to Stardog)
│   ├── Pipelines.csv
│   ├── Refineries.csv
│   ├── StorageTerminals.csv
│   ├── CustomerRegions.csv
│   ├── CSV_LINKING_GUIDE.md   # Guide for how CSVs link together
│   ├── nodes_from_stardog.csv # Extracted nodes (generated after Stardog setup)
│   ├── edges_from_stardog.csv # Extracted edges (generated after Stardog setup)
│   └── model.pt               # Trained model (generated)
├── src/
│   ├── extract_from_stardog.py  # SPARQL queries to fetch data from Stardog
│   ├── build_pyg_graph.py       # Convert RDF data to PyTorch Geometric format
│   ├── model.py                 # GraphSAGE encoder + link predictor
│   ├── train.py                 # Training loop
│   ├── predict.py               # Run inference to find missing links
│   └── write_back.py            # Insert predicted triples into Stardog
├── requirements.txt
├── README.md
├── INTEGRATION_DOCUMENTATION.md  # Detailed technical documentation
├── PRESENTATION_SLIDES.md         # Presentation slides
├── CUSTOMER_STORY.md              # Business use case story
└── Pre-demo setup.md              # Demo execution guide
```

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Stardog database:**
   - Connect to [cloud.stardog.com](https://cloud.stardog.com)
   - Create a new database (e.g., `JF-gnn`)
   - Using **Stardog Designer**, add the CSV resources (`OilFields.csv`, `Pipelines.csv`, etc.) to your project
   - Using **Create and Map**, have Voicebox create an ontology for the oil supply chain domain with classes:
     - `OilField`, `Pipeline`, `Refinery`, `StorageTerminal`, `CustomerRegion`
   - Voicebox will create object properties based on the foreign key relationships:
     - `isSuppliedBy` (OilField → Pipeline)
     - `transportsTo` (Pipeline → Refinery)
     - `storesAt` (Refinery → StorageTerminal)
     - `storesFor` (StorageTerminal → CustomerRegion)
   - Publish the database to make it available for queries

3. **Update Stardog connection details:**
   - Edit `src/extract_from_stardog.py` and `src/write_back.py`
   - Update `ENDPOINT`, `USERNAME`, `PASSWORD`, and `DATABASE` variables

## Usage

### Step 1: Extract Data from Stardog

Once you have your data in Stardog and have updated the connection details:

```bash
cd src
python extract_from_stardog.py
```

This will query Stardog and save the extracted data to CSV files.

### Step 2: Build PyTorch Geometric Graph

```bash
python build_pyg_graph.py
```

This converts the CSV data into a PyG Data object (or you can use it directly in training).

### Step 3: Train the Model

```bash
python train.py
```

This trains a GraphSAGE model for link prediction. The model will be saved to `data/model.pt`.

### Step 4: Predict Missing Links

```bash
python predict.py
```

This runs inference and shows the top 10 most likely missing edges.

### Step 5: Write Predictions Back to Stardog

```bash
python write_back.py
```

This inserts the predicted triples into Stardog in the named graph `<urn:predictions>` (only predictions above the probability threshold). All predicted relationships are prefixed with "predicted" (e.g., `predictedStoresFor`, `predictedTransportsTo`) to distinguish them from existing relationships.

### Step 6: Query Predictions in Stardog

You can query the predicted relationships in Stardog using:

```sparql
PREFIX gnn: <urn:Graph_Neural_Network:>

SELECT ?source ?target ?relation
FROM <urn:predictions>
WHERE {
    ?source ?relation ?target .
    FILTER (STRSTARTS(STR(?relation), 'urn:Graph_Neural_Network:predicted'))
}
```

This returns all predicted relationships stored in the `<urn:predictions>` graph.

## Data Format

### Entity CSV Files

Each entity type has its own CSV file with simple IDs (not full URIs):

- **OilFields.csv**: `id`, `name`, `capacity`, `throughput`
- **Pipelines.csv**: `id`, `name`, `capacity`, `throughput`
- **Refineries.csv**: `id`, `name`, `capacity`, `throughput`
- **StorageTerminals.csv**: `id`, `name`, `capacity`, `throughput`
- **CustomerRegions.csv**: `id`, `name`, `region_code`

When you map these to Stardog, you'll create IRIs/URIs for each entity. The `id` column becomes the basis for the URI.

### Edges

Edges are created in Stardog via the foreign key columns in the CSV files when you map them using Stardog Designer. See `data/CSV_LINKING_GUIDE.md` for details on how entities link together:
- `OilField` → `Pipeline` (via `isSuppliedBy`)
- `Pipeline` → `Refinery` (via `transportsTo`)
- `Refinery` → `StorageTerminal` (via `storesAt`)
- `StorageTerminal` → `CustomerRegion` (via `storesFor`)

## Customization

### Adjusting SPARQL Queries

The SPARQL queries in `extract_from_stardog.py` assume a specific ontology structure. You'll need to adjust them based on your actual ontology:

- Update the `PREFIX` declarations
- Modify the `SELECT` clauses to match your property names
- Adjust the `WHERE` clauses to match your RDF structure

### Model Architecture

The model in `model.py` uses:
- **GraphSAGE** for node embeddings (2 layers, 64 hidden dimensions)
- **MLP** for link prediction (simple concatenation + MLP)

You can adjust:
- `embedding_dim`: Size of node embeddings
- `hidden_dim`: Size of hidden layers
- `num_layers`: Number of GNN layers

### Training Parameters

In `train.py`, you can adjust:
- `epochs`: Number of training epochs
- `lr`: Learning rate
- Train/validation split ratio

## Notes

- The model is kept minimal to focus on the Stardog integration workflow
- The SPARQL queries in `extract_from_stardog.py` query from the `<urn:data>` named graph - adjust if your data is in a different graph
- The `write_back.py` script inserts predictions into the `<urn:predictions>` named graph
- Predicted relationships are automatically labeled with "predicted" prefix (e.g., `predictedStoresFor`) and the relationship type is inferred from node types
- For production use, you'd want to add more sophisticated evaluation metrics and model selection

## Next Steps

After this starter project works, you can extend it with:
- More realistic data generation
- Temporal extensions (time-aware predictions)
- Relation type prediction (not just edge existence)
- Integration diagrams and documentation

