# Stardog + PyTorch Geometric Link Prediction Demo

This is a minimal demo project showing how to:
1. Extract RDF graph data from Stardog using SPARQL queries
2. Convert it to PyTorch Geometric format
3. Train a Graph Neural Network (GraphSAGE) for link prediction
4. Write predicted triples back to Stardog

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
└── README.md
```

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up Stardog:**
   - Install and start Stardog
   - Create a database (e.g., `supply_chain`)
   - Create an ontology for the oil supply chain domain with classes:
     - `OilField`, `Pipeline`, `Refinery`, `StorageTerminal`, `CustomerRegion`
   - Map the entity CSV files (`OilFields.csv`, `Pipelines.csv`, etc.) to your ontology
   - Create object properties: `producesInto`, `transportedTo`, `shipsTo`, `deliversTo`
   - Create edges (relationships) in Stardog using SPARQL INSERT queries (see `CSV_LINKING_GUIDE.md`)

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

This inserts the predicted triples into Stardog (only predictions above the probability threshold).

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

Edges are created in Stardog via SPARQL queries, not CSV files. See `data/CSV_LINKING_GUIDE.md` for details on how entities link together:
- `OilField` → `Pipeline` (via `producesInto`)
- `Pipeline` → `Refinery` (via `transportedTo`)
- `Refinery` → `StorageTerminal` (via `shipsTo`)
- `StorageTerminal` → `CustomerRegion` (via `deliversTo`)

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
- The SPARQL queries are placeholders - adjust them to match your ontology
- The `write_back.py` script uses a generic `predictedLink` relation - you may want to predict relation types as well
- For production use, you'd want to add more sophisticated evaluation metrics and model selection

## Next Steps

After this starter project works, you can extend it with:
- More realistic data generation
- Temporal extensions (time-aware predictions)
- Relation type prediction (not just edge existence)
- Integration diagrams and documentation

