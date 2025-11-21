## Pre-demo setup

### 1. Prerequisites
- Python 3.9+ installed
- Stardog Cloud access (endpoint, username, password, database)
- Internet connection

### 2. Initial setup (one-time)
```bash
cd /Users/joefagan/Documents/Cursor/GNN

# Activate virtual environment
source venv/bin/activate

# Verify dependencies are installed
pip list | grep -E "(torch|pystardog|pandas)"
```

---

## Demo steps (in order)

### Step 1: Extract data from Stardog
```bash
cd src
python extract_from_stardog.py
```

What to show:
- Connection to Stardog Cloud
- SPARQL queries executing
- Output: "Found 15 nodes" and "Found 16 edges"
- Files created: `data/nodes_from_stardog.csv` and `data/edges_from_stardog.csv`

What to say:
- "We're querying Stardog using SPARQL to extract our RDF graph data"
- "Notice we're using the `FROM <urn:data>` clause to query the named graph"
- "The data comes out as structured CSV, ready for ML processing"

---

### Step 2: Build PyTorch Geometric graph
```bash
python build_pyg_graph.py
```

What to show:
- Graph statistics: 15 nodes, 16 edges, 7 features per node
- Sample node features (tensor output)
- Edge index structure

What to say:
- "We convert the RDF data into a PyTorch Geometric graph format"
- "Each node gets encoded features: type (one-hot) + normalized numeric properties"
- "We're preparing the graph structure - nodes, edges, and features"
- "Note: We're NOT creating embeddings yet - that happens during training"
- "This is the input format needed for graph neural networks"

---

### Step 3: Train the model
```bash
python train.py
```

What to show:
- Model creation: "Model created with 17,537 parameters"
- Training progress (epochs 10, 20, 30...)
- Final validation accuracy (~50%)
- Model saved: "Model saved to data/model.pt"

What to say:
- "We're training a GraphSAGE model for link prediction"
- "This is where embeddings are created - the model learns node embeddings that capture graph structure"
- "The GraphSAGE encoder processes the graph and creates dense vector representations for each node"
- "Training takes about 1-2 minutes on this small graph"
- "The model learns which types of connections are likely based on the graph structure"

---

### Step 4: Generate predictions
```bash
python predict.py
```

What to show:
- "Evaluating 194 possible edge pairs..."
- Top 10 predictions with probabilities (92.6% - 99.8%)
- Sample predictions showing source → target with confidence scores

What to say:
- "The model evaluates all possible missing links"
- "These are the top predictions ranked by probability"
- "High confidence scores (90%+) indicate strong predictions"
- "Notice it's predicting relationships like Refinery → CustomerRegion"

---

### Step 5: Write predictions back to Stardog
```bash
python write_back.py
```

What to show:
- "Connecting to Stardog..."
- "Generating predictions..."
- "Inserting 15 predicted triples into Stardog..."
- "Successfully inserted 15 triples!"

What to say:
- "Now we write the predictions back to Stardog as RDF triples"
- "The predictions become part of your knowledge graph"
- "You can query them with SPARQL, visualize them, or use them for further analysis"
- "This completes the round-trip: Stardog → ML → Stardog"

---

## Optional: Verify in Stardog

After Step 5, you can show in Stardog Studio:
1. Query for the new triples:
   ```sparql
   SELECT ?source ?target ?relation
   WHERE {
       ?source ?relation ?target .
       FILTER (?relation IN (gnn:predictedLink, gnn:storesFor))
   }
   ```
2. Show the graph visualization with the new predicted edges

---

## Demo script (5-minute version)

1. "Let me show you how Stardog integrates with modern ML frameworks"
2. Run Step 1: "First, we extract data from Stardog using SPARQL"
3. Run Step 2: "We convert it to a format for graph neural networks"
4. Run Step 3: "We train a model to learn graph patterns" (can skip watching all epochs)
5. Run Step 4: "The model predicts missing relationships"
6. Run Step 5: "We write predictions back to Stardog as triples"
7. "This shows how Stardog serves as the semantic layer while ML handles predictions"

---

## Tips

- If training is slow, mention it's a demo; production uses GPU acceleration
- Emphasize the round-trip: data comes from Stardog and goes back to Stardog
- Highlight that no custom ETL is needed—SPARQL handles extraction
- Mention scalability: this pattern works for enterprise-scale graphs
- Show the documentation files if they want technical details

---

## Troubleshooting

- If connection fails: verify Stardog credentials in `extract_from_stardog.py`
- If model not found: run `train.py` first
- If predictions seem odd: that's normal for a small demo graph; explain it's a proof of concept

This should give you a clear, customer-ready demo flow.