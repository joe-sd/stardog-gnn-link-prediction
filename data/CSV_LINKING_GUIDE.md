# CSV Linking Guide

This document explains how the CSV files link together. When you create your ontology in Stardog and map these CSVs, you'll create edges (relationships) via SPARQL queries based on these connections.

## Entity Files

Each entity file represents a class in your ontology:

1. **OilFields.csv** → `OilField` class
   - Primary key: `id` (e.g., "OF001")
   - Properties: `name`, `capacity`, `throughput`
   - Foreign key: `pipeline_id` → references `Pipelines.id` (creates `producesInto` relationship)

2. **Pipelines.csv** → `Pipeline` class
   - Primary key: `id` (e.g., "PL001")
   - Properties: `name`, `capacity`, `throughput`
   - Foreign key: `refinery_id` → references `Refineries.id` (creates `transportedTo` relationships)
   - Note: If a pipeline connects to multiple refineries, it will have multiple rows (one per relationship)

3. **Refineries.csv** → `Refinery` class
   - Primary key: `id` (e.g., "RF001")
   - Properties: `name`, `capacity`, `throughput`
   - Foreign key: `storage_id` → references `StorageTerminals.id` (creates `shipsTo` relationships)
   - Note: If a refinery ships to multiple storage terminals, it will have multiple rows (one per relationship)

4. **StorageTerminals.csv** → `StorageTerminal` class
   - Primary key: `id` (e.g., "ST001")
   - Properties: `name`, `capacity`, `throughput`
   - Foreign key: `customer_id` → references `CustomerRegions.id` (creates `deliversTo` relationships)
   - Note: If a storage terminal delivers to multiple customer regions, it will have multiple rows (one per relationship)

5. **CustomerRegions.csv** → `CustomerRegion` class
   - Primary key: `id` (e.g., "CR001")
   - Properties: `name`, `region_code`
   - No foreign keys (end of the supply chain)

## How to Create Edges in Stardog

The foreign key columns in each CSV file will automatically create the relationships when you map them to your ontology in Stardog. The mapping tool should recognize:

- `OilFields.pipeline_id` → creates `producesInto` triples
- `Pipelines.refinery_id` → creates `transportedTo` triples (multiple rows for multiple relationships)
- `Refineries.storage_id` → creates `shipsTo` triples (multiple rows for multiple relationships)
- `StorageTerminals.customer_id` → creates `deliversTo` triples (multiple rows for multiple relationships)

If you need to create edges manually via SPARQL, you can use queries like:

```sparql
PREFIX ex: <http://example.org/ontology#>

INSERT DATA {
    ex:OF001 ex:producesInto ex:PL001 .
    ex:OF002 ex:producesInto ex:PL001 .
    ex:OF003 ex:producesInto ex:PL002 .
    ex:PL001 ex:transportedTo ex:RF001 .
    ex:PL002 ex:transportedTo ex:RF001 .
    ex:PL002 ex:transportedTo ex:RF002 .
    # ... etc
}
```

## Expected Graph Structure

The supply chain flows in this direction:
```
OilField → Pipeline → Refinery → StorageTerminal → CustomerRegion
```

Some entities have multiple connections (e.g., a Pipeline can connect to multiple Refineries), which creates the graph structure needed for link prediction.

