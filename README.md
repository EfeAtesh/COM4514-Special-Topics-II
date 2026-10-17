# Neo4j AI Graph Analysis

Neo4j-based movie graph analysis project developed for the COM4514 Special Topics II term project.

## Features

- Actor co-occurrence network analysis with NetworkX.
- Movie similarity analysis using genre and actor features.
- t-SNE visualization and K-means clustering.
- Actor-director link prediction using Jaccard similarity.
- Graph visualizations generated as PNG files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set Neo4j connection variables before running:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
python main.py
```

On Windows PowerShell:

```powershell
$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_PASSWORD = "your_password"
python main.py
```

## Outputs

- `actor_network.png`
- `movie_similarity_clusters.png`
- `potential_collaborations.png`
