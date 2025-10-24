# Knowledge Retrieval

This module enables building a Retrieval-Augmented Generation (RAG) vector database using a knowledge base and an embedding model. It supports automatic downloading of embedding models and integrates with the RAGToolbox for vector database creation.

## Requirements

- transformers
- pytorch

## Usage

Run the script to build the vector database:

```bash
python main.py <Arguments>
```

### Arguments

- `--knowledge-path`: Path to the knowledge JSON file (required).
- `--embedding-model-path`: Path to the local embedding model directory (default: `./jina-embeddings-v2-base-zh`).
- `--database-dir`: Directory to store the vector database (default: `./rag_database`).
- `--ragtoolbox-path`: Optional path to the RAGToolbox library (default: relative path in `third_party` folder).
- `--download-model`: Automatically download the embedding model if not found locally.
- `--model-url`: URL of the Hugging Face model repository (default: `https://huggingface.co/jinaai/jina-embeddings-v2-base-zh`).


## Outputs

- **Vector Database**: A directory containing the serialized vector database for knowledge retrieval when the agent performing task execution.
