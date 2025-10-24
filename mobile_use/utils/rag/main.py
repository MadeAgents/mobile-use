import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

from mobile_use.utils.utils import download_hf_model

# -------------------------------
# Helper functions
# -------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Main workflow
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a RAG vector database (optionally auto-download Jina embedding model)."
    )
    parser.add_argument(
        "--knowledge-path",
        type=str,
        required=True,
        help="Path to knowledge JSON file (e.g., knowledge.json)"
    )
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default='./jina-embeddings-v2-base-zh',
        help="Local embedding model directory (will be downloaded if not found and --download-model is set)"
    )
    parser.add_argument(
        "--database-dir",
        type=str,
        default="./rag_database",
        help="Directory to store the vector database"
    )
    parser.add_argument(
        "--ragtoolbox-path",
        type=str,
        default=None,
        help="Optional path to RAGToolbox (if not specified, will use default relative path)"
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Automatically download model if missing"
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default="https://huggingface.co/jinaai/jina-embeddings-v2-base-zh",
        help="Hugging Face model repository URL"
    )

    args = parser.parse_args()

    knowledge_path = Path(args.knowledge_path).expanduser().resolve()
    embedding_model_path = Path(args.embedding_model_path).expanduser().resolve()
    database_dir = Path(args.database_dir).expanduser().resolve()

    # Handle RAGToolbox import
    if args.ragtoolbox_path:
        ragtoolbox_path = Path(args.ragtoolbox_path).expanduser().resolve()
        sys.path.append(str(ragtoolbox_path))
    else:
        this_file = Path(__file__).resolve()
        project_home = this_file.parent.parent.parent.parent
        default_toolbox = project_home / "third_party" / "RAGToolbox"
        sys.path.append(str(default_toolbox))

    try:
        from RAGToolbox import Jinaembedding, Vectordatabase
    except Exception as exc:
        print("[ERROR] Failed to import RAGToolbox. Please check the path or dependencies.")
        raise

    if not knowledge_path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {knowledge_path}")

    if not embedding_model_path.exists():
        if args.download_model:
            print(f"[INFO] Model directory not found. Downloading to: {embedding_model_path}")
            download_hf_model(args.model_url, str(embedding_model_path))
        else:
            raise FileNotFoundError(
                f"Model directory not found: {embedding_model_path}. "
                "Use --download-model to download automatically or prepare manually."
            )

    print(f"[INFO] Loading knowledge file: {knowledge_path}")
    with open(knowledge_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"[INFO] Initializing embedding model from: {embedding_model_path}")
    embedding_model = Jinaembedding(str(embedding_model_path))

    print("[INFO] Building vector database...")
    database = Vectordatabase(docs)
    _ = database.get_vector(embedding_model)

    ensure_dir(database_dir)
    print(f"[INFO] Saving vector database to: {database_dir}")
    database.persist(path=str(database_dir))
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
