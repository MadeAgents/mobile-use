import json
import os, sys

project_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path += os.path.join(project_home, 'third_party', 'RAGToolbox')

from RAGToolbox import Jinaembedding, Vectordatabase


script_dir = os.path.join(project_home, "mobile_use", "default_prompts", "RAG")
docs = json.load(open(os.path.join(script_dir, "knowledge.json"), encoding='utf-8'))
embedding_model=Jinaembedding(os.path.join(script_dir, "jina-embeddings-v2-base-zh")) 
database=Vectordatabase(docs)
Vectors=database.get_vector(embedding_model)
database.persist(path=os.path.join(script_dir, "rag_database"))
