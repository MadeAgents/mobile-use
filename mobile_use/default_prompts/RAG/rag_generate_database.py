from RAGToolbox import Jinaembedding, Vectordatabase

import json
docs = json.load(open("/root/hammer/mobile-use/mobile_use/default_prompts/RAG/knowledge.json", encoding='utf-8'))
embedding_model=Jinaembedding("/root/hammer/mobile-use/mobile_use/default_prompts/RAG/jina-embeddings-v2-base-zh") 
database=Vectordatabase(docs)
Vectors=database.get_vector(embedding_model)
database.persist(path='rag_database')
