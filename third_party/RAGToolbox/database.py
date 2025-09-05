from tqdm import tqdm
import numpy as np 
import os
import json
from typing import List, Tuple

class Vectordatabase:
    def __init__(self,docs:List=[]) -> None:
        self.document = docs
    def get_vector(self,EmbeddingModel)->List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors
    def persist(self,path:str='database')->None:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        self.vectors = [vector.tolist() for vector in self.vectors]
        with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)
    def load_vector(self,path:str='database')->None:
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)
    def get_similarity(self, vector1: List[float], vector2: List[float],embedding_model) -> float:
        return embedding_model.compare_v(vector1, vector2)
    def query(self, query: str, EmbeddingModel, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        similarities = np.array([self.get_similarity(query_vector, vector, EmbeddingModel)
                            for vector in self.vectors])
        sorted_indices = similarities.argsort()[::-1]
        top_k_indices = sorted_indices[:k]
        
        if isinstance(self.document, dict):
            doc_keys = list(self.document.keys())
            top_k_documents = [doc_keys[i] for i in top_k_indices] 
        else:
            top_k_documents = np.array(self.document)[top_k_indices].tolist()
        return top_k_documents
    def query_score(self, query: str, EmbeddingModel, k: int = 1) -> List[Tuple[float, str, List[str]]]:
        query_vector = EmbeddingModel.get_embedding(query)
        similarities = np.array([self.get_similarity(query_vector, vector, EmbeddingModel)
                                for vector in self.vectors])
        sorted_indices = similarities.argsort()[::-1] 
        top_k_indices = sorted_indices[:k]
        top_k_similarities = similarities[top_k_indices].tolist()
        
        if isinstance(self.document, dict):
            doc_keys = list(self.document.keys())
            results = [
                (top_k_similarities[i], doc_keys[top_k_indices[i]], self.document[doc_keys[top_k_indices[i]]])
                for i in range(len(top_k_indices))
            ]
        else:
            results = [
                (top_k_similarities[i], None, np.array(self.document)[top_k_indices[i]].tolist())
                for i in range(len(top_k_indices))
            ]
        return results