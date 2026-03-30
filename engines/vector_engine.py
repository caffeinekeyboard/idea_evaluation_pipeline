import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

class VectorEngine:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cuda"):
        print(f"Loading embedding model '{model_name}' onto {device}...")
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        print("Model loaded successfully.")

    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        
        return embeddings

    def compute_centroid(self, embeddings: torch.Tensor) -> torch.Tensor:
        centroid = torch.mean(embeddings, dim=0, keepdim=True)
        centroid = centroid / torch.linalg.norm(centroid, dim=1, keepdim=True)
        
        return centroid

    def calculate_novelty(self, idea_text: str, baseline_embeddings: torch.Tensor) -> float:
        idea_emb = self.embed(idea_text)
        centroid = self.compute_centroid(baseline_embeddings)
        similarity = torch.mm(idea_emb, centroid.T).item()
        novelty_score = 1.0 - similarity
        
        return max(0.0, float(novelty_score))

    def calculate_diversity(self, idea_texts: List[str]) -> float:
        if len(idea_texts) < 2:
            return 0.0

        embeddings = self.embed(idea_texts)
        n = embeddings.shape[0]
        similarity_matrix = torch.mm(embeddings, embeddings.T)
        distance_matrix = 1.0 - similarity_matrix
        upper_tri_indices = torch.triu_indices(n, n, offset=1)
        pairwise_distances = distance_matrix[upper_tri_indices[0], upper_tri_indices[1]]
        mean_distance = torch.mean(pairwise_distances).item()
        
        return max(0.0, float(mean_distance))

if __name__ == "__main__":
    engine = VectorEngine()
    baselines = [
        "Use a deep convolutional neural network for image classification.",
        "Train a standard ResNet50 model on the dataset.",
        "Implement a standard CNN pipeline using PyTorch."
    ]
    print("\nComputing baseline embeddings...")
    baseline_tensors = engine.embed(baselines)
    derivative_idea = "Build a convolutional network with PyTorch."
    novel_idea = "Employ an unsupervised Vision Transformer using contrastive learning."
    score1 = engine.calculate_novelty(derivative_idea, baseline_tensors)
    score2 = engine.calculate_novelty(novel_idea, baseline_tensors)
    print(f"\nNovelty Score (Derivative): {score1:.4f} (Expect close to 0)")
    print(f"Novelty Score (Novel):      {score2:.4f} (Expect higher value)")
    batch_low_diversity = baselines
    batch_high_diversity = [
        "Use a deep convolutional neural network for image classification.",
        "Apply Q-learning for robotic pathfinding.",
        "Synthesize proteins using a diffusion model."
    ]
    div_low = engine.calculate_diversity(batch_low_diversity)
    div_high = engine.calculate_diversity(batch_high_diversity)
    print(f"\nDiversity Score (Low):  {div_low:.4f}")
    print(f"Diversity Score (High): {div_high:.4f}")