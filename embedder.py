from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from embedding_cache import EmbeddingCache


class Embedder():
    def __init__(self, model_name, use_cache=True, cache_dir="data/cache"):
        self.use_cache = use_cache
        self.cache = EmbeddingCache(cache_dir) if use_cache else None
        self.set_model(model_name)

    def generate_embeddings(self, split_text):
        """
        Generate embeddings for a list of texts with per-text caching.

        Args:
            split_text: List of strings to embed

        Returns:
            numpy.ndarray: Embeddings of shape (len(split_text), embedding_dim)
        """
        if not split_text:
            return np.array([])

        embeddings_list = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each text individually
        if self.use_cache:
            for i, text in enumerate(split_text):
                cached_embedding = self.cache.load_single(text, self.model_name)
                if cached_embedding is not None:
                    embeddings_list.append((i, cached_embedding))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = split_text
            uncached_indices = list(range(len(split_text)))

        # Generate embeddings for uncached texts
        if uncached_texts:
            if self.use_cache:
                print(f"  Generating {len(uncached_texts)}/{len(split_text)} embeddings (rest from cache)...")

            if self.model_type == "sentence-transformer":
                # Use larger batch size and show progress for H100
                new_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=512,  # Increased for H100
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
            else:
                inputs = self.tokenizer(uncached_texts, padding=True, truncation=True, return_tensors="pt")

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state

                # Extract sentence embeddings (using the [CLS] token's embeddings)
                new_embeddings = hidden_states[:, 0, :].numpy()

            # Save new embeddings to cache
            if self.use_cache:
                self.cache.save_batch(uncached_texts, new_embeddings, self.model_name)

            # Add to embeddings list with their indices
            for i, idx in enumerate(uncached_indices):
                embeddings_list.append((idx, new_embeddings[i]))

        # Sort by original index and extract embeddings
        embeddings_list.sort(key=lambda x: x[0])
        embeddings = np.array([emb for _, emb in embeddings_list])

        if self.use_cache and uncached_texts:
            print(f"  ✓ Generated and cached {len(uncached_texts)} new embeddings")

        return embeddings

    def set_model(self, model_name):
        self.model_name = model_name
        print(f"✓ Loading embedding model: {model_name}")
        if model_name in ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-mpnet-base-v2',
                          'multi-qa-mpnet-base-dot-v1', 'paraphrase-mpnet-base-v2',
                          'T-Systems-onsite/cross-en-de-roberta-sentence-transformer',
                          'paraphrase-multilingual-mpnet-base-v2',
                          'paraphrase-multilingual-MiniLM-L12-v2',
                          'BAAI/bge-m3']:
            self.model = SentenceTransformer(model_name)
            self.model_type = "sentence-transformer"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Llama models don't have a pad token by default - set it to eos_token
            # This matches the training configuration
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # For decoder models, pad on the left (matches training)
            self.tokenizer.padding_side = 'left'

            self.model = AutoModel.from_pretrained(model_name)
            self.model_type = "transformer"

        
        