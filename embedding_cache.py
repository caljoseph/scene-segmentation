import os
import hashlib
import numpy as np
import json
from pathlib import Path


class EmbeddingCache:
    """
    Per-sentence embedding cache with individual file storage.

    Each unique (model_name, text) pair is cached separately, enabling
    maximum reuse across training and pipeline workflows.

    Cache structure:
        cache_dir/
        └── {model_name}/
            ├── {hash1}.npz
            ├── {hash1}_meta.json
            ├── {hash2}.npz
            └── {hash2}_meta.json
    """

    def __init__(self, cache_dir="embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_cache_dir(self, model_name):
        """Get cache directory for a specific model."""
        # Sanitize model name for filesystem
        safe_model_name = model_name.replace('/', '-').replace('\\', '-')
        model_dir = self.cache_dir / safe_model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _get_cache_key(self, text, model_name):
        """
        Generate a unique cache key for a single text and model.

        Uses SHA256 hash of model_name + text content.
        """
        combined = f"{model_name}:::{text}"
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        return hash_obj.hexdigest()

    def _get_cache_paths(self, cache_key, model_name):
        """Get file paths for embedding and metadata."""
        model_dir = self._get_model_cache_dir(model_name)
        embedding_path = model_dir / f"{cache_key}.npz"
        metadata_path = model_dir / f"{cache_key}_meta.json"
        return embedding_path, metadata_path

    def has_cached(self, text, model_name):
        """Check if embedding exists in cache for this text and model."""
        cache_key = self._get_cache_key(text, model_name)
        embedding_path, _ = self._get_cache_paths(cache_key, model_name)
        return embedding_path.exists()

    def load_single(self, text, model_name):
        """
        Load cached embedding for a single text.

        Returns:
            numpy.ndarray: Embedding vector, or None if not cached
        """
        cache_key = self._get_cache_key(text, model_name)
        embedding_path, _ = self._get_cache_paths(cache_key, model_name)

        if not embedding_path.exists():
            return None

        try:
            data = np.load(embedding_path)
            embedding = data['embedding']
            return embedding
        except Exception as e:
            print(f"⚠️  Error loading cached embedding: {e}")
            return None

    def save_single(self, text, embedding, model_name):
        """
        Save embedding for a single text to cache.

        Args:
            text: The text that was embedded
            embedding: numpy.ndarray of shape (embedding_dim,)
            model_name: Name of the embedding model
        """
        cache_key = self._get_cache_key(text, model_name)
        embedding_path, metadata_path = self._get_cache_paths(cache_key, model_name)

        try:
            # Save embedding
            np.savez_compressed(embedding_path, embedding=embedding)

            # Save metadata for reference
            metadata = {
                'model_name': model_name,
                'text_length': len(text),
                'embedding_shape': list(embedding.shape),
                'cache_key': cache_key,
                'text_preview': text[:100] + '...' if len(text) > 100 else text
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️  Error saving embedding to cache: {e}")

    def load_batch(self, texts, model_name):
        """
        Load embeddings for multiple texts.

        Returns:
            embeddings: List of embeddings (or None for uncached texts)
            uncached_indices: List of indices for texts not in cache
        """
        embeddings = []
        uncached_indices = []

        for i, text in enumerate(texts):
            embedding = self.load_single(text, model_name)
            embeddings.append(embedding)
            if embedding is None:
                uncached_indices.append(i)

        return embeddings, uncached_indices

    def save_batch(self, texts, embeddings, model_name):
        """
        Save embeddings for multiple texts.

        Args:
            texts: List of texts
            embeddings: numpy.ndarray of shape (num_texts, embedding_dim)
            model_name: Name of the embedding model
        """
        for text, embedding in zip(texts, embeddings):
            self.save_single(text, embedding, model_name)

    def clear_cache(self, model_name=None):
        """
        Remove cached embeddings.

        Args:
            model_name: If provided, only clear cache for this model.
                       If None, clear entire cache.
        """
        if model_name:
            model_dir = self._get_model_cache_dir(model_name)
            if model_dir.exists():
                for file in model_dir.iterdir():
                    try:
                        file.unlink()
                    except Exception as e:
                        print(f"Error deleting {file}: {e}")
                print(f"✓ Cleared cache for model: {model_name}")
        else:
            if self.cache_dir.exists():
                for model_dir in self.cache_dir.iterdir():
                    if model_dir.is_dir():
                        for file in model_dir.iterdir():
                            try:
                                file.unlink()
                            except Exception as e:
                                print(f"Error deleting {file}: {e}")
                print(f"✓ Cleared entire embedding cache")

    def get_cache_stats(self, model_name=None):
        """
        Get statistics about the cache.

        Args:
            model_name: If provided, get stats for this model only.
                       If None, get stats for all models.

        Returns:
            dict: Statistics including num_cached, total_size_mb, models
        """
        if model_name:
            model_dir = self._get_model_cache_dir(model_name)
            if not model_dir.exists():
                return {"model": model_name, "num_cached": 0, "total_size_mb": 0}

            files = list(model_dir.glob("*.npz"))
            total_size = sum(f.stat().st_size for f in files)

            return {
                "model": model_name,
                "num_cached": len(files),
                "total_size_mb": total_size / (1024 * 1024)
            }
        else:
            # Stats for all models
            stats = {
                "models": {},
                "total_cached": 0,
                "total_size_mb": 0
            }

            if not self.cache_dir.exists():
                return stats

            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    files = list(model_dir.glob("*.npz"))
                    total_size = sum(f.stat().st_size for f in files)

                    stats["models"][model_name] = {
                        "num_cached": len(files),
                        "size_mb": total_size / (1024 * 1024)
                    }
                    stats["total_cached"] += len(files)
                    stats["total_size_mb"] += total_size / (1024 * 1024)

            return stats

    def print_stats(self):
        """Print cache statistics in a human-readable format."""
        stats = self.get_cache_stats()

        print("\n" + "=" * 60)
        print("EMBEDDING CACHE STATISTICS")
        print("=" * 60)

        if not stats["models"]:
            print("Cache is empty.")
            return

        print(f"Total cached embeddings: {stats['total_cached']}")
        print(f"Total cache size: {stats['total_size_mb']:.2f} MB")
        print(f"\nPer-model breakdown:")

        for model_name, model_stats in stats["models"].items():
            print(f"  • {model_name}:")
            print(f"      {model_stats['num_cached']} embeddings")
            print(f"      {model_stats['size_mb']:.2f} MB")

        print("=" * 60)
