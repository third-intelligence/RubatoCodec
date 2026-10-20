"""Speaker-embedding cache shared by the speaker similarity metrics.

Embeddings are keyed by the sha1 of the audio file content, so identical
original-side (ground truth) audio is reused by every arm evaluated against
the same set even when each arm keeps its own copy on disk. Each embedder
writes to its own subdirectory, named after the embedder and a hash of its
model identity, so changing the speaker model never reuses stale embeddings.
"""

import hashlib
import os

import torch


def file_sha1(path):
    """Return the sha1 hex digest of a file's content."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EmbeddingCache:
    def __init__(self, cache_dir, embedder_name, model_id):
        """model_id identifies the embedding model (e.g. a source name or weight hash)."""
        self.dir = None
        if cache_dir:
            model_hash = hashlib.sha1(str(model_id).encode()).hexdigest()[:12]
            self.dir = os.path.join(cache_dir, f"{embedder_name}-{model_hash}")
            os.makedirs(self.dir, exist_ok=True)

    def _path(self, audio_path):
        return os.path.join(self.dir, f"{file_sha1(audio_path)}.pt")

    def get_or_compute(self, audio_path, compute_fn):
        """Return the cached embedding for audio_path, computing and storing it if absent."""
        if self.dir is None:
            return compute_fn()
        path = self._path(audio_path)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
        emb = compute_fn().detach().cpu()
        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save(emb, tmp)
        os.replace(tmp, path)
        return emb
