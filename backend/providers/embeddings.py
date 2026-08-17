import os


class EmbeddingModel:
    _MODEL_NAME: str = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

    # Read by the schema: chunks.embedding is VECTOR(n) and changing the model
    # changes n, so it lives beside the model that determines it.
    DIMENSIONS: int = 384

    _loaded_model = None

    def _model(self):
        """Load the ONNX model once per process, on first use.

        It is roughly 100MB and fastembed downloads it on demand, so this must
        not happen at import time in a web process — and it is cached on the
        class, not the instance, because several components each hold their own
        EmbeddingModel.
        """
        if EmbeddingModel._loaded_model is None:
            from fastembed import TextEmbedding

            EmbeddingModel._loaded_model = TextEmbedding(model_name=self._MODEL_NAME)
        return EmbeddingModel._loaded_model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model().embed(texts)]

    def embed_single_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
