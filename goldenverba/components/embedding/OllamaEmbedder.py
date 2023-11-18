from goldenverba.components.embedding.interface import Embedder
from goldenverba.components.reader.document import Document
from langchain.embeddings import OllamaEmbeddings

class OllamaEmbedder(Embedder):
    """
    OllamaEmbedder for Verba
    """

    def __init__(self):
        super().__init__()
        self.name = "OllamaEmbedder"
        self.requires_library = ["langchain"]
        self.description = "Embeds documents using Ollama's large language models"
        self.vectorizer = "Ollama"
        self.model = None
        try:
            self.model = OllamaEmbeddings(
                model="llama:7b",  # You can specify other model parameters here
            )
        except Exception as e:
            print("Error initializing Ollama model:", e)

    def embed(self, documents):
        """
        Embed documents using the Ollama model.
        """
        if not self.model:
            raise RuntimeError("Ollama model is not initialized")

        texts = [doc.content for doc in documents]
        embeddings = self.model.embed_documents(texts)
        return embeddings

