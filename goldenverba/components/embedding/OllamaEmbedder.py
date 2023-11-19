from weaviate import Client

from tqdm import tqdm

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

 
    def embed(
        self,
        documents: list[Document],
        client: Client,
    ) -> bool:
        """Embed verba documents and its chunks to Weaviate
        @parameter: documents : list[Document] - List of Verba documents
        @parameter: client : Client - Weaviate Client
        @parameter: batch_size : int - Batch Size of Input
        @returns bool - Bool whether the embedding what successful
        """
        if not self.model:
            raise RuntimeError("Ollama model is not initialized")

        for document in tqdm(
            documents, total=len(documents), desc="Vectorizing document chunks"
        ):
            for chunk in document.chunks:
                chunk.set_vector(self.model.embed_documents(chunk))
        return self.import_data(documents, client)

