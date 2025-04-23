from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_chroma import Chroma  
import chromadb

class ChromaSingleton:
    _instance = None

    @staticmethod
    def get_chroma_client():
        """Singleton method to ensure a single ChromaDB client instance."""
        if ChromaSingleton._instance is None:
            ChromaSingleton._instance = chromadb.PersistentClient(path="chroma_db")  
        return ChromaSingleton._instance

def get_vector_store():
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    #  Use correct method to list collections in Chroma v0.6.0+
    existing_collections = chroma_client.list_collections()

    return Chroma(
        collection_name="embeddings",
        persist_directory="./chroma_db",
        embedding_function=EmbeddingSingleton.get_embedding_function()
    )
    
def get_chroma_client():
    """Returns the ChromaDB client instance."""
    return ChromaSingleton.get_chroma_client()

#  Singleton for Hugging Face Embeddings
class EmbeddingSingleton:
    _instance = None

    @staticmethod
    def get_embedding_function():
        """Loads Hugging Face Embedding Model only once."""
        if EmbeddingSingleton._instance is None:
            EmbeddingSingleton._instance = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        return EmbeddingSingleton._instance  #  Return the actual instance, not the function

print("Embedding dimension:", len(EmbeddingSingleton.get_embedding_function().embed_query("test")))


# Add persist_chromadb function
def persist_chromadb():
    """Ensures ChromaDB persists stored embeddings."""
    try:
        chroma_client = ChromaSingleton.get_chroma_client()
        # chroma_client.persist()
        # print(" ChromaDB persistence complete.")
    except Exception as e:
        print(f" ERROR during persistence: {e}")
