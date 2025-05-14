''' this code perform 
        chunking, embedding generation, insert into chromaDB
'''
import os
import json
import hashlib
import numpy as np
import logging
import asyncio
import openai
import re
import tiktoken  # Import tiktoken library
import time
import nltk
import string
from db_util import get_chroma_client, get_vector_store, persist_chromadb, EmbeddingSingleton
from langchain_chroma import Chroma  # Required Import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Setup logging to file "log/embedding_log.txt"
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    filename="log/embedding_log.txt",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Helper function to convert NumPy arrays to lists for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    raise TypeError(f"Type {type(obj)} not serializable")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EmbeddingQueryEngine:
    """
    This class generates embeddings from cleaned text and table data,
    stores them in ChromaDB, and also retrieves stored embeddings to generate an answer
    using ChatOpenAI.
    """
    def __init__(self,
                 input_folder="output/CleanedData",
                 output_folder="output/EmbeddedData",
                 text_filename="cleaned_text_data.json",
                 table_filename="cleaned_table_data.json",
                 output_filename="Embedding.json"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.text_filepath = os.path.join(input_folder, text_filename)
        self.table_filepath = os.path.join(input_folder, table_filename)
        self.output_filepath = os.path.join(output_folder, output_filename)
        
        # Create output folder if it does not exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Initialize ChromaDB components and embedding function
        self.chroma_client = get_chroma_client()
        self.vector_store = get_vector_store()
        if self.vector_store is None:
            logging.error(" Vector store is not initialized.")
            raise RuntimeError("Vector store could not be initialized.")

        self.embedding_function = EmbeddingSingleton.get_embedding_function()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        logging.info("Initialized ChromaDB client, vector store, and LLM.")         #Initialization complete
        
        # Initialize Tiktoken tokenizer for BGE model
        self.tokenizer = tiktoken.get_encoding("p50k_base")  

    # Utility function to formats the structured JSON content of a webpage into a single plain text string
    @staticmethod
    def prepare_text_from_content(entry):
        content_text = ""
        for heading, para in entry.get("content", {}).items():
            content_text += f"\n{heading}:\n{para}\n"
        
        combined = f"Title: {entry.get('title', '')}\nURL: {entry.get('url', '')}\n{content_text}"
        return combined.strip()

    #convert table data into plain text
    @staticmethod
    def flatten_table(table):
        """
        Flattens a table (assumed to be a list of lists) into a plain text string.
        Each row is joined with " | " and rows are separated by newline.
        """
        flattened_rows = []
        for row in table:
            flattened_rows.append(" | ".join(str(cell) for cell in row))
        return "\n".join(flattened_rows)
        
    def load_data(self):
        data_chunks = []

        # Load text data
        text_data = self.load_json_data(self.text_filepath)
        if text_data:
            for entry in text_data:
                url = entry.get("url", "")
                title = entry.get("title", "")
                content_dict = entry.get("content", {})

                for section, text in content_dict.items():
                    if text.strip():
                        # Add section name to text before chunking
                        smart_text = f"{section}\n{text}"
                        chunks = self.split_text_into_chunks(text)
                        for chunk in chunks:
                            metadata = {
                                "url": url,
                                "title": title,
                                "section": section,
                                "source": "GitLab Handbook",
                                "type": "text"
                            }
                            data_chunks.append((chunk, metadata))
                    else:
                        logging.warning(f" Skipping empty section '{section}' for URL: {url}")
        
        print(f" Total text entries loaded: {len(text_data)}" if text_data else " No text data loaded.")

        
        table_chunk_count = 0
        # Load table data
        table_data = self.load_json_data(self.table_filepath)
        if table_data:
            for entry in table_data:
                url = entry.get("url", "")
                title = entry.get("title", "")
                tables = entry.get("tables", [])

                for table_entry in tables:
                    section = table_entry.get("section", "Unknown Section")
                    table = table_entry.get("table", [])

                    if not isinstance(table, list):
                        logging.warning(f"Skipping malformed table in URL: {url}, section: {section}")
                        continue  # Skip if table is not in list format                                    

                    flattened = self.flatten_table(table)
                    context = f"Title: {title}\nSection: {section}\nURL: {url}\nTable Data:\n{flattened}"
                    table_chunks = self.split_text_into_chunks(context)

                    metadata = {
                        "url": url,
                        "title": title,
                        "section": section,
                        "source": "GitLab Handbook",
                        "type": "table"
                    }
                    
                    for chunk in table_chunks:
                        logging.info(f"Table chunk added. Length: {len(chunk)} | Content preview: {chunk[:500]}")
                        table_chunk_count += 1
                        data_chunks.append((chunk, metadata))
                        
        print(f" Table chunks stored: {table_chunk_count}" if table_data else " No table data loaded.")
        logging.info("Loaded %d individual entries.", len(data_chunks))
        return data_chunks

    def load_json_data(self, filepath):
        """Helper function to load JSON data from a file."""
        data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logging.error(f"Failed to load {filepath}: {e}")
        return data

    @staticmethod
    def clean_text(text):
        """
        Cleans text by removing unwanted phrases.
        """
        unwanted_phrases = [
            "© 2025 GitLab All Rights Reserved",
            "Privacy Statement",
            "Cookie Settings",
            "Edit this page",
            "please contribute"
        ]
        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")
        return text.strip()

    def split_text_into_chunks(self, text, chunk_size=256, chunk_overlap=20):
        """
        Dynamically splits text based on length and token count using Tiktoken.
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenizer.encode(cleaned_text)
        token_count = len(tokens)

        # Return early if text is already short and meaningful
        if token_count <= chunk_size and not self.is_low_semantic(cleaned_text):
            return [cleaned_text]
        
        if token_count <= chunk_size:
            return [cleaned_text]

        chunks = []
        for i in range(0, token_count, chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk = self.tokenizer.decode(chunk_tokens)
            if not self.is_low_semantic(chunk):
                chunks.append(chunk)

        logging.info("Text split into %d chunks using Tiktoken.", len(chunks))
        return chunks
        
    
    @staticmethod
    def is_low_semantic(text: str, min_tokens: int = 10, symbol_ratio_thresh: float = 0.7) -> bool:
        token_count = len(text.split())  # Approximation of token count
        
        symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
        total_chars = len(text)
        symbol_ratio = symbol_count / total_chars if total_chars else 0

        return token_count < min_tokens or symbol_ratio > symbol_ratio_thresh
    
    @staticmethod
    def filter_chunks_with_bm25(chunks, threshold=0.5):
        corpus = [chunk for chunk, _ in chunks]
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        selected_chunks = []
        for i, doc in enumerate(corpus):
            scores = bm25.get_scores(word_tokenize(doc.lower()))
            avg_score = np.mean(scores)
            if avg_score > threshold:
                selected_chunks.append(chunks[i])

        return selected_chunks


    # Generate embedding for query
    def generate_embedding(self, query):
        """
        Generates an embedding for the given query using the configured embedding function.
        
        Args:
            query (str): The text query for which to generate an embedding.

        Returns:
            list: The generated embedding as a list of floating-point numbers.
        """
        if not query:
            logging.error("generate_embedding: Received an empty query.")
            return None

        try:
            embedding = self.embedding_function.embed_query(query)  # Generate embedding
            logging.info("Generated embedding for query: %s", query[:50])  # Log query preview
            return embedding
        except Exception as e:
            logging.error("Error generating embedding: %s", str(e))
            return None
    def is_meaningful(self, text: str) -> bool:
        # Remove punctuation and check word count
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.strip().split()
        
        # Heuristic: must have at least 5 words and some alphabetic content
        if len(words) < 5:
            return False
        if not any(char.isalpha() for char in text):
            return False
        
        return True

    def generate_and_store_embeddings(self, text_chunks: list[tuple[str, dict]]):
        start_time = time.time()  # Start timer
        batch_size = 10
        check_batch_size = 50
        processed_ids = set()
        chunk_times = []  # Collect individual embedding times

        logging.info("🔹 Total text chunks to process: %d", len(text_chunks))
        
        # Prepare BM25 scorer
        all_raw_texts = [chunk for chunk, _ in text_chunks]
        tokenized_corpus = [word_tokenize(text.lower()) for text in all_raw_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        query_terms = [
            "People Group", "People Operations", "Talent and Engagement", "People Business Partners", "Diversity, Inclusion, and Belonging",
            "HelpLab", "emergency", "EthicsPoint", "Lighthouse Services", "harassment", "Code of Conduct",
            "People Manager Calendar", "task reminders", "calendar updates", "Slack reminders",
            "reach People Group", "team aliases", "people-connect@gitlab.com",
            "Level Up", "CultureAmp", "Learning Evangelist", "training", "career development", "Take Time Out to Learn",
            "Workday", "Culture Amp", "Okta", "accessing Workday",
            "Legal team support", "contract amendments", "country-specific requirements",
            "how to report", "manager responsibilities", "training guide", "anonymous hotline", "policy updates"
        ]
        tokenized_query = word_tokenize(" ".join(query_terms).lower())

        bm25_scores = bm25.get_scores(tokenized_query)

        # Filter chunks with low scores (e.g., below 0.3)
        text_chunks = [
            (chunk, metadata)
            for (chunk, metadata), score in zip(text_chunks, bm25_scores)
            if score > 0.3
        ]

        logging.info(" BM25 filtering applied. Retained %d chunks.", len(text_chunks))
        
        # Further filter out non-meaningful chunks
        filtered_chunks = []
        for chunk_text, metadata in text_chunks:
            if not self.is_meaningful(chunk_text):
                print(f" Skipping chunk: {metadata}")
                continue
            filtered_chunks.append((chunk_text, metadata))

        text_chunks = filtered_chunks
        logging.info(" Meaningful content filtering applied. Retained %d chunks.", len(text_chunks))

        ids, documents, metadatas = [], [], []
        chunk_records = []

        # Precompute (chunk, metadata, unique_id)
        for chunk, metadata in text_chunks:
            unique_id = hashlib.md5((chunk + json.dumps(metadata, sort_keys=True)).encode()).hexdigest()
            ids.append(hashlib.sha1(chunk.encode()).hexdigest())
            documents.append(chunk)
            metadatas.append(metadata)
            chunk_records.append((chunk, metadata, unique_id))
            
            if metadata.get("type") == "table":
                print(f" Confirmed table metadata: {metadata}")

        all_ids = [uid for _, _, uid in chunk_records]

        # Fetch existing IDs from ChromaDB
        existing_ids = set()
        for i in range(0, len(all_ids), check_batch_size):
            batch = all_ids[i : i + check_batch_size]
            try:
                existing_docs = self.vector_store.get(ids=batch)
                if existing_docs and "ids" in existing_docs:
                    existing_ids.update(existing_docs["ids"])
            except Exception as e:
                logging.error(f"Error fetching existing IDs: {e}")
            
        try:
            stored = self.vector_store.get(include=["metadatas"], limit=10000)
            table_chunks = [m for m in stored["metadatas"] if m.get("type") == "table"]
            logging.info(f" Total table chunks already in ChromaDB: {len(table_chunks)}")
            print(f" Table chunks stored: {len(table_chunks)}")
            print(json.dumps(stored["metadatas"][:5], indent=2))
        except Exception as e:
            logging.error(f" Error fetching stored metadata for table check: {e}")
        
        chunks_to_process = []
        embeddings_to_store = []

        for i, (chunk, metadata, unique_id) in enumerate(chunk_records):
            logging.info("Processing Chunk %d/%d - ID: %s", i + 1, len(chunk_records), unique_id)

            if unique_id in existing_ids:
                logging.info("Skipping duplicate chunk: %s...", chunk[:50])
                continue

            # Measure embedding generation time
            chunk_start = time.time()
            embeddings = self.embedding_function.embed_documents([chunk])
            chunk_end = time.time()
            chunk_duration = chunk_end - chunk_start
            chunk_times.append(chunk_duration)
            
            logging.info(f" Time to embed chunk {i + 1}: {chunk_duration:.4f} sec")
            
            if not embeddings or not isinstance(embeddings[0], (np.ndarray, list)):
                logging.error("Invalid or empty embedding for chunk: %s", chunk[:50])
                continue

            embedding_list = embeddings[0].tolist() if isinstance(embeddings[0], np.ndarray) else embeddings[0]

            chunks_to_process.append(Document(page_content=chunk,
                                              metadata=metadata))
            embeddings_to_store.append({
                "id": unique_id,
                "embedding": embedding_list,
                "metadata": metadata,
                "document": chunk
            })
            processed_ids.add(unique_id)

            # Batch insert if reached batch_size
            if len(chunks_to_process) >= batch_size:
                try:
                    logging.info(" Attempting to store embeddings...")
                    self._store_to_chromadb(chunks_to_process, embeddings_to_store)
                    logging.info(" Successfully stored %d embeddings", len(embeddings_to_store))
                except Exception as e:
                    print(f" Error during batch insert to ChromaDB: {e}")
                chunks_to_process = []
                embeddings_to_store = []  

        # Store any remaining documents
        if chunks_to_process:
            try:
                logging.info(" Attempting to store remaining embeddings...")
                self._store_to_chromadb(chunks_to_process, embeddings_to_store)
                logging.info(" Successfully stored remaining %d embeddings", len(embeddings_to_store))
            except Exception as e:
                print(f" Error during final insert to ChromaDB: {e}")
        
        try:
            #persist_chromadb(self.chroma_client)
            persist_chromadb()
            logging.info(" ChromaDB persisted successfully.")
        except Exception as e:
            logging.error("Error persisting ChromaDB: %s", e)

        elapsed_time = time.time() - start_time
        
        avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
        logging.info(" Embedding and storage complete. Total time: %.2f seconds", elapsed_time)
        logging.info(" Average time per embedding: %.4f seconds", avg_chunk_time)
        print(f" Average time per chunk embedding: {avg_chunk_time:.4f} seconds")
           
    def _store_to_chromadb(self, docs, embeddings_info):
        try:
            ids = [item["id"] for item in embeddings_info]
            embeddings = [item["embedding"] for item in embeddings_info]
            metadatas = [item["metadata"] for item in embeddings_info]
            documents = [item["document"] for item in embeddings_info]  # These are strings

            self.vector_store.add_texts(
                texts=documents,
                ids=ids,
                metadatas=metadatas,
                embeddings=embeddings
            )

            logging.info(" Stored %d embeddings to ChromaDB.", len(embeddings))
            for meta in metadatas:
                logging.info(f" Inserted chunk of type: {meta.get('type')}")
        except Exception as e:
            logging.error(" Error storing embeddings to ChromaDB: %s", str(e))
        
    def prepare_chunks_with_metadata(self, data_chunks):
        """
        Takes loaded data (text/table) and returns list of (chunk, metadata) tuples.
        Includes relevant metadata like URL, title, section (if available), and type.
        """
        result = []

        for content, original_metadata in data_chunks:
            entry_type = original_metadata.get("type", "text")
            
            # Extract metadata from the content using regex
            url_match = re.search(r'URL:\s*(.*)', content)
            title_match = re.search(r'Title:\s*(.*)', content)

            url = url_match.group(1).strip() if url_match else original_metadata.get("url", "")
            title = title_match.group(1).strip() if title_match else original_metadata.get("title", "")
            section = ""

            if entry_type == "table":
                section_match = re.search(r'Section:\s*(.*)', content)
                section = section_match.group(1).strip() if section_match else original_metadata.get("section", "")

            chunks = self.split_text_into_chunks(content)

            for chunk in chunks:
                if not chunk.strip():
                    continue

                # Skip low semantic chunks
                if self.is_low_semantic(chunk):
                    chunks.append(chunk)

                metadata = {
                    "source": "GitLab Handbook",
                    "type": entry_type,
                    "url": url,
                    "title": title,
                }

                if entry_type == "table":
                    metadata["section"] = section

                result.append((chunk, metadata))

        logging.info(" Prepared %d chunks with metadata.", len(result))
        return result
   
if __name__ == "__main__":
    engine = EmbeddingQueryEngine()
    
    # Load all text and table chunks
    all_chunks = engine.load_data()

    # Separate text and table chunks
    text_chunks = [chunk for chunk in all_chunks if chunk[1].get("type") != "table"]
    table_chunks = [chunk for chunk in all_chunks if chunk[1].get("type") == "table"]

    print(f" Total text entries loaded: {len(text_chunks)}")
    print(f" Total table entries loaded: {len(table_chunks)}")

    # Apply BM25 filtering only to text chunks
    filtered_text_chunks = engine.filter_chunks_with_bm25(text_chunks)

    # Combine filtered text with unfiltered tables
    final_chunks = filtered_text_chunks + table_chunks

    # Store the final filtered + table chunks
    engine.generate_and_store_embeddings(final_chunks)



