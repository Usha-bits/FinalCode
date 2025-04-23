## re-ranking for low semantic chunk, Cosine Similarity Version of retrieve_and_answer
## search for table related data
import sys
import time
import json
import logging
import asyncio
import numpy as np
import os

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity

# Disable unwanted telemetry
os.environ["LANGCHAIN_TELEMETRY_ENABLED"] = "false"
os.environ["POSTHOG_DISABLED"] = "true"

# Set up module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom embedding engine
from GenerateLLM_new import get_vector_store, EmbeddingQueryEngine

# Clear existing logging handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup logging
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    filename="log/Retrieve_answer_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

class QueryEngine:
    def __init__(self):
        self.vector_store = get_vector_store()
        if not self.vector_store:
            raise ValueError("ChromaDB vector store not initialized.")

        self.embedding_generator = EmbeddingQueryEngine()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key= os.getenv("OPENAI_API_KEY"))
                             
        stored_data = self.vector_store.get(include=["documents"], limit=5)
        if not stored_data.get("documents"):
            logging.warning(" No documents found in ChromaDB.")
        else:
            logging.info(" Sample documents found in ChromaDB.")
           
    async def generate_response(self, query: str, context_docs: list[str]) -> str:
        context_text = "\n---\n".join(doc.strip() for doc in context_docs if doc.strip())

        messages = [
            SystemMessage(content="You are an AI assistant. Answer questions based only on the given context."),
            #SystemMessage(content="You are an AI assistant. Answer questions **only** using the context below. If the answer is not in the context, reply: 'I don't know.'"),
            HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {query}")
        ]

        try:
            # Inside generate_response (add timing for GPT-4o answer generation)
            start_llm = time.time()
            logging.info(" Sending query to LLM. Context length: %d characters", len(context_text))
            response = self.llm.invoke(messages)
            end_llm = time.time()
            llm_time = end_llm - start_llm
            logging.info(" GPT-4o answer time: %.4f seconds", llm_time)
            print(f"\n GPT-4o answer generation took {llm_time:.4f} seconds")
            if response and hasattr(response, "content"):
                return response.content.strip()
            else:
                logging.error(" Empty or malformed response from LLM.")
                return " LLM returned an empty response. Please try again or refine your query."
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(" Error invoking LLM:", e)
            raise        
    def is_table_query(self, query: str) -> bool:
        keywords = [
            "table", "list", "timeline", "responsibility", "deadline", "response time", "schedule","department",
            "duration", "SLA", "frequency", "how long", "when", "matrix"
        ]
        query_lower = query.lower()
        return any(k in query_lower for k in keywords)

    async def retrieve_and_answer(self, query: str, top_k: int = 5) -> str:
        try:
            start_retrieval = time.time()
            # Step 1: Generate query embedding
            logging.info("User query: %s", query)
            query_embedding = self.embedding_generator.generate_embedding(query)
            if query_embedding is None:
                return "Error: Failed to generate query embedding."

            # Step 2: Retrieve top-k relevant docs using ChromaDB + table prioritization
            if self.is_table_query(query):
                logging.info(" Detected table-oriented query: Prioritizing table chunks.")
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=top_k,
                    filter={"type": "table"}
                )
                if len(results) < top_k:
                    text_results = self.vector_store.similarity_search_with_relevance_scores(
                        query=query,
                        k=top_k - len(results),
                        filter={"type": "text"}
                    )
                    results += text_results
            else:
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=top_k
                )
            results = sorted(results, key=lambda x: x[1], reverse=True)  # Sort by score descending
            end_retrieval = time.time()
            retrieval_time = end_retrieval - start_retrieval
            logging.info(" Vector search time: %.4f seconds", retrieval_time)
            print(f"\n Vector search took {retrieval_time:.4f} seconds")

            # Step 3: Parse results into context_docs, scores, metadata
            context_docs = [doc.page_content for doc, score in results]
            top_scores = [score for doc, score in results]
            metadata_list = [doc.metadata for doc, score in results]

            # Step 4: Logging and debug info
            for i in range(len(context_docs)):
                logging.info(f" Retrieved doc type: {metadata_list[i].get('type')} | Score: {top_scores[i]}")

            debug_data = [
                {
                    "content": context_docs[i],
                    "score": float(top_scores[i]),
                    "metadata": metadata_list[i]
                }
                for i in range(len(context_docs))
            ]
            with open("log/cosine_similarity_debug.json", "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)

            print("\n Top similar documents (ChromaDB relevance scores):")
            for i, doc in enumerate(context_docs):
                print(f"\n--- Document {i+1} (Score: {top_scores[i]:.4f}) ---\n{doc[:500]}...\n")

            # Step 5: Generate response from LLM
            return await self.generate_response(query, context_docs)

        except Exception as e:
            logging.error(" Retrieval error: %s", str(e))
            return "An error occurred during similarity search."


def main():
    engine = QueryEngine()
    print("\n Enter your query (type 'exit' to quit):\n")

    while True:
        try:
            query = input(" Query: ").strip()
            if query.lower() in {"exit", "quit"}:
                print(" Exiting.")
                break

            answer = asyncio.run(engine.retrieve_and_answer(query))
            print("\n Answer:", answer, "\n")

        except KeyboardInterrupt:
            print("\n Interrupted by user.")
            break
        except Exception as e:
            logging.error("Unexpected error: %s", str(e))
            print(" Unexpected error. Please try again.")

if __name__ == "__main__":
    main()
