
# # import chromadb
# # from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# # from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
# # import torch
# # from typing import List, Dict

# # class RAGPipeline:
# #     def __init__(self, vector_store_path: str = "../vector_store"):
# #         # Initialize vector store
# #         self.client = chromadb.PersistentClient(path=vector_store_path)
# #         self.collection = self.client.get_collection(
# #             "complaints",
# #             embedding_function=SentenceTransformerEmbeddingFunction(
# #                 model_name="all-MiniLM-L6-v2"
# #             )
# #         )
        
# #         # Initialize open-source LLM (no auth required)
# #         model_name = "facebook/opt-1.3b"  # Fully open model
# #         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
# #         self.model = AutoModelForCausalLM.from_pretrained(
# #             model_name,
# #             torch_dtype=torch.float16,
# #             device_map="auto"
# #         )
        
# #         self.prompt_template = """[INST] You are a financial analyst assistant. 
# # Use ONLY these complaint excerpts to answer. If unsure, say "I don't have enough information."

# # Context:
# # {context}

# # Question: {question} [/INST]
# # Professional Answer:"""

# #     def retrieve(self, question: str, k: int = 5) -> List[Dict]:
# #         """Retrieve relevant chunks"""
# #         results = self.collection.query(
# #             query_texts=[question],
# #             n_results=k
# #         )
# #         return [
# #             {
# #                 "text": doc,
# #                 "complaint_id": meta["complaint_id"],
# #                 "product": meta["product"]
# #             }
# #             for doc, meta in zip(results['documents'][0], results['metadatas'][0])
# #         ]

# #     def generate(self, question: str, retrieved_chunks: List[Dict]) -> str:
# #         """Generate answer using LLM"""
# #         context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
# #         prompt = self.prompt_template.format(
# #             context=context,
# #             question=question
# #         )
        
# #         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
# #         outputs = self.model.generate(
# #             **inputs,
# #             max_new_tokens=200,
# #             do_sample=True,
# #             temperature=0.7
# #         )
        
# #         return self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# #     def query(self, question: str, k: int = 5) -> Dict:
# #         """Complete RAG pipeline"""
# #         chunks = self.retrieve(question, k)
# #         answer = self.generate(question, chunks)
# #         return {
# #             "question": question,
# #             "answer": answer,
# #             "sources": chunks[:2]  # Return top 2 sources
# #         }

# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch
# from typing import List, Dict

# class RAGPipeline:
#     def __init__(self, vector_store_path: str = "../vector_store"):
#         # Initialize vector store
#         self.client = chromadb.PersistentClient(path=vector_store_path)
#         self.collection = self.client.get_collection(
#             "complaints",
#             embedding_function=SentenceTransformerEmbeddingFunction(
#                 model_name="all-MiniLM-L6-v2"
#             )
#         )
        
#         # Initialize open-source LLM
#         model_name = "distilgpt2"  # Lightweight model that works well
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # Initialize model with device management
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
        
#         self.prompt_template = """Answer the question using ONLY the provided context.
# If the answer isn't in the context, say "I don't have enough information."

# Context:
# {context}

# Question: {question}

# Answer:"""

#     def retrieve(self, question: str, k: int = 5) -> List[Dict]:
#         """Retrieve relevant chunks"""
#         results = self.collection.query(
#             query_texts=[question],
#             n_results=k
#         )
#         return [
#             {
#                 "text": doc,
#                 "complaint_id": meta["complaint_id"],
#                 "product": meta["product"]
#             }
#             for doc, meta in zip(results['documents'][0], results['metadatas'][0])
#         ]

#     def generate(self, question: str, retrieved_chunks: List[Dict]) -> str:
#         """Generate answer using LLM"""
#         context = "\n".join([f"- {chunk['text']}" for chunk in retrieved_chunks])
#         prompt = self.prompt_template.format(
#             context=context,
#             question=question
#         )
        
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=150,
#                 do_sample=True,
#                 temperature=0.7,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
        
#         full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return full_response.replace(prompt, "").strip()

#     def query(self, question: str, k: int = 5) -> Dict:
#         """Complete RAG pipeline"""
#         chunks = self.retrieve(question, k)
#         answer = self.generate(question, chunks)
#         return {
#             "question": question,
#             "answer": answer,
#             "sources": chunks[:2]  # Return top 2 sources
#         }

# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from huggingface_hub import login
# import torch
# from typing import List, Dict
# import pandas as pd

# # Authenticate with Hugging Face (remove if using open model)
# login(token="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")  # Get token from https://huggingface.co/settings/tokens

# class RAGPipeline:
#     def __init__(self, vector_store_path: str = "../vector_store"):
#         # Initialize vector store
#         self.client = chromadb.PersistentClient(path=vector_store_path)
#         self.collection = self.client.get_collection(
#             "complaints",
#             embedding_function=SentenceTransformerEmbeddingFunction(
#                 model_name="all-MiniLM-L6-v2"
#             )
#         )
        
#         # Initialize LLM - Choose ONE of these options:
        
#         # Option 1: Open model (no auth required)
#         # model_name = "google/flan-t5-large"
        
#         # Option 2: Better but requires auth
#         model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
        
#         self.prompt_template = """[INST] <<SYS>>
# You are a financial complaint analyst. Answer using ONLY the provided context.
# Respond concisely in 1-3 sentences. If unsure, say "I cannot determine this from the complaints data."
# <</SYS>>

# Context:
# {context}

# Question: {question} 

# Answer: [/INST]"""

#     def retrieve(self, question: str, k: int = 5) -> List[Dict]:
#         """Retrieve relevant chunks with product filtering"""
#         # Extract likely product from question
#         product_keywords = {
#             "credit card": "Credit card",
#             "bnpl": "Buy Now, Pay Later (BNPL)",
#             "loan": "Personal loan",
#             "savings": "Savings account",
#             "money transfer": "Money transfers"
#         }
        
#         product_filter = None
#         for kw, product in product_keywords.items():
#             if kw in question.lower():
#                 product_filter = product
#                 break
        
#         # Build query
#         query_params = {
#             "query_texts": [question],
#             "n_results": k
#         }
        
#         if product_filter:
#             query_params["where"] = {"product": product_filter}
        
#         results = self.collection.query(**query_params)
        
#         return [
#             {
#                 "text": doc,
#                 "complaint_id": meta["complaint_id"],
#                 "product": meta["product"]
#             }
#             for doc, meta in zip(results['documents'][0], results['metadatas'][0])
#         ]

#     def generate(self, question: str, retrieved_chunks: List[Dict]) -> str:
#         """Generate answer using LLM"""
#         context = "\n".join([f"- {chunk['text']}" for chunk in retrieved_chunks])
#         prompt = self.prompt_template.format(
#             context=context,
#             question=question
#         )
        
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=150,
#                 temperature=0.3,  # Lower for more deterministic answers
#                 repetition_penalty=1.5,  # Reduce repetition
#                 do_sample=False,  # More focused answers
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
        
#         full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return full_response.replace(prompt, "").strip()

#     def query(self, question: str, k: int = 5) -> Dict:
#         """Complete RAG pipeline"""
#         chunks = self.retrieve(question, k)
#         answer = self.generate(question, chunks)
#         return {
#             "question": question,
#             "answer": answer,
#             "sources": chunks[:2]  # Return top 2 sources
#         }

# def evaluate_rag():
#     print("Initializing RAG pipeline...")
#     rag = RAGPipeline()
#     results = []
    
#     EVALUATION_QUESTIONS = [
#         "What are common issues with credit card payments?",
#         "How are customers experiencing problems with BNPL services?",
#         "What complaints exist about money transfer delays?",
#         "Are there any systemic issues with personal loan approvals?",
#         "How do customers report problems with savings accounts?"
#     ]
    
#     for question in EVALUATION_QUESTIONS:
#         print(f"\nProcessing: {question}")
#         try:
#             response = rag.query(question)
#             print("Retrieved sources:", [s['product'] for s in response['sources']])
#             print("Generated answer:", response['answer'][:200] + "...")
            
#             results.append({
#                 "Question": question,
#                 "Generated Answer": response["answer"],
#                 "Retrieved Sources": [
#                     f"{src['product']} (ID: {src['complaint_id']}): {src['text'][:100]}..." 
#                     for src in response["sources"]
#                 ],
#                 "Quality Score": "",  # Fill manually
#                 "Analysis": ""  # Fill manually
#             })
#         except Exception as e:
#             print(f"Error: {str(e)}")
#             results.append({
#                 "Question": question,
#                 "Generated Answer": f"Error: {str(e)}",
#                 "Retrieved Sources": [],
#                 "Quality Score": "0",
#                 "Analysis": "Processing failed"
#             })
    
#     # Save results
#     df = pd.DataFrame(results)
#     df.to_markdown("evaluation_results.md", index=False)
#     print("\nEvaluation complete. Results saved to evaluation_results.md")
#     return df

# if __name__ == "__main__":
#     evaluate_rag()

# import chromadb
# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch
# from typing import List, Dict

# class RAGPipeline:
#     def __init__(self, vector_store_path: str = "../vector_store"):
#         # Initialize vector store
#         self.client = chromadb.PersistentClient(path=vector_store_path)
#         self.collection = self.client.get_collection(
#             "complaints",
#             embedding_function=SentenceTransformerEmbeddingFunction(
#                 model_name="all-MiniLM-L6-v2"
#             )
#         )
        
#         # Initialize open LLM - using EleutherAI/gpt-neo-1.3B (no auth required)
#         model_name = "EleutherAI/gpt-neo-1.3B"
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
        
#         self.prompt_template = """[INSTRUCTION]
# You are a financial complaint analyst. Answer the question using ONLY the context below.
# Respond concisely in 1-3 sentences. If the answer isn't in the context, say "I cannot determine this from the complaints data."

# CONTEXT:
# {context}

# QUESTION: {question}

# ANSWER:"""

#     def retrieve(self, question: str, k: int = 5) -> List[Dict]:
#         """Retrieve relevant chunks with product filtering"""
#         product_map = {
#             "credit card": "Credit card",
#             "bnpl": "Buy Now, Pay Later (BNPL)",
#             "loan": "Personal loan",
#             "savings": "Savings account",
#             "money transfer": "Money transfers"
#         }
        
#         product_filter = None
#         for kw, product in product_map.items():
#             if kw in question.lower():
#                 product_filter = product
#                 break
        
#         query_params = {
#             "query_texts": [question],
#             "n_results": k
#         }
        
#         if product_filter:
#             query_params["where"] = {"product": product_filter}
        
#         results = self.collection.query(**query_params)
        
#         return [
#             {
#                 "text": doc,
#                 "complaint_id": meta["complaint_id"],
#                 "product": meta["product"]
#             }
#             for doc, meta in zip(results['documents'][0], results['metadatas'][0])
#         ]

#     def generate(self, question: str, retrieved_chunks: List[Dict]) -> str:
#         """Generate answer using LLM"""
#         context = "\n".join([f"- {chunk['text']}" for chunk in retrieved_chunks])
#         prompt = self.prompt_template.format(
#             context=context,
#             question=question
#         )
        
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=100,
#                 temperature=0.3,
#                 repetition_penalty=1.5,
#                 do_sample=False,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
        
#         full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return full_response.split("ANSWER:")[-1].strip()

#     def query(self, question: str, k: int = 5) -> Dict:
#         """Complete RAG pipeline"""
#         chunks = self.retrieve(question, k)
#         answer = self.generate(question, chunks)
#         return {
#             "question": question,
#             "answer": answer,
#             "sources": chunks[:2]  # Return top 2 sources
#         }
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from transformers import pipeline
from typing import List, Dict

class RAGPipeline:
    def __init__(self, vector_store_path: str = "../vector_store"):
        """Initialize with tiny, fast model"""
        # Vector store setup
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection = self.client.get_collection(
            "complaints",
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        # Tiny model (only 82MB download)
        self.generator = pipeline(
            "text-generation",
            model="distilgpt2",  # Very small model
            device="cpu",  # No GPU needed
            framework="pt",
            max_length=100
        )
        
        self.prompt_template = """Use this context to answer. If unsure, say "I don't know."

Context: {context}

Question: {question}

Short Answer:"""

    def retrieve(self, question: str, k: int = 3) -> List[Dict]:
        """Faster retrieval with fewer chunks"""
        results = self.collection.query(
            query_texts=[question],
            n_results=k
        )
        return [
            {
                "text": doc,
                "complaint_id": meta["complaint_id"],
                "product": meta["product"]
            }
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]

    def generate(self, question: str, chunks: List[Dict]) -> str:
        """Fast generation with small model"""
        context = "\n".join([f"- {c['text']}" for c in chunks])
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        try:
            response = self.generator(
                prompt,
                max_length=150,
                temperature=0.3,
                do_sample=False
            )
            return response[0]["generated_text"].split("Short Answer:")[-1].strip()
        except:
            return "Could not generate answer"

    def query(self, question: str) -> Dict:
        """Optimized complete pipeline"""
        chunks = self.retrieve(question)
        answer = self.generate(question, chunks)
        return {
            "question": question,
            "answer": answer,
            "sources": chunks
        }