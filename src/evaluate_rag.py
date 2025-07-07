# # # from rag_pipeline import RAGPipeline
# # # import pandas as pd
# # # # Sample evaluation questions
# # # EVALUATION_QUESTIONS = [
# # #     "What are common issues with credit card payments?",
# # #     "How are customers experiencing problems with BNPL services?",
# # #     "What complaints exist about money transfer delays?",
# # #     "Are there any systemic issues with personal loan approvals?",
# # #     "How do customers report problems with savings accounts?"
# # # ]

# # # def evaluate_rag():
# # #     rag = RAGPipeline()
# # #     results = []
    
# # #     for question in EVALUATION_QUESTIONS:
# # #         response = rag.query(question)
# # #         results.append({
# # #             "Question": question,
# # #             "Generated Answer": response["answer"],
# # #             "Retrieved Sources": [
# # #                 f"{src['product']} (ID: {src['complaint_id']}): {src['text'][:100]}..." 
# # #                 for src in response["sources"][:2]  # Show top 2 sources
# # #             ],
# # #             "Quality Score": "",  # To be filled manually
# # #             "Analysis": ""  # To be filled manually
# # #         })
    
# # #     # Save results to DataFrame and Markdown
# # #     df = pd.DataFrame(results)
# # #     df.to_markdown("evaluation_results.md")
# # #     print(df)
# # #     return df

# # # if __name__ == "__main__":
# # #     evaluate_rag()

# # from rag_pipeline import RAGPipeline
# # import pandas as pd

# # # Sample evaluation questions
# # EVALUATION_QUESTIONS = [
# #     "What are common issues with credit card payments?",
# #     "How are customers experiencing problems with BNPL services?",
# #     "What complaints exist about money transfer delays?",
# #     "Are there any systemic issues with personal loan approvals?",
# #     "How do customers report problems with savings accounts?"
# # ]

# # def evaluate_rag():
# #     rag = RAGPipeline()
# #     results = []
    
# #     for question in EVALUATION_QUESTIONS:
# #         try:
# #             response = rag.query(question)
# #             results.append({
# #                 "Question": question,
# #                 "Generated Answer": response["answer"],
# #                 "Retrieved Sources": [
# #                     f"{src['product']} (ID: {src['complaint_id']}): {src['text'][:100]}..." 
# #                     for src in response["sources"]
# #                 ],
# #                 "Quality Score": "",  # To be filled manually
# #                 "Analysis": ""  # To be filled manually
# #             })
# #         except Exception as e:
# #             print(f"Error processing question: {question}")
# #             print(e)
    
# #     # Save results
# #     df = pd.DataFrame(results)
# #     df.to_markdown("evaluation_results.md", index=False)
# #     print("Evaluation results saved to evaluation_results.md")
# #     return df

# # if __name__ == "__main__":
# #     evaluate_rag()

# from rag_pipeline import RAGPipeline
# import pandas as pd
# import time

# def evaluate_rag():
#     print("Setting up (this should be fast)...")
#     try:
#         rag = RAGPipeline()
#         print("Pipeline ready!\n")
#     except Exception as e:
#         print(f"Setup failed: {e}")
#         return

#     questions = [
#         "What are common credit card issues?",
#         "BNPL service problems?",
#         "Money transfer complaints?",
#         "Personal loan approval issues?",
#         "Savings account problems?"
#     ]

#     results = []
    
#     for q in questions:
#         print(f"Processing: {q}")
#         start = time.time()
        
#         try:
#             response = rag.query(q)
#             time_taken = time.time() - start
            
#             print(f"Answer ({time_taken:.1f}s): {response['answer'][:80]}...")
#             print(f"Sources: {[s['product'] for s in response['sources']]}\n")
            
#             results.append({
#                 "Question": q,
#                 "Answer": response["answer"],
#                 "Time (s)": f"{time_taken:.1f}",
#                 "Sources": [s['product'] for s in response['sources']]
#             })
#         except Exception as e:
#             print(f"Error: {e}")
#             results.append({
#                 "Question": q,
#                 "Answer": f"Error: {e}",
#                 "Time (s)": "0.0",
#                 "Sources": []
#             })

#     # Save simple results
#     pd.DataFrame(results).to_markdown("quick_results.md")
#     print("\nDone! Results saved to quick_results.md")

# if __name__ == "__main__":
#     evaluate_rag()

# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain_openai import ChatOpenAI
# import os

# class RAGPipeline:
#     def __init__(self, chroma_db_path, openai_api_key=None):
#         # Initialize embedding model
#         self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
#         # Load Chroma vector store
#         self.vectorstore = Chroma(
#             persist_directory=chroma_db_path,
#             embedding_function=self.embedding_model
#         )
        
#         # Initialize retriever
#         self.retriever = self.vectorstore.as_retriever(
#             search_kwargs={"k": 5},
#             search_type="mmr"  # Maximal marginal relevance for better diversity
#         )
        
#         # Initialize OpenAI LLM
#         self.llm = ChatOpenAI(
#             model="gpt-3.5-turbo",
#             temperature=0.7,
#             openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
#         )
        
#         # Define prompt template
#         self.prompt_template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. 
# Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

# Context: {context}

# Question: {question}

# Please provide a concise answer based on the context. If the answer isn't in the context, say "I don't have enough information to answer this question."

# Answer:"""
        
#         self.qa_prompt = PromptTemplate(
#             template=self.prompt_template,
#             input_variables=["context", "question"]
#         )
        
#         # Set up retrieval QA chain
#         self.qa_chain = RetrievalQA.from_chain_type(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=self.retriever,
#             return_source_documents=True,
#             chain_type_kwargs={"prompt": self.qa_prompt}
#         )
    
#     def answer_question(self, question):
#         """Answer a question using the RAG pipeline"""
#         try:
#             result = self.qa_chain.invoke({"query": question})
#             return {
#                 "answer": result["result"],
#                 "sources": [doc.page_content for doc in result["source_documents"]]
#             }
#         except Exception as e:
#             print(f"Error answering question: {e}")
#             return {
#                 "answer": "Sorry, I encountered an error processing your request.",
#                 "sources": []
#             }


# if __name__ == "__main__":
#     # Initialize pipeline with error handling
#     try:
#         print("Initializing RAG pipeline with ChromaDB and OpenAI...")
        
#         # Configuration
#         CHROMA_DB_PATH = "../vector_store"  # Your ChromaDB path
#         OPENAI_API_KEY = "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"  # Or set as environment variable
        
#         # Initialize RAG pipeline
#         rag = RAGPipeline(
#             chroma_db_path=CHROMA_DB_PATH,
#             openai_api_key=OPENAI_API_KEY
#         )
        
#         # Test questions
#         test_questions = [
#             "What are common complaints about credit card fees?",
#             "How do customers feel about the mobile app experience?",
#             "What issues do customers report about fraud detection?",
#             "Are there complaints about customer service wait times?",
#             "What problems do customers have with balance transfers?"
#         ]
        
#         # Process each question
#         for question in test_questions:
#             print("\n" + "="*50)
#             print(f"Question: {question}")
            
#             # Get answer from RAG pipeline
#             result = rag.answer_question(question)
            
#             # Print results
#             print(f"\nAnswer: {result['answer']}")
            
#             if result['sources']:
#                 print("\nTop sources:")
#                 for i, source in enumerate(result['sources'][:2], 1):
#                     # Truncate long sources for display
#                     display_source = (source[:250] + "...") if len(source) > 250 else source
#                     print(f"{i}. {display_source}")
            
#     except Exception as e:
#         print(f"\nFailed to initialize RAG pipeline: {e}")
#         print("\nTroubleshooting steps:")
#         print("1. Ensure ChromaDB directory exists at:", CHROMA_DB_PATH)
#         print("2. Verify your OpenAI API key is valid")
#         print("3. Check all required packages are installed")
#         print("4. Ensure you have internet connectivity for OpenAI API access")


from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class FreeRAGPipeline:
    def __init__(self, chroma_db_path):
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Load Chroma vector store
        self.vectorstore = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=self.embedding_model
        )
        
        # Initialize retriever with MMR for better diversity
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5},
            search_type="mmr"
        )
        
        # Initialize local LLM (completely free)
        self.llm = self._initialize_local_llm()
        
        # Define optimized prompt template
        self.prompt_template = """You are a helpful financial assistant analyzing customer complaints.
Context: {context}
Question: {question}
Based on the above context, provide a concise answer. If unsure, say "I don't have enough information."
Answer:"""
        
        self.qa_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
        
        # Set up retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt}
        )
    
    def _initialize_local_llm(self):
        """Initialize a free local LLM with efficient settings"""
        try:
            # Using a smaller, more efficient model
            model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
            model_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            
            # Alternative smaller model if needed:
            # model_name = "TheBloke/zephyr-7B-beta-GGUF"
            # model_file = "zephyr-7b-beta.Q4_K_M.gguf"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                model_file=model_file,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"Local LLM initialization failed: {e}")
            raise RuntimeError("Failed to initialize local LLM")
    
    def answer_question(self, question):
        """Answer a question using the RAG pipeline"""
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]]
            }
        except Exception as e:
            print(f"Error answering question: {e}")
            return {
                "answer": "Sorry, I encountered an error processing your request.",
                "sources": []
            }


if __name__ == "__main__":
    # Initialize pipeline with error handling
    try:
        print("Initializing FREE RAG pipeline with ChromaDB...")
        
        # Configuration - point to your ChromaDB directory
        CHROMA_DB_PATH = "../vector_store"
        
        # Initialize RAG pipeline
        rag = FreeRAGPipeline(chroma_db_path=CHROMA_DB_PATH)
        
        # Test questions
        test_questions = [
            "What are common complaints about credit card fees?",
            "How do customers feel about the mobile app experience?",
            "What issues do customers report about fraud detection?",
            "Are there complaints about customer service wait times?",
            "What problems do customers have with balance transfers?"
        ]
        
        # Process each question
        for question in test_questions:
            print("\n" + "="*50)
            print(f"Question: {question}")
            
            # Get answer from RAG pipeline
            result = rag.answer_question(question)
            
            # Print results
            print(f"\nAnswer: {result['answer']}")
            
            if result['sources']:
                print("\nTop sources:")
                for i, source in enumerate(result['sources'][:2], 1):
                    # Truncate long sources for display
                    display_source = (source[:250] + "...") if len(source) > 250 else source
                    print(f"{i}. {display_source}")
            
    except Exception as e:
        print(f"\nFailed to initialize RAG pipeline: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure ChromaDB directory exists at:", CHROMA_DB_PATH)
        print("2. Verify you have sufficient GPU memory (8GB+ recommended)")
        print("3. Check all required packages are installed")
        print("4. For CPU-only systems, use smaller GGUF models")