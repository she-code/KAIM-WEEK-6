# KAIM-WEEK-6

## Project Overview

This project develops an AI-driven complaint analysis tool for CrediTrust Financial to turn unstructured customer feedback into actionable insights. By leveraging NLP and RAG, the system enables non-technical teams to quickly identify emerging issues across core products—reducing manual review time and supporting real-time, proactive decision-making.

---

## Project Structure

```
KAIM-WEEK-6/
├── .github/
│ └── workflows/ # GitHub Actions workflows
├── data/
│ ├── raw/ # Raw data (should never be modified)
│ └── processed/ # Processed/cleaned data (gitignored)
├── notebooks/
  │ └── 1_data_preprocessing_cfpb.ipynb # performs eda and narrative cleaning
│ └── README.md # Documentation for notebooks
├── scripts/
│ └── README.md # Documentation for scripts
├── src/
│ └── utils/ # Utility functions
│  │ └── data_loader.py # loads csv files
│ └── vectorize_cimplaints.py # chunking and vectorization
│ └── inspect_chroma.py # checks if chroma db is created
│ └── rag_pipeline.py # embedding and llm integration
│ └── app.py # llm with streamlit
│ └── README.md # Documentation for src
├── tests/
│ └── __init__.py # 
│ └── test_vectorization.py # test compaint vectorization and store creation
│ └── README.md # Testing documentation
├── vector_store/
│ └── chroma.sqlite3
├── .gitattributes
├── .gitignore
├── README.md # Main project documentation
└── requirements.txt # Python dependencies
```
---
## 🛠 Tools & Technologies

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn (visualization)  
- Jupyter Notebook  
- Git  

---

## Key Task Completed 

### ✅ Task 1: Exploratory Data Analysis and Data Preprocessing 

Loaded the CFPB complaint dataset and conducted initial EDA. Analyzed complaint volume by product and narrative length distribution. Filtered data to include five target products and non-empty narratives. Cleaned text by lowercasing, removing special characters, and stripping boilerplate to prepare for embedding.

### ✅ Task 2: Text Chunking, Embedding, and Vector Store Indexing 

Divided lengthy consumer complaint narratives into manageable text chunks to optimize embedding quality. Generated embeddings for each chunk using a SentenceTransformer model. Created a persistent ChromaDB vector store, ensuring previous collections are deleted to maintain consistency. Verified data integrity by testing chunking output, embedding shapes, and vector store indexing and querying functionality.

### ✅ Task 3: Building the RAG Core Logic and Evaluation 

Loaded the persistent Chroma vector store containing chunked complaint embeddings. Initialized a HuggingFace embedding retriever and integrated it with a high-performance LLM (LLaMA3-70B via Groq). Constructed a Retrieval-Augmented Generation (RAG) chain by combining a context-aware system prompt, the retriever, and the LLM. Enabled the chain to answer domain-specific questions using only the indexed complaint excerpts. Evaluated pipeline quality by running representative queries, inspecting retrieved documents and response accuracy, and measuring retrieval and generation latency.

🔗 Live App: [CrediTrust Complaint Assistant](https://crditrustcomplaintanalysis.streamlit.app/)

---

## ⚙️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/she-code/KAIM-WEEK-6
cd KAIM-WEEK-6
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
3. **Install dependencies:**

```bash
pip install -r requirements.txt

```
---
## Contributors
- Frehiwot Abebie
