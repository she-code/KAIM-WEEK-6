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
├── tests/
│ └── __init__.py # 
│ └── README.md # Testing documentation
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
