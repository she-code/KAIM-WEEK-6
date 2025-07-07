# evaluate_rag.py

from rag_pipeline import generate_answer
import pandas as pd

# Step 1: Define representative questions
questions = [
    "How are credit card disputes usually handled?",
    "What are the most common issues with mortgage services?",
    "How long does it typically take to resolve a complaint?",
    "What kind of problems do customers have with student loans?",
    "Are there recurring issues with credit reporting?",
    "Do customers face problems after paying off loans?",
    "What are common complaints about debt collectors?",
    "How do customers describe auto loan issues?",
    "What kinds of errors do people report in credit reports?",
    "Are there complaints related to loan application denials?"
]

# Step 2: Run the RAG pipeline and collect outputs
results = []

for q in questions:
    print(f"Processing: {q}")
    try:
        answer, sources = generate_answer(q)
        results.append({
            "Question": q,
            "Generated Answer": answer,
            "Retrieved Sources": "\n---\n".join(sources[:2]),  # only show 2 for brevity
            "Quality Score": "",  # to be filled manually
            "Comments/Analysis": ""  # to be filled manually
        })
    except Exception as e:
        print(f"Error processing question: {q}\n{e}")
        results.append({
            "Question": q,
            "Generated Answer": "ERROR",
            "Retrieved Sources": "N/A",
            "Quality Score": "",
            "Comments/Analysis": f"Error: {e}"
        })

# Step 3: Save results to markdown-friendly table
df = pd.DataFrame(results)

# Output as Markdown
markdown_table = df.to_markdown(index=False)
with open("rag_evaluation.md", "w", encoding="utf-8") as f:
    f.write("# RAG Evaluation Table\n\n")
    f.write(markdown_table)

print("\n✅ Evaluation table saved to 'rag_evaluation.md'")
