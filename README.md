# RAG Evaluation with RAGAS, Groq and Hugging Face

A hands-on Jupyter Notebook project that evaluates a **Retrieval-Augmented Generation (RAG)** example using four RAGAS metrics and an additional **derived, faithfulness-based unsupported-claim proxy**.

## Overview

The notebook uses:

- **RAGAS** (`ragas.metrics.collections`) to calculate evaluation scores.
- **Groq** through its OpenAI-compatible API, with `openai/gpt-oss-120b`, as the **evaluation LLM**.
- **Hugging Face sentence-transformers** (`all-MiniLM-L6-v2`) as local CPU embeddings for Answer Relevancy.
- **AsyncOpenAI**, `pandas`, and `python-dotenv`.

> **Scope:** This is a small evaluation demonstration using manually supplied questions, contexts, answers, and references. It does **not** yet retrieve documents from a vector database or generate answers with a separate RAG pipeline.

## Evaluation metrics

| Metric | Evaluates | In the notebook |
|---|---|---|
| **Faithfulness** | Whether claims in the generated answer are supported by retrieved context | `Faithfulness(llm=evaluator_llm)` |
| **Answer Relevancy** | Whether the generated answer addresses the user's query | `AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)` |
| **Context Recall** | Whether retrieved contexts cover the claims needed for the reference answer | `ContextRecall(llm=evaluator_llm)` |
| **Context Precision** | Whether relevant retrieved contexts are ranked ahead of irrelevant ones | `ContextPrecision(llm=evaluator_llm)` |
| **Derived unsupported-claim proxy** | `1 - faithfulness` for this evaluation | Calculated with pandas/Python; **not** a separate RAGAS hallucination metric |

Context Precision is ranking-aware. With **only one retrieved context**, this example cannot meaningfully demonstrate the effect of chunk ordering; use several relevant and irrelevant chunks to test it more thoroughly.

## Example data

**Question:** Within how many days of purchase can customers request a full software refund if unsatisfied?

**Retrieved context:** Customers who are unsatisfied with the software can request a full refund within 14 days of purchase.

**Generated answer:** Customers who are unsatisfied with the software can request a full refund within 14 days of purchase.

**Reference (`ground_truth`):** Customers who are unsatisfied with the software can request a full refund within 14 days of purchase.

The notebook evaluates this one sample. Its answer and reference match the retrieved context, so high scores may be expected, **but actual scores depend on the evaluator and must be measured**.

## Project structure

```text
rag-evaluation/
├── README.md
├── rag_evaluation.ipynb   # Replace with your actual notebook filename
├── .env
```

## Setup

**1. Create and activate a Python virtual environment:**

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

**2. Install dependencies:**

```bash
pip install ragas openai pandas python-dotenv sentence-transformers torch jupyter ipykernel
```

These are the main packages used in the supplied notebook. Since RAGAS interfaces can change between versions, record tested versions after running successfully:


**3. Create a `.env` file in the project folder:**

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Never commit `.env` or real API keys. Include this in `.gitignore`:

```gitignore
.env
.venv/
.ipynb_checkpoints/
__pycache__/
```

Optional `.env.example` (safe to commit):

```dotenv
GROQ_API_KEY=replace_with_your_key
```

**4. Run the notebook:**

```bash
jupyter notebook
```

Open your `.ipynb` file and execute its cells in order. The supplied code uses top-level `await`, which works in a typical Jupyter/IPython notebook.

## How the notebook works

1. Reads `GROQ_API_KEY` using `python-dotenv`.
2. Creates an asynchronous `AsyncOpenAI` client with Groq's OpenAI-compatible base URL.
3. Wraps the Groq-hosted `openai/gpt-oss-120b` model as the RAGAS evaluation LLM through `llm_factory`.
4. Initializes local `all-MiniLM-L6-v2` embeddings on CPU.
5. Defines a sample question, retrieved contexts, generated answer and reference answer.
6. Initializes the four RAGAS metrics.
7. Calls each metric's asynchronous `.ascore(...)` method and collects `.value`.
8. Calculates `hallucination_rate = 1 - faithfulness_score` as a **derived proxy**, not an independently validated hallucination measurement.
9. Displays sample scores and dataset-wide mean scores in a pandas DataFrame.
10. Optionally exports results to `rag_evaluation_results.csv` by uncommenting the CSV line.

### Example scoring calls

```python
faithfulness_result = await faithfulness_metric.ascore(
    user_input=question,
    response=answer,
    retrieved_contexts=contexts,
)

relevancy_result = await answer_relevancy_metric.ascore(
    user_input=question,
    response=answer,
)

recall_result = await context_recall_metric.ascore(
    user_input=question,
    retrieved_contexts=contexts,
    reference=reference,
)

precision_result = await context_precision_metric.ascore(
    user_input=question,
    retrieved_contexts=contexts,
    reference=reference,
)
```

## Results

The notebook produces columns similar to these:

| question | faithfulness | answer_relevancy | context_recall | context_precision | hallucination_rate |
|---|---:|---:|---:|---:|---:|
| Software refund policy question | *Run notebook* | *Run notebook* | *Run notebook* | *Run notebook* | *Run notebook* |

**No scores are fabricated in this README.** After running the notebook, replace the placeholders with your actual results or include a screenshot of the DataFrame.

The notebook also displays **average scores**. With the current dataset of only **one row**, the average of each metric equals that row's score; averages become meaningful when testing multiple cases.

## Interpretation and limitations

- **Faithfulness ≠ factual correctness:** an answer can be supported by a retrieved document even if that document is wrong.
- **Answer Relevancy ≠ answer correctness:** an answer can be on-topic but still incorrect.
- **Context Recall** here is reference-answer-based; its quality depends on the reference.
- **Context Precision** should be assessed on multi-chunk retrievals to demonstrate ranking quality.
- **`1 - faithfulness` is a convenience proxy** for the fraction of answer claims not supported by context under the metric's assumptions. It is **not** a universal hallucination rate and should be labeled accordingly.
- This is a **single-example demo**, not evidence of production performance. Include diverse examples, ambiguous queries, misleading contexts and failure cases for serious evaluation.
- LLM-based evaluation can vary with evaluator model, prompts, rate limits and library versions.

## Next steps

- Add a test dataset with multiple queries and several retrieved chunks per query.
- Include good and bad examples for each metric.
- Save the actual evaluation scores to CSV and show plots if helpful.
- Connect a real retriever/vector database and evaluate its output.
- Pin the working package versions in `requirements.txt`.

## References

- [RAGAS: Available metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
- [RAGAS: v0.3 to v0.4 metric collections migration](https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/)
- [RAGAS: Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)

---

**Suggested LinkedIn project title:** *Evaluating RAG Responses with RAGAS, Groq and Hugging Face Embeddings*
