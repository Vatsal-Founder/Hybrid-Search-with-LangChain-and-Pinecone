"""
RAGAS Evaluation for Hybrid Search RAG
=======================================
Runs the retrieval + generation pipeline against a test dataset
and computes RAGAS metrics: Faithfulness, Answer Relevancy,
and Context Precision (reference-free).

Usage:
    python -m eval.evaluate

Requirements:
    - PINECONE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY in .env
    - Papers already indexed in Pinecone (run searchapp.py first)
    - OPENAI_API_KEY is needed because RAGAS uses OpenAI as the
      evaluator LLM (judge) to score faithfulness and relevancy.

Output:
    - eval/results/eval_results.json  (per-question scores)
    - eval/results/eval_summary.json  (aggregate scores)
    - Console summary table
"""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_NAME = "hybrid-search-research"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-safeguard-20b"
EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def setup_retriever():
    """Initialize Pinecone retriever with existing index."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    bm25_path = Path("bm25_values.json")
    if not bm25_path.exists():
        raise FileNotFoundError(
            "bm25_values.json not found. Run searchapp.py and index papers first."
        )

    bm25_encoder = BM25Encoder().load(str(bm25_path))
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index,
    )
    return retriever


def setup_rag_chain(retriever):
    """Build the RAG chain (same as searchapp.py but without history).
    
    Uses a concise system prompt to keep responses short — this prevents
    RAGAS from hitting max_tokens when extracting statements.
    """
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model=LLM_MODEL,
    )

    # Keep answers concise to avoid RAGAS max_tokens issues
    system_prompt = (
        "You are a research assistant. Answer the question using the "
        "retrieved context below. Be concise — use 2-4 sentences maximum. "
        "If you don't know the answer, say so.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def run_pipeline(rag_chain, questions: list[dict]) -> list[dict]:
    """Run RAG pipeline on each question, collecting responses and contexts."""
    results = []

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['user_input'][:60]}...")

        response = rag_chain.invoke({"input": q["user_input"]})

        # Extract retrieved context strings
        retrieved_contexts = [
            doc.page_content for doc in response.get("context", [])
        ]

        results.append({
            "user_input": q["user_input"],
            "response": response.get("answer", ""),
            "retrieved_contexts": retrieved_contexts,
            "reference": q.get("reference", ""),
        })

    return results


def build_ragas_dataset(pipeline_results: list[dict]) -> EvaluationDataset:
    """Convert pipeline results into RAGAS EvaluationDataset."""
    samples = []
    for r in pipeline_results:
        samples.append(
            SingleTurnSample(
                user_input=r["user_input"],
                response=r["response"],
                retrieved_contexts=r["retrieved_contexts"],
                reference=r["reference"],
            )
        )
    return EvaluationDataset(samples=samples)


def main():
    print("=" * 60)
    print("RAGAS Evaluation — Hybrid Search RAG")
    print("=" * 60)

    # Validate env
    for key in ["PINECONE_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"]:
        if not os.getenv(key):
            raise EnvironmentError(f"Missing {key} in environment. Add to .env")

    # Load eval dataset
    print("\n1. Loading evaluation dataset...")
    with open(EVAL_DATASET_PATH) as f:
        eval_questions = json.load(f)
    print(f"   Loaded {len(eval_questions)} test questions")

    # Setup pipeline
    print("\n2. Setting up retriever & RAG chain...")
    retriever = setup_retriever()
    rag_chain = setup_rag_chain(retriever)

    # Run pipeline on eval questions
    print("\n3. Running pipeline on eval questions...")
    pipeline_results = run_pipeline(rag_chain, eval_questions)

    # Build RAGAS dataset
    print("\n4. Building RAGAS evaluation dataset...")
    ragas_dataset = build_ragas_dataset(pipeline_results)

    # Setup RAGAS evaluator LLM and embeddings
    print("\n5. Running RAGAS evaluation (this may take a few minutes)...")

    
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]

    ragas_result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-question results
    results_df = ragas_result.to_pandas()
    detailed_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
    results_df.to_json(detailed_path, orient="records", indent=2)

    # Aggregate summary
    summary = {
        "timestamp": timestamp,
        "num_questions": len(eval_questions),
        "metrics": {
            "faithfulness": float(results_df["faithfulness"].mean()),
            "answer_relevancy": float(results_df.filter(like="relevancy").iloc[:, 0].mean()),
            "context_precision": float(
                results_df["llm_context_precision_without_reference"].mean()
            ),
        },
    }

    print(results_df.columns.tolist()) 

    summary_path = RESULTS_DIR / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Questions evaluated:  {summary['num_questions']}")
    print(f"  Faithfulness:         {summary['metrics']['faithfulness']:.4f}")
    print(f"  Answer Relevancy:     {summary['metrics']['answer_relevancy']:.4f}")
    print(f"  Context Precision:    {summary['metrics']['context_precision']:.4f}")
    print(f"\n  Detailed results:     {detailed_path}")
    print(f"  Summary:              {summary_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()