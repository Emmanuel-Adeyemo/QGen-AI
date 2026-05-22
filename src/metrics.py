import os
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# LangChain Indexing Engine (For Running Queries Against Vector Store)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# DeepEval Native Framework
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,

)
from deepeval.evaluate import evaluate
from deepeval.evaluate import DisplayConfig
import deepeval

load_dotenv()


def get_metrics():
    ROOT = Path(__file__).resolve().parent.parent
    CHROMADB_DIR = ROOT / 'chromadb'
    BM25_PATH = ROOT / 'bm25_index.pkl'
    CSV_PATH = ROOT / 'output/metrics/deepeval_ground_truth_testset.csv'
    RESULTS_PATH = ROOT / 'output/metrics/final_deepeval_evaluation_metrics.csv'
    SUBSET_DIR = ROOT / 'subset'
    #
    print("Gathering source literature files for testset generation...")
    if not SUBSET_DIR.exists():
        raise FileNotFoundError(f"Source directory missing at {SUBSET_DIR}")

    pdf_paths = [str(p) for p in SUBSET_DIR.glob("*.pdf")]
    if not pdf_paths:
        raise ValueError(f"No PDF files discovered in target directory: {SUBSET_DIR}")

    print(f"Found {len(pdf_paths)} target files. Spinning up DeepEval Synthesizer...")

    context_config = ContextConstructionConfig(
        max_contexts_per_document=3,  # max logical chunks parsed out per file
        chunk_size=1024,
        chunk_overlap=200
    )
    #

    synthesizer = Synthesizer()

    print("Generating ground-truth QA evaluation pairs in parallel...")
    generated_goldens = synthesizer.generate_goldens_from_docs(
        document_paths=pdf_paths,
        max_goldens_per_context=1,  # generates 1 question scenario per chunk
        context_construction_config=context_config
    )


    records = []
    for g in generated_goldens:
        records.append({
            "user_input": g.input,
            "reference": g.expected_output,
            "context": "\n\n".join(g.context) if g.context else ""
        })
    #
    test_set_df = pd.DataFrame(records)
    # check dir exist
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_set_df.to_csv(CSV_PATH, index=False)
    print(f"Test set compiled successfully and saved to {CSV_PATH}!")

    # put questn into active vector loading
    print('\nLoading ground-truth test set for pipeline ingestion...')
    test_df = pd.read_csv(CSV_PATH)
    queries = test_df["user_input"].tolist()
    ground_truths = test_df["reference"].tolist()

    generated_answers = []
    retrieved_contexts_matrix = []

    print('Executing test queries against active RAG vector index...')
    if not CHROMADB_DIR.exists() or not BM25_PATH.exists():
        print("Warning: Vector index files missing! Verification metrics might drop.")

    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    vector_store = Chroma(persist_directory=str(CHROMADB_DIR), embedding_function=embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    llm_pipeline = ChatOpenAI(model='gpt-4o', temperature=0.0)

    for query in queries:
        retrieved_docs = vector_retriever.invoke(query)
        chunks_text = [doc.page_content for doc in retrieved_docs]
        retrieved_contexts_matrix.append(chunks_text)

        context_str = "\n\n".join(chunks_text)
        # prompt = f"Answer the query using only the provided context.\nContext:\n{context_str}\nQuery: {query}"
        #
        # response = llm_pipeline.invoke(prompt)
        # generated_answers.append(response.content)

        prompt = f"""You are a precise scientific assistant translating quantitative genetics literature. Analyze the provided context chunks to answer the user query.

      CRITICAL RULES FOR TRUTH & PRECISION:
        1. **No Property Bleeding:** If the text describes specific traits, locations, or values for one group (e.g., "dent"), do NOT assume or extrapolate that those same traits, locations, or values apply to any other group (e.g., "flint") unless explicitly stated.
        2. **Chunk Isolation:** Treat each chunk as an isolated fact. If a chunk contains irrelevant details, ignore it entirely. Do not attempt to merge unrelated data points into a cohesive narrative.
        3. **Handle Missing Specifics Safely:** If the query asks for a specific value or comparison at a certain point and the text only provides it for one trait, explicitly state what is present and specify exactly what is missing.
        4. **Strict Verbatim Grounding:** Do not guess, smooth over discrepancies, or generalize. Every name, location, and numeric assertion must map cleanly to an explicit statement in the context.

        OUTPUT STYLE:
        - State answers directly and concisely.
        - Avoid conversational filler, excessive nested bulleting, or repeating parenthetical qualifiers if a simple list or sentence captures the context perfectly.
        Context:
        {context_str}

        Query: {query}
        Answer:"""

        response = llm_pipeline.invoke(prompt)
        generated_answers.append(response.content)


    print('\nPacking evaluation parameters into DeepEval test blocks...')
    test_cases = []
    for i in range(len(queries)):
        test_case = LLMTestCase(
            input=queries[i],
            actual_output=generated_answers[i],
            expected_output=ground_truths[i],
            retrieval_context=retrieved_contexts_matrix[i]
        )
        test_cases.append(test_case)
    #

    print('Computing analytical scores via DeepEval Judge Engine...')


    faithfulness = FaithfulnessMetric(threshold=0.5, model="gpt-4o-mini")
    context_precision = ContextualPrecisionMetric(threshold=0.5, model="gpt-4o-mini")
    context_recall = ContextualRecallMetric(threshold=0.5, model="gpt-4o-mini")
    answer_relevancy = AnswerRelevancyMetric(threshold=0.5, model="gpt-4o-mini")
    # correctness = CorrectnessMetric(threshold=0.5, model="gpt-4o-mini")

    metrics_list = [
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy,
        # correctness
    ]

    display_config = DisplayConfig(
        verbose_mode=False
    )

    # run evaluation
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics_list,
        display_config=display_config
    )


    # evaluate(test_cases, metrics_list)
    #
    # test_run = deepeval.get_test_run()
    #

    summary_data = []
    for metric in metrics_list:
        summary_data.append({
            "Metric": metric.__class__.__name__,
            "Average Score": getattr(metric, 'score', None),
            "Total Cases": len(test_cases)
        })

    results_df = pd.DataFrame(summary_data)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nAggregated matrix successfully logged to {RESULTS_PATH}")

if __name__ == '__main__':
    get_metrics()