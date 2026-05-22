import os
import pickle
import shutil
from os import mkdir
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_classic.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever


ROOT = Path(__file__).resolve().parent.parent

def generate_paper_title(pdf_path, llm, scan_depth_limit=5, char_slice=2500):
    """
    Gets actual paper title for one article. Takes first page and extract meta
    :param pdf_path: str path to df
    :param llm: chatopenai model
    :return: str - actual academic title of article
    """
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        if not pages:
            return pdf_path.name

        total_pages = len(pages)

        # 1. Dynamic depth: Scan up to the limit, or fewer if it's a short 3-page article
        actual_scan_depth = min(scan_depth_limit, total_pages)

        front_matter_chunks = []

        for i in range(actual_scan_depth):
            page_text = pages[i].page_content.strip()

            # Only include pages that actually contain readable text characters
            if page_text:
                # Take a slice of each page to prevent context bloating
                front_matter_chunks.append(f"--- START FRONT MATTER PAGE {i + 1} ---\n{page_text[:char_slice]}")

        # Combine the pages into a single contextual wall for the LLM metadata parser
        combined_context = "\n\n".join(front_matter_chunks)

        # cover_page = pages[0].page_content[:3000]

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert digital archivist for an agricultural genomics library.\n"
                "Analyze the provided text from the cover page of a scientific paper. "
                "Extract the primary authors, the year of publication, and the scientific journal.\n\n"
                "CRITICAL OUTPUT FORMATTING:\n"
                "- CRITICAL RULE 1: If the year, journal name, or author details are missing or unknown from the source metadata, DO NOT invent placeholders like 'Year', 'Journal', or request data from the user (e.g., do NOT say 'Please provide cover page text').\n"
                "- CRITICAL RULE 2: Fall back gracefully. If metadata is missing, simply use the clean filename and page number. Example: (Hallauer_Maize_Breeding.pdf, p. 593) or (Hallauer et al., Hallauer_Maize_Breeding.pdf, p. 12).\n"
                "- If there are more than two authors, format exactly as: Lastname et al., Year (Journal)\n"
                "- If there are exactly two authors, format exactly as: Author1 & Author2, Year (Journal)\n"
                "- If there is only one author, format exactly as: Lastname, Year (Journal)\n"
                "- Keep the journal name abbreviated if standard, or use its full title (e.g., Crop Science, Genetics).\n"
                "- Output ONLY the final citation string. Do not include introductory text, markdown quotes, formatting wrappers, or pleasantries."
            )),
            ("human", "Cover Page Text:\n{text}")
        ])
        #     ("system", (
        #         "You are an elite quantitative geneticist and AI co-investigator.\n\n"
        #         "INSTRUCTIONS FOR CITATIONS:\n"
        #         "- For every factual assertion directly sourced from the chunks, append a strict inline citation based on the chunk metadata provided.\n"
        #         "- FORMAT: (Author/Source, Year, Page) or (Filename, Page).\n"
        #         "- CRITICAL RULE 1: If the year, journal name, or author details are missing or unknown from the source metadata, DO NOT invent placeholders like 'Year', 'Journal', or request data from the user (e.g., do NOT say 'Please provide cover page text').\n"
        #         "- CRITICAL RULE 2: Fall back gracefully. If metadata is missing, simply use the clean filename and page number. Example: (Hallauer_Maize_Breeding.pdf, p. 593) or (Hallauer et al., Hallauer_Maize_Breeding.pdf, p. 12).\n"
        #         "- Never break character or output system errors inside citations."
        #     )),
        #     ("human", "{query}")
        # ])

        chain_response = prompt | llm | StrOutputParser()
        # print('here1')

        citation = chain_response.invoke({'text': combined_context})
        # print('here2')
        clean_citation = citation.strip().replace('"', '').replace("'", "")

        print(f'Citation generated! {clean_citation}')
        return clean_citation

    except Exception as e:
        print(f'Problem generating citation, defaulting to original filename {pdf_path.name}: {e}')
        return pdf_path.name


def build_and_save_vector(data_dir, chromadb_dir, bm25_dir):
    load_dotenv()

    pdf_files = list(data_dir.glob('*.pdf'))
    if not pdf_files:
        print(f'Error: No file found in {data_dir} ')
        return

    if chromadb_dir.exists():
        print('Clearing previous chromadb dir...')
        shutil.rmtree(chromadb_dir)

    # Path(ROOT / 'output/chromadb').mkdir(exist_ok=True)
    # mkdir(ROOT / 'output/chromadb')
    os.makedirs(chromadb_dir, exist_ok=True)
    os.makedirs(bm25_dir, exist_ok=True)

    get_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')

    split_chunk = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
        is_separator_regex=False
    )

    processed_chunks = []

    for pdf_file in pdf_files:

        clean_citation = generate_paper_title(pdf_file, get_llm) # keep for later

        loader = PyPDFLoader(str(pdf_file)) # gets actual pdf file name in data/
        full_docs = loader.load()

        chunks = split_chunk.split_documents(full_docs)

        for chunk in chunks:
            # replace the source with actual citation
            chunk.metadata['source'] = clean_citation

            # page is off by 1
            if 'page' in chunk.metadata:
                chunk.metadata['page'] = chunk.metadata['page'] + 1
            else:
                chunk.metadata['page'] = 1

        processed_chunks.extend(chunks)
        # print('Vectorization complete')

    # embed processed chunks (vector matrix) into chromadb
    Chroma.from_documents(
        documents=processed_chunks,
        embedding=embedding_model,
        persist_directory=str(chromadb_dir)
    )

    # vector_db = Chroma(
    #     persist_directory=str(chromadb_dir),
    #     embedding_function=embedding_model
    # )
    #
    # vector_db.add_documents(processed_chunks)

    print('starting bm25 retriever')
    # separate bm25 to keep hybrid
    pull_bm25 = BM25Retriever.from_documents(processed_chunks)

    # save bm25
    bm25_path = bm25_dir / 'bm25_index.pkl'
    with open(bm25_path, 'wb') as f:
        pickle.dump(pull_bm25, f)

    print(f'Success, Hybrid Indices Compiled!!')

    # return processed_chunks

if __name__ == '__main__':
    DATA_DIR = ROOT / 'data'
    CHROMADB_DIR = ROOT / 'output/chromadb'
    BM25_DIR = ROOT / 'output/bm25'
    build_and_save_vector(DATA_DIR, CHROMADB_DIR, BM25_DIR)

    print('Hybrid Index Built!')
