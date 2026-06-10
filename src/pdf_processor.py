import os
import tempfile
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from src.logging_config import logger
from src.error_logging import  ValidationException
from src.config import config_inv
from src.validation import QueryValidation


class PDFProcessor:
    def __init__(self, chunk_size, chunk_overlap, llm_client):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_client = llm_client

    def _temp_pdf(self, file_bytes):
        # tmp_file_path = None
        # hf containers have a read-only root system, making /tmp the only open sandbox
        os.makedirs("/tmp", exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(dir="/tmp", delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
                # logger.info(f'Processing uploaded file: {file_name}')
                return tmp_file_path
        except Exception as e:
            raise ValidationException(
                message=f'Encountered a problem while uploading PDF file into the QGenAI. File may be corrupted or encrypted.',
                validation_field='uploaded_file',
                metadata={'original_error': e}
            )


    def _extract_citation(self, cover_text, fallback_name):
        if not cover_text.strip():
            return fallback_name.replace(".pdf", "").replace("_", " ")

        openai_api_key = config_inv.openai_api_key

        # llm_client = ChatOpenAI(
        #     model=config_inv.llm_model,
        #     temperature=config_inv.temperature,
        #     api_key=openai_api_key
        # )

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an expert digital archivist for an agricultural genomics library.\n"
                    "Analyze the provided text from the cover page of a scientific paper. "
                    "Extract the primary authors, the year of publication, and the scientific journal.\n\n"
                    "CRITICAL OUTPUT FORMATTING:\n"
                    "- If there are more than two authors, format exactly as: Lastname et al., Year (Journal)\n"
                    "- If there are exactly two authors, format exactly as: Author1 & Author2, Year (Journal)\n"
                    "- If there is only one author, format exactly as: Lastname, Year (Journal)\n"
                    "- Keep the journal name abbreviated if standard, or use its full title (e.g., Crop Science, Genetics).\n"
                    "- Output ONLY the final citation string. Do not include introductory text, markdown quotes, formatting wrappers, or pleasantries."
                )),
                ("human", "Cover Page Text:\n{text}")
            ])

            chain_response = prompt | self.llm_client | StrOutputParser()
            citation = chain_response.invoke({'text': cover_text[:3000]})
            return citation.strip().replace('"', '').replace("'", "")

        except Exception as e:
            logger.warning(
                f"Metadata extraction failed for {str(fallback_name)}, falling back to file name. Error: {str(e)}")
            return fallback_name.replace(".pdf", "").replace("_", " ")


    def process_pdf(self, file_bytes, file_name):

        tmp_pdf_path = self._temp_pdf(file_bytes)

        # try:
        loader = PyPDFLoader(str(tmp_pdf_path))
        pages = loader.load()

        if not pages:
            raise ValidationException(
                message=f'There are no readable text to extract in the uploaded PDF file.',
                validation_field='uploaded_file'

            )

        cover_text = pages[0].page_content if pages else ""

        # cover_page = pages[0].page_content[:3000] if pages else ''
        clean_citation = self._extract_citation(cover_text, file_name)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config_inv.chunk_size,
            chunk_overlap=config_inv.chunk_overlap
        )

        chunks = text_splitter.split_documents(pages)


        logger.info(f'Successfully generated chunks from the PDF file: {clean_citation}')

        for chunk in chunks:
            chunk.metadata['source'] = clean_citation

            raw_p = chunk.metadata.get('page', 0) + 1
            chunk.metadata['page'] = int(raw_p)

            # sanitize
            chunk.metadata = QueryValidation.sanitize_metadata(chunk.metadata)
        # print(chunks[0].metadata['source'])
        return chunks


        # except Exception as e:
        #     if not isinstance(e, ValidationException):
        #         raise ValidationException(
        #             message=f"Problem parsing uploaded PDF file structure: {str(e)}",
        #             validation_field="uploaded_file",
        #             metadata={"original_error": e}
        #         )
        #     raise e
        # finally:
        #     if tmp_pdf_path.exists():
        #         os.remove(tmp_pdf_path)




