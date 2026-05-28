import logging

# RAGException: base
# RetrievalException: vector retrieval from pinecone
# LLMException: openai issues
# ValidationException: issues with query inputs
class RAGException(Exception):
    def __init__(self, message, metadata=None):
        self.message = message
        self.metadata = metadata or {}

        super().__init__(message)
        log_msg = f'{self.__class__.__name__}: {self.message}'
        if metadata:
            log_msg += f' | Context metadata : {self.message}'

        logger = logging.error(log_msg)

class RetrievalException(RAGException):
    def __init__(self, message, metadata=None, index_name=True):
        meta = metadata or {}
        if index_name:
            meta['targeted_index'] = index_name
        super().__init__(message, metadata=meta)


class LLMException(RAGException):
    def __init__(self, message, metadata=None, provider=True,status_code=True):
        meta = metadata or {}
        meta.update({'llm_provider': provider, 'http_status': status_code})
        super().__init__(message, metadata=meta)


class ValidationException(RAGException):
    def __init__(self, message, metadata=None, validation_field=None):
        meta = metadata or {}
        if validation_field:
            meta['invalid_field'] = validation_field
        super().__init__(message, metadata=meta)