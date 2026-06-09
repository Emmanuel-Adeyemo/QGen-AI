import os
from dataclasses import dataclass, field

from src.error_logging import ValidationException

def _safe_int(env_var, default_val):
    """helper function to cast into int with fallback safety."""
    val = os.getenv(env_var)
    if val is None:
        return default_val
    try:
        return int(val)
    except ValueError:
        return default_val

@dataclass(frozen=True)
class QGenConfig:

    pinecone_api_key: str = field(
        default_factory=lambda: os.getenv('PINECONE_API_KEY', '') or ""
    )
    pinecone_index: str = field(
        default_factory=lambda: os.getenv('PINECONE_INDEX', 'qgen-ai-index')
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv('OPENAI_API_KEY', '') or ""
    )


    # llm
    llm_model: str = field(default_factory=lambda: os.getenv('LLM_MODEL', 'gpt-4o-mini'))
    embeddings: str = field(default_factory=lambda: os.getenv('TEXT_EMBEDDINGS', 'text-embedding-3-large'))

    chunk_size: int = field(default_factory=lambda: _safe_int('CHUNK_SIZE', 2000))
    chunk_overlap: int = field(default_factory=lambda: _safe_int('CHUNK_OVERLAP', 400))
    temperature:int = field(default_factory=lambda: _safe_int('TEMPERATURE', 0))

    max_query_len: int = field(default_factory=lambda: _safe_int('MAX_QUERY_LEN', 500))
    max_size_lim: int = field(default_factory=lambda: _safe_int('MAX_SIZE_LIM', 52428800))


    def validate(self):
        missing_things = []

        if not self.pinecone_api_key.strip():
            missing_things.append('Pinecone_API_Key')
        if not self.openai_api_key.strip():
            missing_things.append('OpenAI_API_Key')

        if missing_things:
            raise ValidationException(
                message='Fast Boot Faliure: check Env file to confirm keys.',
                validation_field='ENVIRONMENT_BOOT'
            )

        if self.max_size_lim <= 0 :
            raise ValidationException(
                message='Sanity Check: File Size Max Limit must be positive non zero.',
            )
        if self.max_query_len <= 0:
            raise ValidationException(
                message='Sanity Check: Query Length must be positive non zero.'
            )

        if self.chunk_size < self.chunk_overlap:
            raise ValidationException(
                message='Sanity Check: Chunk size must be larger than overlap.'
            )

        if self.temperature < 0 or self.temperature > 1:
            raise ValidationException(
                message='Sanity Check: Model Temperature must be between 0 and 1.'
            )


try:
    config_inv = QGenConfig()
    config_inv.validate()
    config_valid = True
    config_error_msg = ''
except ValidationException as ve:
    config_inv = QGenConfig()
    config_valid = False
    config_error_msg = f'{ve.message}'
except Exception as e:
    config_inv = QGenConfig()
    config_valid = False
    config_error_msg = f'Unexpected error occured during fast boot: {str(e)}'




