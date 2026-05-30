import os
import re
from src.error_logging import ValidationException
from src.logging_config import logger

class QueryValidation:
    @staticmethod
    def validate_query(query):

        # is query?
        if not query or not query.strip():
            return False, 'Query string cannot be empty or whitespace.'

        # pick max len from env file
        try:
            max_len = int(os.getenv('MAX_QUERY_LEN', 500))
        except ValueError:
            logger.warning('MAX_QUERY_LEN is not correctly set in env file, falling back to default (500). ')
            max_len = 500 # set fail

        if len(query) > max_len:
            return False, f'Query cannot exceed {max_len} characters.'

        injection_words = [
            r"ignore\s+(previous|above)\s+instruction",
            r"override\s+guardrail",
            r"system\s+prompt",
            r"you\s+are\s+now\s+a",
            r"act\s+as\s+an",
            r"dan\s+mode"
        ]

        for pattern in injection_words:
            if re.search(pattern, query.lower()):
                return False, ('**Security Block:** Malicious content/ injection pattern detected!')

        # everything good
        return True, ''

