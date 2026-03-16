"""
Shared Cypher query safety utilities.

Provides a compiled regex for detecting write keywords in Cypher queries.
Used by both the MCP tools layer and the web API routes to enforce
read-only query execution.
"""

from re import DOTALL, IGNORECASE, MULTILINE
from re import compile as re_compile

_COMMENT_PATTERN = re_compile(r"//.*?$|/\*.*?\*/", MULTILINE | DOTALL)

WRITE_KEYWORDS = re_compile(
    r"\b(DELETE|DROP|CREATE|SET|REMOVE|MERGE|DETACH|INSTALL|LOAD|COPY)\b",
    IGNORECASE,
)


def sanitize_cypher(query: str) -> str:
    """Strip comments from a Cypher query before safety checking."""
    return _COMMENT_PATTERN.sub("", query)
