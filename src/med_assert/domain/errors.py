"""Domain-level errors for PubMed collection."""


class ArticleMinerError(Exception):
    """Base error for the article miner package."""


class NcbiError(ArticleMinerError):
    """NCBI E-utilities returned an error or unusable response."""


class NcbiTransportError(NcbiError):
    """Network or HTTP-layer failure when calling NCBI."""


class NcbiRateLimitError(NcbiError):
    """NCBI indicated rate limiting (HTTP 429)."""


class MalformedResponseError(NcbiError):
    """Response body could not be parsed or failed validation."""


class ArticleParseError(ArticleMinerError):
    """A PubMed XML fragment could not be turned into a structured article."""
