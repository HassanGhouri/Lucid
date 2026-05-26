from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

class Settings(BaseSettings):
    """
    Application settings for the Lucid ingestion pipeline.
    """

    app_name: str = "Lucid"

    # Ingestion
    max_tokens_per_chunk: int = 350
    overlap_paragraphs: int = 1

    # Retrieval
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "lucid_chunks"

    dense_embedding_model_checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_embedding_batch_size: int = 64
    dense_vector_size: int = 384
    sparse_embedding_model_checkpoint: str = "Qdrant/bm25"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"

    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 30
    rrf_k: int = 60

    cross_encoder_model_checkpoint: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 8

    # Generation
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    max_output_tokens: int = 500

    # System intelligence
    query_rewrite_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"

    # Tracing/Observability
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "Lucid"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    trace_max_text_chars: int = 1200

    # Concurrency
    workers: int = 2

    # Rate limiting (slowapi format: "<count>/<period>")
    rate_limit_ask: str = "20/minute"
    rate_limit_ingest: str = "20/hour"

    # Timeouts (seconds)
    openai_timeout_seconds: float = 45.0
    request_timeout_seconds: float = 90.0
    ingest_timeout_seconds: float = 600.0

    # CORS
    cors_allowed_origins: str = "*"

    # Temporary private uploads
    private_upload_ttl_hours: int = 24
    private_upload_cleanup_interval_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
