"""
Pydantic v2 Models for Every Pipeline Stage Boundary.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class StageTiming(BaseModel):
    """Execution timing and status metrics for an individual pipeline stage."""
    stage: str
    ms: float
    success: bool
    fallback_used: bool = False
    details: Optional[str] = None


class RetrievedChunk(BaseModel):
    """Schema for retrieved and re-ranked context chunk."""
    chunk_id: str
    text: str
    source_lang: str
    chunk_strategy: str
    dense_score: float
    bm25_score: Optional[float] = None
    final_score: float
    contributing_strategies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GuardrailFlags(BaseModel):
    """Structured audit log for all pre- and post-retrieval guardrail checks."""
    unsafe_detected: bool = False
    unsafe_reason: Optional[str] = None
    off_topic_detected: bool = False
    off_topic_distance: Optional[float] = None
    off_topic_reason: Optional[str] = None
    grounding_passed: bool = True
    grounding_score: Optional[float] = None
    grounding_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class QueryRequest(BaseModel):
    """API Request payload supporting audio file or direct text bypass."""
    audio_path: Optional[str] = None
    text: Optional[str] = None  # Text bypass for benchmark / testing
    language_hint: Optional[str] = None


class QueryResponse(BaseModel):
    """Standardized pipeline output schema."""
    query: str
    transcript: str
    language_detected: str
    answer: str
    answer_source: Literal["extractive", "generated", "declined"]
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    guardrail_flags: Dict[str, Any] = Field(default_factory=dict)
    stage_timings: List[StageTiming] = Field(default_factory=list)
    retrieval_ms: float = 0.0  # Isolated retrieval-stage latency (target: ~200ms)
    total_ms: float = 0.0  # Full end-to-end latency including STT/LLM
