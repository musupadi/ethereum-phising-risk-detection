"""
api/schemas.py
--------------
Pydantic request/response models.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """
    Either provide raw feature values (features dict) OR a row_id from
    data/processed/test.csv for fast demo mode.
    wallet is required in both cases.
    """
    wallet: str = Field(..., example="0xAbCd...1234")
    row_id: Optional[int] = Field(
        None,
        description="Row index in test.csv (0-based). If provided, features are loaded from file.",
        example=42,
    )
    features: Optional[Dict[str, Any]] = Field(
        None,
        description="Manual feature values. Used when row_id is None.",
    )


class AnalyzeResponse(BaseModel):
    wallet: str
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: str = Field(..., example="HIGH")
    tx_hash: Optional[str] = Field(None, description="Chain tx hash, if CHAIN_ENABLED=true")


class RiskRecord(BaseModel):
    wallet: str
    risk_score: int
    timestamp: int
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
