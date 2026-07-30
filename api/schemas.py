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


class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float = Field(
        ..., description="SHAP contribution to the phishing log-odds (+ raises risk, - lowers it).",
    )


class AnalyzeResponse(BaseModel):
    wallet: str
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: str = Field(..., example="HIGH")
    is_phishing: bool = Field(..., description="Binary verdict: probability >= 0.5.")
    verdict: str = Field(..., example="Phishing", description="'Phishing' or 'Bukan Phishing'.")
    top_contributors: Optional[list[FeatureContribution]] = Field(
        None, description="Top features driving the score (SHAP), explaining where the score comes from.",
    )
    tx_hash: Optional[str] = Field(None, description="Chain tx hash, if CHAIN_ENABLED=true")
    true_label: Optional[int] = Field(
        None,
        description="Ground-truth label from dataset (1=phishing, 0=normal) when the "
                    "wallet is found in the dataset; null for manual feature input.",
    )
    dataset_split: Optional[str] = Field(
        None, description="Which dataset split the wallet was found in (train/test), if any.",
    )


class RiskRecord(BaseModel):
    wallet: str
    risk_score: int
    timestamp: int
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
