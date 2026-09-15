"""Pydantic-контракты health/readiness endpoints."""

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    status: str = Field(description="Статус probe: live | ready")


class HealthResponse(BaseModel):
    model_loaded: bool = Field(description="ASR-модель загружена в память")
    device: str | None = Field(
        default=None, description="Устройство inference (cpu/cuda)"
    )
    compute_type: str | None = Field(
        default=None, description="Тип вычислений whisperx (int8/float16)"
    )
