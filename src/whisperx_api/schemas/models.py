"""Pydantic-контракты GET /v1/models."""

from pydantic import BaseModel, Field


class ModelObject(BaseModel):
    id: str = Field(description="Идентификатор модели (OpenAI-compatible)")
    object: str = Field(default="model", description="Тип объекта OpenAI API")
    created: int = Field(description="Unix timestamp создания записи")
    owned_by: str = Field(description="Владелец модели (local/openai)")


class ModelsListResponse(BaseModel):
    object: str = Field(default="list", description="Тип объекта OpenAI API")
    data: list[ModelObject] = Field(description="Список доступных моделей")
