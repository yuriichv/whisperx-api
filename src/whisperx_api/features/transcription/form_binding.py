"""Привязка Pydantic wire-моделей к FastAPI multipart Form()."""

import inspect
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Form, HTTPException
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined


def build_form_command_dependency(
    form_cls: type[BaseModel],
    resolve_command: Callable[[BaseModel], Any],
) -> Callable[..., Any]:
    """Собрать FastAPI dependency: Form-поля из model_fields → resolve_command(form)."""
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}

    for name, field_info in form_cls.model_fields.items():
        description = field_info.description or ""
        ann = Annotated[field_info.annotation, Form(description=description)]
        annotations[name] = ann
        default = (
            field_info.default
            if field_info.default is not PydanticUndefined
            else None
        )
        parameters.append(
            inspect.Parameter(
                name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=ann,
            )
        )

    def dependency(**kwargs: Any) -> Any:
        try:
            form = form_cls.model_validate(kwargs)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        try:
            return resolve_command(form)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return_annotation = inspect.signature(resolve_command).return_annotation
    if return_annotation is inspect.Signature.empty:
        return_annotation = Any

    dependency.__name__ = f"resolve_{form_cls.__name__}"
    dependency.__annotations__ = {**annotations, "return": return_annotation}
    dependency.__signature__ = inspect.Signature(
        parameters,
        return_annotation=return_annotation,
    )
    return dependency
