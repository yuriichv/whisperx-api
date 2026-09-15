"""Contract tests for models/health HTTP Pydantic schemas."""

from whisperx_api.models_router import get_models_list
from whisperx_api.schemas.health import HealthResponse
from whisperx_api.schemas.models import ModelObject, ModelsListResponse


def test_models_list_response():
    models = get_models_list()
    response = ModelsListResponse(data=models)
    assert response.object == "list"
    assert all(isinstance(m, ModelObject) for m in response.data)


def test_health_response():
    body = HealthResponse(model_loaded=True, device="cuda", compute_type="float16")
    assert body.model_dump() == {
        "model_loaded": True,
        "device": "cuda",
        "compute_type": "float16",
    }
