from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WHISPERX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    default_model: str = "large-v3"
    default_language: str = ""  # empty => auto
    default_device: str = ""    # empty => auto
    default_compute_type: str = ""  # empty => auto
    api_token: str = "" # auth token for http 
    no_auth: bool = False # disable topken authentication

    log_level: str = "INFO"
    default_align: bool = False
    default_diarize: bool = False
    batch_size: int = 16
    debug: bool = False
    hf_token: str = "" # HUGGINGFACE_TOKEN for download dearize model


config = Config()
