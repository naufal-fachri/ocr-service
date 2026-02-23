from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="/home/naufal/ocr_service/.env", extra="ignore", env_file_encoding="utf-8"
    )

    # --- OCR Engine Configuration ---
    USER_DOC_ORIENTATION_CLASSIFY: bool = Field(
        default=True,
        description="Whether to use document orientation classification"
    )
    USER_DOC_UNWARPING: bool = Field(
        default=True,
        description="Whether to use document unwarping"
    )
    USER_TEXTLINE_ORIENTATION: bool = Field(
        default=True,
        description="Whether to use textline orientation correction"
    )
    OCR_DEVICE: str = Field(
        default="gpu",
        description="Device to run the OCR engine on (e.g., 'cpu' or 'cuda')"
    )
    OCR_PRECISION: str = Field(
        default="fp32",
        description="Precision mode for the OCR engine (e.g., 'fp32', 'fp16')"
    )
    TEXT_DETECTION_MODEL_NAME: str = Field(
        default="PP-OCRv5_mobile_det",
        description="Name of the text detection model to use in the OCR engine"
    )
    TEXT_RECOGNITION_MODEL_NAME: str = Field(
        default="PP-OCRv5_mobile_rec",
        description="Name of the text recognition model to use in the OCR engine"
    )
    POST_PROCESSING_CONFIG: dict = Field(
        default={"y_threshold": 10, "column_threshold": 0.3},
        description="Configuration for post-processing OCR results"
    )
    SET_IMG_SIZE_CONSTANT: bool = Field(
        default=False,
        description="Make every page has the same dimension before OCR"
    )

    # --- Redis Configuration ---
    REDIS_HOST: str = Field(
        description="Redis Host"
    )
    REDIS_PORT: int = Field(
        default=6380,
        description="Port number for connecting to the Redis server"
    )
    REDIS_PASSWORD: str = Field(
        description="Password for authenticating with the Redis server"
    )

    # --- RabbitMQ Configuration ---
    RABBITMQ_HOST: str = Field(
        description="RabbitMQ Host"
    )
    RABBITMQ_USERNAME: str = Field(
        description="Username for authenticating with RabbitMQ"
    )
    RABBITMQ_PASSWORD: str = Field(
        description="Password for authenticating with RabbitMQ"
    )
    RABBITMQ_PORT: int = Field(
        default=5672,
        description="Port number for connecting to RabbitMQ server"
    )
    RABBITMQ_VHOST: str = Field(
        default="/",
        description="Virtual host for RabbitMQ connection"
    )

settings = Settings()