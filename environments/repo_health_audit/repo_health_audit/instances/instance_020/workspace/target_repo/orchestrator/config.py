import os


def get_config() -> dict:
    return {
        "env": os.getenv("PIPELINE_ENV", "dev"),
        "engine_binary": os.getenv("ENGINE_PATH", "./engine/target/release/engine"),
        "s3_bucket": os.getenv("S3_BUCKET", "data-pipeline-dev"),
        "sqs_queue": os.getenv("SQS_QUEUE_URL", ""),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }


def is_production() -> bool:
    return get_config()["env"] == "prod"
