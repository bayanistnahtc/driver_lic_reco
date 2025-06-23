import os
import yaml

from typing import Dict, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


def get_config(config_path: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration file contents as a dictionary.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            if not isinstance(config, dict):
                raise ValueError(
                    "Configuration file does not contain a valid dictionary"
                )
            return config
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {config_path}") from e


class Settings(BaseSettings):
    config_dir: str = Field("../../app_configs")
    config_name: str = Field("app_config.yaml")
    config: Optional[Dict[str, str]] = {}
    triton_host: str = Field("triton.k8s-dev.taximaxim.com")
    triton_port_http: str = Field("8000")
    triton_port_grpc: str = Field("8001")


settings = Settings()
settings.config = get_config(os.path.join(settings.config_dir, settings.config_name))
