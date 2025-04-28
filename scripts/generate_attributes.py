import argparse
import asyncio
import json
import logging
from typing import List, Dict, Any

import yaml
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.scene_graph_builder import AttributeGenerator, AttributeGenerationConfig
from utils import setup_logger


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input data from JSON file with validation"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Input file should contain a JSON list of categories")

            required_category_keys = ["category_id", "category_name", "objects"]
            required_object_keys = ["id", "name"]

            for category in data:
                if not all(k in category for k in required_category_keys):
                    raise ValueError("Each category must have category_id, category_name, and objects")

                if not isinstance(category["objects"], list):
                    raise ValueError("Category objects must be a list")

                for obj in category["objects"]:
                    if not all(k in obj for k in required_object_keys):
                        raise ValueError("Each object must have id and name")

            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid input file: {e}")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration file with validation"""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Config should be a YAML dictionary")
            return config
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid config file: {e}")


async def run_generation(config: AttributeGenerationConfig, main_config: Dict[str, Any]):
    """Run the attribute generation pipeline with unified generator"""
    # Load input data
    input_file = main_config.get("input_file")
    if not input_file:
        raise ValueError("input_file must be specified in config")
    input_data = load_input_data(input_file)

    generator = AttributeGenerator(config)

    # Single-step generation that produces both concepts and values
    input_data = await generator.generate_attributes(
        input_data=input_data,
        concepts_per_object=main_config.get("concepts_per_object", 5),
        values_per_concept=main_config.get("values_per_concept", 5)
    )

    return input_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate object attributes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_file",
        help="Path to the main YAML configuration file"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    setup_logger(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    try:
        main_config = load_config(args.config_file)

        keys_path = main_config.get("keys_path", "configs/keys.yaml")
        with open(keys_path, "r", encoding="utf-8") as f:
            api_keys = yaml.safe_load(f).get("keys", [])

        config = AttributeGenerationConfig(
            model_name=main_config.get("model_name", "deepseek-v3-0324"),
            api_keys=api_keys,
            system_prompt=main_config.get("system_prompt", ""),
            output_file=main_config.get("output_file", "attributes.json"),
            max_concurrent_per_key=main_config.get("max_concurrent_per_key", 300),
            max_retries=main_config.get("max_retries", 5)
        )

        result = asyncio.run(run_generation(config, main_config))
        logger.info(f"Attribute generation completed. Results saved to {config.output_file}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()