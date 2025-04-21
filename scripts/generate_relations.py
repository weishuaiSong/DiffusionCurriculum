import argparse
import asyncio
import json
import logging
from typing import List, Dict, Any

import yaml
from scene_graph_builder import RelationGenerator, RelationGenerationConfig
from utils import setup_logger


def load_categories(category_file: str) -> List[str]:
    """Load categories from JSON file with validation"""
    try:
        with open(category_file, "r", encoding="utf-8") as f:
            categories = json.load(f)
            if not isinstance(categories, list):
                raise ValueError("Category file should contain a JSON list")
            if not all(isinstance(c, str) for c in categories):
                raise ValueError("All categories should be strings")
            return categories
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid category file: {e}")


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate scene graph relations",
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

    # Setup logging
    setup_logger(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    try:
        # Load main configuration
        main_config = load_config(args.config_file)

        # Load API keys from the specified keys file
        keys_path = main_config.get("keys_path", "configs/keys.yaml")
        with open(keys_path, "r", encoding="utf-8") as f:
            api_keys = yaml.safe_load(f).get("keys", [])

        # Create generator config
        config = RelationGenerationConfig(
            model_name=main_config.get("model_name", "deepseek-v3-0324"),
            api_keys=api_keys,
            system_prompt=main_config.get("system_prompt", ""),
            output_file=main_config.get("output_file", "relations.json"),
            max_concurrent_per_key=main_config.get("max_concurrent_per_key", 300),
            max_retries=main_config.get("max_retries", 5)
        )

        generator = RelationGenerator(config)

        async def run_generation():
            mode = main_config.get("mode", "both")

            if mode in ["independent", "both"]:
                relation_types = main_config.get(
                    "independent_relation_types",
                    ["positional", "spatial", "temporal"]
                )
                await generator.generate_object_independent_relations(
                    relation_types=relation_types,
                    examples_per_type=main_config.get("examples_per_type", 5)
                )

            if mode in ["dependent", "both"]:
                category_file = main_config.get("category_file")
                if not category_file:
                    raise ValueError("category_file must be specified in config for dependent mode")
                categories = load_categories(category_file)
                await generator.generate_object_dependent_relations(
                    categories=categories,
                    examples_per_pair=main_config.get("examples_per_pair", 5)
                )

        asyncio.run(run_generation())
        logger.info(f"Relation generation completed. Results saved to {config.output_file}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()