import argparse
import asyncio
import json
import logging
from typing import Dict, Any

import yaml
from template_generator import TemplateGenerator, TemplateGenerationConfig
from utils import setup_logger


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


async def generate_and_process_templates(generator: TemplateGenerator, config: Dict[str, Any]):
    """Generate and process templates based on configuration"""
    # Generate meta-templates
    meta_templates_count = config.get("meta_templates_count", 10)
    logger.info(f"Generating {meta_templates_count} meta-templates...")
    
    meta_templates_results = await generator.generate_meta_templates(meta_templates_count)
    
    if not meta_templates_results:
        logger.error("Failed to generate meta-templates")
        return
    
    # Extract templates from the first result (since we expect only one result from the prompt)
    meta_templates = {"templates": []}
    for result in meta_templates_results:
        if "templates" in result and isinstance(result["templates"], list):
            meta_templates["templates"].extend(result["templates"])
    
    logger.info(f"Successfully generated {len(meta_templates['templates'])} meta-templates")
    
    # Generate placeholder options
    logger.info("Generating placeholder options...")
    enhanced_templates = await generator.generate_placeholder_options(meta_templates)
    
    # Generate concrete templates
    logger.info("Generating concrete templates...")
    concrete_templates = generator.generate_concrete_templates(enhanced_templates)
    
    logger.info(f"Successfully generated {len(concrete_templates)} concrete templates")
    
    # Save results
    output_file = config.get("concrete_templates_output", "concrete_templates.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "meta_templates": enhanced_templates,
            "concrete_templates": concrete_templates
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Templates saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate VQA templates",
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
    global logger
    logger = logging.getLogger(__name__)

    try:
        # Load main configuration
        main_config = load_config(args.config_file)

        # Load API keys from the specified keys file
        keys_path = main_config.get("keys_path", "configs/keys.yaml")
        with open(keys_path, "r", encoding="utf-8") as f:
            api_keys = yaml.safe_load(f).get("keys", [])

        # Create generator config
        config = TemplateGenerationConfig(
            model_name=main_config.get("model_name", "deepseek-v3-0324"),
            api_keys=api_keys,
            system_prompt=main_config.get("system_prompt", ""),
            output_file=main_config.get("output_file", "templates.json"),
            max_concurrent_per_key=main_config.get("max_concurrent_per_key", 300),
            max_retries=main_config.get("max_retries", 5)
        )

        generator = TemplateGenerator(config)
        
        # Run the template generation
        asyncio.run(generate_and_process_templates(generator, main_config))
        
    except Exception as e:
        logger.error(f"Template generation failed: {e}")
        raise


if __name__ == "__main__":
    main()