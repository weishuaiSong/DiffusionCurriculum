import os
import yaml
from typing import Dict, Any

def load_config(config_file: str, difficulty: int) -> Dict[str, Any]:
    """Load config file, update paths with specified difficulty, and ensure directories exist
    
    Args:
        config_file: Path to config file
        difficulty: Difficulty level (3-22)
    
    Returns:
        Updated config dictionary
    """
    if not 3 <= difficulty <= 22:
        raise ValueError(f"Difficulty must be between 3 and 22, got {difficulty}")
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Config should be a YAML dictionary")
            
            difficulty_str = f"difficulty_{difficulty}"
            
            def update_and_ensure_path(path: str) -> str:
                """Update difficulty in path and ensure directory exists"""
                if not path:
                    return path
                
                # Update difficulty in path
                new_path = (path.replace("difficulty_3", difficulty_str)
                              .replace("diff3", f"diff{difficulty}"))
                
                # Extract directory path and create if needed
                dir_path = os.path.dirname(new_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"Created directory: {dir_path}")
                
                return new_path
            
            # Update and ensure paths exist
            if "input_file" in config:
                config["input_file"] = update_and_ensure_path(config["input_file"])
            if "output_file" in config:
                config["output_file"] = update_and_ensure_path(config["output_file"])
            
            config["difficulty"] = difficulty_str
            return config
            
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Invalid config file: {e}")
    except OSError as e:
        raise ValueError(f"Failed to create directories: {e}")
                     
def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts from scene graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_file",
        help="Path to the main YAML configuration file (configs/prompts.yaml)"
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
        # Process all difficulty levels from 3 to 22
        for difficulty in range(3, 23):
            logger.info(f"Processing difficulty level: {difficulty}")
            
            # Load config with updated difficulty
            main_config = load_config(args.config_file, difficulty)

            keys_path = main_config.get("keys_path", "configs/keys.yaml")
            with open(keys_path, "r", encoding="utf-8") as f:
                api_keys = yaml.safe_load(f).get("keys", [])

            input_file = main_config.get("input_file")
            if input_file is None:
                raise ValueError("input_file not specified in the config file")
            output_file = main_config.get("output_file")
            if output_file is None:
                raise ValueError("output_file not specified in the config file")
            difficulty_str = main_config.get("difficulty", f"difficulty_{difficulty}")
            model_name = main_config.get("model_name", "deepseek-v3-0324")
            max_concurrent_per_key = main_config.get("max_concurrent_per_key", 300)
            max_retries = main_config.get("max_retries", 5)
            system_prompt = main_config.get("system_prompt", "")

            config = PromptGenerationConfig(
                model_name=model_name,
                api_keys=api_keys,
                system_prompt=system_prompt,
                output_file=output_file,
                max_concurrent_per_key=max_concurrent_per_key,
                max_retries=max_retries
            )

            generator = PromptGenerator(config)
            scene_graphs = load_scene_graphs(input_file, difficulty_str)

            async def run_generation():
                return await generator.generate_prompts(scene_graphs)

            results = asyncio.run(run_generation())

            for i, result in enumerate(results):
                scene_graphs[i]["prompt"] = result.get("prompt", "")

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                output_data = {difficulty_str: scene_graphs}
                json.dump(output_data, f, indent=2)

            logger.info(f"Prompt generation for {difficulty_str} completed. Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()