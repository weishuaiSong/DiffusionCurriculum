import argparse
import asyncio
import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any
import sys
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.promptqa_generator.qa_generator import QAGenerator, QAGenerationConfig
from utils import setup_logger


def load_scene_graphs(scene_graph_file: str, difficulty: str) -> List[Dict]:
    try:
        with open(scene_graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(difficulty, [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid scene graph file: {e}")


def load_config(config_file: str, difficulty: int) -> Dict[str, Any]:
    """Load config file and update paths with specified difficulty
    
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
            
            # Update paths with current difficulty
            difficulty_str = f"difficulty_{difficulty}"
            
            def update_and_ensure_path(path: str) -> str:
                """Update difficulty in path and ensure directory exists"""
                if not path:
                    return path
                
                # Update both formats
                new_path = path.replace("difficulty_3", difficulty_str) \
                            .replace("diff3", f"diff{difficulty}")
                
                # Extract directory path and create if needed
                dir_path = os.path.dirname(new_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    logging.info(f"Created directory: {dir_path}")
                
                return new_path
            # Update paths
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
        description="Generate QA pairs from scene graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "config_file",
        help="Path to the main YAML configuration file (configs/qa.yaml)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--start-difficulty",
        type=int,
        default=3,
        help="Starting difficulty level (3-22)"
    )
    parser.add_argument(
        "--end-difficulty",
        type=int,
        default=22,
        help="Ending difficulty level (3-22)"
    )

    args = parser.parse_args()

    setup_logger(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    try:
        # Validate difficulty range
        if not (3 <= args.start_difficulty <= args.end_difficulty <= 22):
            raise ValueError("Difficulty range must be between 3 and 22")
        
        # Process each difficulty level
        for difficulty in range(args.start_difficulty, args.end_difficulty + 1):
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing difficulty level: {difficulty}")
            logger.info(f"{'='*40}")
            
            try:
                # Load config with updated difficulty paths
                main_config = load_config(args.config_file, difficulty)
                
                # Load API keys
                keys_path = main_config.get("keys_path", "configs/keys.yaml")
                with open(keys_path, "r", encoding="utf-8") as f:
                    api_keys = yaml.safe_load(f).get("keys", [])
                
                # Get file paths
                input_file = main_config["input_file"]
                output_file = main_config["output_file"]
                difficulty_str = main_config["difficulty"]
                
                # Check if input file exists
                if not os.path.exists(input_file):
                    logger.warning(f"Input file not found: {input_file}. Skipping...")
                    continue
                
                # Load configuration parameters
                config = QAGenerationConfig(
                    model_name=main_config.get("model_name", "deepseek-v3-0324"),
                    api_keys=api_keys,
                    difficulty=difficulty_str,
                    system_prompt=main_config.get("system_prompt", ""),
                    output_file=output_file,
                    max_concurrent_per_key=main_config.get("max_concurrent_per_key", 300),
                    max_retries=main_config.get("max_retries", 5),
                    num_qa_obj=main_config.get("num_qa_obj", 2),
                    num_qa_att=main_config.get("num_qa_att", 2),
                    num_qa_rel=main_config.get("num_qa_rel", 2),
                    num_qa_yes=main_config.get("num_qa_yes", 1)
                )
                
                # Load scene graphs
                scene_graphs = load_scene_graphs(input_file, difficulty_str)
                if not scene_graphs:
                    logger.warning(f"No scene graphs found for {difficulty_str}. Skipping...")
                    continue
                
                # Generate QA pairs
                generator = QAGenerator(config)
                
                async def run_generation():
                    return await generator.generate_qa(scene_graphs)
                
                results = asyncio.run(run_generation())
                
                # Save results
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump({difficulty_str: results}, f, indent=2)
                
                logger.info(
                    f"Successfully generated QA pairs for {difficulty_str}\n"
                    f"Object QAs: {config.num_qa_obj}\n"
                    f"Attribute QAs: {config.num_qa_att}\n"
                    f"Relation QAs: {config.num_qa_rel}\n"
                    f"Saved to: {output_file}\n"
                )
                
            except Exception as e:
                logger.error(f"Error processing difficulty {difficulty}: {str(e)}")
                continue  # Continue with next difficulty level

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()