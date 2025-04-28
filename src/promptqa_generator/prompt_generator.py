import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass
import sys
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.api import StreamGenerator
from utils import JSONParser

logger = logging.getLogger(__name__)


@dataclass
class PromptGenerationConfig:
    model_name: str
    api_keys: List[str]
    system_prompt: str = ""
    max_concurrent_per_key: int = 300
    max_retries: int = 5
    output_file: str = "prompts.json"


class PromptGenerator:
    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1"]

    def __init__(self, config: PromptGenerationConfig):
        self.config = config
        self.existing_data = self._load_existing_data()

    def _load_existing_data(self) -> Dict[str, Any]:
        """Load existing data from output file if it exists"""
        try:
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_data(self, data: Dict[str, Any]):
        """Save data to output file with atomic write"""
        temp_file = f"{self.config.output_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(temp_file).replace(self.config.output_file)

    async def _generate_prompts(
            self,
            prompts_with_index: List[tuple],
            validate_func: Optional[Callable[[str], bool]] = None
    ) -> List[tuple]:
        """Generate prompts from given prompts using LLM"""
        generator = StreamGenerator(
            model_name=self.config.model_name,
            api_keys=self.config.api_keys,
            max_concurrent_per_key=self.config.max_concurrent_per_key,
            max_retries=self.config.max_retries,
            rational=self.config.model_name in self.SUPPORTED_REASONING_MODELS,
        )

        results = []
        async for index, result in generator.generate_stream_with_index(prompts_with_index, self.config.system_prompt, validate_func):
            if result is not None:
                results.append((index, result))
        # 按索引对结果进行排序
        results.sort(key=lambda x: x[0])
        return results

    def _validate_prompt(self, response: str) -> Union[Dict[str, Any], bool]:
        """Validate prompt format"""
        parsed = JSONParser.parse(response)
        if isinstance(parsed, dict) and "prompt" in parsed:
            return parsed
        return False

    def _remove_uuid_like(self, text):
        """Remove UUID-like strings from text"""
        uuid_pattern = re.compile(r'\\u[0-9a-fA-F]{4}')
        return uuid_pattern.sub('', text)

    def _create_prompts(
            self,
            scene_graphs: List[Dict]
    ) -> List[tuple]:
        """Create detailed prompts based on scene graphs"""
        generated_prompts = []
        for index, scene_graph in enumerate(scene_graphs):
            # 直接使用场景图的JSON作为输入
            scene_graph_json = json.dumps(scene_graph["scene_graph"], ensure_ascii=False, indent=2)

            # prompt = f"""Given the following scene graph in JSON format:
            #             {scene_graph_json}

            #             Please craft a single, natural, smooth, and reasonable prompt that comprehensively describes the entire scene. Make sure to:
            #             1. Incorporate all the objects, their attributes, and relations
            #             2. Maintain logical consistency with the given scene graph
            #             3. Vary your sentence patterns for better quality
            #             4. Make the description vivid and engaging

            #             Only output the result as a JSON object with the key 'prompt' like {{"prompt": "Your generated prompt here"}}, and do not include any additional explanations or remarks."""
            
#             prompt=f"""Given the following scene graph in JSON format:
# {scene_graph_json}

# Please generate a single, natural, smooth, and reasonable prompt that comprehensively describes the entire scene. Strictly follow these requirements:
# 1. For all objects, attributes and relations mentioned, you MUST use the exact original words from the scene graph (only allowing minor grammatical adjustments like tense changes)
# 2. Ensuring 100% coverage of all relations, objects, and their attributes (nothing added or omitted)
# 3. With the given scene graph, maintain logical consistency while varying sentence structure for readability
# 4. Avoid unnecessary embellishments
# 5. Keep the generated prompt concise yet smooth

# Output only a JSON object with the key 'prompt' like {{"prompt": "Your generated prompt here"}}. Do not include any explanations or additional remarks. The generated prompt must strictly adhere to using the original words from the scene graph for all objects, attributes and relations."""
            
            prompt = f"""Given the following scene graph in JSON format:
{scene_graph_json}

Generate a description that EXACTLY and COMPLETELY covers all elements from the scene graph:
1. MUST include EVERY object, attribute (with exact values), and relation from the scene graph
2. MUST use the ORIGINAL terms from the scene graph (only allowing minimal grammatical adjustments)
3. MUST NOT add any information not present in the scene graph (no embellishments, interpretations, or assumptions)
4. Keep the generated prompt concise, logical and smooth
5. Vary sentence structures creatively while maintaining accuracy - use different phrasing patterns, active/passive voice alternation, and diverse clause arrangements

Output ONLY this JSON format with NO additional text:
{{"prompt": "Generated description here"}}"""
            # 移除可能的 UUID 类似内容
            clean_prompt = self._remove_uuid_like(prompt)
            generated_prompts.append((index, clean_prompt))
        return generated_prompts

    async def generate_prompts(
            self,
            scene_graphs: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate prompts based on scene graphs"""
        logger.info(f"Generating prompts for {len(scene_graphs)} scene graphs")

        prompts_with_index = self._create_prompts(scene_graphs)
        results = await self._generate_prompts(prompts_with_index, self._validate_prompt)

        # 根据索引将结果正确放回原场景图数据中
        updated_scene_graphs = [{}] * len(scene_graphs)
        for index, result in results:
            if index < len(scene_graphs):
                scene_graph = scene_graphs[index].copy()
                # 移除结果中的 UUID 类似内容
                if isinstance(result, dict) and "prompt" in result:
                    result["prompt"] = self._remove_uuid_like(result["prompt"])
                scene_graph["prompt"] = result.get("prompt", "")
                updated_scene_graphs[index] = scene_graph

        # 过滤掉空字典
        updated_scene_graphs = [graph for graph in updated_scene_graphs if graph]

        self.existing_data = {"difficulty": updated_scene_graphs}
        self._save_data(self.existing_data)

        return updated_scene_graphs
    