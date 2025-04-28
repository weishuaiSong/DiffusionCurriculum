import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.api import StreamGenerator
from utils import JSONParser

logger = logging.getLogger(__name__)


@dataclass
class QAGenerationConfig:
    model_name: str
    api_keys: List[str]
    difficulty: str
    system_prompt: str = ""
    max_concurrent_per_key: int = 300
    max_retries: int = 5
    output_file: str = "qa.json"
    num_qa_obj: int = 2  # object类型QA对数量
    num_qa_att: int = 2  # attribute类型QA对数量
    num_qa_rel: int = 2  # relation类型QA对数量
    num_qa_yes: int = 1  # 每种问题类型中answer为yes的QA对数量


class QAGenerator:
    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1"]

    def __init__(self, config: QAGenerationConfig):
        self.config = config
        self.existing_data = self._load_existing_data()

    def _load_existing_data(self) -> Dict[str, Any]:
        """Load existing data from output file if it exists"""
        try:
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"difficulty_xxx": []}

    def _save_data(self, data: Dict[str, Any]):
        """Save data to output file with atomic write in jsonl format"""
        temp_file = f"{self.config.output_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            for difficulty, items in data.items():
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        Path(temp_file).replace(self.config.output_file)

    async def _generate_qa_stream(
            self,
            prompts_with_index: List[tuple],
            validate_func: Optional[Callable[[str], bool]] = None
    ) -> List[tuple]:
        """Generate QA pairs from given prompts using LLM"""
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
        return results

    def _validate_qa(self, response: str) -> Union[Dict[str, Any], bool]:
        """Validate QA pair format"""
        parsed = JSONParser.parse(response)
        if isinstance(parsed, dict) and "qa" in parsed:
            qa_dict = parsed["qa"]
            required_keys = {"object", "attribute", "relation"}
            if not all(key in qa_dict for key in required_keys):
                return False

            # 验证所有答案都是yes/no
            for qa_type in required_keys:
                for qa in qa_dict[qa_type]:
                    answer = qa.get("answer", "").lower()
                    if answer not in {"yes", "no"}:
                        return False
            return parsed
        return False

    def _validate(self, qa_dict: Dict[str, List[Dict[str, str]]]) -> bool:
        """Validate the number of generated QA pairs and the number of 'yes' answers"""
        required_keys = {"object", "attribute", "relation"}
        for key in required_keys:
            if len(qa_dict[key]) != getattr(self.config, f"num_qa_{key[:3]}"):
                return False
            yes_count = sum(1 for qa in qa_dict[key] if qa["answer"].lower() == "yes")
            if yes_count != self.config.num_qa_yes:
                return False
        return True

    def _create_qa_prompts(
            self,
            scene_graphs: List[Dict]
    ) -> List[tuple]:
        """Create prompts for generating QA pairs based on scene graphs"""
        generated_prompts = []

        for index, scene_graph in enumerate(scene_graphs):
            scene_graph_json = json.dumps(scene_graph["scene_graph"], ensure_ascii=False, indent=2)

            # 提取所有对象名称和属性用于验证
            object_names = [obj["name"] for obj in scene_graph["scene_graph"]["objects"]]
            attributes = []
            for obj in scene_graph["scene_graph"]["objects"]:
                for attr in obj.get("attributes", []):
                    # attributes.append(f"{obj['name']}.{attr['concept']}={attr['value']}")
                    attributes.append(f"{obj['name']}.{attr}")

            relations = []
            for rel in scene_graph["scene_graph"].get("relations", []):
                relations.append(f"{rel['subject']} {rel['relation']} {rel['object']}")

            prompt = f"""Given the following scene graph in JSON format:
                    {scene_graph_json}

                    Generate EXACTLY {self.config.num_qa_obj} object questions, {self.config.num_qa_att} attribute questions, and {self.config.num_qa_rel} relation questions based STRICTLY on the provided scene graph.
                    For each question type, generate EXACTLY {self.config.num_qa_yes} questions with the answer 'yes'.

                    OBJECTS: {", ".join(object_names)}
                    ATTRIBUTES: {", ".join(attributes)}
                    RELATIONS: {", ".join(relations)}

                    RULES:
                    1. Questions MUST ONLY reference objects/attributes/relations that exist in the scene graph
                    2. All answers must be "yes" or "no" based on the scene graph content. Strive to balance the number of "yes" and "no" answers as much as possible while still adhering strictly to the scene graph information.
                    3. For attribute questions, use exact attribute values from the scene graph
                    4. For relation questions, use exact relation phrases from the scene graph

                    Output format (JSON ONLY):
                    {{
                    "qa": {{
                        "object": [
                        {{"question": "Is there a [VALID_OBJECT_NAME]?", "answer": "yes/no"}},
                        ...
                        ],
                        "attribute": [
                        {{"question": "Is the [ATTRIBUTE_NAME] of [OBJECT_NAME] [EXACT_VALUE]?", "answer": "yes/no"}},
                        ...
                        ],
                        "relation": [
                        {{"question": "Does [SUBJECT] [EXACT_RELATION] [OBJECT]?", "answer": "yes/no"}},
                        ...
                        ]
                    }}
                    }}

                    Example VALID questions for this scene:
                    - Object: "Is there a yacht?" (answer: "yes")
                    - Attribute: "Is the fur_color of wolf silver-gray?" (answer: "yes") 
                    - Relation: "Does photographer use camera?" (only if such relation exists in scene_graph)"""

            generated_prompts.append((index, prompt))

        return generated_prompts

    async def generate_qa(
            self,
            scene_graphs: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Generate QA pairs based on scene graphs"""
        logger.info(
            f"Generating QA pairs for {len(scene_graphs)} scene graphs with "
            f"obj={self.config.num_qa_obj}, att={self.config.num_qa_att}, rel={self.config.num_qa_rel}, yes={self.config.num_qa_yes}"
        )

        prompts_with_index = self._create_qa_prompts(scene_graphs)
        results = await self._generate_qa_stream(prompts_with_index, self._validate_qa)

        # 根据索引将结果正确放回原场景图数据中
        updated_scene_graphs = [{}] * len(scene_graphs)
        for index, result in results:
            if index < len(scene_graphs):
                scene_graph = scene_graphs[index].copy()
                qa_dict = result.get("qa", {})
                if self._validate(qa_dict):
                    scene_graph["qa"] = qa_dict
                    updated_scene_graphs[index] = scene_graph

        # 过滤掉空字典
        updated_scene_graphs = [graph for graph in updated_scene_graphs if graph]

        # 保存数据时使用配置中的复杂度
        output_data = {self.config.difficulty: updated_scene_graphs}
        self._save_data(output_data)

        return updated_scene_graphs

    