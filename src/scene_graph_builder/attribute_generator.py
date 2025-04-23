import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass

from api import StreamGenerator
from utils import JSONParser

logger = logging.getLogger(__name__)


@dataclass
class AttributeGenerationConfig:
    model_name: str
    api_keys: List[str]
    system_prompt: str = ""
    max_concurrent_per_key: int = 300
    max_retries: int = 5
    output_file: str = "attributes.json"


class AttributeConceptGenerator:
    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1"]

    def __init__(self, config: AttributeGenerationConfig):
        self.config = config
        self.existing_data = self._load_existing_data()

    def _load_existing_data(self) -> List[Dict[str, Any]]:
        """Load existing data from output file if it exists"""
        try:
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_data(self, data: List[Dict[str, Any]]):
        """Save data to output file with atomic write"""
        temp_file = f"{self.config.output_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(temp_file).replace(self.config.output_file)

    async def _generate_concepts(
            self,
            prompts: List[str],
            validate_func: Optional[Callable[[str], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Generate attribute concepts from prompts using LLM"""
        generator = StreamGenerator(
            model_name=self.config.model_name,
            api_keys=self.config.api_keys,
            max_concurrent_per_key=self.config.max_concurrent_per_key,
            max_retries=self.config.max_retries,
            rational=self.config.model_name in self.SUPPORTED_REASONING_MODELS,
        )

        results = []
        async for result in generator.generate_stream(prompts, self.config.system_prompt, validate_func):
            if result is not None:
                results.append(result)
        return results

    def _validate_concept_response(self, response: str) -> Union[Dict[str, List[str]], bool]:
        """Validate attribute concept format"""
        parsed = JSONParser.parse(response)
        if not isinstance(parsed, dict):
            return False

        if "attributes" not in parsed:
            return False

        if not isinstance(parsed["attributes"], list):
            return False

        for attr in parsed["attributes"]:
            if not isinstance(attr, str):
                return False
        return parsed

    def _create_concept_prompts(
            self,
            objects: List[Dict[str, Any]],
            concepts_per_object: int
    ) -> List[str]:
        """Create prompts for generating attribute concepts"""
        return [
            f"""Generate a JSON response with exactly {concepts_per_object} attribute concepts for the object '{obj['name']}'. 
The response should follow this exact structure:
{{
    "object_id": {obj['id']},
    "object_name": "{obj['name']}",
    "attributes": ["concept1", "concept2", ...]
}}

## Requirements:
1. Provide exactly {concepts_per_object} distinct attribute concepts
2. Concepts should be relevant to the specific object
3. Return ONLY the JSON dictionary

## Object: {obj['name']}"""
            for obj in objects
        ]

    async def generate_attribute_concepts(
            self,
            input_data: List[Dict[str, Any]],
            concepts_per_object: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate attribute concepts for each object"""
        logger.info(f"Generating {concepts_per_object} attribute concepts per object")

        # Extract all objects from categories
        objects = []
        for category in input_data:
            for obj in category["objects"]:
                objects.append({
                    "id": obj["id"],
                    "name": obj["name"],
                    "category_id": category["category_id"],
                    "category_name": category["category_name"]
                })

        prompts = self._create_concept_prompts(objects, concepts_per_object)
        results = await self._generate_concepts(prompts, self._validate_concept_response)

        output_data = self._organize_results(input_data, results)
        self._save_data(output_data)

        return output_data

    def _organize_results(
            self,
            input_data: List[Dict[str, Any]],
            concept_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Organize concept results into the original category structure"""
        concept_map = {res["object_id"]: res["attributes"] for res in concept_results}

        output_data = []
        for category in input_data:
            category_entry = {
                "category_id": category["category_id"],
                "category_name": category["category_name"],
                "objects": []
            }

            for obj in category["objects"]:
                obj_entry = {
                    "id": obj["id"],
                    "name": obj["name"],
                    "attributes": {}
                }

                if obj["id"] in concept_map:
                    # Initialize empty lists for each attribute concept
                    for concept in concept_map[obj["id"]]:
                        obj_entry["attributes"][concept] = []

                category_entry["objects"].append(obj_entry)

            output_data.append(category_entry)

        return output_data


class AttributeValueGenerator(AttributeConceptGenerator):
    def _validate_value_response(self, response: str) -> Union[Dict[str, List[str]], bool]:
        """Validate attribute value format"""
        parsed = JSONParser.parse(response)
        if not isinstance(parsed, dict):
            return False

        required_keys = {"object_id", "object_name", "attribute", "values"}
        if not all(k in parsed for k in required_keys):
            return False

        if not isinstance(parsed["values"], list):
            return False

        for value in parsed["values"]:
            if not isinstance(value, str):
                return False
        return parsed

    def _create_value_prompts(
            self,
            objects: List[Dict[str, Any]],
            values_per_concept: int
    ) -> List[str]:
        """Create prompts for generating attribute values"""
        prompts = []
        for obj in objects:
            for attribute in obj["attributes"]:
                prompts.append(
                    f"""Generate a JSON response with exactly {values_per_concept} possible values for the attribute '{attribute}' 
of object '{obj['name']}'. The response should follow this exact structure:
{{
    "object_id": {obj['id']},
    "object_name": "{obj['name']}",
    "attribute": "{attribute}",
    "values": ["value1", "value2", ...]
}}

## Requirements:
1. Provide exactly {values_per_concept} distinct values
2. Values should be appropriate for the specific object and attribute
3. Use common, realistic values
4. Return ONLY the JSON dictionary

## Object: {obj['name']}
## Attribute: {attribute}"""
                )
        return prompts

    async def generate_attribute_values(
            self,
            input_data: List[Dict[str, Any]],
            values_per_concept: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate attribute values for each concept"""
        logger.info(f"Generating {values_per_concept} values per attribute concept")

        objects = []
        for category in input_data:
            for obj in category["objects"]:
                if obj.get("attributes"):
                    objects.append({
                        "id": obj["id"],
                        "name": obj["name"],
                        "attributes": list(obj["attributes"].keys())
                    })

        prompts = self._create_value_prompts(objects, values_per_concept)
        results = await self._generate_concepts(prompts, self._validate_value_response)

        self._update_with_values(input_data, results)
        self._save_data(input_data)

        return input_data

    def _update_with_values(
            self,
            input_data: List[Dict[str, Any]],
            value_results: List[Dict[str, Any]]
    ):
        """Update the input data structure with generated attribute values"""
        value_map = {}
        for res in value_results:
            key = (res["object_id"], res["attribute"])
            value_map[key] = res["values"]

        for category in input_data:
            for obj in category["objects"]:
                if "attributes" in obj:
                    for attribute in list(obj["attributes"].keys()):
                        key = (obj["id"], attribute)
                        if key in value_map:
                            obj["attributes"][attribute] = value_map[key]