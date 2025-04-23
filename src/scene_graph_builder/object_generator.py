import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass

from api import StreamGenerator
from utils import JSONParser

logger = logging.getLogger(__name__)

@dataclass
class ObjectGenerationConfig:
    model_name: str
    api_keys: List[str]
    system_prompt: str = ""
    max_concurrent_per_key: int = 300
    max_retries: int = 5
    output_file: str = "objects.json"
    objects_per_category: int = 5

class ObjectGenerator:
    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1"]

    def __init__(self, config: ObjectGenerationConfig):
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

    async def _generate_objects(
            self,
            prompts: List[str],
            validate_func: Optional[Callable[[str], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Generate objects from prompts using LLM"""
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

    def _validate_objects(self, response: str) -> Union[Dict[str, Any], bool]:
        """Validate generated objects format"""
        parsed = JSONParser.parse(response)
        if not isinstance(parsed, dict):
            return False

        required_keys = {"category_id", "category_name", "objects"}
        if not all(k in parsed for k in required_keys):
            return False

        if not isinstance(parsed["objects"], list):
            return False

        # Check we have exactly the requested number of objects
        if len(parsed["objects"]) != self.config.objects_per_category:
            return False

        # Validate each object and collect names for uniqueness check
        names = set()
        prev_id = -1

        for obj in parsed["objects"]:
            if not isinstance(obj, dict) or "id" not in obj or "name" not in obj:
                return False
            if not isinstance(obj["id"], int) or not isinstance(obj["name"], str):
                return False

            # Check ID is strictly increasing
            if obj["id"] <= prev_id:
                return False
            prev_id = obj["id"]

            if not obj["name"].strip():
                return False

            # Convert to lowercase and check for duplicates
            lower_name = obj["name"].lower().strip()
            if lower_name in names:
                return False
            names.add(lower_name)

        return parsed

    def _create_category_prompts(self, categories: List[str]) -> List[str]:
        """Create detailed prompts for object generation with category IDs"""
        return [
            f"""Generate exactly {self.config.objects_per_category} unique objects for the category '{category}'. 
    The response must be a JSON dictionary with this exact structure:
    {{
        "category_id": {idx + 1},  // Must be exactly {idx + 1}
        "category_name": "{category}",
        "objects": [
            {{"id": 1, "name": "object1"}},
            {{"id": 2, "name": "object2"}},
            ...
        ]
    }}

    ## Requirements:
    1. Assign category_id exactly {idx + 1}
    2. Provide exactly {self.config.objects_per_category} objects
    3. Each object must have a unique integer ID starting from 1
    4. IDs must be strictly increasing
    5. All object names must be unique (case-insensitive comparison)
    6. Object names should be common examples of the category
    7. Use lowercase names for consistency
    8. Return ONLY the JSON dictionary

    ## Category: {category}
    Example objects:"""
            for idx, category in enumerate(categories)
        ]

    async def generate_objects_for_categories(
            self,
            categories: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate objects for each category"""
        logger.info(f"Generating {self.config.objects_per_category} objects for {len(categories)} categories")

        prompts = self._create_category_prompts(categories)
        results = await self._generate_objects(prompts, self._validate_objects)

        # Update and save data
        self.existing_data = results
        self._save_data(self.existing_data)

        return results
