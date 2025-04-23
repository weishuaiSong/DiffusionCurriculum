import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass

from api import StreamGenerator
from utils import JSONParser

logger = logging.getLogger(__name__)


@dataclass
class RelationGenerationConfig:
    model_name: str
    api_keys: List[str]
    system_prompt: str = ""
    max_concurrent_per_key: int = 300
    max_retries: int = 5
    output_file: str = "relations.json"


class RelationGenerator:
    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1"]

    def __init__(self, config: RelationGenerationConfig):
        self.config = config
        self.existing_data = self._load_existing_data()

    def _load_existing_data(self) -> Dict[str, Any]:
        """Load existing data from output file if it exists"""
        try:
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "object-independent-relations": [],
                "object-dependent-relations": []
            }

    def _save_data(self, data: Dict[str, Any]):
        """Save data to output file with atomic write"""
        temp_file = f"{self.config.output_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        Path(temp_file).replace(self.config.output_file)

    async def _generate_relations(
            self,
            prompts: List[str],
            validate_func: Optional[Callable[[str], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Generate relations from prompts using LLM"""
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

    def _validate_independent_relation(self, response: str) -> Union[Dict[str, List[str]], bool]:
        """Validate object-independent relation format"""
        parsed = JSONParser.parse(response)
        if not isinstance(parsed, dict):
            return False

        for rel_type, examples in parsed.items():
            if not isinstance(examples, list):
                return False
            for example in examples:
                if not isinstance(example, str):
                    return False
        return parsed

    def _validate_dependent_relation(self, response: str) -> Union[Dict[str, Any], bool]:
        """Validate object-dependent relation format"""
        required_keys = {"category1", "category2", "relations"}
        parsed = JSONParser.parse(response)

        if (
                isinstance(parsed, dict)
                and all(k in parsed for k in required_keys)
                and isinstance(parsed["relations"], list)
                and all(isinstance(r, str) for r in parsed["relations"])
        ):
            return parsed
        else:
            return False

    def _create_independent_prompts(
            self,
            relation_types: List[str],
            examples_per_type: int
    ) -> List[str]:
        """Create detailed prompts for independent relations"""
        return [
            f"""Generate a JSON dictionary containing exactly {examples_per_type} examples for the 
relation type '{relation_type}'. The dictionary should have this structure:
{{"{relation_type}": ["example1", "example2", ...]}}

## Requirements:
1. Provide exactly {examples_per_type} distinct examples
2. Examples should be generally applicable to any objects
3. Return ONLY the JSON dictionary

## Relation type: {relation_type}
Examples should describe: {self._get_relation_description(relation_type)}"""
            for relation_type in relation_types
        ]

    def _get_relation_description(self, relation_type: str) -> str:
        """Get description for different relation types"""
        descriptions = {
            "positional": "spatial relationships between objects (e.g., near, far from)",
            "spatial": "directional/orientation relationships (e.g., above, below, inside)",
            "temporal": "time-based relationships (e.g., before, after, during)",
        }
        return descriptions.get(relation_type, "general relationships between objects")

    def _create_dependent_prompts(
            self,
            category_pairs: List[Tuple[str, str]],
            examples_per_pair: int
    ) -> List[str]:
        """Create detailed prompts for dependent relations"""
        return [
            f"""Generate a JSON response with {examples_per_pair} possible relations between 
'{category1}' and '{category2}'. Follow this exact structure:
{{
    "category1": "{category1}",
    "category2": "{category2}",
    "relations": ["relation1", "relation2", ...]
}}

## Requirements:
1. Provide exactly {examples_per_pair} distinct relations
2. Relations should be specific to these object types
3. Use short verb phrases where appropriate (e.g., 'holding', 'looking at'), without any explanations
4. Relations should be plausible real-world interactions
5. Return ONLY the JSON dictionary

Example interactions between {category1} and {category2}:"""
            for category1, category2 in category_pairs
        ]

    async def generate_object_independent_relations(
            self,
            relation_types: List[str],
            examples_per_type: int = 5
    ) -> List[Dict[str, List[str]]]:
        """Generate object-independent relations"""
        logger.info(f"Generating {examples_per_type} examples for each relation type: {relation_types}")

        prompts = self._create_independent_prompts(relation_types, examples_per_type)
        results = await self._generate_relations(prompts, self._validate_independent_relation)

        # Update and save data
        self.existing_data["object-independent-relations"] = results
        self._save_data(self.existing_data)

        return results

    async def generate_object_dependent_relations(
            self,
            categories: List[str],
            examples_per_pair: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate relations between category pairs"""
        logger.info(f"Generating relations for {len(categories)} categories")

        # Generate all unique ordered pairs
        pairs = [(c1, c2) for i, c1 in enumerate(categories) for c2 in categories[i + 1:]]

        prompts = self._create_dependent_prompts(pairs, examples_per_pair)
        results = await self._generate_relations(prompts, self._validate_dependent_relation)
        #print(results)

        # Update and save data
        self.existing_data["object-dependent-relations"] = results
        self._save_data(self.existing_data)

        return results