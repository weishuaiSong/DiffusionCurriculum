import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass

from api import StreamGenerator
from utils import JSONParser

from itertools import product
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


logger = logging.getLogger(__name__)

@dataclass
class TemplateGenerationConfig:
    model_name: str
    api_keys: List[str]
    system_prompt: str = ""
    max_concurrent_per_key: int = 300
    max_retries: int = 5
    output_file: str = "templates.json"


class TemplateGenerator:
    SUPPORTED_REASONING_MODELS = ["DeepSeek-R1"]
    def __init__(self, config: TemplateGenerationConfig):
        self.config = config
        self.existing_data = self._load_existing_data()

    def _load_existing_data(self) -> List[str]:
        """Load existing concrete templates from output file
        
        Returns:
            List of template strings preserving {{question}} and {{choices}} placeholders
            Returns empty list if file doesn't exist or has invalid format
        """
        try:
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Validate file format: must be a list of strings
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    return data
                    
                logger.warning("Invalid file format: Expected list of strings. Returning empty list.")
                return []
                
        except FileNotFoundError:
            logger.info(f"Template file {self.config.output_file} not found. Starting fresh.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading templates: {str(e)}")
            return []

    def _save_data(self, data: List[str]):
        """Save generated templates to file with atomic write
        
        Args:
            data: List of concrete template strings to save
        """
        try:
            # Create parent directories if not exist
            output_path = Path(self.config.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write using temporary file
            temp_file = f"{self.config.output_file}.tmp"
            with open(temp_file, "w", encoding="utf-8", newline="\n") as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True
                )
                
            # Replace original file atomically
            Path(temp_file).replace(output_path)
            logger.info(f"Successfully saved {len(data)} templates to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save templates: {str(e)}")
            raise


    async def _generate_response(
            self,
            prompts: List[str],
            validate_func: Optional[Callable[[str], bool]] = None
    ) -> List[Dict[str, Any]]:
        """Generate Meta_templates from prompts using LLM"""
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

    def _parse_json_from_response(self, response: str) -> dict:
        try:
            start_index = response.find('{')
            response = response[start_index:]
            return json.loads(response.encode('utf-8', errors='ignore'))
        except Exception as e:
            print("JSON parse error:", e)
            return {}
        
    def _validate_metatemplate_response(response: str) -> Union[Dict[str, List[Dict[str, Any]]], bool]:
        """Validate if the response is a valid VQA meta-template JSON structure."""
        parsed = JSONParser.parse(response)
        if not isinstance(parsed, dict):
            return False

        templates = parsed.get("templates")
        if not isinstance(templates, list):
            return False

        for item in templates:
            if not isinstance(item, dict):
                return False

            metatemplate = item.get("metatemplate")
            placeholders = item.get("placeholders")

            if not isinstance(metatemplate, str):
                return False
            if not isinstance(placeholders, list):
                return False

            if "{{question}}" not in metatemplate or "{{choices}}" not in metatemplate:
                return False

            if not all(isinstance(ph, str) for ph in placeholders):
                return False

            # Ensure placeholders list does not contain 'question' or 'choices'
            if "question" in placeholders or "choices" in placeholders:
                return False

        return parsed
    
    def _create_meta_templates_prompts(self,counts: int) -> List[str]:
        """
        Create a list of prompts for generating VQA meta templates.
        
        Args:
            counts (int): Number of prompts (each asking for a meta template).
        
        Returns:
            List[str]: List of formatted prompt strings.
        """
        prompt_template = f"""
        Please generate {counts} different instruction meta templates structures for Visual Question Answering (VQA) tasks that include multiple choice options.
        
        Each meta template should meet the following requirements:
        1. Clearly express the instruction to look at an image, understand a question, and select from provided choices
        2. Include the following two ESSENTIAL placeholders:
        - {{{{question}}}} for the specific question being asked
        - {{{{choices}}}} for the multiple choice options
        3. Use descriptive placeholder names inside curly braces, e.g., {{instruction_modifier}}, {{viewing_action}}, etc.
        4. Include varied instruction structures (direct commands, polite requests, question forms, etc.)
        
        Additional placeholders to consider(including but not limited to):
        - {{instruction_modifier}}
        - {{viewing_action}} 
        - {{image_type}} 
        - {{answering_action}} 
        - {{specificity_modifier}} 
        - {{choice_instruction}} 
        - etc
        ...

        Please return in JSON format as follows:
        {{
  "templates":
        [
            {{
                "metatemplate": "{{instruction_modifier}} {{viewing_action}} this {{image_type}} and {{answering_action}} {{{{question}}}}\\n\\nChoices:\\n{{{{choices}}}}",
                "placeholders": ["image_type", "instruction_modifier", "viewing_action", "answering_action"],
            }},
            {{
                "metatemplate": "Based on the {{image_type}}, {{choice_instruction}}:\\n{{{{question}}}}\\n{{{{choices}}}}",
                "placeholders": ["image_type", "choice_instruction"],
            }},
            {{
                "metatemplate": "When processing the {{image_type}}, first consider the available options: {{{{choices}}}}, then {{answering_action}} to the following query with {{specificity_modifier}}:\\n{{{{question}}}}. {{instruction_modifier}}",
                "placeholders": ["image_type","answering_action","specificity_modifier","instruction_modifier"]
            }},
            
            ...
        ]}}
        
        Ensure meta templates are diverse, covering different expression styles, formats.
        Make sure EVERY template includes all two essential elements: question, and choices,And don't add these two key in placeholders's list.
        Make sure you generate content that satisfies the json syntax and don't replace placeholders with actual words in the meta-template And there are only two types of keys, metatemplate and placeholders.
        """

        return [prompt_template]
    
    def _create_placeholder_options_prompt(self, meta_templates: Dict[str, Any]) -> str:
        """
        Create a prompt to generate placeholder options for meta-templates.
        
        Args:
            meta_templates (Dict[str, Any]): Dictionary containing meta-templates
            
        Returns:
            str: Formatted prompt string
        """
        base_prompt = json.dumps(meta_templates, indent=2)
        instruction = """
        
        The dictionary above contains templates with various placeholders. Please analyze each metatemplate and identify all unique placeholders (words surrounded by curly braces like {{placeholder}}).
        
        Create a new key called "placeholder_options" in the dictionary. For each unique placeholder you identified, create an entry in "placeholder_options" where the key is the placeholder name and the value is an array containing as many synonymous alternatives as possible for that placeholder.
        
        Generate as many meaningful synonyms and alternative phrasings as you can for each placeholder while maintaining the semantic intent. There is no limit to how many alternatives you should provide - the more diverse options, the better.
        
        All the output should be in English please, don't use Chinese.
        
        Please return in JSON format as follows:
        {
        "templates": [
            {
            "metatemplate": "{{instruction_modifier}} {{viewing_action}} this {{image_type}} and {{answering_action}} {{{{question}}}}\\n\\nChoices:\\n{{{{choices}}}}",
            "placeholders": ["image_type", "instruction_modifier", "viewing_action", "answering_action"],
            },
            {
            "metatemplate": "Based on the {{image_type}}, {{choice_instruction}}:\\n{{{{question}}}}\\n{{{{choices}}}}",
            "placeholders": ["image_type", "choice_instruction"],
            },
            {
            "metatemplate": "When processing the {{image_type}}, first consider the available options: {{{{choices}}}}, then {{answering_action}} to the following query with {{specificity_modifier}}:\\n{{{{question}}}}. {{instruction_modifier}}",
            "placeholders": ["image_type","answering_action","specificity_modifier","instruction_modifier"]
            },
            
            ...
        ],
        "placeholder_options": {
            "choice_modifier": ["look", "view", "examine"],
            "image_type": ["picture", "photo", "imaging", "visual", "graphic"],
            "answering_action": ["answer", "respond", "resolve", "process"],
            "question": ["query", "request", "inquiry"],
            "choices": ["options"],
            "specific_modifier": ["precise", "specific", "clear"]
        }
        }
        """
        return base_prompt + instruction
    
    async def generate_meta_templates(self, counts: int) -> List[Dict[str, Any]]:
        """
        Generate meta-templates for VQA tasks.
        
        This method sends a request to the LLM to generate meta-templates based on the provided prompt.
        
        Args:
            counts (int): Number of meta-templates to generate
            
        Returns:
            List[Dict[str, Any]]: List of generated meta-templates
        """
        # Create the prompt
        prompts = self._create_meta_templates_prompts(counts)
        
        # Generate the meta-templates
        results = await self._generate_response(
            prompts=prompts,
            validate_func=self._validate_metatemplate_response
        )
        
        if not results:
            logger.warning("Failed to generate meta templates")
            return []
        
        # Parse the response - we can trust the result since it's been validated
        try:
            parsed_results = [self._parse_json_from_response(result) for result in results]
            return parsed_results
        except Exception as e:
            logger.error(f"Error processing meta templates: {e}")
            return []
    
    def _validate_placeholder_options_response(self, response: str) -> Union[Dict[str, Any], bool]:
        """
        Validate if the response contains proper placeholder options JSON structure.
        
        Args:
            response (str): Response from the LLM
            
        Returns:
            Union[Dict[str, Any], bool]: Parsed JSON if valid, False otherwise
        """
        parsed = JSONParser.parse(response)
        if not isinstance(parsed, dict):
            return False
            
        # Check if it has templates
        templates = parsed.get("templates")
        if not isinstance(templates, list):
            return False
            
        # Check if it has placeholder_options
        placeholder_options = parsed.get("placeholder_options")
        if not isinstance(placeholder_options, dict):
            return False
            
        # Check if placeholder_options has proper structure
        for key, value in placeholder_options.items():
            if not isinstance(key, str) or not isinstance(value, list):
                return False
            if not all(isinstance(item, str) for item in value):
                return False
                
        return parsed

    async def generate_placeholder_options(self, meta_templates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate varied placeholder options for existing meta-templates.
        
        This method sends the current meta-templates to the LLM and asks it to generate
        synonym options for each placeholder, creating a new 'placeholder_options' section
        in the returned data.
        
        Args:
            meta_templates (Dict[str, Any]): Dictionary containing current meta-templates
            
        Returns:
            Dict[str, Any]: Updated dictionary with added 'placeholder_options'
        """
        # Create the prompt
        prompt = self._create_placeholder_options_prompt(meta_templates)
        
        # Generate the enhanced templates with placeholder options
        results = await self._generate_response(
            prompts=[prompt],
            validate_func=self._validate_placeholder_options_response
        )
        
        if not results:
            logger.warning("Failed to generate placeholder options")
            return meta_templates
        
        # Parse the response - we can trust the result since it's been validated
        try:
            enhanced_templates = results[0]
            if isinstance(enhanced_templates, dict) and "placeholder_options" in enhanced_templates:
                logger.info(f"Successfully generated options for {len(enhanced_templates.get('placeholder_options', {}))} placeholders")
                return enhanced_templates
            else:
                logger.warning("Generated response doesn't contain placeholder_options")
                return meta_templates
        except Exception as e:
            logger.error(f"Error processing placeholder options: {e}")
            return meta_templates
        
    def generate_concrete_templates(self, meta_templates: Dict[str, Any]) -> List[str]:
        """
        Replace placeholders in meta-templates with concrete options to generate final template list.

        Args:
            meta_templates: Dictionary containing 'templates' and 'placeholder_options'
                - templates: List of meta-templates
                - placeholder_options: Dictionary of candidate options for each placeholder

        Returns:
            List of concrete template strings (retaining {{question}} and {{choices}} placeholders)
        """
        concrete_templates = []
        placeholder_options = meta_templates.get("placeholder_options", {})

        for template in meta_templates.get("templates", []):
            metatemplate = template.get("metatemplate", "")
            placeholders = template.get("placeholders", [])

            # Collect candidate list for each placeholder
            options_lists = []
            for ph in placeholders:
                if ph_options := placeholder_options.get(ph):
                    options_lists.append(ph_options)
                else:
                    logger.warning(f"Missing options for placeholder: {ph}")
                    options_lists.append([ph])  # Fallback to placeholder name

            # Generate all possible combinations (Cartesian product)
            for combination in product(*options_lists):
                filled_template = metatemplate
                # Replace each placeholder with concrete value
                for ph_name, ph_value in zip(placeholders, combination):
                    filled_template = filled_template.replace(
                        f"{{{{{ph_name}}}}}",  # Match placeholder wrapped in double curly braces
                        ph_value
                    )
                concrete_templates.append(filled_template)

        return concrete_templates

                                                            

