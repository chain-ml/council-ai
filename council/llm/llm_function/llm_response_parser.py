from __future__ import annotations

import abc
import json
import os
import re
from typing import Any, Callable, Dict, Final, Literal, Type, TypeVar, Union, get_args, get_origin

import yaml
from council.llm.base import LLMParsingException
from council.utils import CodeParser
from pydantic import BaseModel, ConfigDict, ValidationError

from .llm_middleware import LLMResponse

T_Response = TypeVar("T_Response")

LLMResponseParser = Callable[[LLMResponse], T_Response]

T = TypeVar("T", bound="BaseModelResponseParser")

RESPONSE_HINTS_FILE_PATH: Final[str] = os.path.join(os.path.dirname(__file__), "data", "response_hints.yaml")


class ResponseHints:
    def __init__(self, hints: Dict[str, str]):
        self.hints_common = hints["hints_common"]
        self.parser_hints_start = hints["parser_hints_start"] if "parser_hints_start" in hints else ""
        self.parser_hints_end = hints["parser_hints_end"] if "parser_hints_end" in hints else ""
        self.block_parser_hints_start = hints["block_parser_hints_start"] if "block_parser_hints_start" in hints else ""

    @classmethod
    def from_yaml(cls, path: str, prefix: str) -> ResponseHints:
        with open(path, "r", encoding="utf-8") as file:
            hints = yaml.safe_load(file)
        return cls(hints[prefix])

    @property
    def parser(self) -> str:
        return self.parser_hints_start + self.hints_common

    @property
    def block_parser(self) -> str:
        return self.block_parser_hints_start + self.hints_common

    @property
    def parser_end(self) -> str:
        return self.parser_hints_end


class ResponseHintsHelper:
    yaml = ResponseHints.from_yaml(RESPONSE_HINTS_FILE_PATH, "yaml")
    json = ResponseHints.from_yaml(RESPONSE_HINTS_FILE_PATH, "json")
    code_blocks = ResponseHints.from_yaml(RESPONSE_HINTS_FILE_PATH, "code_blocks")


class EchoResponseParser:
    @staticmethod
    def from_response(response: LLMResponse) -> LLMResponse:
        """LLMFunction ResponseParser returning LLMResponse"""
        return response


class StringResponseParser:
    @staticmethod
    def from_response(response: LLMResponse) -> str:
        """LLMFunction ResponseParser for plain text responses"""
        return response.value


class BaseModelResponseParser(BaseModel, abc.ABC):
    """Base class for parsing LLM responses into structured data models"""

    model_config = ConfigDict(frozen=True)  # to preserve field order

    @classmethod
    @abc.abstractmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """
        Parse an LLM response into a structured data model.
        Must be implemented by subclasses to define specific parsing logic.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def to_response_template(cls: Type[T]) -> str:
        """Convert an instance to the response template for the LLM."""
        raise NotImplementedError()

    def validator(self) -> None:
        """
        Implement custom validation logic for the parsed data.
        Can be overridden by subclasses to add specific validation rules.
        Raise LLMParsingException to trigger local correction.
        Alternatively, use pydantic validation.
        """
        pass

    @classmethod
    def create_and_validate(cls: Type[T], **kwargs) -> T:
        instance = cls._try_create(**kwargs)
        instance.validator()
        return instance

    @classmethod
    def _try_create(cls: Type[T], **kwargs) -> T:
        """
        Attempt to create a BaseModel object instance.
        Raises an LLMParsingException if a ValidationError occurs during instantiation.
        """

        try:
            return cls(**kwargs)
        except ValidationError as e:
            # LLM-friendlier version of pydantic error message without "For further information visit..."
            clean_exception_message = re.sub(r"For further information visit.*", "", str(e))
            raise LLMParsingException(clean_exception_message)

    @classmethod
    def _is_of_type(cls, obj: Type, type_to_check: Type) -> bool:
        try:
            return issubclass(obj, type_to_check)
        except TypeError:  # for typing.Literal
            return False

    @classmethod
    def _is_list_of_type(cls, obj: Type, type_to_check: Type) -> bool:
        return getattr(obj, "__origin__", None) is list and cls._is_of_type(obj.__args__[0], type_to_check)


T_CodeBlocksResponseParserBase = TypeVar("T_CodeBlocksResponseParserBase", bound="CodeBlocksResponseParser")


class CodeBlocksResponseParser(BaseModelResponseParser):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing multiple named code blocks"""
        llm_response = response.value
        parsed_blocks: Dict[str, Any] = {}

        for field_name in cls.model_fields.keys():
            block = CodeParser.find_first(field_name, llm_response)
            if block is None:
                raise LLMParsingException(f"`{field_name}` block is not found")
            parsed_blocks[field_name] = block.code.strip()

        return cls.create_and_validate(**parsed_blocks)

    @classmethod
    def to_response_template(cls: Type[T_CodeBlocksResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate code blocks response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal code blocks formatting hints.
        """

        template_parts = [ResponseHintsHelper.code_blocks.block_parser] if include_hints else []
        template_parts.append(cls._to_response_template())
        return "\n".join(template_parts)

    @classmethod
    def _to_response_template(cls: Type[T_CodeBlocksResponseParserBase]) -> str:
        """Generate code blocks response template based on the model's fields and their descriptions."""
        template_parts = []

        for field_name, field in cls.model_fields.items():
            if not cls._is_primitive(field.annotation):
                raise ValueError(
                    f"Field `{field_name}` has complex type {field.annotation}. "
                    "Only primitive types (str, int, float, bool) are supported for CodeBlocksResponseParser."
                )

            description = field.description
            if description is None:
                raise ValueError(f"Description is required for field `{field_name}` in {cls.__name__}")

            template_parts.extend([f"```{field_name}", description, "```", ""])

        template_parts.pop()  # delete last \n

        return "\n".join(template_parts)

    @staticmethod
    def _is_primitive(field_type: Any) -> bool:
        """
        Check if a type is primitive (str, int, float, bool) and not a complex type
        with support of Optional and Literal.
        """

        primitive_types = (str, int, float, bool, type(None))

        if field_type in primitive_types:
            return True

        # get the origin type for annotations like Optional[str] and Literal["abc"]
        origin = get_origin(field_type)
        if origin == Literal:
            return True
        elif origin == Union:
            args = get_args(field_type)
            return all(arg in primitive_types for arg in args)

        return False


T_YAMLResponseParserBase = TypeVar("T_YAMLResponseParserBase", bound="YAMLResponseParserBase")


class YAMLResponseParserBase(BaseModelResponseParser, abc.ABC):
    @classmethod
    def _to_response_template(cls: Type[T], indent_level: int = 0) -> str:
        """
        Generate a YAML response template based on the model's fields and their descriptions.
        Supports nested objects/list of objects but all of them must inherit from YAMLResponseParserBase.
        """
        template_parts = []
        indent = "  " * indent_level

        for field_name, field in cls.model_fields.items():
            description = field.description
            if description is None:
                raise ValueError(f"Description is required for field `{field_name}` in {cls.__name__}")
            field_type = field.annotation
            if field_type is None:
                raise ValueError(f"Type annotation is required for field `{field_name}` in {cls.__name__}")

            is_multiline_description = "\n" in description

            # nested BaseModel
            if cls._is_of_type(field_type, YAMLResponseParserBase):
                template_parts.append(f"{indent}{field_name}: # {description}")
                nested_template = field_type._to_response_template(indent_level + 1)
                template_parts.append(nested_template)
            # list of BaseModels
            elif cls._is_list_of_type(field_type, YAMLResponseParserBase):
                template_parts.append(f"{indent}{field_name}: # {description}")
                template_parts.append(f"{indent}- # Each element being:")
                nested_template = field_type.__args__[0]._to_response_template(indent_level + 1)
                template_parts.append(nested_template)
            # multiline string description
            elif field_type is str and is_multiline_description:
                template_parts.append(f"{indent}{field_name}: |")
                for line in description.split("\n"):
                    template_parts.append(f"{indent}  {line.strip()}")
            # regular fields
            else:
                template_parts.append(f"{indent}{field_name}: # {description}")

        return "\n".join(template_parts)

    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise LLMParsingException(f"Error while parsing yaml: {e}")


class YAMLBlockResponseParser(YAMLResponseParserBase):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing a single YAML code block"""
        llm_response = response.value

        yaml_block = CodeParser.find_first("yaml", llm_response)
        if yaml_block is None:
            raise LLMParsingException("yaml block is not found")

        yaml_content = YAMLResponseParserBase.parse(yaml_block.code)
        return cls.create_and_validate(**yaml_content)

    @classmethod
    def to_response_template(cls: Type[T_YAMLResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate YAML block response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal YAML block formatting hints.
        """
        template_parts = [ResponseHintsHelper.yaml.block_parser] if include_hints else []
        template_parts.extend(["```yaml", cls._to_response_template(), "```"])
        return "\n".join(template_parts)


class YAMLResponseParser(YAMLResponseParserBase):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing raw YAML content"""
        llm_response = response.value

        yaml_content = YAMLResponseParserBase.parse(llm_response)
        return cls.create_and_validate(**yaml_content)

    @classmethod
    def to_response_template(cls: Type[T_YAMLResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate YAML response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal YAML formatting hints.
        """
        template_parts = [ResponseHintsHelper.yaml.parser] if include_hints else []
        template_parts.append(cls._to_response_template())
        if include_hints:
            template_parts.extend(["", ResponseHintsHelper.yaml.parser_end])
        return "\n".join(template_parts)


T_JSONResponseParserBase = TypeVar("T_JSONResponseParserBase", bound="JSONResponseParserBase")


class JSONResponseParserBase(BaseModelResponseParser, abc.ABC):
    @classmethod
    def _to_response_template(cls: Type[T]) -> str:
        """
        Generate a JSON response template based on the model's fields and their descriptions.
        Supports nested objects/list of objects but all of them must inherit from JSONResponseParserBase.

        Field descriptions for lists are ignored.
        """
        template_dict = {}

        for field_name, field in cls.model_fields.items():
            field_type = field.annotation
            if field_type is None:
                raise ValueError(f"Type annotation is required for field `{field_name}` in {cls.__name__}")

            # nested BaseModel
            if cls._is_of_type(field_type, JSONResponseParserBase):
                nested_template = json.loads(field_type._to_response_template())
                template_dict[field_name] = nested_template
            # list of BaseModels
            elif cls._is_list_of_type(field_type, JSONResponseParserBase):
                nested_template = json.loads(field_type.__args__[0]._to_response_template())
                template_dict[field_name] = [nested_template]
            # regular fields
            else:
                if field.description is None:
                    raise ValueError(f"Description is required for field `{field_name}` in {cls.__name__}")

                template_dict[field_name] = field.description

        return json.dumps(template_dict, indent=2)

    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMParsingException(f"Error while parsing json: {e}")


class JSONBlockResponseParser(JSONResponseParserBase):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing a single JSON code block"""
        llm_response = response.value

        json_block = CodeParser.find_first("json", llm_response)
        if json_block is None:
            raise LLMParsingException("json block is not found")

        json_content = JSONResponseParserBase.parse(json_block.code)
        return cls.create_and_validate(**json_content)

    @classmethod
    def to_response_template(cls: Type[T_JSONResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate JSON block response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal JSON block formatting hints.
        """
        template_parts = [ResponseHintsHelper.json.block_parser] if include_hints else []
        template_parts.extend(["```json", cls._to_response_template(), "```"])
        return "\n".join(template_parts)


class JSONResponseParser(JSONResponseParserBase):

    @classmethod
    def from_response(cls: Type[T], response: LLMResponse) -> T:
        """LLMFunction ResponseParser for response containing raw JSON content"""
        llm_response = response.value

        json_content = JSONResponseParserBase.parse(llm_response)
        return cls.create_and_validate(**json_content)

    @classmethod
    def to_response_template(cls: Type[T_JSONResponseParserBase], include_hints: bool = True) -> str:
        """
        Generate JSON response template based on the model's fields and their descriptions.

        Args:
            include_hints: If True, returned template will include universal JSON formatting hints.
        """
        template_parts = [ResponseHintsHelper.json.parser] if include_hints else []
        template_parts.append(cls._to_response_template())
        if include_hints:
            template_parts.extend(["", ResponseHintsHelper.json.parser_end])
        return "\n".join(template_parts)
