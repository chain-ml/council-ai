from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import yaml
from council.utils import DataObject, DataObjectSpecBase
from typing_extensions import Self


class PromptTemplateBase(ABC):
    def __init__(self, *, model: Optional[str], model_family: Optional[str]) -> None:
        """Initialize prompt template with at least one of `model` or `model-family`."""
        self._model: Optional[str] = model
        self._model_family: Optional[str] = model_family

        if self._model is None and self._model_family is None:
            raise ValueError("At least one of `model` or `model-family` must be defined")

        if self._model is not None and self._model_family is not None:
            if not self._model.startswith(self._model_family):
                raise ValueError(
                    f"model `{self._model}` and model-family `{self._model_family}` are not compliant."
                    f"Please use separate prompt templates"
                )

    @property
    @abstractmethod
    def template(self) -> str:
        """Return prompt template as a string."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, values: Dict[str, Any]) -> Self:
        pass

    @staticmethod
    def _extract_template_and_models(values: Dict[str, Any]) -> Tuple[Any, Optional[str], Optional[str]]:
        """Extract template, model, and model-family from the values dictionary."""
        template = values.get("template", None)
        if template is None:
            raise ValueError("`template` must be defined")

        model = values.get("model", None)
        model_family = values.get("model-family", None)
        return template, model, model_family

    @staticmethod
    def _parse_template_sections(template: Any) -> List[PromptSection]:
        """Parse template into a list of PromptSection objects."""
        if not isinstance(template, list):
            raise ValueError("`template` must be a list of sections")

        return [PromptSection.from_dict(section) for section in template]

    def is_compatible(self, model: str) -> bool:
        """Check if the prompt template is compatible with the given model."""
        if self._model is not None and self._model == model:
            return True

        if self._model_family is not None and model.startswith(self._model_family):
            return True
        return False


class StringPromptTemplate(PromptTemplateBase):
    """Prompt template implementation where template is a simple string."""

    def __init__(self, *, template: str, model: Optional[str], model_family: Optional[str]) -> None:
        super().__init__(model=model, model_family=model_family)
        self._template = template

    @property
    def template(self) -> str:
        """Return prompt template as a string."""
        return self._template

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> StringPromptTemplate:
        template, model, model_family = cls._extract_template_and_models(values)
        if not isinstance(template, str):
            raise ValueError("`template` must be string for StringPromptTemplate")

        return StringPromptTemplate(template=template, model=model, model_family=model_family)


class PromptSection:
    """Represents a section in an section-based prompt, e.g. XML, markdown, etc."""

    def __init__(
        self, *, name: str, content: Optional[str] = None, sections: Optional[List[PromptSection]] = None
    ) -> None:
        self.name = name
        self.content = content.strip() if content else None
        self.sections = list(sections) if sections else []

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> PromptSection:
        name = values.get("name")
        if name is None:
            raise ValueError("'name' must be defined")

        content = values.get("content")
        sections_data = values.get("sections", [])

        if not isinstance(sections_data, list):
            raise ValueError("'sections' must be a list")

        sections = [PromptSection.from_dict(section) for section in sections_data]

        return PromptSection(name=name, content=content, sections=sections)

    def to_xml(self, indent: str = "") -> str:
        """XML representation of the prompt section with proper indentation."""
        intent_diff = "  "
        name_snake_case = self.name.lower().strip().replace(" ", "_")
        parts = [f"{indent}<{name_snake_case}>"]

        if self.content:
            content_lines = self.content.split("\n")
            content = "\n".join(f"{indent}{intent_diff}{line}" for line in content_lines)
            parts.append(content)

        parts.extend(section.to_xml(indent + intent_diff) for section in self.sections)
        parts.append(f"{indent}</{name_snake_case}>")
        return "\n".join(parts)

    def to_md(self, level: int = 1) -> str:
        """Markdown representation of the prompt section with proper heading levels."""
        parts = [f"{'#' * level} {self.name}"]

        if self.content:
            parts.append(self.content)

        parts.extend(section.to_md(level + 1) for section in self.sections)
        return "\n".join(parts)


class XMLPromptTemplate(PromptTemplateBase):
    """Prompt template implementation where template consists of XML sections."""

    def __init__(self, *, template: Sequence[PromptSection], model: Optional[str], model_family: Optional[str]) -> None:
        super().__init__(model=model, model_family=model_family)
        self._sections = list(template)

    @property
    def template(self) -> str:
        """Return prompt template as a string, formatting each section to XML."""
        return "\n".join(section.to_xml() for section in self._sections)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> XMLPromptTemplate:
        template, model, model_family = cls._extract_template_and_models(values)
        sections = cls._parse_template_sections(template)
        return XMLPromptTemplate(template=sections, model=model, model_family=model_family)


class MarkdownPromptTemplate(PromptTemplateBase):
    """Prompt template implementation where template consists of markdown sections."""

    def __init__(self, *, template: Sequence[PromptSection], model: Optional[str], model_family: Optional[str]) -> None:
        super().__init__(model=model, model_family=model_family)
        self._sections = list(template)

    @property
    def template(self) -> str:
        """Return prompt template as a string, formatting each section to markdown."""
        return "\n".join(section.to_md() for section in self._sections)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> MarkdownPromptTemplate:
        template, model, model_family = cls._extract_template_and_models(values)
        sections = cls._parse_template_sections(template)
        return MarkdownPromptTemplate(template=sections, model=model, model_family=model_family)


class LLMPromptConfigSpecBase(DataObjectSpecBase, ABC):
    def __init__(self, system: Sequence[PromptTemplateBase], user: Optional[Sequence[PromptTemplateBase]]) -> None:
        self.system_prompts = list(system)
        self.user_prompts = list(user or [])

    @classmethod
    @abstractmethod
    def get_template_class(cls) -> Type[PromptTemplateBase]:
        pass

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        system_prompts = values.get("system", [])
        if not system_prompts:
            raise ValueError("System prompt(s) must be defined")

        template_class = cls.get_template_class()

        system = [template_class.from_dict(p) for p in system_prompts]
        user_prompts = values.get("user", [])
        user = [template_class.from_dict(p) for p in user_prompts] if user_prompts else None

        return cls(system, user)

    def to_dict(self) -> Dict[str, Any]:
        result = {"system": self.system_prompts}
        if not self.user_prompts:
            result["user"] = self.user_prompts
        return result

    def __str__(self):
        msg = f"{len(self.system_prompts)} system prompt(s)"
        if self.user_prompts is not None:
            msg += f"; {len(self.user_prompts)} user prompt(s)"
        return msg


class LLMPromptConfigSpec(LLMPromptConfigSpecBase):
    @classmethod
    def get_template_class(cls) -> Type[PromptTemplateBase]:
        return StringPromptTemplate


class XMLLLMPromptConfigSpec(LLMPromptConfigSpecBase):
    @classmethod
    def get_template_class(cls) -> Type[PromptTemplateBase]:
        return XMLPromptTemplate


class MarkdownLLMPromptConfigSpec(LLMPromptConfigSpecBase):
    @classmethod
    def get_template_class(cls) -> Type[PromptTemplateBase]:
        return MarkdownPromptTemplate


class LLMPromptConfigObject(DataObject[LLMPromptConfigSpecBase]):
    """
    Helper class to instantiate a LLMPrompt object from a YAML file.
    """

    _kind_to_spec: Mapping[str, Type[LLMPromptConfigSpecBase]] = {
        "LLMPrompt": LLMPromptConfigSpec,
        "XMLLLMPrompt": XMLLLMPromptConfigSpec,
        "MarkdownLLMPrompt": MarkdownLLMPromptConfigSpec,
    }

    @classmethod
    def from_dict(cls, spec: Type[LLMPromptConfigSpecBase], values: Dict[str, Any]) -> LLMPromptConfigObject:
        return super()._from_dict(spec, values)

    @classmethod
    def from_yaml(cls, filename: str) -> LLMPromptConfigObject:
        with open(filename, "r", encoding="utf-8") as f:
            values = yaml.safe_load(f)

        kind = cls._get_kind(values)
        if kind not in cls._kind_to_spec:
            raise ValueError(f"Unexpected kind `{kind}`. Expected one of: {', '.join(cls._kind_to_spec.keys())}")

        return LLMPromptConfigObject.from_dict(cls._kind_to_spec[kind], values)

    @property
    def has_user_prompt_template(self) -> bool:
        """Return True, if user prompt template was specified in yaml file."""
        return bool(self.spec.user_prompts)

    def get_system_prompt_template(self, model: str) -> str:
        """Return system prompt template for a given model."""
        return self._get_prompt_template(self.spec.system_prompts, model)

    def get_user_prompt_template(self, model: str) -> str:
        """
        Return user prompt template for a given model.
        Raises ValueError if no user prompt template was provided.
        """

        if not self.has_user_prompt_template:
            raise ValueError("No user prompt template provided")
        return self._get_prompt_template(self.spec.user_prompts, model)

    @staticmethod
    def _get_prompt_template(prompts: List[PromptTemplateBase], model: str) -> str:
        """
        Get the first prompt compatible to the given `model` (or `default` prompt).

        Args:
            prompts (List[PromptTemplateBase]): List of prompts to search from

        Returns:
            str: prompt template

        Raises:
            ValueError: if both prompt template for a given model and default prompt template are not provided
        """
        try:
            return next(prompt.template for prompt in prompts if prompt.is_compatible(model))
        except StopIteration:
            try:
                return next(prompt.template for prompt in prompts if prompt.is_compatible("default"))
            except StopIteration:
                raise ValueError(f"No prompt template for a given model `{model}` nor a default one")
