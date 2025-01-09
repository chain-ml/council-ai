import unittest

import yaml

from council.prompt import (
    LLMPromptConfigObject,
    LLMPromptConfigSpec,
    PromptSection,
    XMLPromptTemplate,
    StringPromptTemplate,
)
from council.prompt.llm_prompt_config_object import XMLLLMPromptConfigSpec

from tests import get_data_filename
from .. import LLMPrompts, XMLPrompts


class TestLLMPromptConfig(unittest.TestCase):
    def test_llm_prompt_from_yaml(self):
        filename = get_data_filename(LLMPrompts.sample)
        actual = LLMPromptConfigObject.from_yaml(filename)

        assert isinstance(actual, LLMPromptConfigObject)
        assert actual.kind == "LLMPrompt"
        assert isinstance(actual.spec.system_prompts[0], StringPromptTemplate)

    def test_llm_prompt_templates(self):
        filename = get_data_filename(LLMPrompts.sample)
        actual = LLMPromptConfigObject.from_yaml(filename)

        system_prompt_gpt4o = actual.get_system_prompt_template("gpt-4o")
        assert system_prompt_gpt4o.rstrip("\n") == "System prompt template specific for gpt-4o"
        system_prompt_gpt35 = actual.get_system_prompt_template("gpt-3.5-turbo")
        assert system_prompt_gpt35.rstrip("\n") == "System prompt template for gpt-3.5-turbo and other gpt models"
        system_prompt_gpt = actual.get_system_prompt_template("gpt-4-turbo-preview")
        assert system_prompt_gpt.rstrip("\n") == "System prompt template for gpt-3.5-turbo and other gpt models"
        with self.assertRaises(ValueError) as e:
            _ = actual.get_system_prompt_template("claude-3-opus-20240229")
        assert str(e.exception) == "No prompt template for a given model `claude-3-opus-20240229` nor a default one"

        user_prompt_gpt4_turbo = actual.get_user_prompt_template("gpt-4-turbo-preview")
        assert user_prompt_gpt4_turbo.rstrip("\n") == "User prompt template for gpt-4-turbo-preview"
        user_prompt_gpt4o = actual.get_user_prompt_template("gpt-4o")
        assert user_prompt_gpt4o.rstrip("\n") == "User prompt template for default model"
        user_prompt_claude = actual.get_user_prompt_template("claude-3-opus-20240229")
        assert user_prompt_claude.rstrip("\n") == "User prompt template for default model"

    def test_parse_no_system(self):
        prompt_config_spec = """
        spec:
          user:
            - model: gpt-4o
              template: |
                User prompt template specific for gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception) == "System prompt(s) must be defined"

    def test_parse_no_user(self):
        prompt_config_spec = """
        spec:
          system:
            - model: gpt-4o
              template: |
                System prompt template specific for gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        _ = LLMPromptConfigSpec.from_dict(values["spec"])

    def test_parse_no_template(self):
        prompt_config_spec = """
        spec:
          system:
            - model: gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception) == "`template` must be defined"

    def test_parse_no_model_model_family(self):
        prompt_config_spec = """
        spec:
          system:
            - template: template
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception) == "At least one of `model` or `model-family` must be defined"

    def test_no_compliant(self):
        prompt_config_spec = """
        spec:
          system:
            - model: gpt-4o
              model-family: claude
              template: |
                System prompt template specific for gpt-4o or claude models
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = LLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception).startswith("model `gpt-4o` and model-family `claude` are not compliant")


class TestXMLPrompt(unittest.TestCase):
    def test_xml_section(self):
        section = PromptSection(name="test", content="content")
        assert section.to_xml() == "<test>\ncontent\n</test>"

    def test_xml_section_snake_case(self):
        section = PromptSection(name="  Complex nAmE    ", content="Complex Content 123")
        assert section.to_xml() == "<complex_name>\nComplex Content 123\n</complex_name>"

    def test_xml_prompt_from_yaml(self):
        filename = get_data_filename(XMLPrompts.sample)
        actual = LLMPromptConfigObject.from_yaml(filename)

        assert isinstance(actual, LLMPromptConfigObject)
        assert actual.kind == "XMLLLMPrompt"
        assert isinstance(actual.spec.system_prompts[0], XMLPromptTemplate)

    def test_sample_xml_prompt(self):
        filename = get_data_filename(XMLPrompts.sample)
        actual = LLMPromptConfigObject.from_yaml(filename)

        assert (
            actual.get_system_prompt_template("default")
            == """<role>
You are a helpful assistant.
</role>
<context>
The user is asking about programming concepts.
</context>"""
        )

        assert (
            actual.get_user_prompt_template("default")
            == """<question>
Explain what is object-oriented programming.
</question>
<response_template>
Provide the answer in simple terms.
</response_template>"""
        )

    def test_parse_no_system(self):
        prompt_config_spec = """
        spec:
          user:
            - model: default
              template:
                - name: user
                  content: |
                    User prompt template specific for gpt-4o
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = XMLLLMPromptConfigSpec.from_dict(values["spec"])
        assert str(e.exception) == "System prompt(s) must be defined"

    def test_parse_no_user(self):
        prompt_config_spec = """
        spec:
          system:
            - model: default
              template:
                - name: system
                  content: |
                    System prompt template
        """
        values = yaml.safe_load(prompt_config_spec)
        _ = XMLLLMPromptConfigSpec.from_dict(values["spec"])

    def test_parse_no_class(self):
        prompt_config_spec = """
        spec:
          system:
            - model: default
              template:
                name: system
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = XMLLLMPromptConfigSpec.from_dict(values["spec"])
        self.assertEquals(str(e.exception), "`template` must be a list of sections")

    def test_mixed_classes(self):
        prompt_config_spec = """
        spec:
          system:
            - model: default
              template: This is a string template
          user:
            - model: default
              template:
                - name: section
                  content: This is xml template
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = XMLLLMPromptConfigSpec.from_dict(values["spec"])
        self.assertEquals(str(e.exception), "`template` must be a list of sections")

        prompt_config_spec = """
        spec:
          system:
            - model: default
              template:
                - name: section
                  content: This is xml template
          user:
            - model: default
              template: This is a string template
        """
        values = yaml.safe_load(prompt_config_spec)
        with self.assertRaises(ValueError) as e:
            _ = XMLLLMPromptConfigSpec.from_dict(values["spec"])
        self.assertEquals(str(e.exception), "`template` must be a list of sections")
