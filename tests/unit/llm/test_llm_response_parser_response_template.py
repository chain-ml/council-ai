import unittest
from council.llm import (
    YAMLBlockResponseParser,
    YAMLResponseParser,
    JSONBlockResponseParser,
    JSONResponseParser,
    CodeBlocksResponseParser,
)
from pydantic import Field, BaseModel
from typing import Literal, List, Union, Optional, Dict, Any


class MissingDescriptionField(YAMLBlockResponseParser):
    _multiline_description = "\n".join(
        [
            "You can define multi-line description inside the response class",
            "Like that",
        ]
    )
    number: float = Field(..., description="Number from 1 to 10")
    reasoning: str


multiline_description = "Carefully\nreason about the number"


class BaseResponse:
    reasoning: str = Field(..., description=multiline_description)
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    abc: str = Field(..., description="Not multiline description")


class BaseResponseReordered:
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    abc: str = Field(..., description="Not multiline description")
    reasoning: str = Field(..., description=multiline_description)


class BaseResponseReorderedAgain:
    abc: str = Field(..., description="Not multiline description")
    number: float = Field(..., ge=1, le=10, description="Number from 1 to 10")
    reasoning: str = Field(..., description=multiline_description)


class CodeBlocksResponse(CodeBlocksResponseParser, BaseResponse):
    pass


class YAMLBlockResponse(YAMLBlockResponseParser, BaseResponse):
    pass


class YAMLResponse(YAMLResponseParser, BaseResponse):
    pass


class JSONBlockResponse(JSONBlockResponseParser, BaseResponse):
    pass


class JSONResponse(JSONResponseParser, BaseResponse):
    pass


class CodeBlocksResponseReordered(CodeBlocksResponseParser, BaseResponseReordered):
    pass


class YAMLBlockResponseReordered(YAMLBlockResponseParser, BaseResponseReordered):
    pass


class YAMLResponseReordered(YAMLResponseParser, BaseResponseReordered):
    pass


class JSONBlockResponseReordered(JSONBlockResponseParser, BaseResponseReordered):
    pass


class JSONResponseReordered(JSONResponseParser, BaseResponseReordered):
    pass


class CodeBlocksResponseReorderedAgain(CodeBlocksResponseParser, BaseResponseReorderedAgain):
    pass


class YAMLBlockResponseReorderedAgain(YAMLBlockResponseParser, BaseResponseReorderedAgain):
    pass


class YAMLResponseReorderedAgain(YAMLResponseParser, BaseResponseReorderedAgain):
    pass


class JSONBlockResponseReorderedAgain(JSONBlockResponseParser, BaseResponseReorderedAgain):
    pass


class JSONResponseReorderedAgain(JSONResponseParser, BaseResponseReorderedAgain):
    pass


class YAMLComplexResponse(YAMLBlockResponseParser):
    mode: Literal["mode_one", "mode_two"] = Field(..., description="Mode of operation, one of `mode_one` or `mode_two`")
    pairs: List[YAMLBlockResponse] = Field(..., description="List of number and reasoning pairs")


class JSONComplexResponse(JSONBlockResponseParser):
    mode: Literal["mode_one", "mode_two"] = Field(..., description="Mode of operation, one of `mode_one` or `mode_two`")
    pairs: List[JSONBlockResponse] = Field(..., description="List of number and reasoning pairs")


class YAMLBlockNestedResponse(YAMLBlockResponseParser):
    score: float = Field(..., description="Float score")
    response: YAMLComplexResponse = Field(..., description="Complex response")


class YAMLNestedResponse(YAMLResponseParser):
    score: float = Field(..., description="Float score")
    response: YAMLComplexResponse = Field(..., description="Complex response")


class JSONBlockNestedResponse(JSONBlockResponseParser):
    score: float = Field(..., description="Float score")
    response: JSONComplexResponse = Field(..., description="Complex response")


class JSONNestedResponse(JSONResponseParser):
    score: float = Field(..., description="Float score")
    response: JSONComplexResponse = Field(..., description="Complex response")


class TestCodeBlocksResponseParserTemplate(unittest.TestCase):
    def test_missing_description_field(self):
        with self.assertRaises(ValueError) as e:
            MissingDescriptionField.to_response_template()
        self.assertEqual(str(e.exception), "Description is required for field `reasoning` in MissingDescriptionField")

    def test_template(self):
        template = CodeBlocksResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```reasoning
Carefully
reason about the number
```

```number
Number from 1 to 10
```

```abc
Not multiline description
```""",
        )

    def test_reordered_template(self):
        template = CodeBlocksResponseReordered.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```number
Number from 1 to 10
```

```abc
Not multiline description
```

```reasoning
Carefully
reason about the number
```""",
        )

    def test_reordered_again_template(self):
        template = CodeBlocksResponseReorderedAgain.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```abc
Not multiline description
```

```number
Number from 1 to 10
```

```reasoning
Carefully
reason about the number
```""",
        )

    def test_with_hints(self):
        template = CodeBlocksResponse.to_response_template(include_hints=True)
        self.assertEqual(
            template,
            """- Provide your response in a the following code blocks.
- All keys must be present in the response, even when their values are empty.
- For empty values, leave them blank as follows:
  ```empty_field
  ```
- Your output outside of code blocks will not be parsed.

```reasoning
Carefully
reason about the number
```

```number
Number from 1 to 10
```

```abc
Not multiline description
```""",
        )


class TestCodeBlocksResponseParserTypes(unittest.TestCase):
    def test_correct(self) -> None:
        class Response(CodeBlocksResponseParser):
            number: Union[int, float] = Field(..., description="Number from 1 to 10")
            reasoning: Optional[str] = Field(..., description="Reasoning")
            abc: Literal["abc", "def"] = Field(..., description="ABC")
            boolean: bool = Field(..., description="Boolean")

        _ = Response.to_response_template()

    def test_incorrect_list(self) -> None:
        class Response(CodeBlocksResponseParser):
            numbers: List[int] = Field(..., description="Numbers from 1 to 10")

        with self.assertRaises(ValueError) as e:
            _ = Response.to_response_template()
        assert str(e.exception).startswith("Field `numbers` has complex type typing.List[int].")

    def test_incorrect_dict(self) -> None:
        class Response(CodeBlocksResponseParser):
            numbers: Dict[str, Any] = Field(..., description="Numbers from 1 to 10")

        with self.assertRaises(ValueError) as e:
            _ = Response.to_response_template()
        assert str(e.exception).startswith("Field `numbers` has complex type typing.Dict[str, typing.Any].")

    def test_incorrect_basemodel(self) -> None:
        class Number(BaseModel):
            number: int = Field(..., description="Number from 1 to 10")

        class Response(CodeBlocksResponseParser):
            number: Number = Field(..., description="Number from 1 to 10")

        with self.assertRaises(ValueError) as e:
            _ = Response.to_response_template()
        assert str(e.exception).startswith("Field `number` has complex type")
        assert "Number" in str(e.exception)


class TestYAMLBlockResponseParserResponseTemplate(unittest.TestCase):
    def test_missing_description_field(self):
        with self.assertRaises(ValueError) as e:
            MissingDescriptionField.to_response_template()
        self.assertEqual(str(e.exception), "Description is required for field `reasoning` in MissingDescriptionField")

    def test_template(self):
        template = YAMLBlockResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```yaml
reasoning: |
  Carefully
  reason about the number
number: # Number from 1 to 10
abc: # Not multiline description
```""",
        )

    def test_reordered_template(self):
        template = YAMLBlockResponseReordered.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```yaml
number: # Number from 1 to 10
abc: # Not multiline description
reasoning: |
  Carefully
  reason about the number
```""",
        )

    def test_reordered_again_template(self):
        template = YAMLBlockResponseReorderedAgain.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```yaml
abc: # Not multiline description
number: # Number from 1 to 10
reasoning: |
  Carefully
  reason about the number
```""",
        )

    def test_with_hints(self):
        template = YAMLBlockResponse.to_response_template(include_hints=True)
        self.assertEqual(
            template,
            """- Provide your response in a single YAML code block.
- Make sure you respect YAML syntax, particularly for lists and dictionaries.
- All keys must be present in the response, even when their values are empty.
- For empty values, include empty quotes ("") rather than leaving them blank.
- Always wrap string values in double quotes (") to ensure proper parsing, except when using the YAML pipe operator (|) for multi-line strings.

```yaml
reasoning: |
  Carefully
  reason about the number
number: # Number from 1 to 10
abc: # Not multiline description
```""",
        )

    def test_nested_response_template(self):
        template = YAMLBlockNestedResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```yaml
score: # Float score
response: # Complex response
  mode: # Mode of operation, one of `mode_one` or `mode_two`
  pairs: # List of number and reasoning pairs
  - # Each element being:
    reasoning: |
      Carefully
      reason about the number
    number: # Number from 1 to 10
    abc: # Not multiline description
```""",
        )


class TestYAMLResponseParserResponseTemplate(unittest.TestCase):
    def test_template(self):
        template = YAMLResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """reasoning: |
  Carefully
  reason about the number
number: # Number from 1 to 10
abc: # Not multiline description""",
        )

    def test_reordered_template(self):
        template = YAMLResponseReordered.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """number: # Number from 1 to 10
abc: # Not multiline description
reasoning: |
  Carefully
  reason about the number""",
        )

    def test_reordered_again_template(self):
        template = YAMLResponseReorderedAgain.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """abc: # Not multiline description
number: # Number from 1 to 10
reasoning: |
  Carefully
  reason about the number""",
        )

    def test_with_hints(self):
        template = YAMLResponse.to_response_template(include_hints=True)
        self.assertEqual(
            template,
            """- Provide your response as a parsable YAML.
- Make sure you respect YAML syntax, particularly for lists and dictionaries.
- All keys must be present in the response, even when their values are empty.
- For empty values, include empty quotes ("") rather than leaving them blank.
- Always wrap string values in double quotes (") to ensure proper parsing, except when using the YAML pipe operator (|) for multi-line strings.

reasoning: |
  Carefully
  reason about the number
number: # Number from 1 to 10
abc: # Not multiline description

Only respond with parsable YAML. Do not output anything else. Do not wrap your response in ```yaml```.""",
        )

    def test_nested_response_template(self):
        template = YAMLNestedResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """score: # Float score
response: # Complex response
  mode: # Mode of operation, one of `mode_one` or `mode_two`
  pairs: # List of number and reasoning pairs
  - # Each element being:
    reasoning: |
      Carefully
      reason about the number
    number: # Number from 1 to 10
    abc: # Not multiline description""",
        )


class TestJSONBlockResponseParserResponseTemplate(unittest.TestCase):
    def test_template(self):
        template = JSONBlockResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```json
{
  "reasoning": "Carefully\\nreason about the number",
  "number": "Number from 1 to 10",
  "abc": "Not multiline description"
}
```""",
        )

    def test_reordered_template(self):
        template = JSONBlockResponseReordered.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```json
{
  "number": "Number from 1 to 10",
  "abc": "Not multiline description",
  "reasoning": "Carefully\\nreason about the number"
}
```""",
        )

    def test_reordered_again_template(self):
        template = JSONBlockResponseReorderedAgain.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```json
{
  "abc": "Not multiline description",
  "number": "Number from 1 to 10",
  "reasoning": "Carefully\\nreason about the number"
}
```""",
        )

    def test_with_hints(self):
        template = JSONBlockResponse.to_response_template(include_hints=True)
        self.assertEqual(
            template,
            """- Provide your response in a single JSON code block.
- Make sure you respect JSON syntax, particularly for lists and dictionaries.
- All keys must be present in the response, even when their values are empty.
- For empty values, include empty quotes ("") rather than leaving them blank.

```json
{
  "reasoning": "Carefully\\nreason about the number",
  "number": "Number from 1 to 10",
  "abc": "Not multiline description"
}
```""",
        )

    def test_nested_response_template(self):
        template = JSONBlockNestedResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """```json
{
  "score": "Float score",
  "response": {
    "mode": "Mode of operation, one of `mode_one` or `mode_two`",
    "pairs": [
      {
        "reasoning": "Carefully\\nreason about the number",
        "number": "Number from 1 to 10",
        "abc": "Not multiline description"
      }
    ]
  }
}
```""",
        )


class TestJSONResponseParserResponseTemplate(unittest.TestCase):
    def test_template(self):
        template = JSONResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """{
  "reasoning": "Carefully\\nreason about the number",
  "number": "Number from 1 to 10",
  "abc": "Not multiline description"
}""",
        )

    def test_reordered_template(self):
        template = JSONResponseReordered.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """{
  "number": "Number from 1 to 10",
  "abc": "Not multiline description",
  "reasoning": "Carefully\\nreason about the number"
}""",
        )

    def test_reordered_again_template(self):
        template = JSONResponseReorderedAgain.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """{
  "abc": "Not multiline description",
  "number": "Number from 1 to 10",
  "reasoning": "Carefully\\nreason about the number"
}""",
        )

    def test_with_hints(self):
        template = JSONResponse.to_response_template(include_hints=True)
        self.assertEqual(
            template,
            """- Provide your response as a parsable JSON.
- Make sure you respect JSON syntax, particularly for lists and dictionaries.
- All keys must be present in the response, even when their values are empty.
- For empty values, include empty quotes ("") rather than leaving them blank.

{
  "reasoning": "Carefully\\nreason about the number",
  "number": "Number from 1 to 10",
  "abc": "Not multiline description"
}

Only respond with parsable JSON. Do not output anything else.""",
        )

    def test_nested_response_template(self):
        template = JSONNestedResponse.to_response_template(include_hints=False)
        self.assertEqual(
            template,
            """{
  "score": "Float score",
  "response": {
    "mode": "Mode of operation, one of `mode_one` or `mode_two`",
    "pairs": [
      {
        "reasoning": "Carefully\\nreason about the number",
        "number": "Number from 1 to 10",
        "abc": "Not multiline description"
      }
    ]
  }
}""",
        )
