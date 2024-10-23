from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import httpx
from council.contexts import Consumption, LLMContext

from ..utils import truncate_dict_values_to_str
from . import ChatGPTConfigurationBase
from .llm_base import LLMBase, LLMCostCard, LLMCostManager, LLMResult
from .llm_exception import LLMCallException
from .llm_message import LLMessageTokenCounterBase, LLMMessage


class Provider(Protocol):
    def __call__(self, payload: dict[str, Any]) -> httpx.Response: ...


class Message:
    def __init__(self, role: str, content: str) -> None:
        self._content = content
        self._role = role

    @property
    def content(self) -> str:
        return self._content

    @staticmethod
    def from_dict(obj: Any) -> Message:
        _role = str(obj.get("role"))
        _content = str(obj.get("content"))
        return Message(_role, _content)


class Choice:

    def __init__(self, index: int, finish_reason: str, message: Message) -> None:
        self._index = index
        self._finish_reason = finish_reason
        self._message = message

    @property
    def message(self) -> Message:
        return self._message

    @staticmethod
    def from_dict(obj: Any) -> Choice:
        _index = int(obj.get("index"))
        _finish_reason = str(obj.get("finish_reason"))
        _message = Message.from_dict(obj.get("message"))
        return Choice(_index, _finish_reason, _message)


class Usage:
    def __init__(self, completion_tokens: int, prompt_tokens: int, total_tokens: int) -> None:
        self._completion = completion_tokens
        self._prompt = prompt_tokens
        self._total = total_tokens

    def __str__(self) -> str:
        return f'prompt_tokens="{self._prompt}" total_tokens="{self._total}" completion_tokens="{self._completion}"'

    @property
    def prompt_tokens(self) -> int:
        return self._prompt

    @property
    def completion_tokens(self) -> int:
        return self._completion

    @property
    def total_tokens(self) -> int:
        return self._total

    @staticmethod
    def from_dict(obj: Any) -> Usage:
        _completion_tokens = int(obj.get("completion_tokens"))
        _prompt_tokens = int(obj.get("prompt_tokens"))
        _total_tokens = int(obj.get("total_tokens"))
        return Usage(_completion_tokens, _prompt_tokens, _total_tokens)


class OpenAICostManager(LLMCostManager):
    # https://openai.com/api/pricing/
    COSTS_GPT_35_turbo_FAMILY: Mapping[str, LLMCostCard] = {
        "gpt-3.5-turbo-0125": LLMCostCard(input=0.50, output=1.50),
        "gpt-3.5-turbo-instruct": LLMCostCard(input=1.50, output=2.00),
        "gpt-3.5-turbo-1106": LLMCostCard(input=1.00, output=2.00),
        "gpt-3.5-turbo-0613": LLMCostCard(input=1.50, output=2.00),
        "gpt-3.5-turbo-16k-0613": LLMCostCard(input=3.00, output=4.00),
        "gpt-3.5-turbo-0301": LLMCostCard(input=1.50, output=2.00),
    }

    COSTS_GPT_4_FAMILY: Mapping[str, LLMCostCard] = {
        "gpt-4-turbo": LLMCostCard(input=10.00, output=30.00),
        "gpt-4-turbo-2024-04-09": LLMCostCard(input=10.00, output=30.00),
        "gpt-4": LLMCostCard(input=30.00, output=60.00),
        "gpt-4-32k": LLMCostCard(input=60.00, output=120.00),
        "gpt-4-0125-preview": LLMCostCard(input=10.00, output=30.00),
        "gpt-4-1106-preview": LLMCostCard(input=10.00, output=30.00),
        "gpt-4-vision-preview": LLMCostCard(input=10.00, output=30.00),
    }

    COSTS_GPT_4o_FAMILY: Mapping[str, LLMCostCard] = {
        "gpt-4o": LLMCostCard(input=2.50, output=10.00),
        "gpt-4o-2024-08-06": LLMCostCard(input=2.50, output=10.00),
        "gpt-4o-2024-05-13": LLMCostCard(input=5.00, output=15.00),
        "gpt-4o-mini": LLMCostCard(input=0.150, output=0.60),
        "gpt-4o-mini-2024-07-18": LLMCostCard(input=0.150, output=0.60),
    }

    COSTS_o1_FAMILY: Mapping[str, LLMCostCard] = {
        "o1-preview": LLMCostCard(input=15.00, output=60.00),
        "o1-preview-2024-09-12": LLMCostCard(input=15.00, output=60.00),
        "o1-mini": LLMCostCard(input=3.00, output=12.00),
        "o1-mini-2024-09-12": LLMCostCard(input=3.00, output=12.00),
    }

    MODEL_FAMILY_TO_COSTS: Mapping[str, Mapping[str, LLMCostCard]] = {
        "gpt-3.5-turbo": COSTS_GPT_35_turbo_FAMILY,
        "gpt-4": COSTS_GPT_4_FAMILY,
        "gpt-4o": COSTS_GPT_4o_FAMILY,
        "o1": COSTS_o1_FAMILY,
    }

    def find_model_costs(self, model_name: str) -> Optional[LLMCostCard]:
        model = model_name.split(" with fallback")[0]

        for model_family in self.MODEL_FAMILY_TO_COSTS.keys():
            if model.startswith(model_family):
                return self.MODEL_FAMILY_TO_COSTS[model_family].get(model)

        return None


class OpenAIChatCompletionsResult:
    def __init__(
        self,
        id: str,
        object: str,
        created: int,
        model: str,
        choices: List[Choice],
        usage: Usage,
        raw_response: Dict[str, Any],
    ) -> None:
        self._id = id
        self._object = object
        self._usage = usage
        self._model = model
        self._choices = choices
        self._created = created

        self._raw_response = raw_response

    @property
    def id(self) -> str:
        return self._id

    @property
    def model(self) -> str:
        return self._model

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def choices(self) -> Sequence[Choice]:
        return self._choices

    @property
    def raw_response(self) -> Dict[str, Any]:
        return self._raw_response

    def to_consumptions(self) -> Sequence[Consumption]:
        return [
            Consumption(1, "call", f"{self.model}"),
            Consumption(self.usage.prompt_tokens, "token", f"{self.model}:prompt_tokens"),
            Consumption(self.usage.completion_tokens, "token", f"{self.model}:completion_tokens"),
            Consumption(self.usage.total_tokens, "token", f"{self.model}:total_tokens"),
        ]

    @staticmethod
    def from_response(response: Dict[str, Any]) -> OpenAIChatCompletionsResult:
        _id = str(response.get("id"))
        _object = str(response.get("object"))
        _created = int(response.get("created", -1))
        _model = str(response.get("model"))
        _choices = [Choice.from_dict(y) for y in response.get("choices", [])]
        _usage = Usage.from_dict(response.get("usage"))
        return OpenAIChatCompletionsResult(_id, _object, _created, _model, _choices, _usage, response)


class OpenAIChatCompletionsModel(LLMBase[ChatGPTConfigurationBase]):
    """
    Represents an OpenAI language model hosted on Azure.
    """

    def __init__(
        self,
        config: ChatGPTConfigurationBase,
        provider: Provider,
        token_counter: Optional[LLMessageTokenCounterBase],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(configuration=config, token_counter=token_counter, name=name)
        self._provider = provider

    def _post_chat_request(self, context: LLMContext, messages: Sequence[LLMMessage], **kwargs: Any) -> LLMResult:

        payload = self._build_payload(messages)
        for key, value in kwargs.items():
            payload[key] = value

        context.logger.debug(
            f'message="Sending chat GPT completions request to {self._name}" payload="{truncate_dict_values_to_str(payload, 100)}"'
        )
        r = self._post_request(payload)
        context.logger.debug(
            f'message="Got chat GPT completions result from {self._name}" id="{r.id}" model="{r.model}" {r.usage}'
        )
        return LLMResult(
            choices=[c.message.content for c in r.choices],
            consumptions=r.to_consumptions(),
            raw_response=r.raw_response,
        )

    def _post_request(self, payload) -> OpenAIChatCompletionsResult:
        response = self._provider.__call__(payload)
        if response.status_code != httpx.codes.OK:
            raise LLMCallException(response.status_code, response.text, self._name)

        return OpenAIChatCompletionsResult.from_response(response.json())

    def _build_payload(self, messages: Sequence[LLMMessage]):
        payload = self._configuration.build_default_payload()
        msgs = []
        for message in messages:
            content: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]
            result: Dict[str, Any] = {"role": message.role.value}
            if message.name is not None:
                result["name"] = message.name
            for data in message.data:
                if data.is_image:
                    content.append(
                        {"type": "image_url", "image_url": {"url": f"data:{data.mime_type};base64,{data.content}"}}
                    )
                elif data.is_url:
                    content.append({"type": "image_url", "image_url": {"url": f"{data.content}"}})
            result["content"] = content
            msgs.append(result)
        payload["messages"] = msgs
        return payload
