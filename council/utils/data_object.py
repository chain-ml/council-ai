from __future__ import annotations

import abc
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

import yaml
from typing_extensions import Self

Label = Optional[Union[str, List[str]]]


class DataObjectMetadata:
    def __init__(self, name: str, labels: Dict[str, Any], description: Optional[str] = None) -> None:
        self.name = name
        self.description = description
        self.labels = labels

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def get_label_value(self, label: str) -> Optional[Any]:
        if label in self.labels:
            return self.labels[label]
        return None

    def is_matching_labels(self, labels: Dict[str, Label]) -> bool:
        """
        Returns true if the test_case_object satisfies any of the following:
            - if value_to_check is None, check if the label exists
            - exact match of label-value pair
            - when a label maps to a list, check if value_to_check is in the list of values for that label
        """
        for label, value_to_check in labels.items():
            if not self.has_label(label):
                return False

            value = self.get_label_value(label)
            if value_to_check is not None:
                if type(value_to_check) is not type(value):
                    return False
                elif isinstance(value_to_check, str) and isinstance(value, str) and value_to_check != value:
                    return False
                elif isinstance(value_to_check, list) and isinstance(value, list):
                    if not all(v in value for v in value_to_check):
                        return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "labels": self.labels}

        if self.description is not None:
            result["description"] = self.description

        return result

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> DataObjectMetadata:
        return DataObjectMetadata(values["name"], values.get("labels", {}), values.get("description", None))


T = TypeVar("T", bound="DataObjectSpecBase")


class DataObjectSpecBase(abc.ABC):
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, values: Dict[str, Any]) -> Self:
        pass


class DataObject(Generic[T]):
    def __init__(self, kind: str, version: str, metadata: DataObjectMetadata, spec: T) -> None:
        self.kind = kind
        self.version = version
        self.metadata = metadata
        self.spec = spec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "spec": self.spec.to_dict(),
        }

    @classmethod
    def _get_kind(cls, values: Dict[str, Any]):
        return values.get("kind", None)

    @classmethod
    def _check_kind(cls, values: Dict[str, Any], expected: str) -> None:
        kind = cls._get_kind(values)
        if kind != expected:
            raise ValueError(f"Expected kind: `{expected}`, found `{kind}` instead.")

    @classmethod
    def _from_dict(cls, inner: Type[T], values: Dict[str, Any]) -> Self:
        metadata = DataObjectMetadata.from_dict(values["metadata"])
        spec = inner.from_dict(values["spec"])
        return cls(values["kind"], values["version"], metadata, spec)

    def to_yaml(self, filename: str) -> None:
        values = self.to_dict()
        with open(filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(values, f, default_flow_style=False)
