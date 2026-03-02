from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class TTSProviderError(RuntimeError):
    """Raised when a text-to-speech provider cannot fulfil a request."""


@dataclass(frozen=True)
class TTSRequest:
    """Represents a text-to-speech job expressed in a provider-agnostic form."""

    text_content: str
    output_file: Path
    raw_parameters: str = "{}"

    def parsed_parameters(self) -> dict[str, Any]:
        """Return the parameters JSON string as a dictionary."""
        return parse_parameters(self.raw_parameters)


def parse_parameters(params_json: str | None) -> dict[str, Any]:
    """Decode a provider parameter payload into a dictionary."""
    payload = params_json or "{}"
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise TTSProviderError(f"Invalid parameters JSON: {exc}") from exc

    if not isinstance(data, Mapping):
        raise TTSProviderError("Parameters JSON must decode to an object.")

    # Normalise keys to str for downstream consumers.
    return {str(key): value for key, value in data.items()}


class TTSProvider(ABC):
    """Interface that all text-to-speech backends must implement."""

    @abstractmethod
    def tts(self, request: TTSRequest) -> Path:
        """Generate audio for the supplied request and return the output path."""

    @abstractmethod
    def get_english_voices(self) -> list[dict[str, str]]:
        """Return a list of available English voices as dicts with 'name' and 'code' keys."""

