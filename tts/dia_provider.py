from __future__ import annotations

from pathlib import Path

import torch
try:
    from transformers import AutoProcessor, DiaForConditionalGeneration
except ImportError as exc:
    AutoProcessor = None  # type: ignore[assignment]
    DiaForConditionalGeneration = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from tts.tts_provider import TTSProvider, TTSProviderError, TTSRequest


def _resolve_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DiaTTSProvider(TTSProvider):
    """Generate dialogue audio using the Dia-1.6B model."""

    def __init__(
        self,
        model_id: str = "nari-labs/Dia-1.6B-0626",
        *,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = _resolve_device(device)

        self._processor: AutoProcessor | None = None
        self._model: DiaForConditionalGeneration | None = None

    def _load_assets(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        if AutoProcessor is None or DiaForConditionalGeneration is None:
            raise TTSProviderError(
                "DiaTTSProvider requires the transformers library with Dia support."
            ) from _TRANSFORMERS_IMPORT_ERROR

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:  # noqa: BLE001
            raise TTSProviderError(f"Failed to load Dia processor: {exc}") from exc

        try:
            self._model = DiaForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        except Exception as exc:  # noqa: BLE001
            raise TTSProviderError(f"Failed to load Dia model: {exc}") from exc

        self._model.eval()

    def tts(self, request: TTSRequest) -> Path:
        self._load_assets()
        assert self._processor is not None
        assert self._model is not None

        params = request.parsed_parameters()
        max_new_tokens = int(params.get("max_new_tokens", 3072))
        guidance_scale = float(params.get("guidance_scale", 3.0))
        temperature = float(params.get("temperature", 1.8))
        top_p = float(params.get("top_p", 0.9))
        top_k = int(params.get("top_k", 45))

        text = request.text_content.strip()
        if not text:
            raise TTSProviderError("Dia provider received empty text content.")

        try:
            inputs = self._processor(text=[text], padding=True, return_tensors="pt").to(self.device)
        except Exception as exc:  # noqa: BLE001
            raise TTSProviderError(f"Failed to prepare Dia inputs: {exc}") from exc

        with torch.no_grad():
            try:
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    guidance_scale=guidance_scale,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            except Exception as exc:  # noqa: BLE001
                raise TTSProviderError(f"Dia generation failed: {exc}") from exc

        try:
            decoded = self._processor.batch_decode(generated)
        except Exception as exc:  # noqa: BLE001
            raise TTSProviderError(f"Failed to decode Dia outputs: {exc}") from exc

        output_path = request.output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._processor.save_audio(decoded, str(output_path))
        except Exception as exc:  # noqa: BLE001
            raise TTSProviderError(f"Failed to save Dia audio: {exc}") from exc

        return output_path
