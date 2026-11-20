"""OpenAI structured-output adapter."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from openai import OpenAI, OpenAIError
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
)

from .config import load_settings
from .errors import JudgeCode, JudgeError, fail_judge as fail
from .models import StructuredCompleter
from .utils import JsonValue


@dataclass(frozen=True, slots=True)
class _OpenAICompleter:
    client: OpenAI
    model: str
    temperature: float

    def complete(
        self,
        *,
        instructions: str,
        content: str,
        schema_name: str,
        schema: Mapping[str, JsonValue],
        label: str,
    ) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                temperature=self.temperature,
                instructions=instructions,
                input=content,
                stream=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except OpenAIError as exc:
            raise JudgeError(JudgeCode.MODEL_UNAVAILABLE, f"{label} failed: {exc}.") from exc
        return _response_text(response, label)


def _refusal_fallback(label: str) -> str:
    if label == "Rubric generation":
        return "The model refused to generate a rubric."
    return "The model refused to judge evidence."


def _response_text(response: Response, label: str) -> str:
    if response.status == "failed":
        if response.error is None:
            fail(JudgeCode.MODEL_UNAVAILABLE, f"{label} failed.")
        fail(
            JudgeCode.MODEL_UNAVAILABLE,
            f"{label} failed: {response.error.code}: {response.error.message}.",
        )
    if response.status != "completed":
        reason = response.status if response.status is not None else "unfinished"
        details = response.incomplete_details
        if details is not None and details.reason is not None:
            reason = details.reason
        fail(JudgeCode.MODEL_UNAVAILABLE, f"{label} stopped: {reason}.")
    for item in response.output:
        if not isinstance(item, ResponseOutputMessage):
            continue
        for part in item.content:
            if isinstance(part, ResponseOutputRefusal):
                reason = part.refusal.strip()
                if reason == "":
                    reason = _refusal_fallback(label)
                fail(JudgeCode.MODEL_REFUSAL, reason)
            if isinstance(part, ResponseOutputText):
                if part.text.strip() == "":
                    fail(JudgeCode.MODEL_UNAVAILABLE, f"{label} returned empty text.")
                return part.text
    fail(JudgeCode.MODEL_UNAVAILABLE, f"{label} returned no text.")


def openai_completer(
    api_key: str | None = None,
    model: str | None = None,
) -> StructuredCompleter:
    """Build a completer for one configured OpenAI model.

    Omitted values are read from the environment after loading .env.
    """

    settings = load_settings(api_key, model)
    if settings.api_key == "":
        fail(JudgeCode.MODEL_UNAVAILABLE, "OPENAI_API_KEY is required.")
    if settings.model == "":
        fail(JudgeCode.MODEL_UNAVAILABLE, "OPENAI_MODEL is required.")
    return _OpenAICompleter(
        client=OpenAI(api_key=settings.api_key),
        model=settings.model,
        temperature=settings.temperature,
    )


__all__ = ["openai_completer"]
