"""Failures raised while loading inputs or judging a transcript."""

from __future__ import annotations

from enum import StrEnum
from typing import NoReturn


class Outcome(StrEnum):
    INPUT_ERROR = "input_error"
    UNSUPPORTED = "unsupported"


class ErrorCode(StrEnum):
    MALFORMED_JSON = "malformed_json"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    UNSUPPORTED_ROLE = "unsupported_role"
    INVALID_TIMESTAMP = "invalid_timestamp"
    UNSORTED_TURNS = "unsorted_turns"
    DUPLICATE_NODE_ID = "duplicate_node_id"
    DANGLING_EDGE = "dangling_edge"
    UNSUPPORTED_GRAPH = "unsupported_graph"


class ContractError(Exception):
    """A workflow or transcript that cannot be evaluated."""

    outcome: Outcome
    code: ErrorCode

    def __init__(self, outcome: Outcome, code: ErrorCode, message: str) -> None:
        self.outcome = outcome
        self.code = code
        super().__init__(message)


class JudgeCode(StrEnum):
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_REFUSAL = "model_refusal"
    INVALID_RUBRIC = "invalid_rubric"
    INVALID_EVIDENCE = "invalid_evidence"


class JudgeError(Exception):
    """A semantic judgment result that cannot be used."""

    code: JudgeCode

    def __init__(self, code: JudgeCode, message: str) -> None:
        self.code = code
        super().__init__(message)


def fail_contract(code: ErrorCode, message: str) -> NoReturn:
    outcome = Outcome.UNSUPPORTED if code == ErrorCode.UNSUPPORTED_GRAPH else Outcome.INPUT_ERROR
    raise ContractError(outcome, code, message)


def fail_judge(code: JudgeCode, message: str) -> NoReturn:
    raise JudgeError(code, message)


__all__ = [
    "ContractError",
    "ErrorCode",
    "JudgeCode",
    "JudgeError",
    "Outcome",
    "fail_contract",
    "fail_judge",
]
