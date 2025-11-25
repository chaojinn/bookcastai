from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.FileHandler("debug.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

_ROMAN_NUMERAL_PATTERN = re.compile(
    r"(\s)(X{0,3}(?:IX|IV|V?I{0,3}))(?=[\s.?!])"
)


def _int_to_roman(value: int) -> str:
    mapping = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    result: list[str] = []
    remainder = value
    for numeral_value, numeral_symbol in mapping:
        while remainder >= numeral_value:
            result.append(numeral_symbol)
            remainder -= numeral_value
    return "".join(result)


def _roman_to_int(roman: str) -> int | None:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    roman_upper = roman.upper()
    total = 0
    previous = 0
    for char in reversed(roman_upper):
        value = values.get(char)
        if value is None:
            return None
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    if not (0 < total <= 3999):
        return None
    if _int_to_roman(total) != roman_upper:
        return None
    return total


def _should_convert_single_i(sentence: str, start: int, end: int) -> bool:
    """
    Placeholder for more robust disambiguation of the single-letter numeral "I".
    Currently always returns True as per the design note.
    """
    return False


def _convert_sentence(sentence: str, sentence_number: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert Roman numerals within a sentence to decimal digits.

    Returns the updated sentence and a list of change records.
    """
    changes: List[Dict[str, Any]] = []
    cursor = 0
    parts: List[str] = []
    for match in _ROMAN_NUMERAL_PATTERN.finditer(sentence):
        start, end = match.span()
        boundary = match.group(1)
        token = match.group(2)
        token_upper = token.upper()

        if any(char not in {"I", "V", "X"} for char in token_upper):
            continue
        if token_upper == "I" and not _should_convert_single_i(sentence, start, end):
            continue

        number = _roman_to_int(token_upper)
        if number is None:
            continue

        parts.append(sentence[cursor:start])
        parts.append(boundary)
        parts.append(str(number))
        cursor = end

        changes.append(
            {
                "number_sentence": sentence_number,
                "original": sentence,
                "pos": start,
                "from": token,
                "to": str(number),
            },
        )

    if not changes:
        return sentence, []

    parts.append(sentence[cursor:])
    new_sentence = "".join(parts)
    return new_sentence, changes


def convert_roman_numerals(text: str) -> Dict[str, Any]:
   
    if not isinstance(text, str) or not text:
        return {"changed_text": "" if text is None else text, "changes": []}

    # Local import to avoid circular dependency when chunk_chapter_content imports this module.
    from .chunk_chapter_content import split_sentences

    sentences = split_sentences(text, strip_parentheticals=False)
    if not sentences:
        return {"changed_text": text, "changes": []}

    changes: List[Dict[str, Any]] = []
    
    changed_text = ""
    for idx, sentence in enumerate(sentences, start=1):
        new_sentence, sentence_changes = _convert_sentence(sentence, idx)
        changes.extend(sentence_changes)
        changed = bool(sentence_changes)
        if(changed):
            changed_text += new_sentence +" "
        else:
            changed_text += sentence+" "
        
    return {"changed_text": changed_text, "changes": changes}


__all__ = ["convert_roman_numerals"]
