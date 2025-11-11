
"""Utilities for working with Roman numerals."""

from __future__ import annotations

import re

_ROMAN_NUMERAL_PATTERN = re.compile(
    r"\b(?=[MDCLXVI]+\b)M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\b",
    re.IGNORECASE,
)
_LABELLED_CHAPTER_PATTERN = re.compile(
    r"\bchapter\s+(?P<num>(?:[MDCLXVI]+|\d+))\b",
    re.IGNORECASE,
)
_NUMERIC_TOKEN_PATTERN = re.compile(
    r"\b(?P<num>(?:[MDCLXVI]+|\d+))\b",
    re.IGNORECASE,
)

_ONES = {
    0: "",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}
_TEENS = {
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}


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


def _number_to_words(value: int) -> str:
    if value == 0:
        return "zero"
    parts: list[str] = []
    remainder = value

    if remainder >= 1000:
        thousands = remainder // 1000
        parts.append(_number_to_words(thousands))
        parts.append("thousand")
        remainder %= 1000

    if remainder >= 100:
        hundreds = remainder // 100
        parts.append(_ONES[hundreds])
        parts.append("hundred")
        remainder %= 100

    if remainder >= 20:
        tens = remainder // 10
        parts.append(_TENS[tens])
        remainder %= 10
        if remainder:
            parts.append(_ONES[remainder])
        remainder = 0
    elif remainder >= 10:
        parts.append(_TEENS[remainder])
        remainder = 0

    if remainder > 0:
        parts.append(_ONES[remainder])

    return " ".join(token for token in parts if token)


def _format_chapter_phrase(number: int) -> str:
    words = _number_to_words(number).upper()
    return f"CHAPTER {words}".strip()


def _convert_token(token: str) -> str | None:
    if not token:
        return None
    if token.isdigit():
        number = int(token)
        if number <= 0:
            return None
        return _format_chapter_phrase(number)
    if _ROMAN_NUMERAL_PATTERN.fullmatch(token):
        number = _roman_to_int(token)
        if number is None:
            return None
        return _format_chapter_phrase(number)
    return None


def replace_numeric_titles(text: str) -> str:
    """Replace Roman numerals and decimal digits with their English word equivalents."""

    def _replace(match: re.Match[str]) -> str:
        token = match.group("num")
        converted = _convert_token(token)
        return converted or match.group(0)

    text = _LABELLED_CHAPTER_PATTERN.sub(_replace, text)
    text = _NUMERIC_TOKEN_PATTERN.sub(_replace, text)
    return re.sub(r"(CHAPTER [A-Z\s]+)\.(?=\s|$)", r"\1", text)


# Backwards compatibility for existing imports.
replace_roman_numerals = replace_numeric_titles


__all__ = ["replace_numeric_titles", "replace_roman_numerals"]
