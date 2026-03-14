"""
text_cleaner.py

Utilities for cleaning lyric text before tokenization.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean a single lyric string.

    Steps:
    - lowercase
    - remove section labels (e.g., [chorus])
    - remove punctuation
    - collapse extra whitespace

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove section labels like [chorus], [verse 1]
    text = re.sub(r"\[.*?\]", "", text)

    # remove punctuation
    text = re.sub(r"[^\w\s']", "", text)

    # collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_lyrics(lyrics: list[str]) -> list[str]:
    """
    Clean a list of lyric strings.

    Parameters
    ----------
    lyrics : list[str]

    Returns
    -------
    list[str]
    """

    cleaned = []

    for lyric in lyrics:
        text = clean_text(lyric)

        if text:  # remove empty strings
            cleaned.append(text)

    return cleaned
