import re

import fasttext
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

from cs336_data.common import get_shared_assets_path

CLASSIFIERS_PATH = get_shared_assets_path() / "classifiers"


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html = html_bytes.decode(detect_encoding(html_bytes), errors="replace")
    return extract_plain_text(html)


models = {}


def predict(model_name: str, text: str) -> tuple[str, float]:
    if model_name not in models:
        models[model_name] = fasttext.load_model(str(CLASSIFIERS_PATH / model_name))
    labels, probs = models[model_name].predict(text.replace("\n", " "))
    return labels[0].replace("__label__", ""), float(probs[0])


def identify_language(text: str) -> tuple[str, float]:
    return predict("lid.176.bin", text)


def classify_nsfw(text: str) -> tuple[str, float]:
    return predict("dolma_fasttext_nsfw_jigsaw_model.bin", text)


def classify_toxic_speech(text: str) -> tuple[str, float]:
    return predict("dolma_fasttext_hatespeech_jigsaw_model.bin", text)


def classify_quality(text: str) -> tuple[str, float]:
    return predict("quality.bin", text)


def mask_emails(text: str) -> tuple[str, int]:
    return re.subn(r"[\w.+-]+@[\w-]+\.[\w.-]+", "|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return re.subn(r"(?<!\d)(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)", "|||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> tuple[str, int]:
    octet = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    return re.subn(rf"\b{octet}\.{octet}\.{octet}\.{octet}\b", "|||IP_ADDRESS|||", text)


def gopher_quality_filter(text: str) -> bool:
    words = text.split()
    if len(words) < 50 or len(words) > 100000:
        return False
    mean_word_length = sum(len(word) for word in words) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    lines = text.splitlines()
    if sum(line.rstrip().endswith("...") for line in lines) / len(lines) > 0.3:
        return False
    if sum(bool(re.search(r"[a-zA-Z]", word)) for word in words) / len(words) < 0.8:
        return False
    return True
