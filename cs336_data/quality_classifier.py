import bz2
import random
import re
import subprocess
import urllib.request

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.common import get_shared_assets_path
from cs336_data.filter import extract_text_from_html_bytes, gopher_quality_filter, identify_language

DATA_PATH = get_shared_assets_path()
WIKI_DUMP_URL = "https://dumps.wikimedia.org/enwiki/20260501/enwiki-20260501-pages-articles-multistream1.xml-p1p41242.bz2"


def get_wiki_urls(num_urls: int) -> list[str]:
    url_re = re.compile(
        r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
    )
    urls = set()
    request = urllib.request.Request(WIKI_DUMP_URL, headers={"User-Agent": "cs336-a4"})
    with bz2.open(urllib.request.urlopen(request), "rt", errors="ignore") as f:
        for line in f:
            if refs := re.search("&lt;ref&gt(.*)&lt;/ref&gt;", line):
                urls.update(url_re.findall(refs.group(0)))
            if len(urls) >= num_urls:
                break
    return sorted(urls)


def download_pages(urls: list[str], num_workers: int = 8):
    procs = []
    for i in range(num_workers):
        url_path = DATA_PATH / f"wiki/urls_{i}.txt"
        url_path.write_text("\n".join(urls[i::num_workers]))
        procs.append(subprocess.Popen(
            ["wget", "-q", "--timeout=5", "--tries=1", "-i", url_path, f"--warc-file={DATA_PATH / f'wiki/pages_{i}'}", "-O", "/dev/null"]
        ))
    for p in procs:
        p.wait()


def keep(text: str) -> bool:
    return identify_language(text)[0] == "en" and gopher_quality_filter(text)


def get_wiki_texts() -> list[str]:
    texts = []
    for path in sorted((DATA_PATH / "wiki").glob("pages_*.warc.gz")):
        with open(path, "rb") as f:
            for record in ArchiveIterator(f, record_types=WarcRecordType.response):
                if record.http_content_type and record.http_content_type.startswith("text/html"):
                    text = extract_text_from_html_bytes(record.reader.read())
                    if keep(text):
                        texts.append(text)
    return texts


def get_cc_texts(num_texts: int) -> list[str]:
    texts = []
    with open(DATA_PATH / "CC/example.warc.wet.gz", "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.conversion):
            text = record.reader.read().decode(errors="ignore")
            if keep(text):
                texts.append(text)
            if len(texts) >= num_texts:
                break
    return texts


def train(wiki_texts: list[str], cc_texts: list[str]):
    data = [f"__label__wiki {' '.join(t.split())}" for t in wiki_texts]
    data += [f"__label__cc {' '.join(t.split())}" for t in cc_texts]
    random.shuffle(data)
    num_valid = len(data) // 10
    (DATA_PATH / "quality.valid").write_text("\n".join(data[:num_valid]) + "\n")
    (DATA_PATH / "quality.train").write_text("\n".join(data[num_valid:]) + "\n")
    model = fasttext.train_supervised(input=str(DATA_PATH / "quality.train"), lr=0.1, epoch=25, wordNgrams=2)
    print(model.test(str(DATA_PATH / "quality.valid")))
    model.save_model(str(DATA_PATH / "classifiers/quality.bin"))


if __name__ == "__main__":
    random.seed(0)
    (DATA_PATH / "wiki").mkdir(exist_ok=True)
    download_pages(random.sample(get_wiki_urls(10000), 1000))
    wiki_texts = get_wiki_texts()
    cc_texts = get_cc_texts(len(wiki_texts))
    print(len(wiki_texts), len(cc_texts))
    train(wiki_texts, cc_texts)
