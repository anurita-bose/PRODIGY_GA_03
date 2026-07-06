import os
import random
import re
import shutil
import urllib.request

from markov_chain import MarkovChain

CORPUS_URL = "https://www.gutenberg.org/files/11/11-0.txt"


def load_corpus():
    req = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    start = raw.find("*** START")
    start = raw.find("\n", start) + 1
    end = raw.find("*** END")
    text = raw[start:end]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_word_level(text, out_dir):
    lines = []
    for order in (1, 2, 3):
        model = MarkovChain(order=order, level="word")
        model.train(text)
        sample = model.generate(length=60)
        lines.append(f"[word-level | order={order}]\n{sample}\n")
    result = "\n".join(lines)
    with open(os.path.join(out_dir, "word_level_samples.txt"), "w") as f:
        f.write(result)
    print(result)


def run_char_level(text, out_dir):
    lines = []
    for order in (2, 4, 8):
        model = MarkovChain(order=order, level="char")
        model.train(text)
        sample = model.generate(length=300)
        lines.append(f"[char-level | order={order}]\n{sample}\n")
    result = "\n".join(lines)
    with open(os.path.join(out_dir, "char_level_samples.txt"), "w") as f:
        f.write(result)
    print(result)


def main():
    random.seed(42)
    out_dir = "outputs"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    text = load_corpus()
    print(f"Corpus loaded: {len(text)} characters, {len(text.split())} words\n")
    run_word_level(text, out_dir)
    run_char_level(text, out_dir)
    archive_path = shutil.make_archive("markov_outputs", "zip", out_dir)
    print(f"Samples archived to {archive_path}")


if __name__ == "__main__":
    main()
