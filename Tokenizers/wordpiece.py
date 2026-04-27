

import sys
import re
import time
import unicodedata

from functools import lru_cache



def _build_normalization_table():
    table = {}

    for cp in range(ord("A"), ord("Z") + 1):
        table[cp] = cp + 32

    for cp in range(0x0300, 0x036F + 1):
        table[cp] = None

    table[0x0386] = 0x03B1
    table[0x0388] = 0x03B5
    table[0x0389] = 0x03B7
    table[0x038A] = 0x03B9
    table[0x038C] = 0x03BF
    table[0x038E] = 0x03C5
    table[0x038F] = 0x03C9
    table[0x03AA] = 0x03B9
    table[0x03AB] = 0x03C5

    table[0x0390] = 0x03B9
    table[0x03AC] = 0x03B1
    table[0x03AD] = 0x03B5
    table[0x03AE] = 0x03B7
    table[0x03AF] = 0x03B9
    table[0x03B0] = 0x03C5
    table[0x03CA] = 0x03B9
    table[0x03CB] = 0x03C5
    table[0x03CC] = 0x03BF
    table[0x03CD] = 0x03C5
    table[0x03CE] = 0x03C9

    for cp in range(0x0391, 0x03A9 + 1):
        if cp == 0x03A2:
            continue
        if cp not in table:
            table[cp] = cp + 32

    return table


_NORMALIZE_TABLE = _build_normalization_table()


def normalize_text(text):
    out = text.translate(_NORMALIZE_TABLE)

    needs_nfd = False
    for ch in out:
        cp = ord(ch)
        if 0x1F00 <= cp <= 0x1FFF:
            needs_nfd = True
            break
        if 0x00C0 <= cp <= 0x017F:
            needs_nfd = True
            break

    if needs_nfd:
        nfd = unicodedata.normalize("NFD", out)
        out = "".join(
            ch for ch in nfd
            if unicodedata.category(ch) != "Mn"
        )
        out = out.lower()

    return out



_PUNCT_RE = re.compile(
    r"["
    r"!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~"
    r"\u00ab\u00bb"
    r"\u2010-\u2015"
    r"\u2018\u2019\u201c\u201d"
    r"\u2026"
    r"\u037e\u0387"
    r"]"
)


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]")


_WS_RE = re.compile(r"\s+")


def basic_tokenize(text):
    text = _CONTROL_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    text = normalize_text(text)


    out = []

    for word in text.split(" "):

        if not word:
            continue

        parts = _PUNCT_RE.split(word)
        puncs = _PUNCT_RE.findall(word)


        for i, part in enumerate(parts):
            if part:
                out.append(part)
            if i < len(puncs):
                out.append(puncs[i])

    return out



class WordPieceTokenizer:

    def __init__(self, unk_token="[UNK]", max_chars_per_word=200):

        self.vocab = {}
        self.ids_to_tokens = {}

        self.unk_token = unk_token
        self.unk_id    = -1

        self.max_chars_per_word = max_chars_per_word


    def load_vocab(self, vocab_file):

        with open(vocab_file, "r", encoding="utf-8") as f:

            for token_id, line in enumerate(f):
                token = line.rstrip("\n").rstrip("\r")
                self.vocab[token] = token_id
                self.ids_to_tokens[token_id] = token

        self.unk_id = self.vocab.get(self.unk_token, -1)

        print(f"Loaded vocab: {len(self.vocab)} tokens")
        print(f"  unk_token = {self.unk_token} (id={self.unk_id})")


    def _wordpiece_one_word(self, word):
        if len(word) > self.max_chars_per_word:
            return [self.unk_token]


        sub_tokens = []
        start  = 0
        n      = len(word)
        is_bad = False
        vocab  = self.vocab

        while start < n:

            end = n
            cur_substr = None

            while start < end:

                substr = word[start:end]

                if start > 0:
                    substr = "##" + substr

                if substr in vocab:
                    cur_substr = substr
                    break

                end -= 1


            if cur_substr is None:
                is_bad = True
                break

            sub_tokens.append(cur_substr)
            start = end


        if is_bad:
            return [self.unk_token]

        return sub_tokens


    def _setup_cache(self, maxsize=200_000):
        self._cached = lru_cache(maxsize=maxsize)(self._wordpiece_one_word)


    def tokenize(self, text):

        if not hasattr(self, "_cached"):
            self._setup_cache()

        out = []

        for word in basic_tokenize(text):
            pieces = self._cached(word)
            out.extend(pieces)

        return out


    def encode(self, text):

        pieces = self.tokenize(text)
        unk_id = self.unk_id
        vocab  = self.vocab

        return [vocab.get(p, unk_id) for p in pieces]


    def decode(self, ids):
        toks = [self.ids_to_tokens.get(i, self.unk_token) for i in ids]

        out_parts = []

        for t in toks:
            if t.startswith("##"):
                if out_parts:
                    out_parts[-1] = out_parts[-1] + t[2:]
                else:
                    out_parts.append(t[2:])
            else:
                out_parts.append(t)

        return " ".join(out_parts)




def main():

    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)


    vocab_file = sys.argv[1]
    text_file  = sys.argv[2]


    tokenizer = WordPieceTokenizer()
    tokenizer.load_vocab(vocab_file)


    print(f"\n--- Tokenizing {text_file} ---")


    t_start = time.perf_counter()

    total_lines  = 0
    total_tokens = 0


    with open(text_file, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.rstrip("\n")
            if not line:
                continue


            pieces = tokenizer.tokenize(line)
            ids    = tokenizer.encode(line)


            print(f"\n[line {line_number}] {line}")
            print("  pieces:", " ".join(pieces))
            print("  ids:   ", " ".join(str(x) for x in ids))
            print(f"  count:  {len(ids)} tokens")


            total_lines  += 1
            total_tokens += len(ids)


    t_end = time.perf_counter()
    elapsed_us = (t_end - t_start) * 1_000_000.0


    print("\n--- Summary ---")
    print(f"Lines tokenized: {total_lines}")
    print(f"Total tokens:    {total_tokens}")

    if total_lines > 0:
        avg = total_tokens / total_lines
        print(f"Avg tokens/line: {avg:.2f}")

    print(f"Elapsed:         {elapsed_us:.0f} us"
          f" ({elapsed_us / 1000:.2f} ms)")

    if total_lines > 0:
        per_line_us = elapsed_us / total_lines
        print(f"Per line:        {per_line_us:.1f} us")



if __name__ == "__main__":
    main()
