"""
Add an explicit sociodemographic preamble ("As a woman, ...") to the opinions of a dataset
(--dataset: hby, the Habermas-hybrid of the pilot; hb or vp, from generate_opinions.py), one
sociodemographic variable at a time.

For hby and hb, references are the human Habermas opinions (origin == human), candidates the
synthetic ones (origin == synthetic). For vp, all the opinions are synthetic and are both
references and candidates (the two output files hold the same opinions). Each opinion is written once per level of the variable, plus a neutral
level whose identity phrase names no sociodemographic ("a person"): it adds the same preamble,
so it controls for the extra length and framing of the other levels.

Pattern-based: each opinion is first turned into a frame with an {identity} slot, built without
looking at the level, and the frame is then filled with each identity phrase. The versions of an
opinion thus differ only in the identity phrase, and the opinion itself is kept verbatim (only
the case of its first letter may change). The slot goes
  - after the leading discourse markers:   "Honestly, I think ..."     -> "Honestly, as {identity}, I think ..."
  - before a first-person opening:         "I don't think ..."         -> "As {identity}, I don't think ..."
  - before a subordinate clause/question:  "While I agree, ..."        -> "As {identity}, while I agree, ..."
  - before a first person in the first 8 words of the first sentence:
                                           "It is my opinion that ..." -> "As {identity}, it is my opinion that ..."
  - otherwise with an "I think" bridge,    "The government should ..." -> "As {identity}, I think the government should ..."
    so that the preamble does not dangle on a non-first-person subject.
The first word after the slot is lowercased, unless it is "I", an acronym or a proper noun (a
word that is mostly capitalised mid-sentence in the corpus).

Output: {output_dir}/{dataset}_reference_{variable}.csv, {output_dir}/{dataset}_candidate_{variable}.csv,
in long format: one row per opinion and level, the level in the {variable} column.

Usage:
    python add_sociodem_preamble.py --variable gender                 # hby
    python add_sociodem_preamble.py --variable gender --dataset hb
    python add_sociodem_preamble.py --variable gender --dataset vp
"""
import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")
log = logging.getLogger("add_sociodem_preamble")

DATA_DIR = Path(__file__).resolve().parent / "data"

SLOT = "{identity}"
NEUTRAL_LEVEL = "neutral"
NEUTRAL_PHRASE = "a person"

# Variable -> levels (level -> identity phrase, as in "As {phrase}, ...") and a pattern of the cues
# of the variable already in the opinion, which a preamble may contradict ("as a woman on my own").
SOCIODEMOGRAPHICS = {
    "gender": {
        "levels": {"female": "a woman", "male": "a man"},
        "cue_pattern": r"\b(?:as|being|i am|i'm|im) an? (?:woman|man|female|male|girl|boy|lady|guy|mother|father|mum"
                       r"|mom|dad|wife|husband)\b|\bmy (?:husband|wife|boyfriend|girlfriend)\b",
    },
}

# Dataset -> its file in data/ and, for each output split, the origin of its opinions. HB: the human
# Habermas opinions are the references, the synthetic ones the candidates (as HBY). VP has only
# synthetic opinions, which are both references and candidates (score.py pairs them with each other).
DATASETS = {
    "hby": {"file": "hby.csv", "splits": {"reference": "human", "candidate": "synthetic"}},
    "hb": {"file": "hb.csv", "splits": {"reference": "human", "candidate": "synthetic"}},
    "vp": {"file": "vp.csv", "splits": {"reference": "synthetic", "candidate": "synthetic"}},
}
SPLITS = DATASETS["hby"]["splits"]

APOS = "'’"
DISCOURSE_MARKERS = ["to be honest", "to be fair", "of course", "honestly", "frankly", "personally", "well", "yes",
                     "yeah", "no", "ok", "okay", "absolutely", "definitely", "sadly", "unfortunately", "obviously",
                     "clearly", "overall", "ultimately", "basically", "generally"]
# A marker counts only when followed by punctuation ("No, ...") or by a word that cannot make it a
# determiner ("No I don't ...", "No this would ..."; not "No one ...", "No child ...").
MARKER_NEXT_WORDS = ["i", "it", "its", "this", "that", "there", "they", "we", "the", "you", "he", "she", "because", "not",
                     "as", "although", "but", "if", "in", "of", "absolutely", "definitely"]
# Dashes only with a space before them, so that "Well-being ..." is not a marker.
MARKER_RE = re.compile(rf"(?P<marker>(?:{'|'.join(DISCOURSE_MARKERS)}))\b"
                       rf"(?:(?P<punct>\s*[,.!?;:]+|\s+[-–—]+)|(?=\s+(?:{'|'.join(MARKER_NEXT_WORDS)})\b))\s*", re.I)
FIRST_PERSON_RE = re.compile(rf"(?:i(?:[{APOS}](?:m|ve|d|ll)|m|ve)?|my|me|myself|in my|for me|to me|speaking)\b", re.I)
FIRST_PERSON_WORD_RE = re.compile(rf"\b(?:i(?:[{APOS}](?:m|ve|d|ll)|m|ve)?|my|me|myself)\b", re.I)
FIRST_PERSON_NEAR_WORDS = 8
SUBORDINATE_RE = re.compile(r"(?:while|whilst|although|though|even though|even if|if|when|whenever|because|since"
                            r"|unless|despite|whether|given|as long as|so long as|once)\b", re.I)
QUESTION_RE = re.compile(r"[^.!?]*\?")
I_FORM_RE = re.compile(rf"I(?:[{APOS}]?(?:m|ve|d|ll))?")
WORD_RE = re.compile(rf"[A-Za-z][A-Za-z{APOS}]*")
MID_SENTENCE_WORD_RE = re.compile(rf"(?<=[a-z0-9,;:] )[A-Za-z][A-Za-z{APOS}]*")


def find_proper_nouns(texts, min_count=3, min_ratio=0.8):
    """Lowercased words that are capitalised in at least min_ratio of their mid-sentence uses."""
    capitalised, lowercase = Counter(), Counter()
    for text in texts:
        for word in MID_SENTENCE_WORD_RE.findall(text):
            (capitalised if word[0].isupper() else lowercase)[word.lower()] += 1
    return {w for w, n in capitalised.items() if n >= min_count and n / (n + lowercase[w]) >= min_ratio}


def lower_first(text, proper_nouns):
    """text with its first letter lowercased, unless its first word is "I", an acronym or a proper noun."""
    match = WORD_RE.match(text)
    if not match:
        return text
    word = match.group(0)
    if I_FORM_RE.fullmatch(word) or (word.isupper() and len(word) > 1) or word.lower() in proper_nouns:
        return text
    return text[0].lower() + text[1:]


def build_frame(text, proper_nouns):
    """(frame, rule): text with an "As {identity}, " preamble in it, and the rule that placed it."""
    text = text.lstrip()
    pos, last_marker = 0, None
    while match := MARKER_RE.match(text, pos):
        pos, last_marker = match.end(), match
    rest = text[pos:]
    if not rest:  # only discourse markers: keep them whole
        return f"Speaking as {SLOT}: {text}", "prefix"

    if last_marker is None:
        head, rule = f"As {SLOT}, ", ""
    else:
        punct = (last_marker.group("punct") or "").rstrip()  # keeps the space of "No - ..."
        before = text[:last_marker.end("marker")]
        if punct and punct[-1] in ".!?":
            head = f"{before}{punct} As {SLOT}, "
        else:
            head = f"{before}{punct or ','} as {SLOT}, "
        rule = "marker+"

    if FIRST_PERSON_RE.match(rest):
        rule += "first_person"
    elif SUBORDINATE_RE.match(rest) or QUESTION_RE.match(rest):
        rule += "subordinate"
    elif FIRST_PERSON_WORD_RE.search(" ".join(re.split(r"[.!?]", rest, maxsplit=1)[0].split()[:FIRST_PERSON_NEAR_WORDS])):
        rule += "first_person_near"  # "In the case of the pandemic, I think ...": no bridge, or "I think" twice
    else:
        head, rule = head + "I think ", rule + "bridge"
    frame = head + lower_first(rest, proper_nouns)
    assert frame.count(SLOT) == 1 and frame.endswith(rest[1:]), text
    return frame, rule


def intervene(df, variable, proper_nouns):
    """One row per opinion of df and level of the variable, with the preamble in "text"."""
    spec = SOCIODEMOGRAPHICS[variable]
    levels = {**spec["levels"], NEUTRAL_LEVEL: NEUTRAL_PHRASE}
    cue_re = re.compile(spec["cue_pattern"], re.I)
    rows = []
    for row in df.to_dict("records"):
        frame, rule = build_frame(row["text"], proper_nouns)
        cue = cue_re.search(row["text"])
        for level, phrase in levels.items():
            rows.append({**row, "text_id": f"{row['text_id']}_{variable}_{level}", "text": frame.replace(SLOT, phrase),
                         variable: level, "identity_phrase": phrase, "original_text_id": row["text_id"],
                         "original_text": row["text"], "preamble_frame": frame, "preamble_method": "pattern",
                         "preamble_rule": rule, f"{variable}_cue_in_original": cue.group(0) if cue else None})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variable", required=True, choices=list(SOCIODEMOGRAPHICS))
    parser.add_argument("--dataset", choices=list(DATASETS), default="hby")
    parser.add_argument("--input", type=Path, default=None, help="Default: data/{dataset file}.")
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    data = pd.read_csv(args.input or DATA_DIR / spec["file"])
    proper_nouns = find_proper_nouns(pd.concat([data["text"], data["situation"].drop_duplicates()]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, origin in spec["splits"].items():
        out = intervene(data[data["origin"] == origin], args.variable, proper_nouns)
        out_path = args.output_dir / f"{args.dataset}_{split}_{args.variable}.csv"
        out.to_csv(out_path, index=False)
        log.info("Wrote %d rows (%d opinions x %d levels) to %s", len(out), out["original_text_id"].nunique(),
                 out[args.variable].nunique(), out_path)
        log.info("[%s] preamble rules: %s", split,
                 out.drop_duplicates("original_text_id")["preamble_rule"].value_counts().to_dict())
        n_cues = out.drop_duplicates("original_text_id")[f"{args.variable}_cue_in_original"].notna().sum()
        if n_cues:
            log.warning("[%s] %d opinions already state a %s cue (column %s_cue_in_original)", split, n_cues,
                        args.variable, args.variable)


if __name__ == "__main__":
    main()
