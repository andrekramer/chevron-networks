import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("nltk is required. Install with `pip install nltk`.") from exc


Pair = Tuple[str, str]


@dataclass
class SplitData:
    train: List[Tuple[int, int, int, int]]
    val: List[Tuple[int, int, int, int]]
    test: List[Tuple[int, int, int, int]]
    vocab: List[str]


class PairDataset(Dataset):
    def __init__(self, rows: Sequence[Tuple[int, int, int, int]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w1, w2, match, polarity = self.rows[idx]
        return {
            "w1": torch.tensor(w1, dtype=torch.long),
            "w2": torch.tensor(w2, dtype=torch.long),
            "match": torch.tensor(match, dtype=torch.float32),
            "polarity": torch.tensor(polarity, dtype=torch.float32),
        }


def _normalize_lemma(text: str) -> str:
    return text.lower().replace("_", " ").strip()


def _load_wordnet() -> None:
    try:
        _ = wn.synsets("good")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def _collect_antonym_pairs() -> List[Pair]:
    _load_wordnet()
    pairs = set()
    for syn in wn.all_synsets():
        for lemma in syn.lemmas():
            head = _normalize_lemma(lemma.name())
            for ant in lemma.antonyms():
                tail = _normalize_lemma(ant.name())
                if head == tail:
                    continue
                a, b = sorted((head, tail))
                pairs.add((a, b))
    return sorted(pairs)


def _sample_negative_pairs(
    positives: Sequence[Pair],
    vocab: Sequence[str],
    multiplier: int,
    rng: random.Random,
) -> List[Pair]:
    positive_set = set(positives)
    negatives = set()
    target = len(positives) * max(multiplier, 1)

    while len(negatives) < target:
        a = vocab[rng.randrange(len(vocab))]
        b = vocab[rng.randrange(len(vocab))]
        if a == b:
            continue
        p = tuple(sorted((a, b)))
        if p in positive_set:
            continue
        negatives.add(p)

    return list(negatives)


def _encode_rows(
    positives: Sequence[Pair],
    negatives: Sequence[Pair],
    token_to_id: Dict[str, int],
    rng: random.Random,
) -> List[Tuple[int, int, int, int]]:
    rows: List[Tuple[int, int, int, int]] = []

    for a, b in positives:
        a_id = token_to_id[a]
        b_id = token_to_id[b]

        canonical_first = rng.random() < 0.5
        if canonical_first:
            rows.append((a_id, b_id, 1, 1))
            rows.append((b_id, a_id, 1, 0))
        else:
            rows.append((b_id, a_id, 1, 0))
            rows.append((a_id, b_id, 1, 1))

    for a, b in negatives:
        a_id = token_to_id[a]
        b_id = token_to_id[b]
        if rng.random() < 0.5:
            rows.append((a_id, b_id, 0, 0))
        else:
            rows.append((b_id, a_id, 0, 0))

    rng.shuffle(rows)
    return rows


def _split_rows(
    rows: Sequence[Tuple[int, int, int, int]],
    val_ratio: float,
    test_ratio: float,
) -> SplitData:
    n = len(rows)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    train = list(rows[:n_train])
    val = list(rows[n_train:n_train + n_val])
    test = list(rows[n_train + n_val:])
    return SplitData(train=train, val=val, test=test, vocab=[])


def build_wordnet_splits(
    seed: int = 7,
    negative_multiplier: int = 1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_pairs: int = 0,
) -> SplitData:
    rng = random.Random(seed)

    positives = _collect_antonym_pairs()
    if max_pairs > 0:
        positives = positives[:max_pairs]

    vocab = sorted({word for pair in positives for word in pair})
    negatives = _sample_negative_pairs(positives, vocab, negative_multiplier, rng)

    token_to_id = {token: i for i, token in enumerate(vocab)}
    rows = _encode_rows(positives, negatives, token_to_id, rng)

    split = _split_rows(rows, val_ratio=val_ratio, test_ratio=test_ratio)
    split.vocab = vocab
    return split
