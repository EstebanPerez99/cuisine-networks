# Detecting perceived authenticity in recipe reviews

The unit of evidence is a sentence in a Food.com review. A candidate review contains at least one sentence in which an authenticity term occurs with a recipe attribute. This includes criticism and comparisons, as well as endorsements. Review IDs and recipe IDs link each result back to the cleaned data.

## Results on the cleaned corpus

Across 132,744 reviews covering 24,469 recipes, the screen flagged 7,281 reviews (5.48%) covering 4,295 recipes. They contain 7,762 distinct matching sentences. The 8,126 evidence rows include ties and multiple authenticity terms within a sentence. Review IDs and recipe IDs were checked against the full input.

The model learned vectors for 13,380 tokens from 608,802 nonempty tokenized sentences. Of the 88 published terms, 45 occurred often enough for neighbor lookup. The recipe lexicon adds 10 authenticity terms and 187 attribute words from the ranked neighbors. For example, the nearest words to *authentic* included *authentically* (0.759), *authenic* (0.696), *comparable* (0.687), and *americanized* (0.679). The first two capture an inflectional variant and a misspelling. The search excludes *americanized* and *faux*, which can describe adaptation and imitation. *Comparable* was excluded because the comparison need not involve authenticity.

### Reading the matches

The audit below used the original lexicon, before removal of *americanized* and *faux*. One audited review no longer qualifies.

I inspected 40 distinct matched sentences sampled uniformly with pandas random state 42. I also inspected 10 sentences sampled with random state 43 from the subset whose first recorded term was *authentic*. These are AI-coded sentence readings. A positive judgment requires the reviewer to say that the prepared dish fits a culinary tradition or reproduces an identifiable reference. A departure requires a corresponding judgment of divergence. The coding unit is the matched sentence, so a review may make a stronger claim elsewhere.

| Sentence reading | Random matches | Targeted *authentic* sample |
| --- | ---: | ---: |
| Endorses the prepared dish | 7 | 6 |
| Describes a departure | 1 | 2 |
| Discusses an attribute alone | 7 | 0 |
| Discusses a reference or preference alone | 4 | 2 |
| Other use | 21 | 0 |
| Total | 40 | 10 |

The sample includes a reviewer who compared the dish to an aunt’s bhindi masala, one who described a Mexican dish as Americanized, and one who used *authentic* to question another review’s authorship. Common incidental matches include homemade accompaniments and the phrase *to be honest*. The two samples have different selection probabilities and should be interpreted separately. Neither measures recall because neither sampled unmatched sentences.

## Method

Le et al. (2022, §3.1) used Word2Vec and contextual screening to identify authenticity terms and attribute words, then searched for both within a sentence. Their companion paper specifies 10 nearest neighbors per authenticity term, 20 per attribute word, and selection of the closest attribute to each term, retaining all ties. The subsequent thematic analysis in the 2022 paper used purposive selection within pair types until themes recurred. [Study](https://doi.org/10.1016/j.ijhm.2021.103102) · [Companion paper, §3.1](https://research-repository.griffith.edu.au/server/api/core/bitstreams/307513bd-e604-468b-8ee7-e6428a46056a/content)

This adaptation starts with the published 88 terms and six recipe-specific attribute groups. The groups cover the dish, ingredients, preparation, sensory qualities, geographic or culinary origin, and personal or restaurant reference points. Starting from the published terms preserves broad words such as *real*, *usual*, and *careful*. An equally close attribute on either side of a term is retained. Each distinct term–attribute–distance combination is recorded once per sentence.

Word2Vec was trained exclusively on the text of the cleaned reviews. HTML entities are decoded. Punctuation and line breaks delimit sentences, tokens are lowercased, and alternate spellings of compound seeds such as *home-made* and *middle eastern* are joined. This rule-based splitter can split on abbreviations or miss punctuation-free boundaries. The chosen training settings are skip-gram, 100 dimensions, a maximum window of five tokens, five negative samples, minimum frequency five, five epochs, one worker, and seed 42. All observed seed words remain eligible for extraction even if too rare for training. Cosine similarity proposes vocabulary for inspection. The extraction rule uses accepted words and sentence positions.

The authors used researchers' judgments on purposive samples of 2,000 reviews in vocabulary development. This implementation uses an explicit recipe seed list, a curated subset of the learned neighbors, and a sentence audit. It does not establish human coding agreement or thematic saturation. A review that passes the screen may discuss another dish or use a term incidentally.

## Reproduce

Run from the repository root with a Python 3.9 virtual environment. The project’s Python 3.14 could not build Gensim 4.4.0 during setup.

```sh
/usr/bin/python3 -m venv .venv/word2vec
.venv/word2vec/bin/python -m pip install gensim==4.3.3
.venv/word2vec/bin/python scripts/recipe_word2vec.py train
.venv/word2vec/bin/python scripts/recipe_word2vec.py detect
```

Outputs are under `data/processed/recipe_word2vec/`. The full model is saved as `word2vec.model`. The neighbor table includes seed words absent from the model with blank ranks. The review table contains every cleaned review, including those with zero matches. The sentence-pair table records the sentence, the matched tokens and their distance in intervening tokens.

The final lexicon records each term’s origin. The pair-frequency table counts each sentence once per term and category. The sentence audit contains the sampled text, assigned reading, and rationale.
