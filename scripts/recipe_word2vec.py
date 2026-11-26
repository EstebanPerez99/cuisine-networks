"""Train recipe-review Word2Vec, inspect neighbors, then extract sentence pairs."""
import argparse
import csv
import html
import re
from collections import Counter
from pathlib import Path
from le_authenticity_terms import TERMS, stable_hash


DATA = Path(__file__).resolve().parents[1] / 'data' / 'processed'
OUT = DATA / 'recipe_word2vec'
SEEDS = {
    'dish': 'recipe dish meal soup curry salsa bread pizza noodles dumplings enchiladas hummus salad',
    'ingredient': 'ingredients spice spices seasoning sauce cheese rice chicken garlic beans',
    'preparation': 'method technique preparation cooking frying baking',
    'sensory': 'taste flavor texture aroma',
    'origin': 'italian mexican chinese greek indian french thai japanese middle_eastern italy mexico china greece india france thailand japan',
    'reference': 'restaurant grandmother mother family childhood',
}
COMPOUNDS = [t.replace('-', '_') for t in TERMS if '-' in t or '_' in t] + ['middle_eastern']
COMPOUND_RE = re.compile(r'\b(?:' + '|'.join(t.replace('_', r'[\s‐‑–—-]*') for t in COMPOUNDS) + r')\b')
JOINED = {t.replace('_', ''): t for t in COMPOUNDS}

def sentences(review):
    clean = re.sub(r'<br\s*/?>', '\n', html.unescape(review), flags=re.I)
    clean = re.sub(r'</?[a-zA-Z][^>]*>', ' ', clean)
    for sentence in re.split(r'(?<=[.!?])(?=\s|[A-Z])|[\r\n]+', clean):
        sentence = sentence.strip()
        if sentence:
            yield sentence

def tokens(sentence):
    sentence = COMPOUND_RE.sub(lambda m: JOINED[re.sub('[^a-z]', '', m[0])], sentence.lower())
    return re.findall(r"[^\W\d_]+(?:[_'][^\W\d_]+)*|\d+", sentence)

def reviews():
    with (DATA / 'reviews.csv').open(newline='') as f:
        yield from csv.DictReader(f)

def dictionaries():
    lex = {'authenticity': {t.replace('-', '_') for t in TERMS}}
    lex.update({k: set(v.split()) for k, v in SEEDS.items()})
    return lex

def train():
    from gensim.models import Word2Vec
    corpus = OUT / 'training_sentences.txt'
    with corpus.open('w') as f:
        for review in reviews():
            for sentence in sentences(review['review']):
                words = tokens(sentence)
                if words:
                    print(*words, file=f)
    from gensim.models.word2vec import LineSentence
    model = Word2Vec(sentences=LineSentence(str(corpus)), vector_size=100, window=5,
                     min_count=5, sg=1, negative=5, epochs=5, workers=1, seed=42,
                     hashfxn=stable_hash)
    # Training is complete. Keep the serialized model loadable outside this script.
    model.hashfxn = hash
    model.save(str(OUT / 'word2vec.model'))
    corpus.unlink()
    with (OUT / 'neighbors.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'seed', 'rank', 'neighbor', 'cosine'])
        for category, seeds in dictionaries().items():
            for seed in sorted(seeds):
                if seed not in model.wv:
                    w.writerow([category, seed, '', '', ''])
                    continue
                n = 10 if category == 'authenticity' else 20
                for rank, (word, score) in enumerate(model.wv.most_similar(seed, topn=n), 1):
                    w.writerow([category, seed, rank, word, score])

def detect():
    lex = dictionaries()
    with (OUT / 'neighbors.csv').open() as f:
        for row in csv.DictReader(f):
            if row['neighbor'] in ACCEPTED.get(row['category'], '').split():
                lex[row['category']].add(row['neighbor'])
    with (OUT / 'lexicon.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'token', 'source'])
        seeds = dictionaries()
        for category, words in lex.items():
            for word in sorted(words):
                source = 'seed' if word in seeds[category] else 'word2vec'
                writer.writerow([category, word, source])
    au = lex.pop('authenticity')
    attributes = {word: cat for cat, words in lex.items() for word in words}
    assert au.isdisjoint(attributes)
    pair_counts = Counter()
    with (OUT / 'sentence_pairs.csv').open('w', newline='') as pf, (OUT / 'reviews.csv').open('w', newline='') as rf:
        pairs, scores = csv.writer(pf), csv.writer(rf)
        keys = ['review_id', 'recipe_id', 'cuisine']
        pairs.writerow(keys + ['sentence_number', 'sentence', 'authenticity_term', 'category', 'attribute', 'intervening_words'])
        scores.writerow(keys + ['matched_sentences', 'candidate'])
        for review in reviews():
            ids = [review[k] for k in keys]
            hits = 0
            for number, sentence in enumerate(sentences(review['review']), 1):
                words = tokens(sentence)
                if au.isdisjoint(words):
                    continue
                attrs = [(i, w) for i, w in enumerate(words) if w in attributes]
                if not attrs:
                    continue
                evidence = set()
                for i, term in enumerate(words):
                    if term in au:
                        gap = min(abs(i-j) for j, _ in attrs)
                        for j, w in attrs:
                            if abs(i-j) == gap:
                                evidence.add((term, attributes[w], w, gap-1))
                for pair in sorted(evidence):
                    pairs.writerow(ids + [number, sentence, *pair])
                pair_counts.update({(term, cat) for term, cat, _, _ in evidence})
                hits += 1
            scores.writerow(ids + [hits, int(hits > 0)])

    with (OUT / 'pair_frequencies.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['authenticity_term', 'category', 'sentences'])
        for (term, category), count in sorted(pair_counts.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow([term, category, count])

# Retain direct variants, terms for reproducing a reference, and concrete attribute words.
# Never accept a neighbor solely because it exceeds a cosine threshold.
ACCEPTED = {
    'authenticity': 'authentically authenic homeade speciality clone copycat recreate duplicate reproduce imitate',
    'dish': 'dishes meals receipe recipie recepie recipee reciepe recpe rcipe breadsticks naan focaccia chapati casserole entree meze dessert dosas dolmas potstickers wontons chimichangas burritos quesadillas tacos tamales fajitas nachos enchilada burrito lasagna meatballs tostadas tzatziki bruschetta guacamole hummous falafel raita tabbouleh quesadilla pasta ramen pizzas taco tamale sandwich salads soups stew minestrone risotto chowder pozole bisque stews stroganoff',
    'ingredient': 'chickpeas garbanzos lentils peas mozzarella cheddar cheeses parmesan feta ricotta beef pork duck shrimp cumin oregano rosemary shallot onion ginger ingrediants ingredents seasonings vegetables basmati quinoa couscous bulgur sauces sriracha cayenne herbs masalas',
    'origin': 'cyprus basque sicily crete tokyo chinatown korean asian gujarati sichuan cantonese israel rome egypt arabian tuscan balkan okinawa hispanic kerala persian lebanese pakistani texmex portuguese creole turkish vietnamese latin mediterranean american arabic caribbean szechuan jamaican malaysian kashmiri',
    'preparation': 'proofing broiling folding simmering sauteing deglazing browning searing steaming sautéing toasting roasting dredging caramelizing methods procedure preperation marination techniques',
    'sensory': 'smell fragrance scent aromas flavour flavours flavors creaminess smokiness tang richness tastes tasted aftertaste consistency crunch crispness chewiness',
    'reference': 'memories grandfather mom aunt grandma nonna father grandmothers mum grandpa grandmas yiayia dad resturant restuarant restaurants restaraunt restraunt resteraunt restaraunts restraunts cafe bakery',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['train', 'detect'])
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    (train if stage == 'train' else detect)()
