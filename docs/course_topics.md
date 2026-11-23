# Course topics and how they serve our report

Reference for everyone working on the report, people and AI agents alike. It lists what 02805 Social Graphs and Interactions taught in weeks 1–7 and which of those tools fit our question.

> **Read this first.** The course does not grade how many methods we use. It grades whether we ask **one question worth answering** in 6 pages (figures and references included), and whether each tool we pick helps answer it. Before adding an analysis, write down which part of the question it answers. If it answers none, leave it out. Week 8's rules: the question comes before the method; three good figures beat eight; every measurement needs a null, a second method or a hand-labelled check; one story, not a pile of techniques.

## Our story and question

Two of us are Italian and one is Mexican, and like many people from those countries we care a lot about our national food. We wanted a project about food. We found the Food.com dataset, which has recipes, ratings and **written reviews**. We considered scraping Danish restaurant menus and Google reviews instead. We dropped the idea: collecting the data would cost too much time, the dataset would be small, and restaurant reviews talk about the place more than the food. While exploring Food.com, we read Ahn et al. (2011), *Flavor network and the principles of food pairing*. That paper measures the "authenticity" of an ingredient or ingredient pair as how much more common it is in one cuisine than in the others: $p^c_{ij} = P^c_{ij} - \langle P^{c'}_{ij} \rangle_{c' \neq c}$.

**Question:** Does what reviewers *call* authentic match what the ingredients *say* is characteristic of a cuisine?

- **Network part:** a recipe–ingredient network. It gives each recipe a score for how characteristic its ingredients and ingredient pairs are of its cuisine (Ahn's formula).
- **NLP part:** the reviews of the same recipes. They give each recipe a score for how people talk about its authenticity (mentions of "authentic", including negations, plus sentiment).
- **The comparison:** do the two scores agree, for which cuisines, and what do the disagreements look like?

## Weeks 1–7

**Legend.** ★ = directly useful for our question. ○ = possibly useful as a check or for a figure. Unmarked = background.

### Week 1 · Networks
Nodes and edges; directed vs. undirected; weighted vs. unweighted; simple networks; degree and average degree $\langle k\rangle = 2m/n$; density; edge list; adjacency matrix; paths and distances; connected components, isolates and the giant component; degree distribution; heavy tails and log-log plots; logarithmic binning. Tools: NetworkX, NumPy, matplotlib.
- ★ Choosing a representation. Ours is a **bipartite recipe–ingredient network**. An ingredient's degree is how many recipes use it.
- ○ The ingredient degree distribution (heavy tail: a few staples, many rare ingredients) is why generic ingredients such as salt and water are filtered out.

### Week 2 · Models and null models
Erdős–Rényi $G(n,p)$; the Poisson degree distribution; the giant-component transition; short paths; local clustering vs. transitivity; lattices; small worlds (Watts–Strogatz); power laws and preferential attachment (Barabási–Albert); CCDF; **null models**; **configuration model**; **degree-preserving shuffle**; **shuffle test with z-score and empirical p-value**; the friendship paradox.
- ★ **Null models are central for us.** A pair can look characteristic of a cuisine only because both ingredients are common there. A degree-preserving randomization of the bipartite network (the "Curveball" swap preserves each recipe's size and each ingredient's frequency) tells "pair is characteristic" apart from "ingredients are popular".
- ★ Report results as a z-score or an empirical p-value against the shuffles.

### Week 3 · Advanced network measures
Walks, paths, BFS; eccentricity, diameter, radius; directed reachability; degree, closeness, harmonic, betweenness and eigenvector centrality; PageRank; **centrality compared to a null**; degree assortativity and $k_{nn}(k)$; **homophily (attribute assortativity, label-shuffle test)**; cliques.
- ○ Homophily: do recipes of the same cuisine connect to each other more than label shuffles predict? This is attribute assortativity on the recipe projection.
- ○ Centrality of ingredients inside one cuisine's ingredient network, for example which ingredients hold a cuisine together. Use it only if it feeds the question.

### Week 4 · Communities
Communities and partitions; edge betweenness and Girvan–Newman; dendrograms; **modularity $Q$ (report against its null, since random networks have $Q > 0$)**; greedy modularity; **Louvain**; stability across seeds; **NMI**; k-clique communities; link communities; weights and strength; global threshold; **disparity-filter backbone**; the five drawing rules (backbone first, colour by community, size by degree or strength, label key nodes).
- ★ **Backbone + drawing rules** for the network figure. The ingredient co-occurrence projection is a hairball, so a disparity-filter backbone is the course's tool to show it.
- ○ Louvain on the projection, compared with cuisine labels via **NMI**. This answers "do ingredients alone recover cuisines?", which sets up the question.

### Week 5 · From language to numbers
Tokens, types, vocabulary; tokenization and subword tokenization; preprocessing (lowercasing, lemmatization, stopwords) and what it removes; frequency and relative frequency; hapax; Zipf's law; **concordance (KWIC)**; `similar()`; n-grams; **collocations (PMI, likelihood ratio)**; documents; Bag of Words; document–term matrix; sparse matrices. Tools: spaCy, tiktoken, NLTK (`FreqDist`, `concordance`, `ngrams`, `BigramCollocationFinder`), `CountVectorizer`.
- ★ **Concordance of "authentic"** shows how reviewers use the word: praise, negation ("not authentic"), or "authentic-ish". It is the basis of the hand-labelled validation sample.
- ★ The review is our **document** unit, aggregated per recipe.
- ○ Collocations around "authentic".

### Week 6 · From counts to meaning
**TF-IDF**; TF, DF, IDF; **cosine similarity**; **Scattertext (comparing two labelled corpora)**; topic models and LDA; the distributional hypothesis; word–context matrix; PMI and PPMI; static embeddings (Word2Vec skip-gram and CBOW, negative sampling, GloVe). Tools: scikit-learn, Scattertext, gensim.
- ★ **Scattertext or TF-IDF**: which words separate reviews of high-score recipes from low-score ones, or "authentic" reviews from the rest?
- ○ Embeddings, to find words used like "authentic" (for example "traditional", "like my grandma's").

### Week 7 · From meaning to measurement
Contextual vs. static representations; document embeddings and pretrained encoders (BERT, sentence-transformers); chunking; PCA, t-SNE and UMAP; embedding clusters vs. LDA topics vs. network communities; **sentiment lexicons (VADER)**; emotional arcs; **text classifiers**; **validation of a measurement against labelled examples**; NER, BIO tags, entity resolution, entity co-occurrence networks; sparse vs. dense retrieval; RAG; bias in NLP pipelines.
- ★ **Sentiment** (VADER or a DistilBERT classifier) as the second part of the review-side score.
- ★ **Validation**: hand-label a sample of "authentic" mentions (affirming, negating, other) and report how well the automatic detection agrees with the labels.
- ○ Entity co-occurrence networks, for example dish or ingredient names mentioned in reviews. This would link the text back to the network.

## Week 8 · Project rules (summary)
- Paper in the PNAS template, at most 6 pages including figures and references. Sections: Abstract (~150 words, written last), Significance, Introduction, Results, Discussion, Methods, References, and specific author contributions.
- Grading categories: research question; data and scope; showcasing course methods; analysis and checks; figures; references; overall impression; presentation.
- Each analysis needs one sentence on what it captures and one on what it misses.
- Hand-in: Friday 27 November, 23:59. Presentation: Wednesday 2 December, with 3 slides in 3 minutes.
