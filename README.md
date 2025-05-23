## 📦 Pretrained Embeddings Attribution

This project includes word embeddings derived from the [GloVe embeddings from the Spanish Billion Word Corpus (SBWC)](https://github.com/dccuchile/spanish-word-embeddings), created by Jorge Pérez. These embeddings are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Additionally, the vocabulary was filtered using the [Spanish word frequency list](https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/es/es_50k.txt) from HermitDave's FrequencyWords project, licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Modifications made:
- Filtered the original SBWC embeddings to include only the top 50,000 most frequent Spanish words.
- Normalized all words using `unidecode` (e.g., removing accents and diacritics).
- Saved in GloVe-style `.txt` format for easy reuse.

### Using the Embeddings in Python

First, here's the core functionality to download and load the embeddings:

```python
import os
import requests
import gzip
import numpy as np

# Configuration
url = "https://raw.githubusercontent.com/fbereilh/spanish-word-embeddings/main/spanish_glove_embeddings.txt.gz"
file_path = os.path.basename(url)

def download_embeddings(file_path: str = file_path, url: str = url):
    """Download the embeddings file if it doesn't exist."""
    if not os.path.exists(file_path):
        print(f"Downloading embeddings to {file_path}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            print("Download complete!")
        except requests.RequestException as e:
            print(f"Download failed: {e}")

def load_embeddings(file_path: str = file_path):
    """Load word embeddings from a gzipped file."""
    word_vectors = {}

    print("Loading embeddings...")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue
            word, *vector_values = values
            vector = np.asarray(vector_values, dtype='float32')
            word_vectors[word] = vector

    print(f"Loaded {len(word_vectors)} word vectors!")
    return word_vectors


download_embeddings()
embeddings = load_embeddings()

```

Here's an example of how to use these functions:

```python
# Example 1: Get vector for a word
word = "hola"
vector = embeddings.get(word)
print(f"\nVector for '{word}':")
print(vector[:10], "...")  # Show first 10 dimensions

# Example 2: Calculate similarity between two words
def word_similarity(word1: str, word2: str, embeddings: Dict[str, np.ndarray]) -> Optional[float]:
    """Calculate cosine similarity between two words."""
    if word1 not in embeddings or word2 not in embeddings:
        return None
    vec1 = embeddings[word1]
    vec2 = embeddings[word2]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Calculate similarity between "hombre" and "mujer"
similarity = word_similarity("hombre", "mujer", embeddings)
if similarity:
    print(f"\nSimilarity between 'hombre' and 'mujer': {similarity:.4f}")

# Example 3: Find similar words
def find_similar_words(word: str, embeddings: Dict[str, np.ndarray], n: int = 5) -> Optional[list]:
    """Find n most similar words to the given word."""
    if word not in embeddings:
        return None
    
    similarities = []
    word_vector = embeddings[word]
    
    for other_word, other_vector in embeddings.items():
        if other_word != word:
            similarity = word_similarity(word, other_word, embeddings)
            if similarity:
                similarities.append((other_word, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

# Find words similar to "casa"
similar_words = find_similar_words("casa", embeddings)
if similar_words:
    print("\nWords similar to 'casa':")
    for word, sim in similar_words:
        print(f"{word}: {sim:.4f}")
```

The embeddings are in GloVe format where each line contains a word followed by its 300-dimensional vector representation. The code above provides functionality for:
- Automatically downloading the embeddings from this repository
- Loading the embeddings into a dictionary
- Finding similar words using cosine similarity
- Easy access to word vectors for custom NLP tasks

You can use these vectors for various NLP tasks such as:
- Finding similar words (as shown in the example)
- Word analogies
- Text classification
- Document similarity
- And more!