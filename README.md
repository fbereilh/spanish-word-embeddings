## 📦 Pretrained Embeddings Attribution

This project includes word embeddings derived from the [GloVe embeddings from the Spanish Billion Word Corpus (SBWC)](https://github.com/dccuchile/spanish-word-embeddings), created by Jorge Pérez. These embeddings are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Additionally, the vocabulary was filtered using the [Spanish word frequency list](https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/es/es_50k.txt) from HermitDave's FrequencyWords project, licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Modifications made:
- Filtered the original SBWC embeddings to include only the top 50,000 most frequent Spanish words.
- Normalized all words using `unidecode` (e.g., removing accents and diacritics).
- Saved in GloVe-style `.txt` format for easy reuse.

### Using the Embeddings in Python

Here's a complete script to download and use the embeddings:

```python
import os
import gzip
import numpy as np
import requests
from typing import Dict, Optional

def download_embeddings(url: str, file_path: str) -> None:
    """Download the embeddings file if it doesn't exist."""
    if not os.path.exists(file_path):
        print(f"Downloading embeddings to {file_path}...")
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
        print("Download complete!")

def load_embeddings(file_path: str = 'spanish_glove_embeddings.txt.gz') -> Dict[str, np.ndarray]:
    """Load word embeddings from a gzipped file."""
    # URL of the embeddings file in the repository
    url = "https://raw.githubusercontent.com/fbereilh/spanish-word-embeddings/main/spanish_glove_embeddings.txt.gz"
    
    # Download the file if it doesn't exist
    download_embeddings(url, file_path)
    
    # Dictionary to store word vectors
    word_vectors = {}
    
    # Read the gzipped file
    print("Loading embeddings...")
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    
    print(f"Loaded {len(word_vectors)} word vectors!")
    return word_vectors

def find_similar_words(word: str, embeddings: Dict[str, np.ndarray], n: int = 5) -> Optional[list]:
    """Find n most similar words to the given word using cosine similarity."""
    if word not in embeddings:
        print(f"Word '{word}' not found in embeddings.")
        return None
    
    # Get the vector for the input word
    word_vector = embeddings[word]
    
    # Calculate cosine similarity with all words
    similarities = {}
    for other_word, other_vector in embeddings.items():
        if other_word != word:
            similarity = np.dot(word_vector, other_vector) / (
                np.linalg.norm(word_vector) * np.linalg.norm(other_vector)
            )
            similarities[other_word] = similarity
    
    # Sort by similarity and get top n
    similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
    return similar_words

# Example usage:
if __name__ == "__main__":
    # Load embeddings
    embeddings = load_embeddings()
    
    # Example 1: Get vector for a word
    word = "hola"
    vector = embeddings.get(word)
    print(f"\nVector for '{word}':")
    print(vector[:10], "...")  # Show first 10 dimensions
    
    # Example 2: Find similar words
    similar = find_similar_words(word, embeddings)
    if similar:
        print(f"\nWords most similar to '{word}':")
        for word, similarity in similar:
            print(f"{word}: {similarity:.4f}")
```

The embeddings are in GloVe format where each line contains a word followed by its 300-dimensional vector representation. The script above provides functionality for:
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