## 📦 Pretrained Embeddings Attribution

This project includes word embeddings derived from the [GloVe embeddings from the Spanish Billion Word Corpus (SBWC)](https://github.com/dccuchile/spanish-word-embeddings), created by Jorge Pérez. These embeddings are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Additionally, the vocabulary was filtered using the [Spanish word frequency list](https://github.com/hermitdave/FrequencyWords/blob/master/content/2018/es/es_50k.txt) from HermitDave's FrequencyWords project, licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Modifications made:
- Filtered the original SBWC embeddings to include only the top 50,000 most frequent Spanish words.
- Normalized all words using `unidecode` (e.g., removing accents and diacritics).
- Saved in GloVe-style `.txt` format for easy reuse.

### Loading the Embeddings

The embeddings are stored in a gzipped text file. Here's how to load them using Python:

```python
import gzip
import numpy as np

def load_embeddings(file_path='spanish_glove_embeddings.txt.gz'):
    # Dictionary to store word vectors
    word_vectors = {}
    
    # Read the gzipped file
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word_vectors[word] = vector
    
    return word_vectors

# Load the embeddings
embeddings = load_embeddings()

# Example usage:
vector = embeddings['hola']  # Get vector for the word "hola"
```

The embeddings are in GloVe format where each line contains a word followed by its 300-dimensional vector representation. The vectors can be used for various NLP tasks such as:
- Finding similar words
- Word analogies
- Text classification
- Document similarity
- And more!