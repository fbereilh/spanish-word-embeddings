import os
import gzip
import shutil
import zipfile
import requests
import numpy as np
import torch
from tqdm import tqdm
from unidecode import unidecode
from typing import List, Dict, Tuple, Optional

# Configuration
embeddings_url = 'http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz'
vocab_url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/es/es_50k.txt"
embedding_dim = 300
unk_token = "<unk>"

class EmbeddingsProcessor:
    def __init__(self, embeddings_url: str = embeddings_url, vocab_url: str = vocab_url):
        """
        Initialize the embeddings processor.
        
        Args:
            embeddings_url: URL to download embeddings from
            vocab_url: URL to download vocabulary from
        """
        self.embeddings_url = embeddings_url
        self.vocab_url = vocab_url
        self.vocab: Optional[List[str]] = None
        self.word_to_idx: Optional[Dict[str, int]] = None
        self.embedding_tensor: Optional[torch.FloatTensor] = None

    def download_file(self, url: str, filename: str) -> bool:
        """
        Download a file from a URL with progress bar.
        
        Args:
            url: URL to download from
            filename: Where to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(filename):
            print(f"File {filename} already exists.")
            return True

        try:
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(filename, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
            return True

        except requests.RequestException as e:
            print(f"Download failed: {e}")
            return False

    def extract_file(self, filename: str) -> Optional[str]:
        """
        Extract a compressed file.
        
        Args:
            filename: File to extract
            
        Returns:
            str: Path to extracted file, or None if extraction failed
        """
        try:
            if filename.endswith('.zip'):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall('.')
                print("Extraction complete.")
                return filename[:-4]  # Remove .zip

            elif filename.endswith('.gz'):
                uncompressed_filename = filename[:-3]
                if not os.path.exists(uncompressed_filename):
                    print(f"Decompressing {filename}...")
                    with gzip.open(filename, 'rb') as f_in:
                        with open(uncompressed_filename, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Decompressed to {uncompressed_filename}")
                return uncompressed_filename

            else:
                print("Unsupported file format. No extraction performed.")
                return filename

        except Exception as e:
            print(f"Extraction failed: {e}")
            return None

    def download_and_extract_embeddings(self) -> Optional[str]:
        """
        Download and extract the embeddings file.
        
        Returns:
            str: Path to the extracted embeddings file, or None if failed
        """
        filename = os.path.basename(self.embeddings_url)
        if self.download_file(self.embeddings_url, filename):
            return self.extract_file(filename)
        return None

    def load_vocab(self) -> List[str]:
        """
        Download and load the vocabulary.
        
        Returns:
            List[str]: List of vocabulary words
        """
        vocab_file = "es_50k.txt"
        if not self.download_file(self.vocab_url, vocab_file):
            raise RuntimeError("Failed to download vocabulary file")

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = list(set(
                unidecode(line.strip().lower().split()[0])
                for line in f
                if line.strip()
            ))
        print(f"Loaded {len(vocab):,} words in vocabulary.")
        return vocab

    def load_embeddings(
        self,
        filepath: str,
        vocab: Optional[List[str]] = None,
        embedding_dim: int = embedding_dim,
        unk_token: str = unk_token
    ) -> Tuple[List[str], Dict[str, int], torch.FloatTensor]:
        """
        Load embeddings and convert to PyTorch format.
        
        Args:
            filepath: Path to embeddings file
            vocab: Optional vocabulary to filter embeddings
            embedding_dim: Dimension of embeddings
            unk_token: Token for unknown words
            
        Returns:
            Tuple containing:
            - List of vocabulary words
            - Dictionary mapping words to indices
            - PyTorch tensor of embeddings
        """
        embeddings: Dict[str, np.ndarray] = {}
        
        print(f"Loading embeddings from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing embeddings"):
                parts = line.strip().split()
                if len(parts) < embedding_dim + 1:
                    continue
                
                word = unidecode(parts[0])
                if vocab is None or word in vocab:
                    try:
                        vector = np.array(parts[1:], dtype=np.float32)
                        embeddings[word] = vector
                    except ValueError as e:
                        print(f"Skipping malformed line for word '{word}': {e}")

        # Add unknown token if needed
        if unk_token not in embeddings:
            embeddings[unk_token] = np.random.normal(
                scale=0.6,
                size=(embedding_dim,)
            )

        # Build vocabulary and mapping
        vocab_list = list(embeddings.keys())
        word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}

        # Create embedding matrix
        embedding_matrix = np.vstack([embeddings[word] for word in vocab_list])
        embedding_tensor = torch.FloatTensor(embedding_matrix)

        print(f"Loaded {len(vocab_list):,} word vectors.")
        return vocab_list, word_to_idx, embedding_tensor

    def save_embeddings_txt(
        self,
        path: str = 'spanish_glove_embeddings.txt',
        vocab_list: Optional[List[str]] = None,
        word_to_idx: Optional[Dict[str, int]] = None,
        embedding_tensor: Optional[torch.FloatTensor] = None
    ) -> bool:
        """
        Save embeddings to a text file in GloVe format.
        
        Args:
            path: Path to save the embeddings file
            vocab_list: List of vocabulary words (uses self.vocab if None)
            word_to_idx: Dictionary mapping words to indices (uses self.word_to_idx if None)
            embedding_tensor: PyTorch tensor of embeddings (uses self.embedding_tensor if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use instance variables if not provided
            vocab_list = vocab_list or self.vocab
            word_to_idx = word_to_idx or self.word_to_idx
            embedding_tensor = embedding_tensor or self.embedding_tensor

            if not all([vocab_list, word_to_idx, embedding_tensor]):
                print("Error: Embeddings not loaded. Run process() first.")
                return False

            print(f"Saving embeddings to {path}...")
            with open(path, 'w', encoding='utf-8') as f:
                for word in tqdm(vocab_list, desc="Writing embeddings"):
                    vec = embedding_tensor[word_to_idx[word]].numpy()
                    vec_str = ' '.join(map(str, vec))
                    f.write(f"{word} {vec_str}\n")
            
            print(f"Successfully saved embeddings to {path}")
            return True

        except Exception as e:
            print(f"Failed to save embeddings: {e}")
            return False

    def process(self) -> bool:
        """
        Run the complete embeddings processing pipeline.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Download and extract embeddings
            filepath = self.download_and_extract_embeddings()
            if not filepath:
                return False

            # Load vocabulary
            self.vocab = self.load_vocab()

            # Load embeddings
            self.vocab, self.word_to_idx, self.embedding_tensor = self.load_embeddings(
                filepath=filepath,
                vocab=self.vocab
            )

            # Save processed embeddings
            if not self.save_embeddings_txt():
                print("Warning: Failed to save embeddings, but processing completed.")

            return True

        except Exception as e:
            print(f"Processing failed: {e}")
            return False


processor = EmbeddingsProcessor()
if processor.process():
    print("Processing completed successfully!")
    print(f"Vocabulary size: {len(processor.vocab):,}")
    print(f"Embedding tensor shape: {processor.embedding_tensor.shape}")
else:
    print("Processing failed!")