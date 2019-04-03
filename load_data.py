import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import random

def load(device = 'cpu',
              pretrained_embedding = 'glove.6B.300d',
              embedding_dim = 300,
              embedding_num = 100,
              batch_size = 16):
    # Our input $x$
    TEXT = NamedField(names=('seqlen',))

    # Our labels $y$
    LABEL = NamedField(sequential=False, names=())

    # create train val test split
    train, val, test = torchtext.datasets.SNLI.splits(TEXT, LABEL)

    # build vocabs
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    # create iters
    train_iter, val_iter = torchtext.data.BucketIterator.splits(
    (train, val), batch_size=batch_size, device=torch.device(device), repeat=False)

    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, device=torch.device(device))


    # Build the vocabulary with word embeddings
    # Out-of-vocabulary (OOV) words are hashed to one of 100 random embeddings each
    # initialized to mean 0 and standarad deviation 1 (Sec 5.1)
    unk_vectors = [torch.randn(embedding_dim) for _ in range(embedding_num)]
    TEXT.vocab.load_vectors(vectors=pretrained_embedding, unk_init=lambda x:random.choice(unk_vectors))

    # normalized to have l_2 norm of 1
    vectors = TEXT.vocab.vectors
    vectors = vectors / vectors.norm(dim=1,keepdim=True)
    vectors = NamedTensor(vectors, ('word', 'embedding'))
    TEXT.vocab.vectors = vectors

    return train_iter, val_iter, test_iter, TEXT, LABEL
