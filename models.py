import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import random
import copy

class LatentVariableMixtureModel(ntorch.nn.Module):
    def __init__(self, model, experts, variational):
        super().__init__()
        try:
            self.models = []
            for _ in range(experts):
                self.models.append(copy.deepcopy(model))
        except RuntimeError:
            raise RuntimeError("model must be newly instantiated")
        self.experts = experts
        self.variational = variational

    def forward(self, premise, hypothesis):
        if self.variational:
            return self.sample(premise, hypothesis)
        else:
            return self.enumerate(premise, hypothesis)

    def self.sample(premise, hypothesis):
        pass

    def enumerate(self, premise, hypothesis):
        predictions = []
        for model in self.models:
            predictions.append(model(premise, hypothesis))
        return (
            ntorch.stack(predictions, "experts")
            .softmax('logit')
            .mean('experts')
            .log()
            .rename('logit','logprob')
        )
class AttentionModel(ntorch.nn.Module):
    def __init__(self,
                 TEXT, LABEL,
                 hidden_attn, hidden_aligned,
                 intra_attn = False, hidden_intra_attn = 200,
                 dropout = 0.5, device = 'cpu', freeze_emb = True):
        super().__init__()
        self.pretrained_emb = TEXT.vocab.vectors.to(device)
        self.embedding = ntorch.nn.Embedding.from_pretrained(self.pretrained_emb.values, freeze=freeze_emb).spec('seqlen','embedding')
        emb_dim = self.pretrained_emb.shape['embedding']

        self.intra_attn = intra_attn
        if self.intra_attn:
            self.feedforward_intra_attn = ntorch.nn.Linear(self.pretrained_emb.shape['embedding'], hidden_intra_attn).spec('embedding', 'hidden')
            emb_dim = 2*emb_dim
        self.feedforward_attn = ntorch.nn.Linear(emb_dim, hidden_attn).spec('embedding', 'hidden')
        self.feedforward_aligned = ntorch.nn.Linear(2*emb_dim, hidden_aligned).spec('embedding', 'hidden')
        self.feedforward_agg = ntorch.nn.Linear(2*hidden_aligned, len(LABEL.vocab)).spec('hidden', 'logit')

        self.dropout = ntorch.nn.Dropout(dropout)

    def forward(self, premise, hypothesis):
        premise = self.embedding(premise)
        premise = premise.rename('seqlen', 'seqlenPremise')
        if self.intra_attn:
            premise = self.intra_attn_layer(premise, 'seqlenPremise')
        premise_hidden = self.feedforward_attn(premise).relu()
        premise_hidden = self.dropout(premise_hidden)


        hypothesis = self.embedding(hypothesis)
        hypothesis = hypothesis.rename('seqlen', 'seqlenHypo')
        if self.intra_attn:
            hypothesis = self.intra_attn_layer(hypothesis, 'seqlenHypo')
        hypothesis_hidden = self.feedforward_attn(hypothesis).relu()
        hypothesis_hidden = self.dropout(hypothesis_hidden)

        self.attn = premise_hidden.dot('hidden', hypothesis_hidden)
        alpha = self.attn.softmax('seqlenHypo').dot('seqlenPremise', premise)
        beta = self.attn.softmax('seqlenPremise').dot('seqlenHypo', hypothesis)

        hypothesis_comparison = self.feedforward_aligned(ntorch.cat([alpha, hypothesis],'embedding')).relu().sum('seqlenHypo')
        premise_comparison = self.feedforward_aligned(ntorch.cat([beta, premise],'embedding')).relu().sum('seqlenPremise')
        agg = ntorch.cat([premise_comparison, hypothesis_comparison], 'hidden')
        agg = self.feedforward_agg(agg)
        agg = self.dropout(agg)

        return agg

    def intra_attn_layer(self, x, seqlen_dimname):
        ## TODO: not sure about distance bias term

        temp_dim = seqlen_dimname+'temp'
        x_hidden1 = self.feedforward_intra_attn(x).relu()
        x_hidden1 = self.dropout(x_hidden1)
        x_hidden2 = x_hidden1.rename(seqlen_dimname, temp_dim)
        intra_attn = x_hidden1.dot('hidden', x_hidden2).softmax(temp_dim)
        x_aligned = intra_attn.dot(temp_dim, x)

        # save attention distribution
        if seqlen_dimname == "seqlenPremise":
            self.intra_attn_premise = intra_attn
        else:
            self.intra_attn_hypo = intra_attn

        return ntorch.cat([x, x_aligned], 'embedding')
