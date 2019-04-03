import torch
import torchtext
from torchtext.vocab import Vectors, GloVe
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import random
import copy
import numpy as np


class VariationalEncoder(ntorch.nn.Module):

    def __init__(self, TEXT, VOCAB, K, lstm_size, device='cpu'):
        super().__init__()
        self.pretrained_emb = TEXT.vocab.vectors.to(device)
        self.embedding = ntorch.nn.Embedding.from_pretrained(
            self.pretrained_emb.values, freeze=True).spec('seqlen','embedding')
        emb_dim = self.pretrained_emb.shape['embedding']
        self.lstm_prem = ntorch.nn.LSTM(emb_dim, lstm_size).spec('embedding', 'seqlen')
        self.lstm_hyp = ntorch.nn.LSTM(emb_dim, lstm_size).spec('embedding', 'seqlen')
        self.linear = ntorch.nn.Linear(lstm_size*2 + 1, K).spec('embedding', 'logprob')
        self.to(device)
        self.device = device
    
    def forward(self, premise, hypothesis, target):
        prem = self.embedding(premise)
        prem = self.lstm_prem(prem)[0][{'seqlen': -1}]
        hyp = self.embedding(hypothesis)
        hyp = self.lstm_hyp(hyp)[0][{'seqlen': -1}]
        tar = ntorch.tensor(target.values.reshape(1, -1).to(torch.float32), names=('embedding', 'batch'))
        flat = ntorch.cat([hyp, prem, tar], 'embedding')
        out = self.linear(flat).log_softmax('logprob')
        return out


class LatentVariableMixtureModel(ntorch.nn.Module):

    def __init__(self, model_const, experts, encoder=None):
        super().__init__()
        self.models = torch.nn.ModuleList([model_const() for _ in range(experts)])
        self.experts = experts
        self.q = encoder

    def forward(self, premise, hypothesis, sample_size=None):
        if sample_size:
            assert self.q is not None, 'No encoder present'
            model_inds = np.random.choice(self.experts, sample_size, replace=True)
            models = [self.models[i] for i in model_inds]
            return self.enumerate(premise, hypothesis, models=models)
        return self.enumerate(premise, hypothesis)

    def reinforce(self, premise, hypothesis, target):
        q_outputs = self.q(premise, hypothesis, target).log_softmax('logprob')
        log_q, model_inds = q_outputs.topk('logprob', 1)
        model_inds = set([int(i.item()) for i in model_inds.values.reshape(-1)])
        elbos = []
        grad_fn_val = torch.tensor(0., device=self.q.device)
        for i in model_inds:
            inds = (q_outputs.topk('logprob', 1)[1] == i).nonzero().rename('elements', 'batch')[{'inputdims': 0}]
            mod_premise = premise.index_select('batch', inds)
            mod_hypothesis = hypothesis.index_select('batch', inds)
            mod_target = target.index_select('batch', inds)
            mod_qout = q_outputs.index_select('batch', inds)
            elbo, grad = self.reinforce_batch(mod_premise, mod_hypothesis, mod_target, self.models[i], mod_qout)
            grad_fn_val = grad_fn_val + grad
            for _ in range(len(mod_premise)):
                elbos.append(elbo)
        grad_fn_val.backward()
        return np.mean(elbos)

    def reinforce_batch(self, premise, hypothesis, target, model, q_outputs):
        """ Calculates the gradients of the models and returns the ELBO. """
        log_q, _ = q_outputs.topk('logprob', 1)
        outputs = model(premise, hypothesis)
        log_p = outputs.log_softmax('logit').rename('logit', 'logprob')
        kl_term = (q_outputs.exp() * (q_outputs - torch.tensor(1 / self.experts).log())).sum('logprob')
        p = log_p.values.gather(1, target.values.reshape(-1, 1))
        elbo = p.mean() - kl_term.values.mean()

        grad_fn = (kl_term.values - p - (p.detach() * log_q.values)).mean()
        return elbo.item(), grad_fn

    def enumerate(self, premise, hypothesis, models=None):
        predictions = []
        for model in (models or self.models):
            predictions.append(model(premise, hypothesis))
        return (
            ntorch.stack(predictions, "experts")
            .softmax('logit')
            .mean('experts')
            .log()
            .rename('logit','logprob')
        )

    def uniform_fit(self, train_iter, val_iter, epochs, lr, verbose):
        crit = torch.nn.NLLLoss()
        self.losses = []
        self.val_losses = []
        opt = torch.optim.Adagrad(self.parameters(), lr=lr)
        for e in range(epochs):
            for ibatch, batch in enumerate(train_iter):
                opt.zero_grad()
                pred = self.forward(batch.premise, batch.hypothesis)
                loss = crit(
                    pred.transpose('batch', 'logprob').values,
                    batch.label.transpose('batch').values)
                self.losses.append(loss.item())
                loss.backward()
                opt.step()
                if verbose and ibatch % verbose == 0 and ibatch > 0:
                    print(f"batch: {ibatch}", end='')
                    print(f', loss: {np.mean(self.losses[-verbose:])}')
            val_loss = self.eval(val_iter)
            self.val_losses.append(val_loss)
            if verbose:
                print(f'epoch: {e}, loss: {self.val_losses[-1]}')

    def eval(self, iter):
        crit = torch.nn.NLLLoss()
        losses = []
        for ibatch, batch in enumerate(iter):
            pred = self.forward(batch.premise, batch.hypothesis)
            loss = crit(
                pred.transpose('batch', 'logprob').values,
                batch.label.transpose('batch').values)
            losses.append(loss.item())
        return np.mean(losses)


    def fit(self, train_iter, val_iter=[], epochs=1, lr=.001, verbose=False):
        if not self.q:
            self.uniform_fit(train_iter, val_iter, epochs, lr, verbose)
            return
        crit = torch.nn.NLLLoss()
        self.elbos = []
        self.losses = []
        self.val_losses = []
        opt = torch.optim.Adagrad(self.parameters(), lr=lr)
        for e in range(epochs):
            for ibatch, batch in enumerate(train_iter):
                opt.zero_grad()
                elbo = self.reinforce(batch.premise, batch.hypothesis, batch.label)
                pred = self.forward(batch.premise, batch.hypothesis)
                loss = crit(
                    pred.transpose('batch', 'logprob').values,
                    batch.label.transpose('batch').values)
                self.elbos.append(elbo)
                self.losses.append(loss.item())
                opt.step()
                if verbose and ibatch % verbose == 0 and ibatch > 0:
                    print(f"batch: {ibatch}", end='')
                    print(f', elbo: {np.mean(self.elbos[-verbose:])}, loss: {np.mean(self.losses[-verbose:])}')
            val_loss = self.eval(val_iter)
            self.val_losses.append(val_loss)
            if verbose:
                print(f'epoch: {e}, loss: {self.val_losses[-1]}')


class AttentionModel(ntorch.nn.Module):
    def __init__(self,
                 TEXT, LABEL,
                 hidden_attn, hidden_aligned, hidden_final,
                 intra_attn = False, hidden_intra_attn = 200,
                 dropout = 0.5, device = 'cpu', freeze_emb = True):
        super().__init__()
        # record parameters
        self.device = device
        self.dropout = dropout

        # initialize embedding
        self.pretrained_emb = TEXT.vocab.vectors.to(device)
        self.embedding = (
            ntorch.nn.Embedding.from_pretrained(self.pretrained_emb.values, freeze=freeze_emb)
            .spec('seqlen','embedding')
        )
        self.embedding.weight[1] = torch.zeros(300)
        self.embedding_projection = ntorch.nn.Linear(self.pretrained_emb.shape['embedding'], 200).spec('embedding', 'embedding')
        emb_dim = 200

        # initialize intra attn
        self.intra_attn = intra_attn
        if self.intra_attn:
            self.feedforward_intra_attn = self.intra_attn_layer()
            emb_dim = 2*emb_dim

        # initialize feedforward modules
        self.feedforward_attn = MLP(emb_dim, hidden_attn,'embedding', 'hidden', self.dropout)
        self.feedforward_aligned = MLP(2*emb_dim, hidden_aligned, 'embedding', 'hidden', self.dropout)
        self.feedforward_agg = MLP(2*hidden_aligned, hidden_final, 'hidden', 'final',self.dropout)
        self.final_linear = ntorch.nn.Linear(hidden_final, len(LABEL.vocab)).spec('final','logit')
        self.to(device)

    def forward(self, premise, hypothesis):
        premise = self.embedding(premise)
        premise = self.embedding_projection(premise).rename('seqlen', 'seqlenPremise')
        hypothesis = self.embedding(hypothesis)
        hypothesis = self.embedding_projection(hypothesis).rename('seqlen', 'seqlenHypo')

        premise_mask = (premise != 0).float()
        hypothesis_mask = (hypothesis != 0).float()

        if self.intra_attn:
            premise = self.feedforward_intra_attn(premise)
            hypothesis = self.feedforward_intra_attn(hypothesis)

        #attend
        premise_hidden = self.feedforward_attn(premise)
        hypothesis_hidden = self.feedforward_attn(hypothesis)

        self.attn = premise_hidden.dot('hidden', hypothesis_hidden)
        alpha = self.attn.softmax('seqlenHypo').dot('seqlenPremise', premise)
        beta = self.attn.softmax('seqlenPremise').dot('seqlenHypo', hypothesis)

        #mask
        alpha = alpha * hypothesis_mask
        beta = beta * premise_mask

        #compare
        hypothesis_comparison = self.feedforward_aligned(ntorch.cat([alpha, hypothesis],'embedding')).sum('seqlenHypo')
        premise_comparison = self.feedforward_aligned(ntorch.cat([beta, premise],'embedding')).sum('seqlenPremise')

        #aggregate
        agg = ntorch.cat([premise_comparison, hypothesis_comparison], 'hidden')
        agg = self.feedforward_agg(agg)
        agg = self.final_linear(agg)
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

    def fit(self, train_iter, val_iter=[], lr=1e-2, verbose=True,
            batch_size=128, epochs=10, interval=1, early_stopping=False):
        self.to(self.device)
        lr = torch.tensor(lr)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adagrad(self.parameters(), lr=lr)
        train_iter.batch_size = batch_size

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            self.train()
            for i, data in enumerate(train_iter, 0):
                premise, hypothesis, labels = (
                    data.premise,
                    data.hypothesis,
                    data.label
                )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(premise, hypothesis)
                loss = criterion(
                    outputs.transpose("batch", "logit").values,
                    labels.transpose("batch").values,
                )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % interval == 0 and i > 0 and verbose:
                    print(f"batch: {i}, loss: {running_loss/interval}")
                    running_loss = 0
            if verbose and val_iter is not None:
                val_loss = self.validate(val_iter)
                print(f"epoch: {epoch}, val loss: {val_loss}")

    def validate(self, val_iter):
        running_loss = 0
        val_count = 0
        self.eval()
        criterion = torch.nn.CrossEntropyLoss()

        for i, data in enumerate(val_iter):
            premise, hypothesis, labels = (
                data.premise,
                data.hypothesis,
                data.label
            )

            outputs = self(premise, hypothesis)
            loss = criterion(
                outputs.transpose('batch','logit').values,
                labels.transpose('batch').values
            )
            running_loss += loss.item()
            val_count += 1
        avg_loss = running_loss / val_count
        return avg_loss

class MLP(ntorch.nn.Module):
    def __init__(self, input_dim, output_dim, input_name, output_name, dropout):
        super(MLP, self).__init__()
        self.l1 = ntorch.nn.Linear(input_dim, output_dim).spec(input_name, output_name)
        self.l2 = ntorch.nn.Linear(output_dim, output_dim).spec(output_name, output_name)
        self.dropout = ntorch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.dropout(x)
        x = self.l2(x).relu()
        x = self.dropout(x)
        return x
