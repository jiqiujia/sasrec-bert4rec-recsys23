"""
Pytorch Lightning Modules.
"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class SeqRecBase(pl.LightningModule):

    def __init__(self, model, lr=1e-3, padding_idx=0,
                 predict_top_k=10, filter_seen=True):

        super().__init__()

        self.model = model
        self.lr = lr
        self.padding_idx = padding_idx
        self.predict_top_k = predict_top_k
        self.filter_seen = filter_seen

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)

        scores = scores.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        user_ids = batch['user_id'].detach().cpu().numpy()

        return {'preds': preds, 'scores': scores, 'user_ids': user_ids}

    def validation_step(self, batch, batch_idx):

        preds, scores = self.make_prediction(batch)
        metrics = self.compute_val_metrics(batch['target'], preds)

        self.log("val_ndcg", metrics['ndcg'], prog_bar=True)
        self.log("val_hit_rate", metrics['hit_rate'], prog_bar=True)
        self.log("val_mrr", metrics['mrr'], prog_bar=True)

    def make_prediction(self, batch):

        outputs = self.prediction_output(batch)

        input_ids = batch['input_ids']
        rows_ids = torch.arange(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        last_item_idx = (input_ids != self.padding_idx).sum(axis=1) - 1

        preds = outputs[rows_ids, last_item_idx, :]

        scores, preds = torch.sort(preds, descending=True)

        if self.filter_seen:
            seen_items = batch['full_history']
            preds, scores = self.filter_seen_items(preds, scores, seen_items)
        else:
            scores = scores[:, :self.predict_top_k]
            preds = preds[:, :self.predict_top_k]

        return preds, scores

    def filter_seen_items(self, preds, scores, seen_items):

        max_len = seen_items.size(1)
        scores = scores[:, :self.predict_top_k + max_len]
        preds = preds[:, :self.predict_top_k + max_len]

        final_preds, final_scores = [], []
        for i in range(preds.size(0)):
            not_seen_indexes = torch.isin(preds[i], seen_items[i], invert=True)
            pred = preds[i, not_seen_indexes][:self.predict_top_k]
            score = scores[i, not_seen_indexes][:self.predict_top_k]
            final_preds.append(pred)
            final_scores.append(score)

        final_preds = torch.vstack(final_preds)
        final_scores = torch.vstack(final_scores)

        return final_preds, final_scores

    def compute_val_metrics(self, targets, preds):

        ndcg, hit_rate, mrr = 0, 0, 0

        for i, pred in enumerate(preds):
            if torch.isin(targets[i], pred).item():
                hit_rate += 1
                rank = torch.where(pred == targets[i])[0].item() + 1
                ndcg += 1 / np.log2(rank + 1)
                mrr += 1 / rank

        hit_rate = hit_rate / len(targets)
        ndcg = ndcg / len(targets)
        mrr = mrr / len(targets)

        return {'ndcg': ndcg, 'hit_rate': hit_rate, 'mrr': mrr}


class SeqRec(SeqRecBase):

    def training_step(self, batch, batch_idx):

        outputs = self.model(batch['input_ids'], batch['attention_mask'])
        loss = self.compute_loss(outputs, batch)

        return loss

    def compute_loss(self, outputs, batch):

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs.view(-1, outputs.size(-1)), batch['labels'].view(-1))

        return loss

    def prediction_output(self, batch):

        return self.model(batch['input_ids'], batch['attention_mask'])


class SeqRecWithSampling(SeqRec):

    def __init__(self, model, lr=1e-3, loss='cross_entropy',
                 padding_idx=0, predict_top_k=10, filter_seen=True,
                 in_batch_negatives=False, log_q_correction=False,
                 temperature=1.0):

        super().__init__(model, lr, padding_idx, predict_top_k, filter_seen)

        self.loss = loss
        self.in_batch_negatives = in_batch_negatives
        self.log_q_correction = log_q_correction
        self.temperature = temperature

        # log_q_correction requires item frequency to be set via set_item_freq()
        # before training starts. The correction is: logit_j -= log(Q(j))
        # where Q(j) is the sampling probability for item j.
        self._log_q = None

        if hasattr(self.model, 'item_emb'):  # for SASRec
            self.embed_layer = self.model.item_emb
        elif hasattr(self.model, 'embed_layer'):  # for other models
            self.embed_layer = self.model.embed_layer

    def set_item_freq(self, item_freq_tensor):
        """Set item frequency counts for log-Q correction.

        Args:
            item_freq_tensor: Tensor of shape [num_items + 1], where index 0
                is padding. Each entry is the interaction count for that item.
                Will be normalized to probabilities internally.
        """
        # Normalize to probabilities, add small epsilon to avoid log(0)
        freq = item_freq_tensor.float()
        freq[0] = 0  # padding index has no frequency
        prob = freq / freq.sum().clamp(min=1)
        self._log_q = torch.log(prob + 1e-10)
        # Register as buffer so it moves to the correct device automatically
        self.register_buffer('log_q', self._log_q)

    def compute_loss(self, outputs, batch):

        if self.in_batch_negatives:
            return self._compute_loss_in_batch(outputs, batch)

        return self._compute_loss_sampled(outputs, batch)

    def _compute_loss_in_batch(self, outputs, batch):
        """Compute loss using in-batch negatives.

        Collect all unique label item ids across the batch as shared negatives.
        This naturally provides popularity-weighted negative sampling because
        popular items appear more frequently as positives in other sequences.

        When log_q_correction is enabled, apply the correction:
            corrected_logit_j = logit_j - log(Q(j))
        where Q(j) is the sampling probability of item j. For in-batch negatives,
        Q(j) is approximately proportional to item j's frequency in the dataset.
        This makes the sampled softmax an unbiased estimate of the full softmax.

        Args:
            outputs: [N, T, D] - hidden states from the model.
            batch: dict with 'labels' [N, T] and optionally 'negatives'.
        """
        labels = batch['labels']  # [N, T]
        valid_mask = labels != -100  # [N, T]

        # Collect all valid label item ids in this batch as the negative pool.
        # This is naturally popularity-weighted: popular items appear in more
        # users' sequences, so they are more likely to show up as in-batch negs.
        all_label_ids = labels[valid_mask]  # [num_valid]
        unique_neg_ids = torch.unique(all_label_ids)  # [U]

        # Embed the shared negative pool: [U, D]
        embeds_negatives = self.embed_layer(unique_neg_ids)

        # Embed positives: [N, T, D]
        labels_for_embed = labels.clone()
        labels_for_embed[~valid_mask] = self.padding_idx
        embeds_labels = self.embed_layer(labels_for_embed)

        # Positive logits: [N, T]
        # [N, T, 1, D] x [N, T, D, 1] -> [N, T, 1, 1] -> [N, T]
        logits_pos = torch.matmul(
            outputs.unsqueeze(2), embeds_labels.unsqueeze(3)).squeeze(-1).squeeze(-1)

        # Negative logits: [N, T, U]
        # [N, T, D] x [D, U] -> [N, T, U]
        logits_neg = torch.matmul(outputs, embeds_negatives.T)

        # Apply log-Q correction: subtract log(Q(item)) from each logit.
        # This debiases the sampled softmax so that it approximates full softmax.
        # Reference: Bengio & Senécal (2008), Yi et al. (2019) "Sampling-Bias-Corrected
        # Neural Modeling for Large Corpus Item Recommendations" (Google)
        if self.log_q_correction and self.log_q is not None:
            # Correction for positive logits: [N, T]
            log_q_pos = self.log_q[labels_for_embed]  # [N, T]
            logits_pos = logits_pos - log_q_pos

            # Correction for negative logits: [U]
            log_q_neg = self.log_q[unique_neg_ids]  # [U]
            logits_neg = logits_neg - log_q_neg.unsqueeze(0).unsqueeze(0)  # [1, 1, U] broadcast

        # Concatenate: [N, T, 1 + U]  (positive at index 0)
        logits = torch.cat([logits_pos.unsqueeze(2), logits_neg], dim=-1)

        # Apply temperature scaling: logits / tau
        logits = logits / self.temperature

        if self.loss == 'cross_entropy':
            # Target is 0 for valid positions (positive at index 0), -100 for padding
            targets = labels.clone()  # [N, T]
            targets[valid_mask] = 0
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        elif self.loss == 'bce':
            # [N, T, 1 + U] target: 1 for positive, 0 for negatives
            targets = torch.zeros_like(logits)
            targets[:, :, 0] = 1
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits, targets)
            loss = loss[valid_mask]
            loss = loss.mean()

        return loss

    def _compute_loss_sampled(self, outputs, batch):

        # embed  and compute logits for negatives
        if batch['negatives'].ndim == 2:  # for full_negative_sampling=False
            # [N, M, D]
            embeds_negatives = self.embed_layer(batch['negatives'].to(torch.int32))
            # [N, T, D] * [N, D, M] -> [N, T, M]
            logits_negatives = torch.matmul(outputs, embeds_negatives.transpose(1, 2))
        elif batch['negatives'].ndim == 3:  # for full_negative_sampling=True
            # [N, T, M, D]
            embeds_negatives = self.embed_layer(batch['negatives'].to(torch.int32))
            # [N, T, 1, D] * [N, T, D, M] -> [N, T, 1, M] -> -> [N, T, M]
            logits_negatives = torch.matmul(
                outputs.unsqueeze(2), embeds_negatives.transpose(2, 3)).squeeze()
            if logits_negatives.ndim == 2:
                logits_negatives = logits_negatives.unsqueeze(2)

        # embed  and compute logits for positives
        # [N, T]
        labels = batch['labels'].clone()
        labels[labels == -100] = self.padding_idx
        # [N, T, D]
        embeds_labels = self.embed_layer(labels)
        # [N, T, 1, D] * [N, T, D, 1] -> [N, T, 1, 1] -> [N, T]
        logits_labels = torch.matmul(outputs.unsqueeze(2), embeds_labels.unsqueeze(3)).squeeze()

        # concat positives and negatives
        # [N, T, M + 1]
        logits = torch.cat([logits_labels.unsqueeze(2), logits_negatives], dim=-1)

        # Apply temperature scaling: logits / tau
        logits = logits / self.temperature

        # prepare targets for loss
        if self.loss == 'cross_entropy':
            # [N, T]
            targets = batch['labels'].clone()
            targets[targets != -100] = 0
        elif self.loss == 'bce':
            # [N, T, M + 1]
            targets = torch.zeros_like(logits)
            targets[:, :, 0] = 1

        if self.loss == 'cross_entropy':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
        elif self.loss == 'bce':
            # loss_fct = nn.BCEWithLogitsLoss()
            # loss = loss_fct(logits, targets)
            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits, targets)
            loss = loss[batch['labels'] != -100]
            loss = loss.mean()

        return loss

    def prediction_output(self, batch):

        outputs = self.model(batch['input_ids'], batch['attention_mask'])
        outputs = torch.matmul(outputs, self.embed_layer.weight.T)

        return outputs
