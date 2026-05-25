import time
from collections import defaultdict
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

from utils import CharVocab
class Trainer:



    def __init__(self, config):
        self.config = config
        self.optimizer = None
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def get_vocabulary(self, data):
        return CharVocab.from_data(data)
    def get_collate_fn(self, model):
        def collate_fn(data):
            tensors = [
                model.string2tensor(s, device='cpu')
                for s in data
            ]

            prevs = [t[:-1] for t in tensors]
            nexts = [t[1:] for t in tensors]

            pad_id = model.vocabulary.pad

            prevs = pad_sequence(
                prevs,
                batch_first=True,
                padding_value=pad_id
            )

            nexts = pad_sequence(
                nexts,
                batch_first=True,
                padding_value=pad_id
            )

            return prevs, nexts

        return collate_fn

    def fit(self, model, train_data, val_data=None):
        model = model.to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        self.optimizer = optimizer

        criterion = nn.CrossEntropyLoss(
            ignore_index=model.vocabulary.pad
        )

        train_loader = DataLoader(
            train_data,
            shuffle=True,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.n_workers,
            collate_fn=self.get_collate_fn(model)
        )

        self.iter_num = 0
        self.iter_time = time.time()

        for epoch in range(self.config.train_epochs):
            model.train()

            tqdm_data = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.train_epochs}"
            )

            for prevs, nexts in tqdm_data:
                prevs = prevs.to(self.device)
                nexts = nexts.to(self.device)

                # Transformer 不能超过 block_size
                if prevs.size(1) > model.block_size:
                    prevs = prevs[:, -model.block_size:]
                    nexts = nexts[:, -model.block_size:]

                logits = model(prevs)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    nexts.reshape(-1)
                )

                model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.grad_norm_clip
                )
                optimizer.step()

                self.loss = loss
                self.trigger_callbacks('on_batch_end')

                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                tqdm_data.set_postfix(loss=loss.item())

                if self.config.max_iters is not None and self.iter_num >= self.config.max_iters:
                    return model

        return model