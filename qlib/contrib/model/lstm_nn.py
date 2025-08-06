# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset.handler import DataHandlerLP
from ...model.utils import ConcatDataset
from ...data.dataset.weight import Reweighter

class LSTMNN(Model):
    def __init__(
        self,
        alpha_feat_dim=6,    # e.g., 6
        embed_vector_dim=64,    # e.g., 64
        embed_feat_count=16,
        hidden_size=64,
        lstm_layers=2,
        nn_out=2,
        dropout=0.1,
        n_epochs=200,
        lr=0.001,
        metric="",
        batch_size=2000,
        early_stop=20,
        loss="mse",
        optimizer="adam",
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs,
    ):
        super().__init__()
        # Set logger.
        self.logger = get_module_logger("LSTM")
        self.logger.info("LSTM pytorch version...")

        # set hyper-parameters.
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.n_jobs = n_jobs
        self.seed = seed
        self.alpha_feat_dim = alpha_feat_dim
        self.embed_vector_dim = embed_vector_dim
        self.embed_feat_count = embed_feat_count
        self.nn_out = nn_out

        self.logger.info(
            "LSTM NN Fusion parameters setting:"
            "\nhidden_size : {}"
            "\ndropout : {}"
            "\nn_epochs : {}"
            "\nlr : {}"
            "\nmetric : {}"
            "\nbatch_size : {}"
            "\nearly_stop : {}"
            "\noptimizer : {}"
            "\nloss_type : {}"
            "\nvisible_GPU : {}"
            "\nuse_GPU : {}"
            "\nseed : {}"
            "\nalpha_feat_dim : {}"
            "\nembed_vector_dim : {}"
            "\nnn_nidden : {}".format(
                hidden_size,
                dropout,
                n_epochs,
                lr,
                metric,
                batch_size,
                early_stop,
                optimizer.lower(),
                loss,
                GPU,
                self.use_gpu,
                seed,
                alpha_feat_dim,
                embed_vector_dim,
                nn_out
            )
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        self.lstm_nn_model = LSTMNNModel(
            alpha_feat_dim=self.alpha_feat_dim,
            embed_vector_dim=self.embed_vector_dim,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            nn_out=self.nn_out,
            dropout=self.dropout,
        )

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.lstm_nn_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.lstm_nn_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        
        self.fitted = False
        self.lstm_nn_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label, weight):
        loss = weight * (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label, weight):
        mask = ~torch.isnan(label)

        if weight is None:
            weight = torch.ones_like(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask], weight[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask], weight=None)

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        self.lstm_nn_model.train()

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.lstm_nn_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.lstm_nn_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):
        self.lstm_nn_model.eval()

        scores = []
        losses = []

        for data, weight in data_loader:
            feature = data[:, :, 0:-1].to(self.device)
            # feature[torch.isnan(feature)] = 0
            label = data[:, -1, -1].to(self.device)

            pred = self.lstm_nn_model(feature.float())
            loss = self.loss_fn(pred, label, weight.to(self.device))
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset,
        evals_result=dict(),
        save_path=None,
        reweighter=None,
    ):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        if reweighter is None:
            wl_train = np.ones(len(dl_train))
            wl_valid = np.ones(len(dl_valid))
        elif isinstance(reweighter, Reweighter):
            wl_train = reweighter.reweight(dl_train)
            wl_valid = reweighter.reweight(dl_valid)
        else:
            raise ValueError("Unsupported reweighter type.")

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=True,
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.lstm_nn_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.lstm_nn_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.lstm_nn_model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.lstm_nn_model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())

class LSTMNNModel(nn.Module):
    def __init__(
        self,
        alpha_feat_dim=6,       # e.g., 6
        embed_vector_dim=64,    # e.g., 64 (normalized)
        embed_feat_count=16,
        nn_out=2,
        hidden_size=64,
        lstm_layers=2,
        dropout=0.1
    ):
        super().__init__()

        # LSTM for Alpha158 features
        self.lstm = nn.LSTM(
            input_size=alpha_feat_dim + embed_feat_count * nn_out,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True,
        )

        # FFN for embedding
        self.ffn = nn.Sequential(
            nn.Linear(embed_vector_dim, embed_vector_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_vector_dim // 2, nn_out),
        )
        self.embed_vector_dim = embed_vector_dim
        self.embed_feat_count = embed_feat_count
        self.alpha_feat_dim = alpha_feat_dim

        # last layer
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x_alpha: [N, F * T] -> flatten 때문에 벡터 넣기가 힘듦
        x_alpha = x

        N = x_alpha.shape[0]
        F = self.embed_feat_count
        T = x_alpha.shape[1]

        # --- 임베딩 따로 불러오는 로직 작성 ---
        vector_size = self.embed_vector_dim
        # x_embed: [N, T, F, vector_size(64)]
        x_embed = torch.zeros((N, T, F, vector_size), dtype=torch.float32, device=x_alpha.device)

        # ---- FFN branch ----
        x_embed_flat = x_embed.view(-1, vector_size)  # [(N*T*F), 64]
        x_embed_transformed = self.ffn(x_embed_flat)  # [(N*T*F), 2]
        x_embed_transformed = x_embed_transformed.view(N, T, F * 2)  # [N, T, F*2]

        # ---- LSTM branch ----
        x_alpha = x_alpha.reshape(len(x_alpha), self.alpha_feat_dim, -1)  # [N, F, T]
        x_alpha = x_alpha.permute(0, 2, 1)  # [N, T, F]

        # ---- concat ----
        x_input = torch.cat([x_alpha, x_embed_transformed], dim=2)  # [N, T, F + F*2]

        # ---- LSTM ----
        out_lstm, _ = self.lstm(x_input)    # [N, T, H]
        h_lstm = out_lstm[:, -1, :]         # [N, H]

        return self.fc_out(h_lstm).squeeze()  # [N]