# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Text, Union
import copy
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP

class LSTMNN(nn.Module):
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

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, x_train, y_train):
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)

        self.lstm_nn_model.train()

        indices = np.arange(len(x_train_values))
        np.random.shuffle(indices)

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_train_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.lstm_nn_model(feature)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.lstm_nn_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y):
        # prepare training data
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)

        self.lstm_nn_model.eval()

        scores = []
        losses = []

        indices = np.arange(len(x_values))

        for i in range(len(indices))[:: self.batch_size]:
            if len(indices) - i < self.batch_size:
                break

            feature = torch.from_numpy(x_values[indices[i : i + self.batch_size]]).float().to(self.device)
            label = torch.from_numpy(y_values[indices[i : i + self.batch_size]]).float().to(self.device)

            pred = self.lstm_nn_model(feature)
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        if isinstance(df_train, pd.DataFrame):
            x_train, y_train = df_train["feature"], df_train["label"]
            x_valid, y_valid = df_valid["feature"], df_valid["label"]
        elif hasattr(df_train, "to_dataframe"):
            df_train = df_train.to_dataframe()
            df_valid = df_valid.to_dataframe()
            x_train, y_train = df_train["feature"], df_train["label"]
            x_valid, y_valid = df_valid["feature"], df_valid["label"]
        else:
            raise TypeError(f"Unsupported data type: {type(df_train)}")

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
            self.train_epoch(x_train, y_train)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid)
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

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        index = x_test.index
        self.lstm_nn_model.eval()
        x_values = x_test.values
        sample_num = x_values.shape[0]
        preds = []

        for begin in range(sample_num)[:: self.batch_size]:
            if sample_num - begin < self.batch_size:
                end = sample_num
            else:
                end = begin + self.batch_size
            x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
            with torch.no_grad():
                pred = self.lstm_nn_model(x_batch).detach().cpu().numpy()
            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)

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
            input_size=alpha_feat_dim,
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