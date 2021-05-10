"""
add cluster layer upon tinysleepnet.
"""
import torch
import torch.nn as nn
import os
import timeit
import numpy as np
import sklearn.metrics as skmetrics
from networks.tinysleepnet import TinySleepNet
from networks.contrastive import MLP, SupClusterConLoss
from torch.optim import Adam
from tensorboardX import SummaryWriter
from script.utils import AverageMeter
import logging
logger = logging.getLogger("default_log")


class ClsuterModel:
    def __init__(self, args, config=None, output_dir="./output", use_rnn=False, testing=False, use_best=False, device=None):
        self.tsn = TinySleepNet(config).to(device)
        self.mlp = MLP(dim_mlp=2048, dim_nce=128, l2_norm=True).to(device)  # todo: try bottle neck layer
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.device = device
        self.rep_m = []  # representation memory bank
        self.lab_m = []  # label memory bank
        self.ptr_m = 0  # memory bank(queue) pointer


        # weight_decay only apply on cnn, and cnn has no bias
        # self.optimizer_all = Adam(
        #     [{'params': [parm for name, parm in self.tsn.cnn.named_parameters() if 'conv' in name], 'weight_decay': self.config['l2_weight_decay']},  # cnn, l2
        #      {'params': [parm for name, parm in self.tsn.cnn.named_parameters() if 'conv' not in name]},
        #      {'params': [parm for name, parm in self.tsn.rnn.named_parameters()]},
        #      {'params': [parm for name, parm in self.tsn.fc.named_parameters()]}],
        #     lr=config['learning_rate'], betas=(config["adam_beta_1"], config["adam_beta_2"]),
        #     eps=config["adam_epsilon"])
        self.optimizer_all = Adam(self.tsn.parameters(),
            lr=config['learning_rate'], betas=(config["adam_beta_1"], config["adam_beta_2"]),
            eps=config["adam_epsilon"])

        self.optimizer_nce = Adam([{'params': self.tsn.cnn.parameters()},
                                   {'params': self.mlp.parameters()}],
            lr=config['learning_rate'], betas=(config["adam_beta_1"], config["adam_beta_2"]),
            eps=config["adam_epsilon"])


        self.CE_loss = nn.CrossEntropyLoss(reduce=False)
        self.infoNCE = SupClusterConLoss()

        self.train_writer = SummaryWriter(os.path.join(self.log_dir, "train"))
        self.train_writer.add_graph(self.tsn, input_to_model=(torch.rand(size=(self.config['batch_size']*self.config['seq_length'], 1, 3000)).to(device), (torch.zeros(size=(1, self.config['batch_size'], 128)).to(device), torch.zeros(size=(1, self.config['batch_size'], 128)).to(device))))
        self.global_epoch = 0
        self.global_step = 0

        if testing and use_best:  # load form best checkpoint
            best_ckpt_path = os.path.join(self.best_ckpt_path, "best_model.ckpt")
            self.tsn.load_state_dict(torch.load(best_ckpt_path))
            logger.info(f'load best model from {best_ckpt_path}')

    def get_current_epoch(self):
        return self.global_epoch

    def pass_one_epoch(self):
        self.global_epoch = self.global_epoch + 1

    def train_with_dataloader(self, minibatches):
        self.tsn.train()
        self.mlp.train()
        start = timeit.default_timer()
        preds, trues, losses, outputs = ([], [], [], {})
        nce_loss_meter = AverageMeter('nce loss', ':6.3f')
        for x, y, w, sl, re in minibatches:
            # w is used to mark whether the sample is true, if the sample is filled with zero, w == 0
            # while calculating loss, multiply with w
            x = torch.from_numpy(x).view(self.config['batch_size'] * self.config['seq_length'], 1, 3000)  # shape(batch_size* seq_length, in_channels, input_length)
            y = torch.from_numpy(y)
            w = torch.from_numpy(w)
            if re:  # Initialize state of RNN
                state = (torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])),
                         torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])))
                state = (state[0].to(self.device), state[1].to(self.device))
            self.optimizer_all.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)
            last_state = state
            y_pred, state, x_nce = self.tsn.forward(x, state)
            state = (state[0].detach(), state[1].detach())
            loss = self.CE_loss(y_pred, y)
            # weight by sample
            loss = torch.mul(loss, w)
            # Weight by class
            one_hot = torch.zeros(len(y), self.config["n_classes"]).to(self.device).scatter_(1, y.unsqueeze(dim=1), 1)
            sample_weight = torch.mm(one_hot, torch.Tensor(self.config["class_weights"]).to(self.device).unsqueeze(dim=1)).view(-1)  # (300, 5) * (5,) = (300,)
            loss = torch.mul(loss, sample_weight).sum() / w.sum()
            cnn_weights = [parm for name, parm in self.tsn.cnn.named_parameters() if 'conv' in name]
            reg_loss = 0
            for p in cnn_weights:
                reg_loss += torch.sum(p ** 2) / 2
            reg_loss = self.config["l2_weight_decay"] * reg_loss
            ce_loss = loss
            # print(f"ce loss {ce_loss:.2f}, reg loss {reg_loss:.2f}")
            loss = loss + reg_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.tsn.parameters(), max_norm=self.config["clip_grad_value"], norm_type=2)
            self.optimizer_all.step()
            losses.append(loss.detach().cpu().numpy())
            self.global_step += 1
            tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1), (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))
            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

            # NCE cluster part
            self.optimizer_all.zero_grad()
            self.optimizer_nce.zero_grad()
            y_pred, _, x_nce = self.tsn.forward(x, last_state)
            x_nce = self.mlp(x_nce)
            x_nce, y_nce = x_nce[w == 1], y[w == 1]

            if self.global_epoch == 0:  # init memory bank
                self.rep_m.append(x_nce.detach())
                self.lab_m.append(y_nce)
            else:
                # calculate centers
                centers = self._calculate_centers()

                # calculate nce loss

                nce_loss = self.infoNCE(x_nce, centers, y_nce)  # todo: consider centers
                if nce_loss != 0:
                    nce_loss_meter.update(nce_loss.item(), x_nce.shape[0])
                    nce_loss.backward()

                # optimize MLP layer and encoder
                    self.optimizer_nce.step()

                # enqueue and dequeue
                self._dequeue_and_enqueue(x_nce.detach(), y_nce)



        if self.global_epoch == 0:  # init memory bank
            self.rep_m = torch.cat(self.rep_m)  # N * C
            self.lab_m = torch.cat(self.lab_m)  # N * 1

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "global_step": self.global_step,
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/loss": all_loss,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,
            "train/nce_loss": nce_loss_meter.avg

        })
        self.global_epoch += 1
        # logger.info(nce_loss_meter)
        return outputs

    def evaluate_with_dataloader(self, minibatches):
        self.tsn.eval()
        start = timeit.default_timer()
        preds, trues, losses, outputs = ([], [], [], {})
        with torch.no_grad():
            for x, y, w, sl, re in minibatches:
                x = torch.from_numpy(x).view(self.config['batch_size'] * self.config['seq_length'], 1,
                                             3000)  # shape(batch_size* seq_length, in_channels, input_length)
                y = torch.from_numpy(y)
                w = torch.from_numpy(w)

                if re:
                    state = (torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])),
                             torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])))
                    state = (state[0].to(self.device), state[1].to(self.device))

                # Carry the states from the previous batches through time  # 在测试时,将上一批样本的lstm状态带入下一批样本

                x = x.to(self.device)
                y = y.to(self.device)
                w = w.to(self.device)

                # summary(self.tsn, x, state)
                # exit(0)
                y_pred, state, _ = self.tsn.forward(x, state)
                state = (state[0].detach(), state[1].detach())
                loss = self.CE_loss(y_pred, y)
                # weight by sample
                loss = torch.mul(loss, w)
                # Weight by class
                one_hot = torch.zeros(len(y), self.config["n_classes"]).to(self.device).scatter_(1, y.unsqueeze(dim=1),
                                                                                                 1)
                sample_weight = torch.mm(one_hot, torch.Tensor(self.config["class_weights"]).to(self.device).unsqueeze(
                    dim=1)).view(-1)  # (300, 5) * (5,) = (300,)
                loss = torch.mul(loss, sample_weight).sum() / w.sum()

                losses.append(loss.detach().cpu().numpy())
                tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1),
                                       (self.config["batch_size"], self.config["seq_length"]))
                tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))
                for i in range(self.config["batch_size"]):
                    preds.extend(tmp_preds[i, :sl[i]])
                    trues.extend(tmp_trues[i, :sl[i]])
        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        all_loss = np.array(losses).mean()
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs = {
            "test/trues": trues,
            "test/preds": preds,
            "test/loss": all_loss,
            "test/accuracy": acc,
            "test/f1_score": f1_score,
            "test/cm": cm,
            "test/duration": duration,
        }
        return outputs

    def save_best_checkpoint(self, name):
        if not os.path.exists(self.best_ckpt_path):
            os.makedirs(self.best_ckpt_path)
        save_path = os.path.join(self.best_ckpt_path, "{}.ckpt".format(name))
        torch.save(self.tsn.state_dict(), save_path)
        logger.info("Saved best checkpoint to {}".format(save_path))

    def _dequeue_and_enqueue(self, rep, lab):
        batch_size = rep.shape[0]
        ptr = self.ptr_m
        if ptr + batch_size <= self.rep_m.shape[0]:
            self.rep_m[ptr:ptr + batch_size] = rep
            self.lab_m[ptr:ptr + batch_size] = lab
            ptr = ptr + batch_size # move pointer
        else:  # moved to the end of queue
            batch_head_size = self.rep_m.shape[0] - ptr
            batch_tail_size = batch_size - batch_head_size
            self.rep_m[ptr:] = rep[: batch_head_size]
            self.lab_m[ptr:] = lab[: batch_head_size]
            self.rep_m[: batch_tail_size] = rep[batch_head_size:]
            self.lab_m[: batch_tail_size] = lab[batch_head_size:]
            ptr = batch_tail_size
        self.ptr_m = ptr

    def _calculate_centers(self):
        centers = []
        for i in range(self.config['n_classes']):
            class_rep = self.rep_m[self.lab_m == i]
            centers.append(torch.mean(class_rep, dim=0))
        return centers




