import torch
import torch.nn as nn
import os
import timeit
import numpy as np
import sklearn.metrics as skmetrics
from network import TinySleepNet
from torch.optim import Adam

class Model:
    def __init__(self, config=None, output_dir="./output", use_rnn=False, testing=False, use_best=False, device=None):
        self.tsn = TinySleepNet(config)
        self.config = config
        self.output_dir = output_dir
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        self.best_ckpt_path = os.path.join(self.output_dir, "best_ckpt")
        self.weights_path = os.path.join(self.output_dir, "weights")
        self.log_dir = os.path.join(self.output_dir, "log")
        self.device = device
        self.tsn.to(device)

        # weight_decay only apply on cnn, and cnn has no bias
        self.optimizer_all = Adam(
            [{'params': [parm for name, parm in self.tsn.cnn.named_parameters() if 'conv' in name], 'weight_decay': self.config['l2_weight_decay']},  # cnn, l2
             {'params': [parm for name, parm in self.tsn.cnn.named_parameters() if 'conv' not in name]},
             {'params': [parm for name, parm in self.tsn.rnn.named_parameters()]},
             {'params': [parm for name, parm in self.tsn.fc.named_parameters()]}],
            lr=config['learning_rate'], betas=(config["adam_beta_1"], config["adam_beta_2"]),
            eps=config["adam_epsilon"])
        self.CE_loss = nn.CrossEntropyLoss(reduce=False)





        self.global_epoch = 0
        self.global_step = 0

    def get_current_epoch(self):
        return self.global_epoch

    def pass_one_epoch(self):
        self.global_epoch = self.global_epoch + 1

    def train_with_dataloader(self, minibatches):

        # self.run(self.metric_init_op)  # 初始化评价指标记录  # todo 编写评价指标部分的代码
        self.tsn.train()
        start = timeit.default_timer()
        preds = []
        trues = []
        outputs = {}
        for x, y, w, sl, re in minibatches:
            # w is used to mark whether the sample is true, if the sample is filled with zero, w == 0
            # while calculating loss, multiply with w
            x = torch.from_numpy(x).view(self.config['batch_size'] * self.config['seq_length'], 1, 3000)  # shape(batch_size* seq_length, in_channels, input_length)
            y = torch.from_numpy(y)
            w = torch.from_numpy(w)


            feed_dict = {
                # self.signals: x,
                # self.labels: y,
                # self.is_training: True,
                # self.loss_weights: w,
                # self.seq_lengths: sl,
            }
            if min(w) == 0:
                print('point')

            if re:
                # Initialize state of RNN
                state = (torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])),
                         torch.zeros(size=(1, self.config['batch_size'], self.config['n_rnn_units'])))
                state = (state[0].to(self.device), state[1].to(self.device))
                # state = self.run(self.init_state)

            # Carry the states from the previous batches through time
            # for i, (c, h) in enumerate(self.init_state):
            #     feed_dict[c] = state[i].c
            #     feed_dict[h] = state[i].h

            # test
            # conv1, max1, conv2, conv3, outputs = self.run([self.conv1, self.max1, self.conv2, self.conv3, self.train_outputs], feed_dict=feed_dict)
            # net2, outputs = self.run([self.net2, self.train_outputs], feed_dict=feed_dict)
            # forward

            self.optimizer_all.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)
            y_pred, state = self.tsn.forward(x, state)
            state = (state[0].detach(), state[1].detach())

            loss = self.CE_loss(y_pred, y)

            # set the loss of meaningless sample 0
            loss = torch.mul(loss, w)

            # Weight by class
            one_hot = torch.zeros(len(y), self.config["n_classes"]).to(self.device).scatter_(1, y.unsqueeze(dim=1), 1)
            sample_weight = torch.mm(one_hot, torch.Tensor(self.config["class_weights"]).to(self.device).unsqueeze(dim=1)).view(-1)  # (300, 5) * (5,) = (300,)
            loss = torch.mean(torch.mul(loss, sample_weight))



            print('loss', loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=list(self.tsn.cnn.parameters()) + list(self.tsn.rnn.parameters()) + list(self.tsn.fc.parameters()), max_norm=self.config["clip_grad_value"], norm_type=2)
            self.optimizer_all.step()


            prediction = torch.argmax(y_pred, 1)

            acc = np.mean(prediction.detach().cpu().numpy() == y.detach().cpu().numpy())
            print('acc', acc)
            # losses.append(loss.detach().cpu().numpy())
            # preds.append(prediction.cpu().numpy())
            # trues.append(y.cpu().numpy())



            # caculate loss and optimize network






            self.global_step += 1


            # print()


            # _, outputs = self.run([self.cnn, self.train_outputs], feed_dict=feed_dict)

            # _, outputs = self.run([self.train_step_op, self.train_outputs], feed_dict=feed_dict)

            # Buffer the final states
            # state = outputs["train/final_state"]


            tmp_preds = np.reshape(np.argmax(y_pred.cpu().detach().numpy(), axis=1), (self.config["batch_size"], self.config["seq_length"]))
            tmp_trues = np.reshape(y.cpu().detach().numpy(), (self.config["batch_size"], self.config["seq_length"]))

            for i in range(self.config["batch_size"]):
                preds.extend(tmp_preds[i, :sl[i]])
                trues.extend(tmp_trues[i, :sl[i]])

        acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
        f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
        cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])
        stop = timeit.default_timer()
        duration = stop - start
        outputs.update({
            "global_step": self.global_step,
            "train/trues": trues,
            "train/preds": preds,
            "train/accuracy": acc,
            "train/f1_score": f1_score,
            "train/cm": cm,
            "train/duration": duration,
        })
        self.global_epoch += 1
        return outputs


if __name__ == '__main__':
    from torchsummaryX import summary
    from config.sleepedf import train
    model = TinySleepNet(config=train)
    summary(model, torch.randn(size=(2, 1, 3000)))



