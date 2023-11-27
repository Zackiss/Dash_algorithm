import math
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm

from .LPKTNet import LPKTNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train_one_epoch(net, optimizer, criterion, dataloader):
    net.train()

    pred_list = []
    target_list = []

    for outcomes, problem_ids, interval_times, time_spent in tqdm.tqdm(dataloader, 'Training'):
        optimizer.zero_grad()

        outcomes = outcomes.to(device)
        problem_ids = problem_ids.to(device)
        interval_times = interval_times.to(device)
        time_spent = time_spent.to(device)

        pred = net(problem_ids, interval_times, outcomes, time_spent)

        mask = problem_ids[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = outcomes[:, 1:][mask]

        loss = criterion(masked_pred, masked_truth).sum()

        loss.backward()
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        pred_list.append(masked_pred)
        target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


def test_one_epoch(net, dataloader):
    net.eval()

    pred_list = []
    target_list = []

    for outcomes, problem_ids, interval_times, time_spent in tqdm.tqdm(dataloader, 'Testing'):
        outcomes = outcomes.to(device)
        problem_ids = problem_ids.to(device)
        interval_times = interval_times.to(device)
        time_spent = time_spent.to(device)

        with torch.no_grad():
            pred = net(problem_ids, interval_times, outcomes, time_spent)

            mask = problem_ids[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = outcomes[:, 1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred)
            target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


class LPKT:
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, batch_size, dropout=0.2):
        super(LPKT, self).__init__()
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        self.lpkt_net = LPKTNet(n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, q_matrix, dropout).to(device)
        self.batch_size = batch_size

    def train(self, train_dataloader, test_dataloader=None, *, epoch: int, lr=0.002, lr_decay_step=15,
              lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.lpkt_net.parameters(), lr=lr, eps=1e-8, betas=(0.1, 0.999), weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss(reduction='none')
        best_train_auc, best_test_auc = .0, .0

        for idx in range(epoch):
            train_loss, train_auc, train_accuracy = train_one_epoch(self.lpkt_net, optimizer, criterion,
                                                                    train_dataloader)
            print("[Epoch %d] LogisticLoss: %.6f" % (idx, train_loss))
            if train_auc > best_train_auc:
                best_train_auc = train_auc

            scheduler.step()

            if test_dataloader is not None:
                test_loss, test_auc, test_accuracy = test_one_epoch(self.lpkt_net, test_dataloader)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (idx, test_auc, test_accuracy))
                if test_auc > best_test_auc:
                    best_test_auc = test_auc

        return best_train_auc, best_test_auc

    def eval(self, test_data_loader) -> ...:
        self.lpkt_net.eval()
        return test_one_epoch(self.lpkt_net, test_data_loader)

    def save(self, filepath) -> ...:
        torch.save(self.lpkt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.lpkt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
