import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience, delta, trace_func=print, type='loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.path1 = path1
        # self.path2 = path2
        self.trace_func = trace_func
        self.type = type

    def __call__(self, loss):

        if self.type == 'loss':
            score = -loss

            if self.best_score is None:
                self.best_score = score
                # self.save_checkpoint(loss, model, model2)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                # self.save_checkpoint(loss, model, model2)
                self.counter = 0

        elif self.type == 'acc':
            score = -loss

            if self.best_score is None:
                self.best_score = score
                # self.save_checkpoint(loss, model, model2)
            elif score > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                # self.save_checkpoint(loss, model, model2)
                self.counter = 0

    # def save_checkpoint(self, loss, model, model2):
    #     # if self.local_rank == 0:
    #     #     torch.save(model.module.state_dict(), self.path)
    #     torch.save(model.state_dict(), self.path1)
    #     torch.save(model2.state_dict(), self.path2)
    #
    #     self.val_loss_min = loss


# 归一化
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# 余弦相似度
def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_theta = 0.5 + 0.5 * cos_theta
    return cos_theta


# 余弦距离
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_theta = float(np.dot(a, b) / (a_norm * b_norm))
    cos_distance = 1 - cos_theta
    # cos_distance = 0.5 - 0.5 * cos_theta
    return cos_distance
