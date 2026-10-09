from sys import path


class RunLogger:
    def __init__(self, sample, ep, bsize, gstep, tp, vp,lr):
        self.path = path
        self.sample = sample
        self.ep = ep
        self.bsize = bsize
        self.gstep = gstep
        self.tp = tp
        self.vp = vp
        self.lr = lr
        self.loss_per_epoch = []
        self.train_loss = []
        self.val_loss = []

