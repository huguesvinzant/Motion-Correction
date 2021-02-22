import numpy as np
import torch
import torch_dct as dct
import torch.nn as nn

from PoseCorrection.softdtw import SoftDTW


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_labels(raw_labels, level=0):
    if level == 0:
        mapping = {'SQUAT': 0, 'Lunges': 2, 'Plank': 4}
        labels = np.zeros(len(raw_labels))
        for i, el in enumerate(raw_labels):
            if el[2] == 1:
                labels[i] = mapping[el[0]]
            else:
                labels[i] = mapping[el[0]] + 1
        return torch.from_numpy(labels).long()
    elif level == 1:
        mapping = {'SQUAT': 0, 'Lunges': 6, 'Plank': 9}
        map_label = {'SQUAT': [1, 2, 3, 4, 5, 10], 'Lunges': [1, 4, 6], 'Plank': [1, 7, 8]}
        labels = np.zeros(len(raw_labels))
        for i, el in enumerate(raw_labels):
            labels[i] = mapping[el[0]] + np.where(np.array(map_label[el[0]]) == el[2])[0].item()
        return torch.from_numpy(labels).long()


def dtw_loss(originals, deltas, targets, criterion, attentions=None, is_cuda=False, test=False):
    loss = 0
    preds = []
    for i, o in enumerate(originals):

        length = o.shape[1]
        org = torch.from_numpy(o).T.unsqueeze(0)
        targ = torch.from_numpy(targets[i]).T.unsqueeze(0)

        if length > deltas[i].shape[1]:
            m = torch.nn.ZeroPad2d((0, length - deltas[i].shape[1], 0, 0))
            delt = dct.idct_2d(m(deltas[i]).T.unsqueeze(0))
        else:
            delt = dct.idct_2d(deltas[i, :, :length].T.unsqueeze(0))

        if attentions is not None:
            delt = torch.mul(delt, attentions[i].T.unsqueeze(0))

        out = org + delt

        if is_cuda:
            out = out.cuda()
            targ = targ.cuda()

        crit = criterion(out, targ) - 1 / 2 * (criterion(out, out) + criterion(targ, targ))
        loss += crit

        if test:
            preds.append(out[0].detach().numpy().T)

    if test:
        return loss, preds
    else:
        return loss


def train_corr(train_loader, model, optimizer, fact=None, is_cuda=False):
    tr_l = AccumLoss()

    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.train()
    for i, (batch_id, inputs) in enumerate(train_loader):

        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()
        targets = [train_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [train_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]

        deltas, att = model(inputs)

        # calculate loss and backward
        if fact is None:
            dtw = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda)
            loss = dtw / batch_size
        else:
            dtw = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1) / batch_size
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return tr_l.avg


def evaluate_corr(val_loader, model, fact=None, is_cuda=False):
    val_l = AccumLoss()

    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.eval()
    for i, (batch_id, inputs) in enumerate(val_loader):

        if is_cuda:
            inputs = inputs.cuda().float()
        else:
            inputs = inputs.float()
        targets = [val_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [val_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]

        deltas, att = model(inputs)

        # calculate loss and backward
        if fact is None:
            dtw = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda)
            loss = dtw
        else:
            dtw = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1)

        # update the training loss
        val_l.update(loss.cpu().data.numpy(), batch_size)

    return val_l.avg


def test_corr(test_loader, model, fact=None, is_cuda=False):
    test_l = AccumLoss()
    preds = {'in': [], 'out': [], 'targ': [], 'att': []}
    criterion = SoftDTW(use_cuda=is_cuda, gamma=0.01)
    model.eval()
    for i, (batch_id, inputs) in enumerate(test_loader):

        if is_cuda:
            inputs = inputs.cuda().float()
        else:
            inputs = inputs.float()
        targets = [test_loader.dataset.targets[int(i)] for i in batch_id]
        originals = [test_loader.dataset.inputs_raw[int(i)] for i in batch_id]
        batch_size = inputs.shape[0]

        deltas, att = model(inputs)

        # calculate loss and backward
        if fact is None:
            dtw, out = dtw_loss(originals, deltas, targets, criterion, is_cuda=is_cuda, test=True)
            loss = dtw
        else:
            dtw, out = dtw_loss(originals, deltas, targets, criterion, attentions=att, is_cuda=is_cuda, test=True)
            l1 = fact * torch.sum(torch.abs(att))
            loss = (dtw + l1)

        preds['in'] = preds['in'] + originals
        preds['out'] = preds['out'] + out
        preds['targ'] = preds['targ'] + targets
        preds['att'] = preds['att'] + [att[j].detach().numpy() for j in range(att.shape[0])]
        test_l.update(loss.cpu().data.numpy(), batch_size)

    return test_l.avg, preds


def train_class(train_loader, model, optimizer, is_cuda=False, level=0):
    tr_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.3, 1, 0.5, 1, 1]))
    else:
        criterion = nn.NLLLoss()
    model.train()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(train_loader):

        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([train_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level)
        batch_size = inputs.shape[0]

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        # update the training loss
        tr_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return tr_l.avg, 100 * correct / total


def evaluate_class(val_loader, model, is_cuda=False, level=0):
    val_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.5, 1, 0.5, 1, 0.5]))
    else:
        criterion = nn.NLLLoss()
    model.eval()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(val_loader):

        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([val_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level)
        batch_size = inputs.shape[0]

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)

        # update the training loss
        val_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

    return val_l.avg, 100 * correct / total


def test_class(test_loader, model, is_cuda=False, level=0):
    te_l = AccumLoss()

    if level == 0:
        criterion = nn.NLLLoss(weight=torch.tensor([1, 0.5, 1, 0.5, 1, 0.5]))
    else:
        criterion = nn.NLLLoss()
    model.eval()

    correct = 0
    total = 0
    for i, (batch_id, inputs) in enumerate(test_loader):

        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=level)
        batch_size = inputs.shape[0]

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # calculate loss and backward
        loss = criterion(outputs, labels)

        # update the training loss
        te_l.update(loss.cpu().data.numpy() * batch_size, batch_size)

        # summary = np.vstack((labels.numpy(), predicted.numpy()))
        summary = torch.stack((labels, predicted), dim=1)
        if level == 0:
            cmt = torch.zeros(6, 6, dtype=torch.int64)
        else:
            cmt = torch.zeros(12, 12, dtype=torch.int64)
        for p in summary:
            tl, pl = p.tolist()
            cmt[tl, pl] = cmt[tl, pl] + 1

    return te_l.avg, 100 * correct / total, summary, cmt
