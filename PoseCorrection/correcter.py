import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

from PoseCorrection.dataset import HV3D
from PoseCorrection.model import GCN_corr
from PoseCorrection.opt import Options
from PoseCorrection.utils import *


def main(opt):
    torch.manual_seed(0)
    np.random.seed(0)

    is_cuda = torch.cuda.is_available()

    print('Loading data...')
    try:
        with open('tmp.pickle', "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']
    except FileNotFoundError:
        sets = [[0, 1, 2], [], [3]]
        data_train = HV3D(opt.gt_dir, sets=sets, split=0, is_cuda=is_cuda)
        data_test = HV3D(opt.gt_dir, sets=sets, split=2, is_cuda=is_cuda)
        with open('Data/tmp.pickle', 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)

    train_loader = DataLoader(dataset=data_train, batch_size=opt.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    model = GCN_corr()
    if is_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    lr_now = opt.lr
    with tqdm(range(opt.epoch_corr), desc=f'Training Correcter', unit="epoch") as tepoch:
        for epoch in tepoch:
            if (epoch + 1) % opt.lr_decay == 0:
                lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
            tr_l = train_corr(train_loader, model, optimizer, is_cuda=is_cuda)
            tepoch.set_postfix(train_loss=tr_l.item())

    torch.save(model.state_dict(), opt.corr_model_dir)

    model = GCN_corr()
    model.load_state_dict(torch.load('Results/model_corr1.pt'))

    if is_cuda:
        model.cuda()

    with torch.no_grad():
        test_l, preds = test_corr(test_loader, model, is_cuda=is_cuda)

    with open('Results/res1.pickle', 'wb') as f:
        pickle.dump({'loss': test_l, 'preds': preds}, f)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
