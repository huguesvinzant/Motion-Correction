import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--vibe_dir', type=str, default='../data_3D_VIBE.pickle', help='path to dataset')
        self.parser.add_argument('--gt_dir', type=str, default='Data/010920/data_3D.pickle', help='path to dataset')
        self.parser.add_argument('--corr_model_dir', type=str, default='Results/model_corr.pt',
                                 help='path to saved file')
        self.parser.add_argument('--class_model_dir', type=str, default='Results/model_class.pt',
                                 help='path to saved file')

        # ===============================================================
        #                     Model & Running options
        # ===============================================================
        self.parser.add_argument('--dct_n', type=float, default=25, help='Number of DCT coefficients')
        self.parser.add_argument('--batch', type=float, default=128, help='Batch size')
        self.parser.add_argument('--hidden', type=float, default=32, help='Number of hidden features')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability, 1 for none')
        self.parser.add_argument('--block', type=float, default=1, help='Number of GC blocks')
        self.parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        self.parser.add_argument('--lr_decay', type=int, default=5, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.9)
        self.parser.add_argument('--epoch_corr', type=float, default=150, help='Number of epochs for correction')
        self.parser.add_argument('--epoch_class', type=float, default=50, help='Number of epochs for classification')

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        return self.opt
