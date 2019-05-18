import torch
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from classification.trainer import Trainer
from netmodels.netmar.resnet import resnet101_mar
from classification.classoption import ClassOptions
from datasets.rap import RAPDataset
from datasets.aic import AICDataset
from other_utils.path_config import PathMng
from other_utils.class_utils import Toperation
from datasets.aic_info import SINGLE_ATTR


SEED = 1821
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def main():
    bo = ClassOptions()
    bo.initializer()
    oparsed = bo.make_parsing()
    pm = PathMng(oparsed.exp_name)
    pm.folders_initialization()

    if oparsed.dataset == 'RAP':
        data_train = RAPDataset(pm.rap_dataset, None, Toperation.classification, True,
                                attributes=oparsed.attributes, dst_size=oparsed.img_size,
                                fold=oparsed.fold)
        data_test = RAPDataset(pm.rap_dataset, None, Toperation.classification, False,
                               attributes=oparsed.attributes, dst_size=oparsed.img_size,
                               fold=oparsed.fold)
        info_eng_data = data_train.attributes_eng[0:oparsed.attributes]
    else:
        data_train = AICDataset(pm.naic_dataset, comp_path=None, train_data=True,
                                type_data=Toperation.classification,
                                dst_size=oparsed.img_size, tran=True, random_flip=1)
        data_test = AICDataset(pm.naic_dataset, comp_path=None, train_data=False,
                               type_data=Toperation.classification,
                               dst_size=oparsed.img_size, tran=True, random_flip=0)
        info_eng_data = SINGLE_ATTR[0: oparsed.attributes]

    train_loader = DataLoader(data_train, batch_size=oparsed.batch_size, shuffle=True, num_workers=oparsed.workers)
    test_loader = DataLoader(data_test, batch_size=oparsed.batch_size, shuffle=False, num_workers=oparsed.workers)

    network = resnet101_mar(num_classes=oparsed.attributes, pretrained=True)

    optm = optim.Adam(network.parameters(), lr=oparsed.lr, betas=(0.5, 0.999))
    tra = Trainer(oparsed, pm, optm, train_loader, test_loader, network, info_eng_data, cons_attr=oparsed.attributes)
    tra.run(oparsed.epochs)


if __name__ == '__main__':
    main()
