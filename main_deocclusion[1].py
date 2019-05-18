import torch
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from deocclusion.trainer import Trainer
from netmodels.gannet.gan_model import Generator, Discriminator
from other_utils.tensor_utils import weights_init
from netmodels.unet.unet_model import UNet, AttriRAP, AttriAiC
from deocclusion.deoccoption import DeocOptions
from datasets.rap import RAPDataset, Toperation
from datasets.aic import AICDataset
from other_utils.path_config import PathMng


SEED = 1821
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


def main():
    do = DeocOptions()
    do.initializer()
    oparsed = do.make_parsing()

    pm = PathMng(oparsed.exp_name)
    pm.folders_initialization()

    print('Experiment \'{}\''.format(oparsed.exp_name))

    if oparsed.dataset == 'RAP':
        attr_resized = True
    else:
        attr_resized = False

    if oparsed.dataset == 'RAP':
        data_train = RAPDataset(pm.rap_dataset, None, typed=Toperation.occlusion, mode=True,
                                attributes=oparsed.attributes, dst_size=oparsed.img_size, tran=True, fold=0,
                                attr_res=attr_resized)
        data_test = RAPDataset(pm.rap_dataset, None, typed=Toperation.occlusion, mode=False,
                               attributes=oparsed.attributes, dst_size=oparsed.img_size, tran=True, fold=0,
                               attr_res=attr_resized)
    else:
        data_train = AICDataset(pm.naic_dataset, comp_path=None, train_data=True,
                                type_data=Toperation.occlusion,
                                dst_size=oparsed.img_size, tran=True, random_flip=0.5)
        data_test = AICDataset(pm.naic_dataset, comp_path=None, train_data=False,
                               type_data=Toperation.occlusion,
                               dst_size=oparsed.img_size, tran=True, random_flip=0)

    train_loader = DataLoader(data_train, batch_size=oparsed.batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(data_test, batch_size=oparsed.batch_size, shuffle=True, num_workers=1)
    print('data loader ready')

    dsc = Discriminator(oparsed.c_out, f=oparsed.f, input_shape=oparsed.img_size)

    if oparsed.gen_type == 'unet':
        gen = UNet(oparsed.c_in, oparsed.c_out, bilinear=True)
    elif oparsed.gen_type == 'attribunet':
        if oparsed.dataset == 'RAP':
            gen = AttriRAP(oparsed.c_in, oparsed.c_out, oparsed.attributes, bilinear=True)
        else:
            gen = AttriAiC(oparsed.c_in, oparsed.c_out, oparsed.attributes, bilinear=True)
    elif oparsed.gen_type == 'classic':
        gen = Generator(oparsed.c_in, oparsed.c_out, f=oparsed.f, network_mode=oparsed.network_mode)
    else:
        print('Invalid generator')
        raise Exception
    print('network loaded')

    # if oparsed.gpu:
    #     dsc = dsc.cuda(oparsed.gpu_id)
    #     gen = gen.cuda()

    if oparsed.w_init:
        gen.apply(weights_init)
        dsc.apply(weights_init)
    print('Weights initialization')

    gen_opt = optim.Adam(gen.parameters(), lr=oparsed.lr, betas=(0.5, 0.999))
    dsc_opt = optim.Adam(dsc.parameters(), lr=oparsed.lr, betas=(0.5, 0.999))

    trainer = Trainer(
        pm=pm,
        parsedoption=oparsed,
        gen_model=gen,
        dsc_model=dsc,
        gen_opt=gen_opt,
        dsc_opt=dsc_opt,
        data_loader_train=train_loader,
        data_loader_test=test_loader,
    )
    trainer.run(oparsed.epochs)


if __name__ == '__main__':
    main()
