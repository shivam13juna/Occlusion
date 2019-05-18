import torch
import numpy as np
import os
from deocclusion.deoccoption import DeocOptions
from netmodels.gannet.gan_model import Generator
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from other_utils.tensor_utils import unorm_print
from netmodels.unet.unet_model import UNet, AttriRAP, AttriAiC
from other_utils.folder_utils import check_generate_dir
from datasets.rap import RAPDataset
from other_utils.path_config import PathMng
from other_utils.class_utils import Toperation
from datasets.aic import AICDataset


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model = model.load_state_dict(checkpoint['gen_model'])
    return model


def save_images(generator, data_loader, output, opa):
    check_generate_dir(output)

    with torch.no_grad():
        generator.eval()
        for iteration, samp in enumerate(data_loader):
            img_in, _, _, img_att, names = samp

            img_in = Variable(img_in)
            img_att = Variable(img_att)

            img_in = img_in.cuda()
            img_att = img_att.cuda()

            if opa.gen_type == 'attribunet':
                img_out = generator(img_in, img_att)
            else:
                img_out = generator(img_in)

            for b_num in np.arange(0, img_out.shape[0]):
                save_image(unorm_print(img_out.data.cpu())[b_num, ...],
                           os.path.join(output, '{}'.format(names[b_num]).split('.')[0] + '.jpg'))


if __name__ == '__main__':
    do = DeocOptions()
    do.initializer()
    oparsed = do.make_parsing()

    pm = PathMng(oparsed.exp_name)
    pm.folders_initialization()

    if oparsed.dataset == 'RAP':
        attr_resized = True
    else:
        attr_resized = False

    if oparsed.dataset == 'RAP':
        data_test = RAPDataset(pm.rap_dataset, None, typed=Toperation.occlusion, mode=False,
                               attributes=oparsed.attributes, dst_size=oparsed.img_size, tran=True, fold=0,
                               attr_res=attr_resized)
    else:

        data_test = AICDataset(pm.naic_dataset, comp_path=None, train_data=False,
                               type_data=Toperation.occlusion,
                               dst_size=oparsed.img_size, tran=True, random_flip=0)

    dl = DataLoader(data_test, batch_size=oparsed.batch_size)

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

    gen = gen.cuda()
    checkpoint = torch.load(os.path.join(pm.experiment_results, 'last.checkpoint'))
    gen.load_state_dict(checkpoint['gen_model'])

    save_images(gen, dl, os.path.join(pm.experiment_results, 'converted'), oparsed)



