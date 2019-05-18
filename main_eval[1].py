import random
import other_utils.tensor_utils as tu
from torch.utils.data import DataLoader
from deocclusion.deoccoption import *
from metrics.evalmetrics import *
from datasets.rap import RAPDataset
from datasets.aic import AICDataset
from netmodels.netmar.resnet import resnet101_mar
from netmodels.unet.unet_model import UNet, AttriRAP, AttriAiC
from other_utils.folder_utils import *
from other_utils.path_config import PathMng
from other_utils.class_utils import Toperation
from torch.autograd import Variable
from netmodels.gannet.gan_model import Generator
from datasets.aic_info import SINGLE_ATTR

SEED = 1821
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

#############################
#           MAR             #
#############################

def mar(parsedp, pm):

    if parsedp.dataset == 'RAP':
        attr_resized = True
    else:
        attr_resized = False

    if parsedp.dataset == 'RAP':
        data = RAPDataset(pm.rap_dataset, None, typed=Toperation.occlusion, mode=False,
                          attributes=parsedp.attributes, dst_size=parsedp.img_size, tran=True, fold=0,
                          attr_res=attr_resized)
        eng_att = data.attributes_eng
        resmar = resnet101_mar(num_classes=parsedp.attributes)
        resmar_weights_path = os.path.join(pm.rap_dataset, 'lastmar.checkpoint')
    else:
        data = AICDataset(pm.naic_dataset, comp_path=None, train_data=False, type_data=Toperation.occlusion,
                          dst_size=oparsed.img_size, tran=True, random_flip=0)
        eng_att = SINGLE_ATTR[0:parsedp.attributes]
        resmar = resnet101_mar(num_classes=parsedp.attributes)
        resmar_weights_path = os.path.join(pm.naic_dataset, 'lastmar.checkpoint')

    with open('./config/th.json', 'r') as fn:
        ath = json.load(fn)

    if parsedp.dataset == 'AIC':
        ath = ath[0]
    else:
        ath = ath[1]

    data_loader = DataLoader(data, batch_size=parsedp.batch_size, num_workers=1)

    resmar_checkpoint = torch.load(resmar_weights_path)
    resmar.load_state_dict(resmar_checkpoint['marnet'])

    if parsedp.gen_type == 'unet':
        gen = UNet(parsedp.c_in, parsedp.c_out, bilinear=True)
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

    gen_checkpoint = torch.load(os.path.join(pm.experiment_results, 'last.checkpoint'))
    gen.load_state_dict(gen_checkpoint['gen_model'])

    if parsedp.gpu:
        gen = gen.cuda()
        resmar = resmar.cuda(device=parsedp.gpu_id)

    ori_precision = RAPPrecision()
    ori_recall = RAPRecall()
    ori_accuracy = RAPAccuracy()

    occ_precision = RAPPrecision()
    occ_recall = RAPRecall()
    occ_accuracy = RAPAccuracy()

    deocc_precision = RAPPrecision()
    deocc_recall = RAPRecall()
    deocc_accuracy = RAPAccuracy()

    sm_orig = [SingleClassMetrics() for _ in range(parsedp.attributes)]
    sm_occ = [SingleClassMetrics() for _ in range(parsedp.attributes)]
    sm_deocc = [SingleClassMetrics() for _ in range(parsedp.attributes)]

    with torch.no_grad():
        resmar.eval()
        gen.eval()
        for iteration, sample in enumerate(data_loader):

            data_in, data_gt, data_att, data_attr_res, _ = sample

            data_in = Variable(data_in)
            data_gt = Variable(data_gt)
            data_att = Variable(data_att)
            data_attr_res = Variable(data_attr_res)

            if parsedp.gpu:
                data_in = data_in.cuda()
                data_gt = data_gt.cuda()
                data_att = data_att.cuda()
                data_attr_res = data_attr_res.cuda()
            if parsedp.gen_type == 'attribunet':
                img_out = gen(data_in, data_attr_res)
            else:
                img_out = gen(data_in)

            ori_predict = resmar(data_gt)
            occ_predict = resmar(data_in)
            deocc_predict = resmar(img_out)

            tmp_gt = data_att.clone()
            tmp_gt = tmp_gt.data.cpu().numpy()

            ori_pred = ori_predict.data.cpu().numpy()
            occ_pred = occ_predict.data.cpu().numpy()
            deocc_pred = deocc_predict.data.cpu().numpy()

            for i in range(parsedp.attributes):

                tmp_ori = ori_pred[:, i, ...]
                tmp_ori[tmp_ori > ath[i]] = 1
                tmp_ori[tmp_ori <= ath[i]] = 0
                ori_pred[:, i, ...] = tmp_ori

                tmp_occ = occ_pred[:, i, ...]
                tmp_occ[tmp_occ > ath[i]] = 1
                tmp_occ[tmp_occ <= ath[i]] = 0
                occ_pred[:, i, ...] = tmp_occ

                tmp_deocc = deocc_pred[:, i, ...]
                tmp_deocc[tmp_deocc > ath[i]] = 1
                tmp_deocc[tmp_deocc <= ath[i]] = 0
                deocc_pred[:, i, ...] = tmp_deocc

            for batch_elem in np.arange(0, img_out.shape[0]):
                ori_precision.append(tmp_gt[batch_elem, ...], ori_pred[batch_elem, ...])
                ori_recall.append(tmp_gt[batch_elem, ...], ori_pred[batch_elem, ...])
                ori_accuracy.append(tmp_gt[batch_elem, ...], ori_pred[batch_elem, ...])

                occ_precision.append(tmp_gt[batch_elem, ...], occ_pred[batch_elem, ...])
                occ_recall.append(tmp_gt[batch_elem, ...], occ_pred[batch_elem, ...])
                occ_accuracy.append(tmp_gt[batch_elem, ...], occ_pred[batch_elem, ...])

                deocc_precision.append(tmp_gt[batch_elem, ...], deocc_pred[batch_elem, ...])
                deocc_recall.append(tmp_gt[batch_elem, ...], deocc_pred[batch_elem, ...])
                deocc_accuracy.append(tmp_gt[batch_elem, ...], deocc_pred[batch_elem, ...])

                for j in range(parsedp.attributes):
                    sm_orig[j].append(tmp_gt[batch_elem, j], ori_pred[batch_elem, j])
                    sm_occ[j].append(tmp_gt[batch_elem, j], occ_pred[batch_elem, j])
                    sm_deocc[j].append(tmp_gt[batch_elem, j], deocc_pred[batch_elem, j])

    info_ori = dict()
    info_ori['precision'] = ori_precision.get
    info_ori['recall'] = ori_recall.get
    info_ori['accuracy'] = ori_accuracy.get
    info_ori['f1'] = 2 * (ori_precision.get * ori_recall.get) / (ori_precision.get + ori_recall.get)

    info_occ = dict()
    info_occ['precision'] = occ_precision.get
    info_occ['recall'] = occ_recall.get
    info_occ['accuracy'] = occ_accuracy.get
    info_occ['f1'] = 2 * (occ_precision.get * occ_recall.get) / (occ_precision.get + occ_recall.get)

    info_deocc = dict()
    info_deocc['precision'] = deocc_precision.get
    info_deocc['recall'] = deocc_recall.get
    info_deocc['accuracy'] = deocc_accuracy.get
    info_deocc['f1'] = 2 * (deocc_precision.get * deocc_recall.get) / (deocc_precision.get + deocc_recall.get)

    info1 = {'original': info_ori, 'occlusion': info_occ, 'deocclusion': info_deocc}

    info2 = {}

    t_mean_accuracy = np.array([0.] * 3)
    t_accuracy = np.array([0.] * 3)
    t_precision = np.array([0.] * 3)
    t_recall = np.array([0.] * 3)
    t_f1 = np.array([0.] * 3)

    for i in range(parsedp.attributes):
        info2['{}'.format(eng_att[i])] = {
            'mean_accuracy': [sm_orig[i].mean_accuracy, sm_occ[i].mean_accuracy, sm_deocc[i].mean_accuracy],
            'accuracy': [sm_orig[i].accuracy, sm_occ[i].accuracy, sm_deocc[i].accuracy],
            'precision': [sm_orig[i].precision, sm_occ[i].precision, sm_deocc[i].precision],
            'recall': [sm_orig[i].recall, sm_occ[i].recall, sm_deocc[i].recall],
            'f1': [sm_orig[i].f1, sm_occ[i].f1, sm_deocc[i].f1]
        }

        t_mean_accuracy[0] += sm_orig[i].mean_accuracy
        t_mean_accuracy[1] += sm_occ[i].mean_accuracy
        t_mean_accuracy[2] += sm_deocc[i].mean_accuracy

        t_accuracy[0] += sm_orig[i].accuracy
        t_accuracy[1] += sm_occ[i].accuracy
        t_accuracy[2] += sm_deocc[i].accuracy

        t_precision[0] += sm_orig[i].precision
        t_precision[1] += sm_occ[i].precision
        t_precision[2] += sm_deocc[i].precision

        t_recall[0] += sm_orig[i].recall
        t_recall[1] += sm_occ[i].recall
        t_recall[2] += sm_deocc[i].recall

        t_f1[0] += sm_orig[i].f1
        t_f1[1] += sm_occ[i].f1
        t_f1[2] += sm_deocc[i].f1

    t_mean_accuracy = t_mean_accuracy / parsedp.attributes
    t_accuracy = t_accuracy / parsedp.attributes
    t_recall = t_recall / parsedp.attributes
    t_precision = t_precision / parsedp.attributes
    t_f1 = t_f1 / parsedp.attributes

    info2['all_metrics'] = {'mean_accuracy': t_mean_accuracy.tolist(), 'accuracy': t_accuracy.tolist(),
                            'precision': t_precision.tolist(),
                            'recall': t_recall.tolist(), 'f1': t_f1.tolist()}

    with open(os.path.join(pm.experiment_results, 'class_mtr.json'), 'w') as fn:
        json.dump(info1, fn)

    with open(os.path.join(pm.experiment_results, 'all_att_mtr.json'), 'w') as fn:
        json.dump(info2, fn)


#############################
#           NOISE           #
#############################
def noise(parsedp, pm):

    if parsedp.dataset == 'RAP':
        attr_resized = True
    else:
        attr_resized = False

    if parsedp.dataset == 'RAP':
        data = RAPDataset(pm.rap_dataset, None, typed=Toperation.occlusion, mode=False,
                          attributes=parsedp.attributes, dst_size=parsedp.img_size, tran=True, fold=0,
                          attr_res=attr_resized)
    else:
        data = AICDataset(pm.naic_dataset, comp_path=None, train_data=False, type_data=Toperation.occlusion,
                          dst_size=oparsed.img_size, tran=True, random_flip=0)

    data_loader = DataLoader(data, batch_size=parsedp.batch_size, num_workers=1)

    psnr_avg_occ = PSNR()
    ssim_avg_occ = SSIM()

    psnr_avg_deocc = PSNR()
    ssim_avg_deocc = SSIM()

    if parsedp.gen_type == 'unet':
        gen = UNet(parsedp.c_in, parsedp.c_out, bilinear=True)
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

    gen_checkpoint = torch.load(os.path.join(pm.experiment_results, 'last.checkpoint'))
    gen.load_state_dict(gen_checkpoint['gen_model'])

    if parsedp.gpu:
        gen = gen.cuda()

    with torch.no_grad():
        gen.eval()

        for step, sample in enumerate(data_loader):

            data_in, data_gt, _, data_attr_res, _ = sample
            data_in = Variable(data_in)
            data_gt = Variable(data_gt)
            data_attr_res = Variable(data_attr_res)

            if parsedp.gpu:
                data_in = data_in.cuda()
                data_gt = data_gt.cuda()
                data_attr_res = data_attr_res.cuda()
            if parsedp.gen_type == 'attribunet':
                img_out = gen(data_in, data_attr_res)
            else:
                img_out = gen(data_in)

            img_gt = tu.unorm_print(data_gt.data.cpu()).squeeze().mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
            img_occ = tu.unorm_print(data_in.data.cpu()).squeeze().mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
            img_deocc = tu.unorm_print(img_out.data.cpu()).squeeze().mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()

            for batch_elem in np.arange(0, img_gt.shape[0]):
                psnr_avg_occ.append(img_gt[batch_elem, ...], img_occ[batch_elem, ...])
                ssim_avg_occ.append(img_gt[batch_elem, ...], img_occ[batch_elem, ...])
                psnr_avg_deocc.append(img_gt[batch_elem, ...], img_deocc[batch_elem, ...])
                ssim_avg_deocc.append(img_gt[batch_elem, ...], img_deocc[batch_elem, ...])

    noise_log = {'occ': {'psnr': psnr_avg_occ.get, 'ssim': ssim_avg_occ.get},
                 'deocc': {'psnr': psnr_avg_deocc.get, 'ssim': ssim_avg_deocc.get}}

    with open(os.path.join(pm.experiment_results, 'noise.json'), 'w') as fn:
        json.dump(noise_log, fn)


if __name__ == '__main__':

    mo = DeocOptions()
    mo.initializer()
    oparsed = mo.make_parsing()
    pmn = PathMng(oparsed.exp_name)
    pmn.folders_initialization()

    print('Start metric calculation')
    mar(oparsed, pmn)
    print('End multi-attribute recognition metrics')
    noise(oparsed, pmn)
    print('End noise metrics')
