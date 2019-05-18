import os
import torch
import torchvision
from tensorboardX import SummaryWriter
from datetime import datetime
from torch import nn
from torch.autograd import Variable
from other_utils.avg_meter import AVGMeter
from netloss.vgg_loss import VggLoss
from pathlib import Path
from other_utils.tensor_utils import unorm_print
from netloss.attr_loss import AttributeLoss


class Trainer(object):
    def __init__(self, pm, parsedoption, gen_model, dsc_model, gen_opt, dsc_opt, data_loader_train, data_loader_test):

        self.gen_name = parsedoption.gen_type

        self.gpu = parsedoption.gpu
        if self.gpu:
            self.gpu_id = parsedoption.gpu_id

        self.exp_path = pm.experiment_results
        self.w1 = parsedoption.w1
        self.w2 = parsedoption.w2
        self.min_test = 10000
        self.tc = parsedoption.tc

        self.fake_real_trick = parsedoption.fake_real_trick
        self.sec_loss = parsedoption.second_loss
        self.thi_loss = parsedoption.third_loss

        self.gen_model = gen_model
        self.dsc_model = dsc_model
        self.adv_loss = nn.BCELoss()

        if self.sec_loss == 'vggLoss':
            self.secondary_loss = VggLoss()
        else:
            self.secondary_loss = nn.MSELoss()

        if parsedoption.dataset == 'RAP':
            dataspath = pm.rap_dataset
            ipath = 'rap'
            mode_loss = 0
        else:
            dataspath = pm.naic_dataset
            ipath = 'aic'
            mode_loss = 1

        if self.w2 != 0:
            resmar_weights_path = os.path.join(dataspath, 'lastmar.checkpoint')
            infopath = os.path.join(pm.project_path, 'config/{}/info.json'.format(ipath))
            self.third_loss = AttributeLoss(resmar_weights_path, infopath, parsedoption.attributes, requires_grad=False
                                            , mode=mode_loss)
        else:
            self.third_loss = None

        if self.gpu:
            self.gen_model = gen_model.cuda(device=self.gpu_id)
            self.dsc_model = dsc_model.cuda(device=self.gpu_id)
            self.adv_loss = self.adv_loss.cuda(device=self.gpu_id)
            self.secondary_loss = self.secondary_loss.cuda(device=self.gpu_id)
            self.third_loss = self.third_loss.cuda(device=self.gpu_id)

        self.gen_opt = gen_opt
        self.dsc_opt = dsc_opt

        self.train_data_loader = data_loader_train
        self.test_data_loader = data_loader_test

        self.g_sec_losses = AVGMeter()
        self.d_losses_real = AVGMeter()
        self.d_losses_gen = AVGMeter()
        self.g_losses = AVGMeter()
        self.g_thi_losses = AVGMeter()

        self.epoch = 0
        self.start_epoch = 0

        self.test_imgs, self.gt_imgs, self.att_imgs, self.att_res, _ = self.test_data_loader.__iter__().__next__()

        checkpoint_path = os.path.join(self.exp_path, 'last.checkpoint')
        if os.path.isfile(checkpoint_path):
            print('[loading checkpoint \'{}\']'.format(checkpoint_path))
            self.load_checkpoint(checkpoint_path)

        self.step = 0

        self.tbx = SummaryWriter(self.exp_path)

    def visualize(self):

        with torch.no_grad():
            self.gen_model.eval()

            img_in = Variable(self.test_imgs)
            img_gt = Variable(self.gt_imgs)
            img_att_res = Variable(self.att_res)

            if self.gpu:
                img_gt = img_gt.cuda(device=self.gpu_id)
                img_in = img_in.cuda(device=self.gpu_id)
                img_att_res = img_att_res.cuda(device=self.gpu_id)

            if self.gen_name == 'attribunet':
                img_out = self.gen_model(img_in, img_att_res)
            else:
                img_out = self.gen_model(img_in)

            if self.epoch == 0:
                torchvision.utils.save_image(unorm_print(img_in.data.cpu()[:, :3, ...]),
                                             os.path.join(self.exp_path, 'IN_RGB_{}.jpg'.format(self.epoch)))

                torchvision.utils.save_image(unorm_print(img_gt.data.cpu()[:, :3, ...]),
                                             os.path.join(self.exp_path, 'GT_RGB_{}.jpg'.format(self.epoch)))

            torchvision.utils.save_image(unorm_print(img_out.data.cpu()[:, :3, ...]),
                                         os.path.join(self.exp_path, 'OUT_RGB_{}.jpg'.format(self.epoch)))

    def save_checkpoint(self, path):
        checkpoint = {
            'gen_model': self.gen_model.state_dict(),
            'dsc_model': self.dsc_model.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'dsc_opt': self.dsc_opt.state_dict(),
            'g_sec_losses': self.g_sec_losses,
            'd_losses_real': self.d_losses_real,
            'd_losses_gen': self.d_losses_gen,
            'g_losses': self.g_losses,
            'epoch': self.epoch,
            'test_imgs': self.test_imgs,
            'gt_imgs': self.gt_imgs,
            'att_imgs': self.att_imgs,
            'att_rsd': self.att_res,
            'g_thi_losses': self.g_thi_losses,
            'min_test': self.min_test

        }
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, path):
        checkpoint = torch.load(str(path))

        self.gen_model.load_state_dict(checkpoint['gen_model'])
        self.dsc_model.load_state_dict(checkpoint['dsc_model'])
        self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        self.dsc_opt.load_state_dict(checkpoint['dsc_opt'])
        self.g_sec_losses = checkpoint['g_sec_losses']
        self.d_losses_real = checkpoint['d_losses_real']
        self.d_losses_gen = checkpoint['d_losses_gen']
        self.g_losses = checkpoint['g_losses']
        self.g_thi_losses = checkpoint['g_thi_losses']

        self.start_epoch = checkpoint['epoch'] + 1
        self.epoch = self.start_epoch

        self.test_imgs = checkpoint['test_imgs']
        self.gt_imgs = checkpoint['gt_imgs']
        self.att_imgs = checkpoint['att_imgs']
        self.att_res = checkpoint['att_rsd']
        self.min_test = checkpoint['min_test']

    def fit_discriminator(self, x_real, y_true_real, x_gen, y_false_gen):

        self.dsc_opt.zero_grad()

        y_pred_real = self.dsc_model(x_real)
        d_loss_real = self.adv_loss(y_pred_real, y_true_real)
        self.d_losses_real.append(d_loss_real.item())

        y_pred_gen = self.dsc_model(x_gen.detach())
        d_loss_gen = self.adv_loss(y_pred_gen, y_false_gen)
        self.d_losses_gen.append(d_loss_gen.item())

        d_loss = (d_loss_real + d_loss_gen) * 0.5
        d_loss.backward()
        self.dsc_opt.step(closure=None)

        if self.step % 10 == 0:
            tmp_val = {
                'd_loss_real': self.d_losses_real.last,
                'd_loss_gen': self.d_losses_gen.last,
                'd_loss': (self.d_losses_real.last + self.d_losses_gen.last) * 0.5
            }
            self.tbx.add_scalars('Discriminator', tmp_val, (self.epoch * len(self.train_data_loader)) + self.step)

    def fit_generator(self, train_in, train_attr, train_attr_res, real_image, y_true):

        self.gen_opt.zero_grad()

        if self.gen_name == 'attribunet':
            generated_img = self.gen_model(train_in, train_attr_res)
        else:
            generated_img = self.gen_model(train_in)

        second_loss = self.w1 * self.secondary_loss(generated_img, real_image)
        self.g_sec_losses.append(second_loss.item())

        if self.third_loss is not None:
            third_loss = self.w2 * self.third_loss(generated_img, train_attr)
            self.g_thi_losses.append(third_loss.item())
        else:
            self.g_thi_losses.append(0)

        y_pred = self.dsc_model(generated_img)
        cls_loss = self.adv_loss(y_pred, y_true)
        self.g_losses.append(cls_loss.item())

        if self.third_loss is not None:
            sum_losses = second_loss + cls_loss + third_loss
        else:
            sum_losses = second_loss + cls_loss

        sum_losses.backward()
        self.gen_opt.step(closure=None)

        if self.step % 10 == 0:
            tmp_val = {
                'adv_gen_loss': self.g_losses.last,
                '{}'.format(self.sec_loss): self.g_sec_losses.last,
                '{}'.format(self.thi_loss): self.g_thi_losses.last,
                'g_sum_loss': (self.g_losses.last+self.g_sec_losses.last+self.g_thi_losses.last)

            }
            self.tbx.add_scalars('Generator', tmp_val, (self.epoch*len(self.train_data_loader))+self.step)

    def test(self):
        self.gen_model.eval()
        self.dsc_model.eval()

        with torch.no_grad():
            vggloss = 0
            attloss = 0
            genloss = 0
            disloss = 0
            tot = 0

            for self.step, sample in enumerate(self.test_data_loader):
                test_in, test_gt, test_attr, test_attr_res, _ = sample

                test_in = Variable(test_in)
                test_gt = Variable(test_gt)
                test_attr = Variable(test_attr)
                test_attr_res = Variable(test_attr_res)
                batch_size = test_in.shape[0]

                if self.fake_real_trick:
                    y_true_real = Variable((torch.rand((batch_size, 1)) * 0.5) + 0.7)
                    y_false_gen = Variable(torch.rand((batch_size, 1)) * 0.3)
                    y_true_gen = Variable((torch.rand((batch_size, 1)) * 0.5) + 0.7)
                else:
                    y_true_real = Variable(torch.ones((batch_size, 1)) * 0.9)
                    y_false_gen = Variable(torch.zeros((batch_size, 1)))
                    y_true_gen = Variable(torch.ones((batch_size, 1)) * 0.9)

                if self.gpu:
                    test_in = test_in.cuda(device=self.gpu_id)
                    test_gt = test_gt.cuda(device=self.gpu_id)
                    test_attr = test_attr.cuda(device=self.gpu_id)
                    test_attr_res = test_attr_res.cuda(device=self.gpu_id)
                    y_true_real = y_true_real.cuda(device=self.gpu_id)
                    y_false_gen = y_false_gen.cuda(device=self.gpu_id)
                    y_true_gen = y_true_gen.cuda(device=self.gpu_id)

                if self.gen_name == 'attribunet':
                    test_generated = self.gen_model(test_in, test_attr_res)
                else:
                    test_generated = self.gen_model(test_in)

                # Generator test loss
                y_pred = self.dsc_model(test_generated)
                adv_loss = self.adv_loss(y_pred, y_true_gen)
                genloss += adv_loss.item()

                sloss = self.secondary_loss(test_generated, test_gt)
                vggloss += sloss.item()

                if self.third_loss is not None:
                    third_loss = self.third_loss(test_generated, test_attr)
                    attloss += third_loss.item()
                else:
                    attloss += 0

                # Discriminator test loss
                y_pred_real = self.dsc_model(test_gt)
                d_loss_real = self.adv_loss(y_pred_real, y_true_real)

                y_pred_gen = self.dsc_model(test_generated)
                d_loss_gen = self.adv_loss(y_pred_gen, y_false_gen)

                disloss += (d_loss_gen.item() + d_loss_real.item()) * 0.5

                tot += 1

        tmp_val_gen = {
            'vgg_loss_test': vggloss / tot,
            'att_loss_test': attloss / tot,
            'adv_loss_test': genloss / tot
        }

        tmp_val_dsc = {
            'adv_loss_test': disloss / tot
        }

        self.tbx.add_scalars('Generator_test', tmp_val_gen, self.epoch)
        self.tbx.add_scalars('Discriminator_test', tmp_val_dsc, self.epoch)

    def train(self):
        self.gen_model.train()
        self.dsc_model.train()

        for self.step, sample in enumerate(self.train_data_loader):

            train_in, train_gt, train_attr, train_attr_res, _ = sample
            batch_size = train_in.shape[0]

            train_in = Variable(train_in)
            train_gt = Variable(train_gt)
            train_attr = Variable(train_attr)
            train_attr_res = Variable(train_attr_res)

            if self.fake_real_trick:
                y_true_real = Variable((torch.rand((batch_size, 1))*0.5)+0.7)
                y_false_gen = Variable(torch.rand((batch_size, 1))*0.3)
                y_true_gen = Variable((torch.rand((batch_size, 1))*0.5)+0.7)
            else:
                y_true_real = Variable(torch.ones((batch_size, 1))*0.9)
                y_false_gen = Variable(torch.zeros((batch_size, 1)))
                y_true_gen = Variable(torch.ones((batch_size, 1))*0.9)

            if self.gpu:
                train_in = train_in.cuda(device=self.gpu_id)
                train_gt = train_gt.cuda(device=self.gpu_id)
                train_attr = train_attr.cuda(device=self.gpu_id)
                train_attr_res = train_attr_res.cuda(device=self.gpu_id)
                y_true_real = y_true_real.cuda(device=self.gpu_id)
                y_false_gen = y_false_gen.cuda(device=self.gpu_id)
                y_true_gen = y_true_gen.cuda(device=self.gpu_id)

            if self.gen_name == 'attribunet':
                train_generated = self.gen_model(train_in, train_attr_res)
            else:
                train_generated = self.gen_model(train_in)

            self.fit_discriminator(train_gt, y_true_real, train_generated, y_false_gen)
            self.fit_generator(train_in, train_attr, train_attr_res, train_gt, y_true_gen)

            progress = (self.step + 1) / len(self.train_data_loader)
            # progress_bar = ('█' * int(30 * progress)) + ('┈' * (30 - int(30 * progress)))
            if (self.step + 1) % 10 == 0 or progress >= 1:
                print('\r[{}] Epoch {:04d}.{:04d}: {:6.2f}% | D_real: {:.4f} | '
                      'D_gen: {:.4f} | G: {:.4f} | G_sec {:.4f} | G_third {:.4f}'
                      .format(datetime.now().strftime("%Y-%m-%d@%H:%M"), self.epoch, self.step,
                              100 * progress,
                              self.d_losses_real.avg,
                              self.d_losses_gen.avg,
                              self.g_losses.avg,
                              self.g_sec_losses.avg,
                              self.g_thi_losses.avg
                              ), end='')

    def run(self, epochs):

        for self.epoch in range(self.start_epoch, epochs):

            self.train()
            self.save_checkpoint(Path(self.exp_path, 'last.checkpoint'))
            if self.tc > 0 and (self.epoch != 0 and (self.epoch % self.tc) == 0):
                self.test()
            self.visualize()
