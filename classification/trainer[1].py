from tensorboardX import SummaryWriter
from datetime import datetime
from other_utils.avg_meter import AVGMeter
from other_utils.folder_utils import *
from other_utils.tensor_utils import *
from netloss.bce_loss import BCECustomLoss
from metrics.evalmetrics import *


class Trainer(object):
    def __init__(self, option, pm, mar_opt, train_data_loader, test_data_loader, marnet, info_eng_data, cons_attr=24):

        self.marnet = marnet
        self.exp_name = option.exp_name
        self.info_eng_data = info_eng_data
        self.cons_attr = cons_attr
        self.pm = pm
        self.option = option

        self.exp_path = pm.experiment_results

        self.gpu = option.gpu
        if self.gpu:
            self.gpu_id = option.gpu_id

        if option.dataset == 'RAP':
            folder_name = 'rap'
            mode_loss = 0
        else:
            folder_name = 'aic'
            mode_loss = 1

        info_path = os.path.join(pm.project_path, 'config/{}/info.json'.format(folder_name))
        with open(info_path, 'r') as tfile:
            a = json.load(tfile)

        weights = torch.zeros(2, self.cons_attr)
        for i in range(self.cons_attr):
            weights[0, i] = np.exp(1-a[0][i])
            weights[1, i] = np.exp(a[0][i])

        self.loss = BCECustomLoss(weights, mode_loss)

        if self.gpu:
            self.marnet = marnet.cuda(device=self.gpu_id)
            self.loss = self.loss.cuda(device=self.gpu_id)

        self.mar_opt = mar_opt

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.sm = [SingleClassMetrics() for _ in range(self.cons_attr)]

        self.best_test_acc = 0
        self.best_test_pre = 0
        self.best_test_rec = 0

        self.loss_avg_meter = AVGMeter()

        self.epoch = 0
        self.start_epoch = 0

        checkpoint_path = os.path.join(self.exp_path, 'lastmar.checkpoint')
        if check_file_existance(checkpoint_path):
            print('[loading checkpoint \'{}\']'.format(str(checkpoint_path)))
            self.load_checkpoint(checkpoint_path)

        self.step = 0
        self.tbx = SummaryWriter(self.exp_path)

    def eval(self):

        test_precision = RAPPrecision()
        test_recall = RAPRecall()
        test_accuracy = RAPAccuracy()
        test_mean_accuracy = MeanAccuracy()

        attr_loss = 0
        counter_loss = 0

        stest = [SingleClassMetrics() for _ in range(self.cons_attr)]

        with torch.no_grad():
            self.marnet.eval()
            for iteration, samp in enumerate(self.test_data_loader):
                img_in, img_att, _ = samp

                img_in = Variable(img_in)
                img_att = Variable(img_att)

                if self.gpu:
                    img_in = img_in.cuda(device=self.gpu_id)
                    img_att = img_att.cuda(device=self.gpu_id)

                test_y_predict = self.marnet(img_in)

                attr_loss += self.loss(test_y_predict, img_att).item()
                counter_loss += 1

                tmp_gt = img_att.clone()
                tmp_gt = tmp_gt.data.cpu().numpy()

                tmp_test_pred = test_y_predict.data.cpu().numpy()
                tmp_test_pred[tmp_test_pred > 0.5] = 1
                tmp_test_pred[tmp_test_pred <= 0.5] = 0

                for batch_elem in np.arange(0, img_in.shape[0]):
                    for j in range(self.cons_attr):
                        stest[j].append(tmp_gt[batch_elem, j], tmp_test_pred[batch_elem, j])
                    test_precision.append(tmp_gt[batch_elem, ...], tmp_test_pred[batch_elem, ...])
                    test_recall.append(tmp_gt[batch_elem, ...], tmp_test_pred[batch_elem, ...])
                    test_accuracy.append(tmp_gt[batch_elem, ...], tmp_test_pred[batch_elem, ...])
                    test_mean_accuracy.append(tmp_gt[batch_elem, ...], tmp_test_pred[batch_elem, ...])

        val_test = [0]*5

        for i in range(self.cons_attr):
            val_test[0] += stest[i].mean_accuracy
            val_test[1] += stest[i].accuracy
            val_test[2] += stest[i].precision
            val_test[3] += stest[i].recall
            val_test[4] += stest[i].f1

        prec = {
            'test_precision': val_test[2]/self.cons_attr,
            'test_precision_2': test_precision.get,
        }
        rec = {
            'test_recall': val_test[3]/self.cons_attr,
            'test_recall_2': test_recall.get,
        }
        acc = {
            'test_accuracy': val_test[1]/self.cons_attr,
            'test_accuracy_2': test_accuracy.get,
        }
        mean_acc = {
            'test_mean_accuracy': val_test[0]/self.cons_attr,
            'test_mean_accuracy_2': test_mean_accuracy.get,
        }
        f1 = {
            'test_f1': val_test[4]/self.cons_attr,
            'test_f1_2': 2 * (test_precision.get * test_recall.get) / (test_precision.get + test_recall.get),

        }
        self.tbx.add_scalars('PRECISION', prec, self.epoch)
        self.tbx.add_scalars('RECALL', rec, self.epoch)
        self.tbx.add_scalars('ACCURACY', acc, self.epoch)
        self.tbx.add_scalars('MEAN ACCURACY', mean_acc, self.epoch)
        self.tbx.add_scalars('F1', f1, self.epoch)
        self.tbx.add_scalars('test', {'loss': attr_loss/counter_loss}, self.epoch)

    def save_checkpoint(self, path):
        checkpoint = {
            'marnet': self.marnet.state_dict(),
            'loss_avgmeter': self.loss_avg_meter,
            'epoch': self.epoch,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.marnet.load_state_dict(checkpoint['marnet'])
        self.loss_avg_meter = checkpoint['loss_avgmeter']
        self.start_epoch = checkpoint['epoch'] + 1
        self.epoch = self.start_epoch

    def train(self):
        self.marnet.train()

        for self.step, sample in enumerate(self.train_data_loader):

            train_in, train_attr, _ = sample

            train_in = Variable(train_in)
            train_attr = Variable(train_attr)

            if self.gpu:
                train_in = train_in.cuda(device=self.gpu_id)
                train_attr = train_attr.cuda(device=self.gpu_id)

            self.mar_opt.zero_grad()

            y_pred = self.marnet(train_in)
            attr_loss = self.loss(y_pred, train_attr)

            self.loss_avg_meter.append(attr_loss.item())

            attr_loss.backward()
            self.mar_opt.step(closure=None)

            progress = (self.step + 1) / len(self.train_data_loader)
            if (self.step + 1) % 10 == 0 or progress >= 1:
                print('\r[{}] Epoch {:04d}.{:04d}: │ {:6.2f}% | train_loss: {:.4f}'
                      .format(datetime.now().strftime("%Y-%m-%d@%H:%M"), self.epoch, self.step, 100 * progress,
                              self.loss_avg_meter.last), end='')

            if self.step % 10 == 0:
                tmp = {'bce_loss': self.loss_avg_meter.last}
                self.tbx.add_scalars('train', tmp,  (self.epoch*len(self.train_data_loader))+self.step)

    def run(self, epochs):

        for self.epoch in range(self.start_epoch, epochs):
            self.train()
            self.save_checkpoint(os.path.join(self.exp_path, 'lastmar.checkpoint'))
            self.eval()
