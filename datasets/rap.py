import scipy.io
import os
from torchvision import transforms
from torch.utils.data import Dataset
from other_utils.folder_utils import read_json
from other_utils.class_utils import Toperation
from other_utils.tensor_utils import *


class RAPDataset(Dataset):

    def __init__(self, path, match_path, typed, mode, attributes=0, dst_size=(320, 128), tran=True, fold=0,
                 attr_res=False):

        self.path = path
        # Original Images
        self.path_to_images = os.path.join(self.path, 'RAP_dataset')
        # Images with occlusion
        self.path_to_occlusion = os.path.join(self.path, 'RAP_occluded')

        # Json info
        self.path_to_broken_train = os.path.join(self.path, 'broken_train.json')
        self.path_to_broken_test = os.path.join(self.path, 'broken_test.json')
        self.match_path = match_path

        # mode train or test dataset
        self.train_mode = mode
        self.dst_size = dst_size
        self.attributes = attributes
        self.typed = typed
        self.att_res = attr_res

        self.tran = tran
        if self.tran:
            self.t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.dst_size),
                transforms.ToTensor()
            ])

        mat = scipy.io.loadmat(os.path.join(self.path, 'RAP_annotation/RAP_annotation.mat'), squeeze_me=True)
        self.imagesname = mat['RAP_annotation']['imagesname'].tolist()
        self.positions = mat['RAP_annotation']['position'].tolist()
        self.attributes_eng = mat['RAP_annotation']['attribute_eng'].tolist()
        labels = mat['RAP_annotation']['label'].tolist()
        partions = mat['RAP_annotation']['partion'].tolist()
        partions_train = []
        partions_test = []
        for p in partions:
            partions_train.append(p.tolist()[0])
            partions_test.append(p.tolist()[1])
        # Loading only the first partion, there are 5 in total!!

        self.imagesname_train = self.imagesname[partions_train[fold] - 1]
        self.labels_train = labels[partions_train[fold] - 1]
        self.imagesname_test = self.imagesname[partions_test[fold] - 1]
        self.labels_test = labels[partions_test[fold] - 1]
        self.positions_train = self.positions[partions_train[fold] - 1]
        self.positions_test = self.positions[partions_test[fold] - 1]

        # cleaning
        self.clear_data()

        self.train_dim = len(self.imagesname_train)
        self.test_dim = len(self.imagesname_test)

    def __len__(self):
        if self.train_mode:
            return self.train_dim
        else:
            return self.test_dim

    def __getitem__(self, idx):

        if self.typed == Toperation.classification:
            return self.classification(idx)
        elif self.typed == Toperation.occlusion:
            return self.occlusion(idx)
        elif self.typed == Toperation.metrics:
            return self.metrics(idx)
        elif self.typed == Toperation.oldmetrics:
            return self.old_metrics(idx)
        else:
            print('Invalid data type')
            raise Exception

    def classification(self, idx):
        if self.train_mode:
            images_to_read = self.imagesname_train
        else:
            images_to_read = self.imagesname_test

        img = cv2.imread(os.path.join(self.path_to_images, images_to_read[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        attrib = torch.zeros([])
        if self.attributes != 0:
            attrib = self.get_labels(idx, numb_attr=self.attributes)

        if self.tran:
            img = self.t(img)

        if self.tran:
            img = torch.add(img, -0.5)
            img = torch.mul(img, 2)

        return img, attrib, images_to_read[idx]

    def occlusion(self, idx):
        if self.train_mode:
            images_to_read = self.imagesname_train
        else:
            images_to_read = self.imagesname_test

        gt = cv2.imread(os.path.join(self.path_to_images, images_to_read[idx]))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        occ = cv2.imread(os.path.join(self.path_to_occlusion, images_to_read[idx]))
        occ = cv2.cvtColor(occ, cv2.COLOR_BGR2RGB)

        if self.attributes != 0:
            labels_tensor = self.get_labels(idx, self.attributes, None)
            concat_size = [self.dst_size[0] // 16, self.dst_size[1] // 16]
            if self.att_res:
                labels_tensor_rsd = self.get_labels(idx, self.attributes, concat_size)
            else:
                labels_tensor_rsd = self.get_labels(idx, self.attributes, None)
        else:
            labels_tensor = None
            labels_tensor_rsd = None

        if self.tran:
            gt = self.t(gt)
            occ = self.t(occ)
            occ = torch.add(occ, -0.5)
            occ = torch.mul(occ, 2)
            gt = torch.add(gt, -0.5)
            gt = torch.mul(gt, 2)

        labels_tensor_rsd = torch.add(labels_tensor_rsd, -0.5)
        labels_tensor_rsd = torch.mul(labels_tensor_rsd, 2)
        return occ, gt, labels_tensor, labels_tensor_rsd, images_to_read[idx]

    def metrics(self, idx):
        if self.train_mode:
            images_to_read = self.imagesname_train
        else:
            images_to_read = self.imagesname_test

        if os.path.isfile(os.path.join(self.match_path, images_to_read[idx])):
            cmp_p = os.path.join(self.match_path, images_to_read[idx])
        else:
            cmp_p = os.path.join(self.match_path, images_to_read[idx].split('.')[0]+'.jpg')

        deocc = cv2.imread(cmp_p, cv2.IMREAD_COLOR)
        deocc = cv2.cvtColor(deocc, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(os.path.join(self.path_to_images, images_to_read[idx]))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        occ = cv2.imread(os.path.join(self.path_to_occlusion, images_to_read[idx]))
        occ = cv2.cvtColor(occ, cv2.COLOR_BGR2RGB)

        if self.attributes != 0:
            labels_tensor = self.get_labels(idx, self.attributes, None)
            concat_size = [self.dst_size[0] // 16, self.dst_size[1] // 16]
        else:
            labels_tensor = None

        if self.tran:
            deocc = self.t(deocc)
            gt = self.t(gt)
            occ = self.t(occ)

        if self.tran:
            deocc = torch.add(deocc, -0.5)
            deocc = torch.mul(deocc, 2)
            occ = torch.add(occ, -0.5)
            occ = torch.mul(occ, 2)
            gt = torch.add(gt, -0.5)
            gt = torch.mul(gt, 2)
        else:
            deocc = cv2.resize(deocc, self.dst_size)
            occ = cv2.resize(occ, self.dst_size)
            gt = cv2.resize(gt, self.dst_size)

        return gt, occ, deocc, labels_tensor, images_to_read[idx]

    def old_metrics(self, idx):
        if self.train_mode:
            images_to_read = self.imagesname_train
        else:
            images_to_read = self.imagesname_test

        gt = cv2.imread(os.path.join(self.path_to_images, images_to_read[idx]))
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        cv_gt = gt.copy()
        cv_gt = cv2.resize(cv_gt, self.dst_size)

        occ = cv2.imread(os.path.join(self.path_to_occlusion, images_to_read[idx]))
        occ = cv2.cvtColor(occ, cv2.COLOR_BGR2RGB)
        cv_oc = occ.copy()
        cv_oc = cv2.resize(cv_oc, self.dst_size)

        if self.attributes != 0:
            concat_size = [self.dst_size[0] // 16, self.dst_size[1] // 16]
            if self.att_res:
                labels_tensor_rsd = self.get_labels(idx, self.attributes, concat_size)
            else:
                labels_tensor_rsd = self.get_labels(idx, self.attributes, None)
        else:
            labels_tensor_rsd = None

        gt = self.t(gt)
        occ = self.t(occ)

        occ = torch.add(occ, -0.5)
        occ = torch.mul(occ, 2)
        gt = torch.add(gt, -0.5)
        gt = torch.mul(gt, 2)

        labels_tensor_rsd = torch.add(labels_tensor_rsd, -0.5)
        labels_tensor_rsd = torch.mul(labels_tensor_rsd, 2)
        cv_gt = torch.from_numpy(cv_gt)
        cv_oc = torch.from_numpy(cv_oc)

        return gt, occ, cv_gt, cv_oc, labels_tensor_rsd

    def get_labels(self, idx, numb_attr, concat_size=None):

        if self.train_mode:
            labels_to_read = self.labels_train
        else:
            labels_to_read = self.labels_test

        if concat_size is not None:
            labels_tensor = torch.zeros(numb_attr, concat_size[0], concat_size[1])
            for i in np.arange(0, numb_attr):
                labels_tensor[i, ...] = int(labels_to_read[idx][i])
        else:
            labels_tensor = torch.zeros(numb_attr)
            for i in np.arange(0, numb_attr):
                labels_tensor[i] = int(labels_to_read[idx][i])

        return labels_tensor

    def clear_data(self):

        jdata = read_json(self.path_to_broken_train)

        broken_train = jdata['broken']
        broken_train_t = jdata['broken_t']
        broken_train_o = jdata['broken_o']

        broken_train = broken_train + broken_train_t
        broken_train = broken_train + broken_train_o

        self.imagesname_train = np.delete(self.imagesname_train, broken_train, 0)
        self.labels_train = np.delete(self.labels_train, broken_train, 0)
        self.positions_train = np.delete(self.positions_train, broken_train, 0)

        jdata = read_json(self.path_to_broken_test)

        broken_test = jdata['broken']
        # broken_test_t = jdata['broken_t']
        broken_test_o = jdata['broken_o']

        # broken_test = broken_test + broken_test_t
        broken_test = broken_test + broken_test_o

        self.imagesname_test = np.delete(self.imagesname_test, broken_test, 0)
        self.labels_test = np.delete(self.labels_test, broken_test, 0)
        self.positions_test = np.delete(self.positions_test, broken_test, 0)

