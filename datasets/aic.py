import os
import json
from torchvision import transforms
from torch.utils.data import Dataset
from other_utils.tensor_utils import *
from other_utils.class_utils import Toperation
from datasets.aic_info import SINGLE_ATTR


RES = [1920, 1080]

SEED = 1821
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)


class AICDataset(Dataset):

    def __init__(self, root_dir, comp_path, type_data, train_data=True, dst_size=(320, 128),
                 tran=None, random_flip=0., attr_rsd=False):

        self.train_data = train_data

        self.random_flip = random_flip
        self.attributes_eng = SINGLE_ATTR
        self.attributes_rsd = attr_rsd

        self.root_dir = root_dir
        self.dst_size = dst_size

        # json and path info
        self.json_folder = os.path.join(self.root_dir)
        self.image_path = os.path.join(self.root_dir, 'crops')

        self.comp_path = comp_path
        self.tran = tran
        self.all_file = {}
        self.type_data = type_data

        with open(os.path.join(self.json_folder, 'annotations.json')) as fn:
            self.dictionary_images = json.load(fn)

        with open('./config/aic/aic_split.json') as fn:
            tmp = json.load(fn)

        self.train_idx = np.array(tmp[0:100000])
        self.test_idx = np.array(tmp[100000:])

        if self.random_flip > 0:
            self.rnd_flip = transforms.RandomHorizontalFlip(self.random_flip)

        if self.tran:
            self.t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.dst_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        if self.train_data:
            return len(self.train_idx)
        else:
            return len(self.test_idx)

    def __getitem__(self, idx):

        if self.train_data:
            idx = self.train_idx[idx]
        else:
            idx = self.test_idx[idx]

        pinfo = self.dictionary_images[idx]

        oc_image_path = os.path.join(self.image_path, '{}'.format(pinfo['id']) + '_occ.jpg')
        gt_image_path = os.path.join(self.image_path, '{}'.format(pinfo['id']) + '.jpg')
        if self.comp_path is not None:
            de_image_path = os.path.join(self.comp_path, '{}.jpeg'.format(pinfo['info']))
        else:
            de_image_path = ''

        gt = self.read_image(gt_image_path)
        occ = self.read_image(oc_image_path)

        attributes = pinfo['attributes']
        name_file = '{}'.format(pinfo['id']) + '.jpg'

        if self.type_data == Toperation.occlusion:
            return self.deocclusion(gt, occ, attributes, name_file)
        elif self.type_data == Toperation.classification:
            return self.classification(gt, attributes, name_file)
        elif self.type_data == Toperation.metrics:
            deocc = self.read_image(de_image_path)
            return self.metrics(gt, occ, deocc, attributes, name_file)
        elif self.type_data == Toperation.demo:
            return self.demo(gt, occ, attributes, pinfo['pose'], name_file)
        else:
            print('Invalid data type')
            raise Exception

    def classification(self, gt, attributes, name):

        csize = [self.dst_size[0] // 16, self.dst_size[1] // 16]
        labels_tensor, _ = self.get_labels_tensors(attributes, concat_size=csize)

        if self.random_flip > 0 and np.random.rand() > 0.5:
            gt = cv2.flip(gt, 1)

        if self.tran:
            gt = self.t(gt)
            gt = torch.add(gt, -0.5)
            gt = torch.mul(gt, 2)

        return gt, labels_tensor, name

    def metrics(self, gt, occ, deocc, attributes, name):

        csize = [self.dst_size[0] // 16, self.dst_size[1] // 16]
        labels_tensor, _ = self.get_labels_tensors(attributes, concat_size=csize)

        if self.tran:
            gt = self.t(gt)
            occ = self.t(occ)
            deocc = self.t(deocc)
            occ = torch.add(occ, -0.5)
            occ = torch.mul(occ, 2)
            gt = torch.add(gt, -0.5)
            gt = torch.mul(gt, 2)
            deocc = torch.add(deocc, -0.5)
            deocc = torch.mul(deocc, 2)
        else:
            deocc = cv2.resize(deocc, (self.dst_size[1], self.dst_size[0]))
            occ = cv2.resize(occ, (self.dst_size[1], self.dst_size[0]))
            gt = cv2.resize(gt, (self.dst_size[1], self.dst_size[0]))

        return gt, occ, deocc, labels_tensor, name

    def deocclusion(self, gt, occ, attributes, name):

        csize = [self.dst_size[0] // 16, self.dst_size[1] // 16]
        labels_tensor, labels_tensor_rsd = self.get_labels_tensors(attributes, csize)

        if self.random_flip > 0 and np.random.rand() > 0.5:
            occ = cv2.flip(occ, 1)
            gt = cv2.flip(gt, 1)

        if self.tran:
            gt = self.t(gt)
            gt = torch.add(gt, -0.5)
            gt = torch.mul(gt, 2)
            occ = self.t(occ)
            occ = torch.add(occ, -0.5)
            occ = torch.mul(occ, 2)
            labels_tensor_rsd = torch.add(labels_tensor_rsd, -0.5)
            labels_tensor_rsd = torch.mul(labels_tensor_rsd, 2)

        return occ, gt, labels_tensor, labels_tensor_rsd, name

    def demo(self, gt, occ, attributes,  pose, name):

        for elem in pose:
            if elem[3] == 0 and elem[4] == 0:
                cv2.circle(gt, (elem[1], elem[2]), 3, (0, 0, 255), thickness=-1)
                cv2.circle(occ, (elem[1], elem[2]), 3, (0, 0, 255), thickness=-1)
            else:
                cv2.circle(gt, (elem[1], elem[2]), 3, (0, 255, 0), thickness=-1)
                cv2.circle(occ, (elem[1], elem[2]), 3, (0, 255, 0), thickness=-1)

        csize = [self.dst_size[0] // 16, self.dst_size[1] // 16]
        labels_tensor, labels_tensor_rsd = self.get_labels_tensors(attributes, csize)

        if self.random_flip > 0 and np.random.rand() > 0.5:
            occ = cv2.flip(occ, 1)
            gt = cv2.flip(gt, 1)

        if self.tran:
            gt = self.t(gt)
            gt = torch.add(gt, -0.5)
            gt = torch.mul(gt, 2)
            occ = self.t(occ)
            occ = torch.add(occ, -0.5)
            occ = torch.mul(occ, 2)

        return gt, occ, labels_tensor, name

    def read_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_labels_tensors(self, attr_labels, concat_size):

        labels_tensor = torch.from_numpy(np.array(attr_labels).astype(np.float32))

        if self.attributes_rsd:
            labels_tensor_resized = torch.zeros(len(attr_labels), concat_size[0], concat_size[1])
            for i in np.arange(0, len(attr_labels)):
                labels_tensor_resized[i, ...] = int(attr_labels[i])
        else:
            labels_tensor_resized = torch.from_numpy(np.array(attr_labels).astype(np.float32))

        return labels_tensor, labels_tensor_resized






