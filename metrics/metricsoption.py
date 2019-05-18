from other_utils.base import *


class MetricsOptions(BaseOptions):
    def initializer(self):
        BaseOptions.initializer(self)

        # fai comp path e exp path nella stessa direcotry
        self.parser.add_argument('--exp_name', type=str, default='mtr_jta_attr_base')
        self.parser.add_argument('--img_size', type=tuple, default=(320, 128))

        # Gpu info
        self.parser.add_argument('--gpu', type=bool, default=True)
        self.parser.add_argument('--gpu_id', type=int, default=0)

        self.parser.add_argument('--dataset', type=str, default='JTA')
        self.parser.add_argument('--epochs', type=int, default=50)
        self.parser.add_argument('--batch_size', type=int, default=10)
        self.parser.add_argument('--f', type=int, default=64)
        self.parser.add_argument('--occ_mode', type=str, default='o_mask')
        self.parser.add_argument('--pose_mode', type=bool, default=True)
        self.parser.add_argument('--attributes', type=int, default=24)
        self.parser.add_argument('--attributes_rsd', type=bool, default=False)
        self.parser.add_argument('--workers', type=int, default=6)
