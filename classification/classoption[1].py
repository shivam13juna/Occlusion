from other_utils.base import *


class ClassOptions(BaseOptions):
    def initializer(self):
        BaseOptions.initializer(self)

        # Dataset and log info
        self.parser.add_argument('--exp_name', type=str, default='mar')
        self.parser.add_argument('--img_size', type=tuple, default=(320, 128))

        # Gpu info
        self.parser.add_argument('--gpu', type=bool, default=True)
        self.parser.add_argument('--gpu_id', type=int, default=0)

        # train info
        self.parser.add_argument('--dataset', type=str, default='JTA')
        self.parser.add_argument('--epochs', type=int, default=10)
        self.parser.add_argument('--fold', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=10)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--attributes', type=int, default=24)
        self.parser.add_argument('--workers', type=int, default=8)
