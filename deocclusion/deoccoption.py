from other_utils.base import *


class DeocOptions(BaseOptions):
    def initializer(self):
        BaseOptions.initializer(self)

        # Dataset and log info
        self.parser.add_argument('--exp_name', type=str, default='jta_base_attr')
        self.parser.add_argument('--img_size', type=tuple, default=(320, 128))

        # Gpu info
        self.parser.add_argument('--gpu', type=bool, default=True)
        self.parser.add_argument('--gpu_id', type=int, default=0)

        # train info
        self.parser.add_argument('--dataset', type=str, default='JTA')
        self.parser.add_argument('--c_in', type=int, default=3)
        self.parser.add_argument('--c_out', type=int, default=3)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--batch_size', type=int, default=20)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--w_init', type=bool, default=True)
        self.parser.add_argument('--attributes', type=int, default=24)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('--gen_type', type=str, default='unet')
        self.parser.add_argument('--network_mode', type=str, default='upsampling')
        self.parser.add_argument('--f', type=int, default=64)
        self.parser.add_argument('--w1', type=float, default=10.)
        self.parser.add_argument('--w2', type=float, default=0.01)
        self.parser.add_argument('--second_loss', type=str, default='cont_loss')
        self.parser.add_argument('--third_loss', type=str, default='attrloss')
        self.parser.add_argument('--fake_real_trick', type=bool, default=True)
        self.parser.add_argument('--tc', type=int, default=-1)
