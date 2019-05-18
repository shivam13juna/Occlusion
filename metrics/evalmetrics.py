import numpy as np
from skimage.measure import compare_psnr, compare_ssim


class EvalMetrics(object):
    def __init__(self):
        self.count = 0
        self.score = 0.
        self.values = []

    def append(self, gt_label, predict_label):
        pass

    @property
    def last(self):
        return self.values[len(self.values)-1]

    @property
    def get(self):
        pass

# CLASSIC METRICS


class SingleClassMetrics(object):
    def __init__(self):
        self._TP_ = 0.
        self._FP_ = 0.
        self._TN_ = 0.
        self._FN_ = 0.
        self.count = 0.

    def append(self, gt_label, predict_label):
        if gt_label == 0:
            if predict_label == 0:
                self._TN_ += 1
            else:
                self._FP_ += 1
        else:
            if predict_label == 0:
                self._FN_ += 1
            else:
                self._TP_ += 1

        self.count += 1

    @property
    def mean_accuracy(self):
        p1 = self._TP_ + self._FN_
        p2 = self._FP_ + self._TN_

        if p1 == 0:
            p1 = 0
        else:
            p1 = self._TP_/(self._TP_ + self._FN_)

        if p2 == 0:
            p2 = 0
        else:
            p2 = self._TN_/(self._FP_ + self._TN_)

        return (p1 + p2)/2

    @property
    def accuracy(self):
        if (self._TP_ + self._FP_ + self._TN_ + self._FN_) == 0:
            return 1
        else:
            return (self._TP_ + self._TN_) / (self._TP_ + self._FP_ + self._TN_ + self._FN_)

    @property
    def precision(self):
        if (self._TP_ + self._FP_) == 0:
            return 1
        else:
            return self._TP_ / (self._TP_ + self._FP_)

    @property
    def recall(self):
        if (self._TP_ + self._FN_) == 0:
            return 1
        else:
            return self._TP_ /(self._TP_ + self._FN_)

    @property
    def f1(self):
        if (self.precision+self.recall) == 0:
            return 1
        else:
            return 2*(self.precision*self.recall)/(self.precision+self.recall)


class MeanAccuracy(EvalMetrics):
    def __init__(self):
        super(MeanAccuracy, self).__init__()

    def append(self, gt_label, predict_label):

        _TP_ = np.sum(predict_label[gt_label == 1])
        _P_ = np.sum(gt_label[gt_label == 1])

        _TN_ = np.sum(predict_label[gt_label == 0] == 0)
        _N_ = np.sum(gt_label[gt_label == 0] == 0)
        # aggiusta

        tmp = (_TP_/_P_ + _TN_/_N_)
        self.score += tmp
        self.values.append(tmp)
        self.count += 1

    @property
    def get(self) -> float:
        return self.score / (2*self.count)


class Accuracy(EvalMetrics):
    def __init__(self):
        super(Accuracy, self).__init__()

    def append(self, gt_label, predict_label):

        _TP_ = np.sum(predict_label[gt_label == 1])
        _FP_ = np.sum(predict_label[gt_label == 0])

        _TN_ = np.sum(predict_label[gt_label == 0] == 0)
        _FN_ = np.sum(predict_label[gt_label == 1] == 0)

        tmp = ((_TP_ + _TN_)/(_TP_ + _FP_ + _TN_ + _FN_))
        self.score += tmp
        self.values.append(tmp)
        self.count += 1

    @property
    def get(self) -> float:
        return self.score / self.count


class SingleClassAccuracy(object):
    def __init__(self):
        super(SingleClassAccuracy, self).__init__()

        self.tp_ = 0.
        self.fp_ = 0.

        self.tn_ = 0.
        self.fn_ = 0.

        self.giusto = 0.

        self.count = 0.

    def append(self, gt_label, predict_label):

        if gt_label == 0:
            if predict_label == 0:
                self.tn_ += 1
            else:
                self.fn_ += 1
        else:
            if predict_label == 0:
                self.fp_ += 1
            else:
                self.tp_ += 1

        if gt_label == predict_label:
            self.giusto += 1

        self.count += 1

    @property
    def get(self) -> float:

        return self.giusto / self.count


class Precision(EvalMetrics):
    def __init__(self):
        super(Precision, self).__init__()

    def append(self, gt_label, predict_label):

        _TP_ = np.sum(predict_label[gt_label == 1])
        _FP_ = np.sum(predict_label[gt_label == 0])

        tmp = (_TP_/(_TP_ + _FP_))
        self.score += tmp
        self.values.append(tmp)
        self.count += 1

    @property
    def get(self) -> float:
        return self.score / self.count


class Recall(EvalMetrics):
    def __init__(self):
        super(Recall, self).__init__()

    def append(self, gt_label, predict_label):

        _TP_ = np.sum(predict_label[gt_label == 1])
        _FN_ = np.sum(predict_label[gt_label == 1] == 0)

        tmp = (_TP_ /(_TP_ + _FN_))
        self.score += tmp
        self.values.append(tmp)
        self.count += 1

    @property
    def get(self) -> float:
        return self.score / self.count

# NOISE METRICS


class PSNR(EvalMetrics):
    def __init__(self):
        super(PSNR, self).__init__()

    def append(self, gt, img):

        tmp = compare_psnr(gt, img)
        if tmp == float('NaN') or tmp == float('Inf'):
            print('')
        else:
            self.score += tmp
            self.values.append(tmp)
            self.count += 1

    @property
    def get(self) -> float:
        return self.score / self.count


class SSIM(EvalMetrics):
    def __init__(self):
        super(SSIM, self).__init__()

    def append(self, gt, img):
        tmp = compare_ssim(gt, img, multichannel=True)
        self.score += tmp
        self.values.append(tmp)
        self.count += 1

    @property
    def get(self) -> float:
        return self.score / self.count


# RAP METRICS

class RAPMetrics(object):
    def __init__(self):
        self.sum = 0.
        self.count = 0.

    def append(self, gt_label, predict_label):
        pass

    @property
    def get(self):
        pass


class RAPAccuracy(RAPMetrics):
    def __init__(self):
        super(RAPAccuracy, self).__init__()

    def append(self, gt_label, predict_label):
        union = 0.
        intersection = 0.
        for i in range(len(gt_label)):
            if gt_label[i] == 1 and gt_label[i] == predict_label[i]:
                intersection += 1

            if gt_label[i] == 1 and gt_label[i] == predict_label[i]:
                union += 1
            elif gt_label[i] == 1 and predict_label[i] == 0:
                union += 1
            elif gt_label[i] == 0 and predict_label[i] == 1:
                union += 1

        if union != 0:
            self.sum += intersection/union
        else:
            self.sum += 1
        self.count += 1

    @property
    def get(self) -> float:
        return self.sum/self.count


class RAPPrecision(RAPMetrics):
    def __init__(self):
        super(RAPPrecision, self).__init__()

    def append(self, gt_label, predict_label):
        intersection = 0.
        for i in range(len(gt_label)):
            if gt_label[i] == 1 and gt_label[i] == predict_label[i]:
                intersection += 1

        denominator = np.sum(predict_label)

        if denominator != 0:
            self.sum += intersection / denominator
        else:
            self.sum += 1
        self.count += 1

    @property
    def get(self) -> float:
        return self.sum / self.count


class RAPRecall(RAPMetrics):
    def __init__(self):
        super(RAPRecall, self).__init__()

    def append(self, gt_label, predict_label):
        intersection = 0.
        for i in range(len(gt_label)):
            if gt_label[i] == 1 and gt_label[i] == predict_label[i]:
                intersection += 1

        denominator = np.sum(gt_label)

        if denominator != 0:
            self.sum += intersection / denominator
        else:
            self.sum += 1
        self.count += 1

    @property
    def get(self) -> float:
        return self.sum / self.count


