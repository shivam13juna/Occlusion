import os
import socket

from .folder_utils import check_generate_dir

PYTHONPATH = '..:.'
if os.environ.get('PYTHONPATH', default=None) is None:
    os.environ['PYTHONPATH'] = PYTHONPATH
else:
    os.environ['PYTHONPATH'] += (':' + PYTHONPATH)


class PathMng(object):
    HOSTNAME = socket.gethostname()
    h_name = 'CURRENT HOSTNAME: {} '.format(HOSTNAME)
    print(h_name)

    if HOSTNAME == 'socket name':
        DS_PATH = '/homes/<name>/datasets'
        PJ_PATH = '/homes/<name>/PycharmProjects/deocclusion'
        LG_PATH = '/homes/<name>/results'
    else:
        print("no valid socket name")

    def __init__(self, experiment_name, boost=False):

        self.experiment_results = os.path.join(PathMng.LG_PATH, '{}'.format(experiment_name))
        self.rap_dataset = os.path.join(self.DS_PATH, 'RAP')
        self.naic_dataset = os.path.join(self.DS_PATH, 'NAIC')
        self.project_path = self.PJ_PATH

    def folders_initialization(self):
        check_generate_dir(self.experiment_results)
