import argparse


class BaseOptions(object):
    def __init__(self, description='describe the aim...'):
        self.parser = argparse.ArgumentParser(description=description)
        self.initialized = False

    def initializer(self):
        self.initialized = True

    def make_parsing(self):

        if self.initialized:
            opt = self.parser.parse_args()
            print(opt)
            return opt
        else:
            print('Problem ==> invalid parser or not initialized parser')
