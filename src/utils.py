import os
import argparse

def create_folder(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path) 

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

class StoreStringList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))

class StoreFloatList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        float_list = [float(x) for x in values.split(',')]
        setattr(namespace, self.dest, float_list)