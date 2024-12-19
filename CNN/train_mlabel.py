import argparse
import functools

from macls.trainer_mlabel import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/fish3.yml',        'Configs')
add_arg("local_rank",       int,    0,                          'Parameters for multi GPU')
add_arg("use_gpu",          bool,   True,                       'Use GPU or not')
add_arg('save_model_path',  str,    'models/',                  'Save dir path')
add_arg('resume_model',     str,    None,                       'Resume model，if None it will not using pretrain model')
add_arg('pretrained_model', str,    None,                       'Pretrain model path， if None it will not use pretrained models')
args = parser.parse_args()
print_arguments(args=args)


trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

trainer.train(save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model)
