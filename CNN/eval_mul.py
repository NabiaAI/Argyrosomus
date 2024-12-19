import argparse
import functools
import time

from macls.trainer_mlabel import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(project_root)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/fish3.yml',         "Configs")
add_arg("use_gpu",          bool,  True,                        "USe GPU or not")
add_arg('save_matrix_path', str,   'output/images/',            "Save dir")
add_arg('resume_model',     str,   'models//Res18_MelSpectrogram/best_model/',  "model path")
args = parser.parse_args()
print_arguments(args=args)

trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)
start = time.time()
loss, accuracy = trainer.evaluate(resume_model=args.resume_model,
                                  save_matrix_path=args.save_matrix_path)
end = time.time()
print('Time：{}s，loss：{:.5f}，accuracy：{:.5f}'.format(int(end - start), loss, accuracy))
