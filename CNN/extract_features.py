import argparse
import functools

from macls.trainer_mlabel import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/fish3.yml','configs')
add_arg('save_dir',         str,    'data/aa_4s_de001_mel','save dir')
args = parser.parse_args()
print_arguments(args=args)

# 获取训练器
trainer = MAClsTrainer(configs=args.configs)

# 提取特征保存文件
trainer.extract_features(save_dir=args.save_dir)
