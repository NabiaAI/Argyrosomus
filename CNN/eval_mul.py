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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs',          str,   'CNN/configs/fish3.yml',         "configure")
    add_arg("use_gpu",          bool,  True,                        "use gpu")
    add_arg('save_matrix_path', str,   'CNN/output/images/',            "save_matrix_path")
    add_arg('resume_model',     str,   'CNN/models_trained/Res18lstm_MelSpectrogram/best_model',  "model path")
    add_arg('output_txt',     str,   'CNN/outputs/test54_e.txt',  "txt path")
    args = parser.parse_args()
    print_arguments(args=args)

    trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)


    start = time.time()


    trainer.evaluate(resume_model=args.resume_model, save_matrix_path=args.save_matrix_path)
    # trainer.evaluate_grouped_confusion_matrix(resume_model=args.resume_model,
    #                                   save_matrix_path=args.save_matrix_path)

    end = time.time()




