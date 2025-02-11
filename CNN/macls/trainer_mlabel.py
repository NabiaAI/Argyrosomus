import json
import os
import platform
import shutil
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchinfo import summary
from tqdm import tqdm
#from visualdl import LogWriter
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from macls import SUPPORT_MODEL, __version__
from macls.data_utils.collate_fn_mulabel import collate_fn
from macls.data_utils.featurizer import AudioFeaturizer
from macls.data_utils.reader_mulabel import MAClsDataset
from macls.data_utils.spec_aug import SpecAug
from macls.metric.metrics import accuracy
from macls.models.campplus import CAMPPlus
from macls.models.ecapa_tdnn import EcapaTdnn
from macls.models.eres2net import ERes2NetV2, ERes2Net
from macls.models.panns import PANNS_CNN6, PANNS_CNN10, PANNS_CNN14
from macls.models.res2net import Res2Net
from macls.models.resnet_se import ResNetSE
from macls.models.tdnn import TDNN
from macls.utils.logger import setup_logger
from macls.utils.scheduler import WarmupCosineSchedulerLR
from macls.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


from macls.models.mulresnet import ResNet
from macls.models.resnet18 import ResNet18
from macls.models.resnet18lstm import ResNet18LSTM
from macls.models.resnet18lstmask import ResNet18LSTMASK
from macls.models.lstm_resnet import LSTMResNetWithAttention
from macls.models.mulcnns import SimpleCNN
from macls.models.res2netmul import Res2Netmul
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score,  f1_score
from collections import defaultdict
from itertools import product
from sklearn.metrics import accuracy_score
import sys
sys.path.append('.')
import utils_eval as eval
import quantification as quant

logger = setup_logger(__name__)

#我修改
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import torch

from sklearn.metrics import precision_recall_curve

import json

threshold_history = []
def shift_spectrogram(features, sample_rate, max_shift_hz):


    num_frequency_bins = features.size(1)
    frequency_per_bin = (sample_rate / 2) / num_frequency_bins
    max_shift_bins = int(max_shift_hz / frequency_per_bin)
    min_shift_bins = int(10 / frequency_per_bin)
    

    shifts = torch.randint(-min_shift_bins, max_shift_bins + 1, (features.size(0),))
    

    for i, shift in enumerate(shifts):
        features[i] = torch.roll(features[i], shifts=shift.item(), dims=0)
    
    return features
def save_threshold_history(threshold_history, save_model_path, model_name, feature_method):
    save_path = os.path.join(save_model_path, f"{model_name}_{feature_method}", 'best_model', "threshold_history.json")
    with open(save_path, 'w') as f:
        json.dump(threshold_history, f)  
        


        
        
class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, background_samples, bg_ratio=0.2):
        self.original_dataset = original_dataset
        self.background_samples = background_samples
        self.bg_ratio = bg_ratio
        self.bg_count = int(len(original_dataset) * bg_ratio) 

    def __len__(self):
        return len(self.original_dataset) + self.bg_count

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            bg_idx = idx - len(self.original_dataset)

            bg_idx = bg_idx % len(self.background_samples)
            return self.background_samples[bg_idx]



from collections import deque

class ThresholdStabilizer:
    def __init__(self, window_size=5):

        self.window_size = window_size
        self.threshold_history = deque(maxlen=window_size)

    def update_thresholds(self, new_thresholds):

        self.threshold_history.append(new_thresholds)
        smoothed_thresholds = np.mean(self.threshold_history, axis=0)
        return smoothed_thresholds


def save_thresholds_to_json(thresholds, save_model_path, model_name, feature_method, filename="optimal_thresholds.json"):

    best_model_dir = os.path.join(save_model_path, f'{model_name}_{feature_method}', 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)  
    filepath = os.path.join(best_model_dir, filename)
    thresholds = [float(th) for th in thresholds]  
    with open(filepath, 'w') as f:
        json.dump(thresholds, f)
    print(f"Optimal thresholds saved to {filepath}")
    

    
def load_thresholds_from_json(save_model_path, model_name, feature_method,filename):


    json_path = os.path.join(
        save_model_path,
        f"{model_name}_{feature_method}",
        "best_model",
        filename
    )
    print(json_path)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Threshold file not found: {json_path}")
    with open(json_path, "r") as f:
        thresholds = json.load(f)
    return thresholds


def label_smoothing(targets, smoothing=0.1):

    with torch.no_grad():
        targets = targets * (1 - smoothing) + smoothing / targets.size(1)
    return targets

from scipy.stats import linregress

def compute_final_thresholds(history):

    history = np.array(history)
    final_thresholds = []


    for class_idx in range(history.shape[1]):

        class_thresholds = history[:, class_idx]


        mean = np.mean(class_thresholds)
        std = np.std(class_thresholds)
        filtered_thresholds = class_thresholds[np.abs(class_thresholds - mean) < 3 * std]


        if len(filtered_thresholds) < 5:
            filtered_thresholds = class_thresholds


        x = np.arange(len(filtered_thresholds))
        slope, intercept, _, _, _ = linregress(x, filtered_thresholds)
        final_threshold = intercept + slope * len(filtered_thresholds)  
        final_thresholds.append(final_threshold)

    return final_thresholds


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None, smoothing=0.1):

        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights  
        self.smoothing = smoothing 

    def forward(self, inputs, targets):


        if self.smoothing > 0:
            targets = label_smoothing(targets, smoothing=self.smoothing)


        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)


        pt = torch.exp(-BCE_loss)


        dynamic_gamma = self.gamma * (1 - pt)


        F_loss = self.alpha * (1 - pt) ** dynamic_gamma * BCE_loss


        if self.class_weights is not None:
            weights = self.class_weights.unsqueeze(0) 
            F_loss = F_loss * weights


        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:  
            return F_loss


logger = setup_logger(__name__)
        
logger = setup_logger(__name__)
        
def find_optimal_thresholds(y_true, y_probs):

    optimal_thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)  
        if np.isnan(f1_scores).all():
            optimal_threshold = [0.61,0.56,0.56]
            
        else:
            optimal_threshold = thresholds[np.argmax(f1_scores)]
        optimal_thresholds.append(optimal_threshold)
    return optimal_thresholds




class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss



class MultiLabelFocalLossWithFreqAttention(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None, smoothing=0.1, freq_band_weights=None):

        super(MultiLabelFocalLossWithFreqAttention, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights  
        self.smoothing = smoothing 
        self.freq_band_weights = freq_band_weights  

    def forward(self, inputs, targets):

        if self.smoothing > 0:
            targets = label_smoothing(targets, smoothing=self.smoothing)

        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss


        if self.class_weights is not None:
            class_weights = self.class_weights.unsqueeze(0) 
            F_loss = F_loss * class_weights


        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class MultiLabelFocalLossWithFreqAttentionAndNoTarget(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None, smoothing=0.1, freq_band_weights=None,background_weight=2.0):

        super(MultiLabelFocalLossWithFreqAttentionAndNoTarget, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights  
        self.smoothing = smoothing  
        self.freq_band_weights = freq_band_weights  
        self.background_weight = background_weight 

 
    def forward(self, inputs, targets,freq_band_activations):

        if self.smoothing > 0:
            targets = label_smoothing(targets, smoothing=self.smoothing)

        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.class_weights is not None:
            class_weights = self.class_weights.unsqueeze(0) 
            F_loss = F_loss * class_weights

 
        if freq_band_activations is not None:
            freq_band_activations = torch.relu(freq_band_activations)
            freq_weights = torch.relu(torch.matmul(freq_band_activations, self.freq_band_weights.T))
            freq_weights = freq_weights / (freq_weights.sum(dim=1, keepdim=True) + 1e-6)  
            F_loss = F_loss * freq_weights

    
        epsilon = self.smoothing + 1e-6  
        is_no_target = (targets.sum(dim=1) < epsilon)
        is_target = ~is_no_target



        F_loss_no_target = F_loss[is_no_target].mean() if is_no_target.any() else torch.tensor(0.0, device=inputs.device)
        F_loss_target = F_loss[is_target].mean() if is_target.any() else torch.tensor(0.0, device=inputs.device)


        F_loss_no_target *= self.background_weight



        total_loss = F_loss_target + F_loss_no_target
    



        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


class MAClsTrainer(object):
    def __init__(self, configs, use_gpu=True):

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else: 
                assert (False), 'Accelerator can not be used'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            #print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'No such model：{self.configs.use_model}'
        self.model = None
        self.audio_featurizer = None
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.amp_scaler = None
        self.save_model_path = 'CNN/models_trained/'
        self.threshold_stabilizer = ThresholdStabilizer(window_size=5)

        6
        # 获取分类标签
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')
            
        # 我修改
        self.optimal_thresholds = [0.6,0.56,0.54]
        # 特征增强
        self.spec_aug = SpecAug(**self.configs.dataset_conf.get('spec_aug_args', {}))
        self.spec_aug.to(self.device)
        self.max_step, self.train_step = None, None
        self.train_loss, self.train_acc = None, None
        self.train_eta_sec = None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.stop_train, self.stop_eval = False, False

    def __setup_dataloader(self, is_train=False):

        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        if is_train:
            self.train_dataset = MAClsDataset(data_list_path=self.configs.dataset_conf.train_list,
                                              audio_featurizer=self.audio_featurizer,
                                              do_vad=self.configs.dataset_conf.do_vad,
                                              max_duration=self.configs.dataset_conf.max_duration,
                                              min_duration=self.configs.dataset_conf.min_duration,
                                              aug_conf=self.configs.dataset_conf.aug_conf,
                                              sample_rate=self.configs.dataset_conf.sample_rate,
                                              use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                              target_dB=self.configs.dataset_conf.target_dB,
                                              mode='train')

            train_sampler = None
            if torch.cuda.device_count() > 1:
                # 设置支持多卡训练
                train_sampler = DistributedSampler(dataset=self.train_dataset)
                
            # 修改   
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           shuffle=(train_sampler is None),
                                           sampler=train_sampler,
                                           **self.configs.dataset_conf.dataLoader)

        self.test_dataset = MAClsDataset(data_list_path=self.configs.dataset_conf.test_list,
                                         audio_featurizer=self.audio_featurizer,
                                         do_vad=self.configs.dataset_conf.do_vad,
                                         max_duration=self.configs.dataset_conf.eval_conf.max_duration,
                                         min_duration=self.configs.dataset_conf.min_duration,
                                         sample_rate=self.configs.dataset_conf.sample_rate,
                                         use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                         target_dB=self.configs.dataset_conf.target_dB,
                                         mode='eval')
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                      num_workers=self.configs.dataset_conf.dataLoader.num_workers)


    def extract_features(self, save_dir='dataset/features'):
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                                method_args=self.configs.preprocess_conf.get('method_args', {}))
        for i, data_list in enumerate([self.configs.dataset_conf.train_list, self.configs.dataset_conf.test_list]):

            test_dataset = MAClsDataset(data_list_path=data_list,
                                        audio_featurizer=self.audio_featurizer,
                                        do_vad=self.configs.dataset_conf.do_vad,
                                        sample_rate=self.configs.dataset_conf.sample_rate,
                                        max_duration=self.configs.dataset_conf.max_duration,
                                        min_duration=self.configs.dataset_conf.min_duration, 
                                        use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                        target_dB=self.configs.dataset_conf.target_dB,
                                        mode='extract_feature')
            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for i in tqdm(range(len(test_dataset))):
                    #add
                    feature, labels = test_dataset[i]

                    if torch.sum(labels) == 0:  
                        save_subdir = "bg"
                    else:
                        save_subdir = '_'.join(map(str, labels.nonzero()[0].tolist()))
                    feature = feature.numpy()
                    label_str = ' '.join(map(str, labels.tolist()))

                    save_path = os.path.join(save_dir, save_subdir, f'{int(time.time() * 1000)}.npy').replace('\\', '/')

                        #add
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, feature)

                    f.write(f'{save_path}\t{label_str}\n')

            logger.info(f'{data_list}列表中的数据已提取特征完成，新列表为：{save_data_list}')



    def __setup_model(self, input_size, is_train=False):

        if self.configs.model_conf.num_class is None:
            self.configs.model_conf.num_class = len(self.class_labels)

        if self.configs.use_model == 'EcapaTdnn':
            self.model = EcapaTdnn(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN6':
            self.model = PANNS_CNN6(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN10':
            self.model = PANNS_CNN10(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN14':
            self.model = PANNS_CNN14(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'Res2Net':
            self.model = Res2Net(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'ResNetSE':
            self.model = ResNetSE(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'TDNN':
            self.model = TDNN(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'ERes2Net':
            self.model = ERes2Net(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'ERes2NetV2':
            self.model = ERes2NetV2(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'CAMPPlus':
            self.model = CAMPPlus(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'ML_Restnet':
            self.model = ResNet(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'Res18':
            self.model = ResNet18(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'Res18lstm':
            self.model = ResNet18LSTM(input_size=input_size, **self.configs.model_conf)       
        elif self.configs.use_model == 'Res18lstmask':
            self.model = ResNet18LSTMASK(input_size=input_size, **self.configs.model_conf)       
        elif self.configs.use_model == 'Lstmaskresnet':
            self.model = LSTMResNetWithAttention(input_size=input_size, **self.configs.model_conf)    
        elif self.configs.use_model == 'SP_CNN':
            self.model = SimpleCNN(input_size=input_size, **self.configs.model_conf)
        elif self.configs.use_model == 'Res2Netmul':
            self.model = Res2Netmul(input_size=input_size, **self.configs.model_conf)
        else:
            raise Exception(f'{self.configs.use_model} 模型不存在！')
        self.model.to(self.device)
 
        feature_dim = self.audio_featurizer.feature_dim



        # 使用Pytorch2.0的编译器
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() == 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")

        weight = torch.tensor(self.configs.train_conf.loss_weight, dtype=torch.float, device=self.device)\
            if self.configs.train_conf.loss_weight is not None else None
            

        class_weights = torch.tensor([5.588300221,20.390417941,20.60752688], device=self.device)


        freq_dim = 32 
        num_classes = 3  


        
        freq_band_weights = torch.full((num_classes, freq_dim), 0.3, device=self.device)
        freq_band_weights[0, 0:20] = 0.8  
        freq_band_weights[1, 4:31] = 1.0  
        freq_band_weights[2, 0:23] = 0.8  
        print(freq_band_weights.shape)
        

        self.loss = MultiLabelFocalLossWithFreqAttentionAndNoTarget(alpha=1,gamma=2,reduction='mean',class_weights=class_weights,smoothing=0.1,freq_band_weights=freq_band_weights)

        
        

        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)

            optimizer = self.configs.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                  lr=self.configs.optimizer_conf.learning_rate,
                                                  weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=self.configs.optimizer_conf.learning_rate,
                                                   weight_decay=self.configs.optimizer_conf.weight_decay)
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 momentum=self.configs.optimizer_conf.get('momentum', 0.9),
                                                 lr=self.configs.optimizer_conf.learning_rate,
                                                 weight_decay=self.configs.optimizer_conf.weight_decay)
            else:
                raise Exception(f'不支持优化方法：{optimizer}')

            scheduler_args = self.configs.optimizer_conf.get('scheduler_args', {}) \
                if self.configs.optimizer_conf.get('scheduler_args', {}) is not None else {}
            if self.configs.optimizer_conf.scheduler == 'CosineAnnealingLR':
                max_step = int(self.configs.train_conf.max_epoch * 1.2) * len(self.train_loader)
                self.scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                                   T_max=max_step,
                                                   **scheduler_args)
            elif self.configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR':
                self.scheduler = WarmupCosineSchedulerLR(optimizer=self.optimizer,
                                                         fix_epoch=self.configs.train_conf.max_epoch,
                                                         step_per_epoch=len(self.train_loader),
                                                         **scheduler_args)
            else:
                raise Exception(f'不支持学习率衰减函数：{self.configs.optimizer_conf.scheduler}')
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() != 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.model.to(self.device)

    def __load_pretrained(self, pretrained_model):

        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pth')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_dict = self.model.module.state_dict()
            else:
                model_dict = self.model.state_dict()
            model_state_dict = torch.load(pretrained_model)

            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(model_state_dict, strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
            logger.info('成功加载预训练模型：{}'.format(pretrained_model))

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_acc = 0
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')
        

        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth'))
                                         and os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):

            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pth')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pth')), "优化方法参数文件不存在！"
            state_dict = torch.load(os.path.join(resume_model, 'model.pth'))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pth')))

            if self.amp_scaler is not None and os.path.exists(os.path.join(resume_model, 'scaler.pth')):
                self.amp_scaler.load_state_dict(torch.load(os.path.join(resume_model, 'scaler.pth')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_acc = json_data['accuracy']
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
            self.optimizer.step()
            [self.scheduler.step() for _ in range(last_epoch * len(self.train_loader))]
        return last_epoch, best_acc


    def __save_checkpoint(self, save_model_path, epoch_id, best_acc=0., best_model=False):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        if best_model:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'best_model')
        else:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(state_dict, os.path.join(model_path, 'model.pth'))

        if self.amp_scaler is not None:
            torch.save(self.amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            data = {"last_epoch": epoch_id, "accuracy": best_acc, "version": __version__}
            f.write(json.dumps(data))
        if not best_model:
            last_model_path = os.path.join(save_model_path,
                                           f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)

            old_model_path = os.path.join(save_model_path,
                                          f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))


    def __train_epoch(self, epoch_id, local_rank, writer, nranks=0):
        train_times, accuracies, loss_sum = [], [], []
        all_labels, all_probs = [], []
        start = time.time()

        for batch_id, (features, labels, input_lens) in enumerate(self.train_loader):
            if self.stop_train: break

            if nranks > 1:
                features = features.to(local_rank)
                labels = labels.to(local_rank)
            else:
                features = features.to(self.device)
                labels = labels.float().to(self.device)




            if self.configs.dataset_conf.use_spec_aug:
                features = self.spec_aug(features)
            
            if self.configs.dataset_conf.use_shift_aug:
                features = shift_spectrogram(features, 
                                            sample_rate=self.configs.dataset_conf.sample_rate, 
                                            max_shift_hz=self.configs.dataset_conf.max_shift_hz)


            with torch.cuda.amp.autocast(enabled=self.configs.train_conf.enable_amp):

                output, freq_band_activations  = self.model(features)
                




                

            loss = self.loss(output, labels, freq_band_activations)



            if self.configs.train_conf.enable_amp:
                scaled = self.amp_scaler.scale(loss)
                scaled.backward()
            else:

                loss.backward()
                
                





            if self.configs.train_conf.enable_amp:
                self.amp_scaler.unscale_(self.optimizer)
                self.amp_scaler.step(self.optimizer)
                self.amp_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            

            

            hidden_class_threshold = 0.2  
            preds = self.apply_optimal_thresholds(output, self.optimal_thresholds, hidden_class_threshold=hidden_class_threshold)
            





            
 
            correct = torch.all(preds == labels, dim=1).float()
            acc = correct.sum().item() / labels.size(0)

            accuracies.append(acc)
            loss_sum.append(loss.data.cpu().numpy())
            train_times.append((time.time() - start) * 1000)
            self.train_step += 1

            all_probs.append(output.detach().cpu().numpy())

            all_labels.append(labels.detach().cpu().numpy())




            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                batch_id = batch_id + 1

                train_speed = self.configs.dataset_conf.dataLoader.batch_size / (sum(train_times) / len(train_times) / 1000)

                self.train_eta_sec = (sum(train_times) / len(train_times)) * (self.max_step - self.train_step) / 1000
                eta_str = str(timedelta(seconds=int(self.train_eta_sec)))
                self.train_loss = sum(loss_sum) / len(loss_sum)
                self.train_acc = sum(accuracies) / len(accuracies)
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {self.train_loss:.5f}, accuracy: {self.train_acc:.5f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', self.train_loss, self.train_log_step)
                writer.add_scalar('Train/Accuracy', self.train_acc, self.train_log_step)
 
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_log_step)
                train_times, accuracies, loss_sum = [], [], []
                self.train_log_step += 1
            start = time.time()
            self.scheduler.step()


        if epoch_id >= 101:  
            all_probs = torch.cat([torch.tensor(prob) for prob in all_probs], dim=0) 
            all_labels = torch.cat([torch.tensor(label) for label in all_labels], dim=0)
            
            new_thresholds = find_optimal_thresholds(all_labels.numpy(), all_probs.numpy())  
            smoothed_thresholds = self.threshold_stabilizer.update_thresholds(new_thresholds)
            
            recall_values = recall_score(
                all_labels.numpy(),
                self.apply_optimal_thresholds(all_probs, smoothed_thresholds).numpy(),
                average=None
            )
            

 

            tolerance = 1e-5


            if np.any(np.isclose(recall_values, 0, atol=tolerance)) or np.any(np.isclose(recall_values, 1, atol=tolerance)):
                logger.warning(f"Threshold adjustment skipped for epoch {epoch_id} due to extreme recall values.")
                threshold_history.append(self.optimal_thresholds)
                save_threshold_history(threshold_history, self.save_model_path, self.configs.use_model, self.configs.preprocess_conf.feature_method)
            else:




                hidden_class_ratio = np.mean([1 if np.all(predicted_probs.cpu().numpy() < hidden_class_threshold) else 0 for predicted_probs in preds])

                logger.info(f"Hidden class ratio for epoch {epoch_id}: {hidden_class_ratio:.2f}")

                if hidden_class_ratio > 0.2:  
                    hidden_class_threshold += 0.01
                elif hidden_class_ratio < 0.05:  
                    hidden_class_threshold -= 0.01
                hidden_class_threshold = np.clip(hidden_class_threshold, 0.01, 0.3)  


                if not (np.all(np.isnan(smoothed_thresholds)) or np.all(np.isinf(smoothed_thresholds))):
                    self.optimal_thresholds = [float(th) for th in smoothed_thresholds]
                    self.optimal_thresholds.append(float(hidden_class_threshold))  
                    threshold_history.append(self.optimal_thresholds)
                    save_threshold_history(threshold_history, self.save_model_path, self.configs.use_model, self.configs.preprocess_conf.feature_method)
                    logger.info(f"Updated optimal thresholds (smoothed) for epoch {epoch_id}: {self.optimal_thresholds}")
                    save_thresholds_to_json(self.optimal_thresholds, self.save_model_path, self.configs.use_model, self.configs.preprocess_conf.feature_method)
                else:
                    logger.warning(f"Invalid smoothed thresholds detected for epoch {epoch_id}. Using current thresholds.")
                    threshold_history.append(self.optimal_thresholds)
                    save_threshold_history(threshold_history, self.save_model_path, self.configs.use_model, self.configs.preprocess_conf.feature_method)            
                        



                        
            
            if epoch_id == self.configs.train_conf.max_epoch:
                print('================================================')
                print(f"Updated optimal thresholds (smoothed) for epoch {epoch_id}: {self.optimal_thresholds}")
                print('================================================')






    def evaluate_and_adjust_thresholds(self, all_probs, all_labels):
        optimal_thresholds = []
        for i in range(all_labels.shape[1]):
            precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])
            f1_scores = 2 * precision * recall / (precision + recall)
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            optimal_thresholds.append(optimal_threshold)
        self.optimal_thresholds = optimal_thresholds
        print("Optimal thresholds for each label:", optimal_thresholds)
        
 


    def apply_optimal_thresholds(self, probs, thresholds,hidden_class_threshold=None):
        """
        应用阈值到概率，生成预测结果
        :param probs: 模型的输出概率值，形状为 (batch_size, num_classes)
        :param thresholds: 每个类别的阈值，形状为 (num_classes,)
        :return: 二值化的预测结果，形状为 (batch_size, num_classes)
        """
        # 初始化与 probs 相同形状的张量，用于存储预测结果
        predictions = torch.zeros_like(probs)

        # 遍历每个类别并应用阈值
        for i, threshold in enumerate(thresholds):
            predictions[:, i] = (probs[:, i] >= threshold).float()
            
      
        if hidden_class_threshold is not None:
            is_hidden_class = torch.all(probs < hidden_class_threshold, dim=1)  
            predictions[is_hidden_class] = 0 
            
        return predictions


    
    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None):

        self.save_model_path = save_model_path

        nranks = torch.cuda.device_count()
        local_rank = 0
        writer = None
        if local_rank == 0:

            writer = SummaryWriter('logs')

        if nranks > 1 and self.use_gpu:

            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])


        self.__setup_dataloader(is_train=True)

        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=True)


        if nranks > 1 and self.use_gpu:
            self.model.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        logger.info('训练数据：{}'.format(len(self.train_dataset)))

        self.__load_pretrained(pretrained_model=pretrained_model)

        last_epoch, best_acc = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        self.train_loss, self.train_acc = None, None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        last_epoch += 1
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)

        self.max_step = len(self.train_loader) * self.configs.train_conf.max_epoch
        self.train_step = max(last_epoch, 0) * len(self.train_loader)

        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            if self.stop_train: break
            epoch_id += 1
            start_epoch = time.time()

            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer, nranks=nranks)

            if local_rank == 0:
                if self.stop_eval: continue
                logger.info('=' * 70)
                self.eval_loss, self.eval_acc = self.evaluate()
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), self.eval_loss, self.eval_acc))
                logger.info('=' * 70)
                writer.add_scalar('Test/Accuracy', self.eval_acc, self.test_log_step)
                writer.add_scalar('Test/Loss', self.eval_loss, self.test_log_step)
                self.test_log_step += 1
                self.model.train()
 
                if self.eval_acc >= best_acc:
                    best_acc = self.eval_acc
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=self.eval_acc,
                                           best_model=True)

                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=self.eval_acc)


    def predict(self, data_or_path, resume_model=None, return_feature_maps=False):
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)

        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model, map_location=self.device)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')

        self.model.eval()
        eval_model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        ds = MAClsDataset(data_list_path=data_or_path,
                                         audio_featurizer=self.audio_featurizer,
                                         do_vad=self.configs.dataset_conf.do_vad,
                                         max_duration=self.configs.dataset_conf.eval_conf.max_duration,
                                         min_duration=self.configs.dataset_conf.min_duration,
                                         sample_rate=self.configs.dataset_conf.sample_rate,
                                         use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                         target_dB=self.configs.dataset_conf.target_dB,
                                         mode='eval')
        loader = DataLoader(dataset=ds,
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    batch_size=self.configs.dataset_conf.eval_conf.batch_size,
                                    num_workers=self.configs.dataset_conf.dataLoader.num_workers)
        
        all_preds = []
        all_outputs = []
        feature_maps = []
        with torch.no_grad():
            for batch_id, (features, _, _) in enumerate(tqdm(loader)):
                if self.stop_eval: break
                features = features.to(self.device)

                assert not self.model.training, "Model is in training mode during forward inference."
                output, _, fm = eval_model.forward(features, return_feature_maps=True)
                
                thresholds = self.optimal_thresholds 
                preds = self.apply_optimal_thresholds(output,thresholds)
                
                all_outputs.append(output.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                feature_maps.append(fm.cpu().numpy())

        all_outputs = np.vstack(all_outputs)
        all_preds = np.vstack(all_preds)
        feature_maps = np.vstack(feature_maps)

        if return_feature_maps:
            return all_outputs, all_preds, feature_maps
        return all_outputs, all_preds


    def evaluate(self, resume_model=None, save_matrix_path=None, label_names=None, noise_threshold=0.5):


        if label_names is None:
            label_names = self.class_labels

        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        

        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model, map_location=self.device)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')

        self.model.eval()
        eval_model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        total_correct = 0
        total_samples = 0
        bg_count = 0 
        all_preds = []
        all_targets = []
        all_probs = []
        losses = []
        num_labels = len(self.class_labels)
        all_combinations = ["_".join(map(str, comb)) for comb in product([0, 1], repeat=num_labels)]
        all_combinations.append("noise")
        label_counts = {label: 0 for label in label_names} 
        
        try:
            self.optimal_thresholds = load_thresholds_from_json(save_model_path=self.save_model_path,model_name=self.configs.use_model,feature_method=self.configs.preprocess_conf.feature_method,filename="optimal_thresholds.json")
        except FileNotFoundError:
            logger.warning("No saved thresholds found, using default thresholds.")
            self.optimal_thresholds =[0.6,0.56,0.54]
        
        with torch.no_grad():

            for batch_id, (features, labels, input_lens) in enumerate(tqdm(self.test_loader)):
                if self.stop_eval: break
                features = features.to(self.device)
                labels = labels.to(self.device).float()


 
                output, freq_band_activations  = eval_model(features)



                loss = self.loss(output, labels, freq_band_activations)
                losses.append(loss.item())

                preds = self.apply_optimal_thresholds(output, self.optimal_thresholds)

                for i in range(preds.size(0)):
                        if (output[i] < torch.tensor(self.optimal_thresholds, device=output.device)).all():
                            preds[i] = torch.zeros_like(preds[i])  

                loss = self.loss(output, labels, freq_band_activations)
                losses.append(loss.item())

  
                correct = torch.all(preds == labels, dim=1).float()
                total_correct += correct.sum().item()
                total_samples += labels.size(0)

                for label_idx, label_name in enumerate(label_names):
                    label_counts[label_name] += int(labels[:, label_idx].sum().item())

                bg_count += (labels.sum(dim=1) == 0).sum().item()


                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                all_probs.append(output.cpu().numpy())
                
        acc = total_correct / total_samples
        loss = sum(losses) / len(losses)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)


        precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
        recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)


        print(f"\nEvaluation Results:")
        print(f"Subset Accuracy: {acc:.5f}")
        print(f"Loss: {loss:.5f}")
        print(f"Precision per label: {precision}")
        print(f"Recall per label: {recall}")
        print(f"F1 Score per label: {f1}")
        print(f"Background samples in dataset: {bg_count}")


        print(f"Background samples: {bg_count}")
        is_background = np.all(all_targets == 0, axis=1)
        background_preds = all_preds[is_background]
        background_targets = all_targets[is_background]

        if len(background_targets) > 0:
            correctly_predicted_bg = (background_preds == 0).all(axis=1).sum()
            total_bg = len(background_targets)
            

            bg_precision = correctly_predicted_bg / total_bg if total_bg > 0 else 0.0
            bg_recall = correctly_predicted_bg / total_bg if total_bg > 0 else 0.0
            bg_f1 = 2 * bg_precision * bg_recall / (bg_precision + bg_recall) if (bg_precision + bg_recall) > 0 else 0.0

            print(f"Background Precision (manual): {bg_precision:.5f}")
            print(f"Background Recall (manual): {bg_recall:.5f}")
            print(f"Background F1 Score (manual): {bg_f1:.5f}")
        else:
            print("No background samples in the evaluation set.")

        print("\n\n")
        which = ['lt', 'm', 'w']
        output_path = "data/output/validation"
        eval.plot_roc_curves(all_targets, all_probs, which, os.path.join(output_path, "cnn_roc.pdf"))
        eval.evaluate_results(all_preds, all_targets, which, "cnn", output_path)
        eval.print_combined_multilabel_confusion_matrix(os.path.join(output_path, "cnn_multi-label_cm.pdf"), all_targets, all_preds, list(range(len(which))), which, title=None)
        all_targets, all_preds, all_targets
        quant.eval_ratio_error(all_targets, all_preds, all_targets, all_targets, all_preds, all_targets, [[0,1], [1,2]])
        
        return loss, acc

        
    def group_labels(label, label_names):
        # Define grouped categories
        groups = {'lt': ['lt', 'ltc', 'ltg'],
                'm': ['chros', 'm'],
                'w': ['w']}
        result = []

        for group, members in groups.items():
            if any(label[label_names.index(member)] == 1 for member in members):
                result.append(group)
        
        # If no group is active, mark as no_target
        if not result:
            result.append('no_target')

        return result


 

    def evaluate_grouped_confusion_matrix(self, resume_model=None, save_matrix_path=None, label_names=['lt', 'm', 'w'], noise_threshold=0.5,output_txt_path="evaluation_results.txt"):

        
        # 设置随机种子
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if label_names is None:
            label_names = self.class_labels

        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)

        # 加载模型
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')

        self.model.eval()
        eval_model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        all_preds = []
        all_targets = []
        all_preds_v = []
        all_targets_v = []
        total_correct = 0
        total_samples = 0

        try:
            threshold_history = load_thresholds_from_json(save_model_path=self.save_model_path, model_name=self.configs.use_model, feature_method=self.configs.preprocess_conf.feature_method, filename="threshold_history.json")
            thresholds = compute_final_thresholds(threshold_history)
            logger.info(f"Final thresholds computed from training history: {thresholds}")
        except FileNotFoundError:
            logger.warning("No saved thresholds found, using default thresholds.")
            thresholds = [0.61,0.56,0.56]

        with torch.no_grad():
            with open(output_txt_path, "w", encoding="utf-8") as output_txt:
                for batch_id, (features, labels, input_lens) in enumerate(tqdm(self.test_loader)):
                    features = features.to(self.device)
                    labels = labels.to(self.device).float()


                    output, freq_band_activations = eval_model(features)

                    preds = self.apply_optimal_thresholds(output, thresholds)
                    
                          
                    for idx in range(features.size(0)):
                        file_name = f"sample_{batch_id}_{idx}.wav"  
                        true_label = labels[idx].cpu().numpy().tolist()
                        predicted_label = preds[idx].cpu().numpy().tolist()
                        scores = output[idx].cpu().numpy().tolist()
                        
         
                        output_txt.write(
                            f"{file_name}\t{true_label}\t{predicted_label}\t{scores}\n"
                        )

      
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
                    all_preds_v.append(preds.cpu().numpy())
                    all_targets_v.append(labels.cpu().numpy())

  
                    correct = torch.all(preds == labels, dim=1).float()
                    total_correct += correct.sum().item()
                    total_samples += labels.size(0)


        acc = total_correct / total_samples
        print(f"Subset Accuracy: {acc:.5f}")


        all_preds_v = np.vstack(all_preds_v)
        all_targets_v = np.vstack(all_targets_v)
        precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
        recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)


        print(f"\nEvaluation Results:")
        print(f"Subset Accuracy: {acc:.5f}")
        print(f"Precision per label: {precision}")
        print(f"Recall per label: {recall}")
        print(f"F1 Score per label: {f1}")

        precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        precision_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        print(f"Overall Precision (Macro Average): {precision_macro:.5f}")
        print(f"Overall Precision (Weighted Average): {precision_weighted:.5f}")






        unique_labels_binary = sorted(set("_".join(map(str, map(int, label))) for label in all_targets + all_preds))

        label_mapping = {}
        for binary_label in unique_labels_binary:
            binary_values = list(map(int, binary_label.split("_")))
            active_labels = [label_names[i] for i, val in enumerate(binary_values) if val == 1]
            if active_labels:
                label_mapping[binary_label] = ", ".join(active_labels)
            else:
                label_mapping[binary_label] = "no_target"


        unique_labels_binary = sorted(set("_".join(map(str, map(int, label))) for label in all_targets + all_preds))


        label_mapping = {}
        for binary_label in unique_labels_binary:
            binary_values = list(map(int, binary_label.split("_")))
            active_labels = [label_names[i] for i, val in enumerate(binary_values) if val == 1]
            if active_labels:
                label_mapping[binary_label] = ", ".join(active_labels)
            else:
                label_mapping[binary_label] = "no_target"


        unique_labels_readable = [label_mapping[binary_label] for binary_label in unique_labels_binary]

 
        cm_detailed = confusion_matrix(
            ["_".join(map(str, map(int, label))) for label in all_targets],
            ["_".join(map(str, map(int, pred))) for pred in all_preds],
            labels=unique_labels_binary
        )

 
        non_zero_indices = [i for i in range(len(cm_detailed)) if not all(cm_detailed[i, :] == 0) and not all(cm_detailed[:, i] == 0)]
        filtered_cm = cm_detailed[np.ix_(non_zero_indices, non_zero_indices)]
        filtered_labels_readable = [unique_labels_readable[i] for i in non_zero_indices]


        plt.figure(figsize=(12, 10))
        sns.heatmap(filtered_cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_labels_readable, yticklabels=filtered_labels_readable)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Filtered Detailed Confusion Matrix')
        plt.show()








