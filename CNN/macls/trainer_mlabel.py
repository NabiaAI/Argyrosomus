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
from macls.metric.metrics import accuracy_imb
from macls.utils.logger import setup_logger
from macls.utils.scheduler import WarmupCosineSchedulerLR
from macls.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments
from macls.models.mulcnns import SimpleCNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from macls.models.resnet18 import ResNet18
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import torch
import sys
sys.path.append('.')
import utils_eval as eval


logger = setup_logger(__name__)

def find_optimal_thresholds(y_true, y_probs):
    optimal_thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_probs[:, i])
        f1_scores = 2 * precision * recall / (precision + recall)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        optimal_thresholds.append(optimal_threshold)
    return optimal_thresholds

class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        loss = self.bce(inputs, targets)
        loss = loss * self.weights
        return loss.mean()

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
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'No such model：{self.configs.use_model}'
        self.model = None
        self.audio_featurizer = None
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.amp_scaler = None
        

        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.dataLoader.num_workers = 0
            logger.warning('Windows cannot support')
            

        self.optimal_thresholds = [0.6,0.7, 0.6]
        self.spec_aug = SpecAug(**self.configs.dataset_conf.get('spec_aug_args', {}))
        self.spec_aug.to(self.device)
        self.max_step, self.train_step = None, None
        self.train_loss, self.train_acc = None, None
        self.train_eta_sec = None
        self.eval_loss, self.eval_acc = None, None
        self.test_log_step, self.train_log_step = 0, 0
        self.stop_train, self.stop_eval = False, False
        self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf.feature_method,
                                        method_args=self.configs.preprocess_conf.get('method_args', {}))

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
                train_sampler = DistributedSampler(dataset=self.train_dataset)
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
                                      shuffle=True,
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
                                        use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                        target_dB=self.configs.dataset_conf.target_dB,
                                        mode='extract_feature')
            save_data_list = data_list.replace('.txt', '_features.txt')
            with open(save_data_list, 'w', encoding='utf-8') as f:
                for i in tqdm(range(len(test_dataset))):
                    feature, labels = test_dataset[i]
                    feature = feature.numpy()
                    label_str = ' '.join(map(str, labels.tolist()))
                    save_path = os.path.join(save_dir, '_'.join(map(str, labels.nonzero()[0].tolist())), f'{int(time.time() * 1000)}.npy').replace('\\', '/')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    np.save(save_path, feature)
                    f.write(f'{save_path}\t{label_str}\n')
            logger.info(f'{data_list}feature are extracted，new list will be：{save_data_list}')



    def __setup_model(self, input_size, is_train=False):
        if self.configs.model_conf.num_class is None:
            self.configs.model_conf.num_class = len(self.class_labels)

        if self.configs.use_model == 'Res18':
            self.model = ResNet18(input_size=input_size, **self.configs.model_conf)

        else:
            raise Exception(f'{self.configs.use_model} Model does not exist')
        self.model.to(self.device)
        summary(self.model, input_size=(1, 98, input_size))

        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() == 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")

        weight = torch.tensor(self.configs.train_conf.loss_weight, dtype=torch.float, device=self.device)\
            if self.configs.train_conf.loss_weight is not None else None
            

        self.loss = FocalLoss(alpha=1, gamma=2)
        
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
                raise Exception(f'Do NOT SUPPORT OPTIMIZER{optimizer}')

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
                raise Exception(f'Do not support function：{self.configs.optimizer_conf.scheduler}')
        if self.configs.train_conf.use_compile and torch.__version__ >= "2" and platform.system().lower() != 'windows':
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.model.to(self.device)

    def __load_pretrained(self, pretrained_model):
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pth')
            assert os.path.exists(pretrained_model), f"{pretrained_model} Model do not exist！"
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
            logger.info('Sucessful upload pretrained model：{}'.format(pretrained_model))

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_acc = 0
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')
        
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth'))
                                         and os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pth')), "resume_model is not exist"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pth')), "optimizer is not exist！"
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
            logger.info('Sucessful upload pretrained model：{}'.format(resume_model))
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
        logger.info('Save model{}'.format(model_path))

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
            
            with torch.cuda.amp.autocast(enabled=self.configs.train_conf.enable_amp):
                output = self.model(features)
            
            loss = self.loss(output, labels)
            
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

            thresholds = [0.6, 0.7, 0.6]
            preds = self.apply_optimal_thresholds(output,thresholds)
            correct = (preds == labels).float()
            acc = correct.sum().item() / (labels.size(0) * labels.size(1))
            accuracies.append(acc)
            loss_sum.append(loss.data.cpu().numpy())
            train_times.append((time.time() - start) * 1000)
            self.train_step += 1

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
            


    def evaluate_and_adjust_thresholds(self, all_probs, all_labels):
        optimal_thresholds = []
        for i in range(all_labels.shape[1]):
            precision, recall, thresholds = precision_recall_curve(all_labels[:, i], all_probs[:, i])
            f1_scores = 2 * precision * recall / (precision + recall)
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            optimal_thresholds.append(optimal_threshold)
        self.optimal_thresholds = optimal_thresholds
        print("Optimal thresholds for each label:", optimal_thresholds)
        
    def apply_optimal_thresholds(self, output, thresholds=None):
        probs = torch.sigmoid(output)
        preds = torch.zeros_like(probs)
        if thresholds is None:
            thresholds = [0.5] * probs.shape[1]  
        for i, threshold in enumerate(thresholds):
            #print(print(probs.shape,preds.shape))
            preds[:, i] = (probs[:, i] >= threshold).float()
        return preds


    
    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None):


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
        logger.info('Training data{}'.format(len(self.train_dataset)))

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

    def _setup_inference(self, resume_model):
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            resume_model_root = resume_model
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} Model does not exist！"
            model_state_dict = torch.load(resume_model, map_location=self.device)
            self.model.load_state_dict(model_state_dict)
            with open(os.path.join(resume_model_root, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                self.optimal_thresholds = json_data.get('optimal_thresholds', self.optimal_thresholds)
            logger.info(f'Sucessful upload model{resume_model}')
        self.model.eval()

    def predict(self, data_or_path, resume_model=None):
        self._setup_inference(resume_model)

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
        with torch.no_grad():
            for batch_id, (features, _, _) in enumerate(tqdm(loader)):
                if self.stop_eval: break
                features = features.to(self.device)

                assert not self.model.training, "Model is in training mode during forward inference."
                output = self.model(features)
                
                thresholds = self.optimal_thresholds #[0.681, 0.517, 0.612, 0.696] # ResNet; [0.6, 0.64, 0.62, 0.6] SP_CNN
                preds = self.apply_optimal_thresholds(output,thresholds)
                
                all_outputs.append(output.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        all_outputs = np.vstack(all_outputs)
        all_preds = np.vstack(all_preds)

        return all_outputs, all_preds


    def evaluate(self, resume_model=None, save_matrix_path=None):
        if self.test_loader is None:
            self.__setup_dataloader()
        self._setup_inference(resume_model)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module
        else:
            eval_model = self.model
        total_correct = 0
        total_samples = 0

        all_preds = []
        all_targets = []
        all_probs = []
        losses = []
        with torch.no_grad():
            for batch_id, (features, labels, input_lens) in enumerate(tqdm(self.test_loader)):
                if self.stop_eval: break
                features = features.to(self.device)
                labels = labels.to(self.device).float()

                output = self.model(features)
                loss = self.loss(output, labels)
                losses.append(loss.item())
                
                thresholds = [0.6, 0.7, 0.6]
                preds = self.apply_optimal_thresholds(output,thresholds)
                correct = (preds == labels).float()
                total_correct += correct.sum().item()
                total_samples += labels.size(0) * labels.size(1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                all_probs.append(output.cpu().numpy())

        acc = total_correct / total_samples
        loss = sum(losses) / len(losses)
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)

        print(f'Test Accuracy: {acc:.5f}')


        if save_matrix_path is not None:
            eval.plot_roc_curves(all_targets, all_probs, self.class_labels, os.path.join(save_matrix_path, 'cnn_roc_curve.pdf'))
            try:
                cm = multilabel_confusion_matrix(all_targets, all_preds)
                os.makedirs(save_matrix_path, exist_ok=True)
                for i, matrix in enumerate(cm):
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix for Label {i}')
                    plt.savefig(os.path.join(save_matrix_path, f'confusion_matrix_label_{i}_{int(time.time())}.pdf'))
                    plt.close()
            except Exception as e:
                logger.error(f'Save confusion martrix{e}')
        self.model.train()
        return loss, acc

    


    def export(self, save_model_path='models/', resume_model='models/EcapaTdnn_Fbank/best_model/'):
        self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pth')
        assert os.path.exists(resume_model), f"{resume_model} Model does not exist！"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info('Sucessful upload resume model{}'.format(resume_model))
        self.model.eval()
        infer_model = self.model.export()
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                        'inference.pth')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info("pretrained model saved{}".format(infer_model_path))
