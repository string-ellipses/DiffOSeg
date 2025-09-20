import importlib
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os
import ignite.distributed as idist
import numpy as np
import torch
# Ignite imports
from ignite.engine import Engine
from ignite.handlers import ModelCheckpoint
from ignite.metrics import ConfusionMatrix, mIoU, IoU, DiceCoefficient
from ignite.utils import setup_logger
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ddpm.models.one_hot_categorical import OneHotCategoricalBCHW
from ddpm.polyak import PolyakAverager
from ddpm.trainer import _build_model, _flatten
from ddpm.utils import expanduservars
from metrics_set import *
from ddpm.trainer import _build_datasets
from sklearn.model_selection import KFold
from torch.utils.data import Subset
LOGGER = logging.getLogger(__name__)

NUM_CLASSES = 2

def iou(x, y, axis=-1):
    iou_ = (x & y).sum(axis) / (x | y).sum(axis)
    iou_[np.isnan(iou_)] = 1.
    return iou_


# exclude background
def batched_distance(x, y):
    try:
        per_class_iou = iou(x[:, :, None], y[:, None, :], axis=-2)
    except MemoryError:
        raise NotImplementedError

    return 1 - per_class_iou[..., 1:].mean(-1)


def calc_batched_generalised_energy_distance(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.reshape(*samples_dist_0.shape[:2], -1)
    samples_dist_1 = samples_dist_1.reshape(*samples_dist_1.shape[:2], -1)

    eye = np.eye(num_classes)

    samples_dist_0 = eye[samples_dist_0].astype(np.bool)
    samples_dist_1 = eye[samples_dist_1].astype(np.bool)
    
    cross = np.mean(batched_distance(samples_dist_0, samples_dist_1), axis=(1,2))
    diversity_0 = np.mean(batched_distance(samples_dist_0, samples_dist_0), axis=(1,2))
    diversity_1 = np.mean(batched_distance(samples_dist_1, samples_dist_1), axis=(1,2))
    return 2 * cross - diversity_0 - diversity_1, diversity_0, diversity_1


def batched_hungarian_matching(samples_dist_0, samples_dist_1, num_classes):
    samples_dist_0 = samples_dist_0.reshape((*samples_dist_0.shape[:2], -1))
    samples_dist_1 = samples_dist_1.reshape((*samples_dist_1.shape[:2], -1))

    eye = np.eye(num_classes)

    samples_dist_0 = eye[samples_dist_0].astype(np.bool)
    samples_dist_1 = eye[samples_dist_1].astype(np.bool)
    
    cost_matrix = batched_distance(samples_dist_0, samples_dist_1)

    h_scores = []
    for i in range(samples_dist_0.shape[0]):
        h_scores.append((1-cost_matrix[i])[linear_sum_assignment(cost_matrix[i])].mean())

    return h_scores

def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(arr.shape + (NUM_CLASSES,), dtype=np.float32)
    h, w = np.ix_(np.arange(arr.shape[0]), np.arange(arr.shape[1]))
    res[h, w, arr] = 1.0

    return res

def dice_coefficient(pred, true):
    smooth = 1.0  # 添加平滑因子以避免除以零
    intersection = np.sum(pred & true, axis=(0, 1))  # 计算交集
    union = np.sum(pred, axis=(0, 1)) + np.sum(true, axis=(0, 1))  # 计算并集
    dice = (2. * intersection + smooth) / (union + smooth)  # 计算Dice系数
    return dice

def batch_dice(prediction, labels):
    prediction = prediction.argmax(dim=2)
    labels = labels.squeeze(1)

    best_dice_scores = []
    best_predictions = []

    for i in range(labels.shape[0]):
        max_dice = 0 
        best_pred = None
        for j in range(prediction.shape[1]):
            dice_score = dice_coefficient(prediction[i,j].cpu().numpy().astype(bool), labels[i].cpu().numpy().astype(bool))
            if dice_score > max_dice:
                max_dice = dice_score 
                best_pred = torch.from_numpy(one_hot_encoding(prediction[i,j].cpu().numpy()).transpose(2,0,1))
        
        best_dice_scores.append(max_dice)
        best_predictions.append(best_pred)
    best_predictions = torch.stack(best_predictions, dim=0)

    return best_dice_scores, best_predictions


@dataclass
class Tester:

    polyak: PolyakAverager
    num_samples: int
    num_classes: int
    geds: float
    Dice_max_reverse: List
    Dice_soft: List
    Dice_match: List
    Dice_each_mean: List
    # similarity_samples: List
    # similarity_experts: List
    # hm_ious: List
    nonzero: int
    counter: int
    stage:int

    @torch.no_grad()
    def test_step(self, _: Engine, batch: Tensor) -> Dict[str, Any]:

        # 手动指定id
        # id = 3
        self.polyak.average_model.eval()

        image, labels = batch
        feature_condition=None
        #labels = labels[:,id][:,None]#.repeat(1,4,1,1,1)
        max_num_samples = np.max(self.num_samples)
        image = image.to(idist.device())
        #np.save("stage2_image/image", image.squeeze(0).squeeze(0).cpu().numpy())
        image = image.repeat_interleave(max_num_samples, dim=0)
        labels.to(idist.device())
        #np.save("stage2_image/labels", labels.squeeze(0).argmax(dim=1).cpu().numpy())
        if self.stage == 2:
            predictions = []
            for id in range(0, 4):
                labels_id = labels[:,id][:, None]
                prompt = torch.tensor(id).repeat_interleave(image.shape[0]).to(idist.device())
                condition_b_enc = {'hint': image, 'txt': prompt}
                x = OneHotCategoricalBCHW(logits=torch.zeros(labels_id[:, 0].repeat_interleave(max_num_samples, dim=0).shape, device=labels_id.device)).sample().to(
                    idist.device())

                prediction = self.polyak.average_model(x,  condition_b_enc, feature_condition=feature_condition)['diffusion_out']
                prediction = prediction.reshape(labels_id.shape[0], -1, *labels_id.shape[2:])#.argmax(dim=2)
                #np.save(f"stage2_image/100_prediction_{id}", prediction.squeeze(0).argmax(dim=1).cpu().numpy())
                labels_id = labels_id.to(idist.device())
                labels_id = labels_id.argmax(dim=2)
                mean_prediction = torch.mean(prediction, dim=1)[:,None]
                #np.save(f"stage2_image/mean_prediction_{id}", mean_prediction.squeeze(0).argmax(dim=1).squeeze(0).squeeze(0).cpu().numpy())
                predictions.append(mean_prediction.to(idist.device()))

            predictions = torch.cat(predictions, dim=1)
            self.counter += 1      
        else:
            prompt = None
            condition_b_enc = {'hint': image, 'txt': prompt}
            x = OneHotCategoricalBCHW(logits=torch.zeros(labels[:, 0].repeat_interleave(max_num_samples, dim=0).shape, device=labels.device)).sample().to(
                idist.device())

            predictions = self.polyak.average_model(x,  condition_b_enc, feature_condition=feature_condition)['diffusion_out']
            predictions = predictions.reshape(labels.shape[0], -1, *labels.shape[2:]).to(idist.device())

            labels = labels.argmax(dim=2).to(idist.device())
            self.counter += 1

        if self.stage == 1:
            for i, num_samples in enumerate(self.num_samples):
                predictions_sub = predictions[:,:num_samples].argmax(dim=2)
                GED_iter = generalized_energy_distance(labels.cpu(), predictions_sub.cpu())
                _, dice_max_reverse_iter, dice_match_iter, dice_each_iter= dice_at_all(labels.cpu(), predictions_sub.cpu(), thresh=0.5)
                dice_soft_iter = dice_at_thresh(labels.float().cpu(), predictions_sub.float().cpu())

                self.geds[i] += GED_iter
                self.Dice_max_reverse[i] += dice_max_reverse_iter
                self.Dice_soft[i] += dice_soft_iter
                self.Dice_match[i] += dice_match_iter
                if num_samples == max_num_samples:
                    self.Dice_each_mean = [self.Dice_each_mean[i] + dice_each_iter[i] for i in range(len(dice_each_iter))]
        else:
            labels = labels.argmax(dim=2).to(idist.device())
            predictions = predictions.repeat(1,4,1,1,1) #评测单个id
            for i in range(len(self.num_samples)):
                GED_iter = generalized_energy_distance(labels.cpu(), predictions.argmax(dim=2).cpu())
                _, dice_max_reverse_iter, dice_match_iter, dice_each_iter= dice_at_all(labels.cpu(), predictions.argmax(dim=2).cpu(), thresh=0.5)
                dice_soft_iter = dice_at_thresh(labels.float().cpu(), predictions.argmax(dim=2).float().cpu())

                self.geds[i] += GED_iter
                self.Dice_max_reverse[i] += dice_max_reverse_iter
                self.Dice_soft[i] += dice_soft_iter
                self.Dice_match[i] += dice_match_iter
                self.Dice_each_mean = [self.Dice_each_mean[i] + dice_each_iter[i] for i in range(len(dice_each_iter))]
        #LOGGER.info("Test batch %d with mean GED (%d) %.2f and mean HM IoU %.2f", self.counter, self.num_samples[-1], np.mean(ged), np.mean(hm_iou))
        LOGGER.info("Test batch %d with mean Dice %.2f ", self.counter, np.mean(dice_each_iter))
        #return {'y': labels[nonzero], 'y_pred': mean_prediction}
        # return {'y': labels[nonzero], 'y_pred': best_predictions}
        nonzero = torch.count_nonzero(labels, dim=(2,3)) > 0
        self.nonzero += torch.sum(nonzero)
        
        #mean_prediction = mean_prediction.repeat_interleave(torch.sum(nonzero, dim=1), dim=0)
        predictions =predictions.repeat_interleave(torch.sum(nonzero, dim=1), dim=0)
        mean_prediction = torch.log(predictions).mean(dim=(1))
        return {'y': labels[nonzero], 'y_pred': mean_prediction}
        

    def objects_to_save(self, engine: Optional[Engine] = None) -> Dict[str, Any]:
        to_save: Dict[str, Any] = {
            "average_model": _flatten(self.polyak.average_model).unet,
        }

        return to_save

def build_engine(tester: Tester, validation_loader: DataLoader, num_classes: int) -> Engine:
    engine_test = Engine(tester.test_step)

    cm = ConfusionMatrix(num_classes=num_classes)

    IoU(cm).attach(engine_test, "IoU")
    mIoU(cm).attach(engine_test, "mIoU")
    DiceCoefficient(cm).attach(engine_test, "Dice")

    return engine_test


def load(filename: str, trainer, engine: Engine):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine)
    ModelCheckpoint.load_objects(to_load, state)



def eval_lidc_uncertainty(params):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    LOGGER.info("%d GPUs available", torch.cuda.device_count())

    # Load the datasets
    dataset_file: str = params['dataset_file']
    #dataset_module = importlib.import_module(dataset_file)

    # test_dataset = dataset_module.test_dataset(params["dataset_val_max_size"])  # type: ignore
    # 获取完整的测试数据集
    all_dataset, all_test_dataset, class_weights, ignore_class, train_ids_to_class_names = _build_datasets(params)

    # split datasets into 4 folds
    kfolds = 4
    kf = KFold(n_splits=kfolds, shuffle=False)

    splits = []
    indices = list(range(len(all_dataset)))
    np.random.seed(1337)
    np.random.shuffle(indices)
    for (train_index, test_index) in kf.split(np.arange(len(all_dataset))):
        splits.append({
            'train_index': [indices[i] for i in train_index.tolist()],
            'test_index': [indices[i] for i in test_index.tolist()]})

    print(f"Fold {len(splits)} training samples: {len(train_index)}, testing samples: {len(test_index)}")
    # 初始化用于存储每折结果的列表
    fold_results = {
    'GED': [],
    'Dice_max_reverse': [],
    'Dice_soft': [],
    'Dice_match': [],
    'Dice_each_mean': np.zeros(4)  
    }
    for fold_idx in range(0, 1):
        print(f"Starting fold {fold_idx}")
        #train_indices = splits[fold_idx]['train_index']
        test_indices = splits[fold_idx]['test_index']
        test_subset = Subset(all_test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=params['batch_size'], shuffle=False)
        
        LOGGER.info("Fold %d: %d images in test dataset '%s'", fold_idx, len(test_indices), dataset_file)

        # test_loader = idist.auto_dataloader(
        #     test_dataset,
        #     batch_size=params["batch_size"],
        #     shuffle=False,
        #     num_workers=params["mp_loaders"]
        # )

        # Build the model, optimizer, trainer and training engine
        input_shapes = [test_loader.dataset[0][0].shape, test_loader.dataset[0][1].shape]
        input_shapes[1] = input_shapes[1][1:]
        LOGGER.info("Input shapes: " + str(input_shapes))
        
        num_classes = input_shapes[1][0]
        model, average_model = [_build_model(params, input_shapes, input_shapes[0]) for _ in range(2)]
        polyak = PolyakAverager(model, average_model, alpha=params["polyak_alpha"])

        tester = Tester(polyak, params["evaluations"], num_classes, np.zeros(len(params["evaluations"])),np.zeros(len(params["evaluations"])), np.zeros(len(params["evaluations"])), np.zeros(len(params["evaluations"])),  [0]*4, 0, 0, stage=params["stage"])
        engine = build_engine(tester, test_loader, num_classes=num_classes)

        # Load a model (if requested in params.yml) to continue training from it
        # fold_dir = f'lidc_concatenate_ca3_learn_fold_{fold_idx}'
        # load_from = os.path.join('logs', fold_dir, 'model_checkpoint_180000.pt')
        load_from = os.path.join(params.get('load_from', None), f"stage{params['stage']}_fold{fold_idx}")

        # 获取目录下所有pt文件并按文件名排序
        pt_files = [f for f in os.listdir(load_from) if f.endswith('.pt')]
        if pt_files:
            # 按文件名排序（假设文件名包含迭代次数）
            pt_files.sort()
            last_pt = pt_files[-1]  # 获取最后一个文件
            load_model_path = os.path.join(load_from, last_pt)
            LOGGER.info("Loading last model: %s", last_pt)
        else:
            LOGGER.warning("No .pt files found in %s", load_model_path)
        # if "pretrained" in load_from:
        #     load_from = load_from + '/' + f"pretrained_from_stage1_{params['finetune']}" + '/' + f"build_optimizer_{params['optim']['build_optimizer']}" + '/' + f"model_checkpoint_{params['iterations']}.pt"
        # else:
        #     load_from = load_from + '/' + f"model_checkpoint_{params['iterations']}.pt"
        if load_model_path is not None:
            load_model_path = expanduservars(load_model_path)
            load(load_model_path, trainer=tester, engine=engine)

        engine.state.max_epochs = None
        engine.run(test_loader, max_epochs=1)
        
        fold_results['GED'].append(tester.geds / tester.counter)
        fold_results['Dice_max_reverse'].append(tester.Dice_max_reverse / tester.counter)
        fold_results['Dice_soft'].append(tester.Dice_soft / tester.counter)
        fold_results['Dice_match'].append(tester.Dice_match / tester.counter)
        fold_results['Dice_each_mean'] += [x / tester.counter for x in tester.Dice_each_mean]  # 累积每一类的Dice mean


        # LOGGER.info(str(params['time_steps']) + ' ' + str(params['load_from']))
        LOGGER.info(str(params['time_steps']) + ' ' + str(load_from))
        for i in range(len(params['evaluations'])):
            # LOGGER.info("GED (%d): %.4g", params["evaluations"][i], tester.geds[i] / len(test_dataset))
            # LOGGER.info("Diversity samples (%d): %.4g", params["evaluations"][i], tester.similarity_samples[i] / len(test_dataset))
            # LOGGER.info("HM IoU (%d):%.4g", params["evaluations"][i], tester.hm_ious[i] / len(test_dataset))
            LOGGER.info("GED (%d): %.4g", params["evaluations"][i], tester.geds[i] / tester.counter)
            LOGGER.info("Dice_max (%d): %.4g", params["evaluations"][i], tester.Dice_max_reverse[i] / tester.counter)
            LOGGER.info("Dice_soft (%d):%.4g", params["evaluations"][i], tester.Dice_soft[i] / tester.counter)
            LOGGER.info("Dice_match (%d):%.4g", params["evaluations"][i], tester.Dice_match[i] / tester.counter)

            if i == len(params['evaluations']) - 1:
                LOGGER.info("Dice_each_mean: %s", ", ".join(["%.4g" % (dice / tester.counter) for dice in tester.Dice_each_mean]))

    avg_GED = np.mean(fold_results['GED'], axis=0)
    avg_Dice_max_reverse = np.mean(fold_results['Dice_max_reverse'], axis=0)
    avg_Dice_soft = np.mean(fold_results['Dice_soft'], axis=0)
    avg_Dice_match = np.mean(fold_results['Dice_match'], axis=0)
    avg_Dice_each_mean = fold_results['Dice_each_mean'] / kfolds  

    # 输出平均值
    LOGGER.info("Average GED: %s", ", ".join(["%.4g" % ged for ged in avg_GED]))
    LOGGER.info("Average Dice_max_reverse: %s", ", ".join(["%.4g" % dice for dice in avg_Dice_max_reverse]))
    LOGGER.info("Average Dice_soft: %s", ", ".join(["%.4g" % dice for dice in avg_Dice_soft]))
    LOGGER.info("Average Dice_match: %s", ", ".join(["%.4g" % dice for dice in avg_Dice_match]))
    LOGGER.info("Average Dice_each_mean per class: %s", ", ".join(["%.4g" % dice for dice in avg_Dice_each_mean]))
