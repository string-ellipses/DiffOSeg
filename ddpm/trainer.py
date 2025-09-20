import importlib
import logging
import os
import pathlib
import pprint
from dataclasses import dataclass
from random import randint
from types import FunctionType
from typing import Callable, Dict, Any, cast
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import random
from sklearn.model_selection import KFold
import ignite.distributed as idist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm 
from metrics_set import *
# import wandb
from PIL import Image
from ignite.contrib.handlers import ProgressBar, WandBLogger
from ignite.contrib.metrics import GpuInfo
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Frequency, ConfusionMatrix, mIoU, IoU
from ignite.utils import setup_logger
from torch.utils.data.sampler import SubsetRandomSampler
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
from torchvision.utils import make_grid  # save_image
from torch.utils.tensorboard import SummaryWriter
from datasets.pipelines.transforms import build_transforms
from .models import DenoisingModel, build_model
from .models.condition_encoder import ConditionEncoder, _build_feature_cond_encoder
from .models.one_hot_categorical import OneHotCategoricalBCHW
from .optimizer import build_optimizer
from .polyak import PolyakAverager
from .utils import WithStateDict, archive_code, expanduservars, worker_init_fn, _onehot_to_color_image, calc_batched_generalised_energy_distance, \
    batched_hungarian_matching

__all__ = ["run_train", "Trainer", "load"]

LOGGER = logging.getLogger(__name__)
Model = Union[DenoisingModel, nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]


def _flatten(m: Model) -> DenoisingModel:
    if isinstance(m, DenoisingModel):
        return m
    elif isinstance(m, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return cast(DenoisingModel, m.module)
    else:
        raise TypeError("type(m) should be one of (DenoisingModel, DataParallel, DistributedDataParallel)")


def _loader_subset(loader: DataLoader, num_images: int, randomize=False) -> DataLoader:
    dataset = loader.dataset
    lng = len(dataset)
    fixed_indices = range(0, lng - lng % num_images, lng // num_images)
    if randomize:
        overlap = True
        fixed_indices_set = set(fixed_indices)
        maxatt = 5
        cnt = 0
        while overlap and cnt < maxatt:
            indices = [randint(0, lng - 1) for _ in range(0, num_images)]
            overlap = len(set(indices).intersection(fixed_indices_set)) > 0
            cnt += 1
    else:
        indices = fixed_indices
    return DataLoader(
        Subset(dataset, indices),
        batch_size=loader.batch_size,
        shuffle=False
    )


@torch.no_grad()
def grid_of_predictions(model: Model, feature_cond_encoder: ConditionEncoder, loader: DataLoader, num_predictions: int, cond_vis_fn: FunctionType,
                        params: dict) -> Tensor:
    model.eval()
    if feature_cond_encoder:
        feature_cond_encoder.eval()

    
    grids_ = []
    for id in range(0,4):
        labels_: List[Tensor] = []
        predictions_: List[Tensor] = []
        conditions_: List[Tensor] = []
        for batch in loader:
            images_b, labels_b= batch
            label_b = labels_b[:,id][:,None]
            images_b = images_b.to(idist.device())
            prompt = torch.tensor(id).repeat_interleave(images_b.shape[0]).to(idist.device())
            condition_b_enc = {'hint': images_b, 'txt': prompt}
            feature_condition_b_enc = feature_cond_encoder(images_b) if feature_cond_encoder else None

            label_b = label_b.to(idist.device())

            # if len(batch) == 3:
            #     # mm cityscapes
            #     label_shape = (labels_b.shape[0], model.diffusion.num_classes, *labels_b.shape[3:])
            # else:
            #     # cityscapes and friends
            #     label_shape = (labels_b.shape[0], model.diffusion.num_classes, *labels_b.shape[2:])
            label_shape = (labels_b.shape[0], model.diffusion.num_classes, *labels_b.shape[3:])
            predictions_b = []
            for _ in range(num_predictions):
                x = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=label_b.device)).sample()
                y = model(x, condition_b_enc, feature_condition_b_enc)["diffusion_out"]
                prediction = _onehot_to_color_image(y, params)
                predictions_b.append(prediction)
            predictions_b = torch.stack(predictions_b, dim=1)
            # if len(batch) == 3:
            #     labels_.append(_onehot_to_color_image(labels_b[:, 0], params))
            # else:
            #     labels_.append(_onehot_to_color_image(labels_b, params))
            labels_.append(_onehot_to_color_image(label_b[:, 0], params))
            predictions_.append(predictions_b)
            conditions_.append(cond_vis_fn(images_b))

        labels = torch.cat(labels_, dim=0).cpu()
        predictions = torch.cat(predictions_, dim=0).cpu()
        conditions = torch.cat(conditions_, dim=0).cpu()

        grid = torch.cat([
            conditions[:, None].expand((-1, -1, 3, -1, -1)),
            labels[:, None].expand((-1, -1, 3, -1, -1)),
            predictions.expand((-1, -1, 3, -1, -1))
        ],
            dim=1
        )
        grids_.append(grid[None,:])
    grids = torch.cat(grids_, dim=0).cpu()
    new_shape = (grids.shape[0], grids.shape[1] * grids.shape[2]) + grids.shape[3:]
    return torch.reshape(grids, new_shape)

# @torch.no_grad()
# def compute_ged(loader, model, num_samples, average_feature_cond_encoder):
#     model.eval()
#     if average_feature_cond_encoder:
#         average_feature_cond_encoder.eval()

    
#     geds = []
#     hm_ious = []
#     sim_sampless = []
#     cnts = []
#     dices = []
#     ious = []
#     for id in range(0,4):
#         ged = 0
#         hm_iou = 0
#         sim_samples = 0
#         cnt = 0
#         dice = 0
#         iou = 0
#         for batch in loader:
#             LOGGER.info(f"{cnt}...")
#             image, labels = batch
#             label = labels[:,id][:, None]
#             image = image.to(idist.device())
#             feature_condition = None
#             if average_feature_cond_encoder:
#                 feature_condition = average_feature_cond_encoder(image)
#                 feature_condition = feature_condition.repeat_interleave(num_samples, dim=0)

#             image = image.repeat_interleave(num_samples, dim=0)
#             prompt = torch.tensor(id).repeat_interleave(image.shape[0]).to(idist.device())
#             condition_b_enc = {'hint': image, 'txt': prompt}
#             x = OneHotCategoricalBCHW(logits=torch.zeros(label[:, 0].repeat_interleave(num_samples, dim=0).shape, device=labels.device)).sample().to(
#                 idist.device())

#             prediction = model(x,  condition_b_enc, feature_condition=feature_condition)['diffusion_out']
#             prediction = prediction.reshape(label.shape[0], -1, *label.shape[2:])

#             label = label.to(idist.device())
#             num_classes = label.shape[2]
#             label = label.argmax(dim=2)

#             batch_ged, _, similarity_samples = calc_batched_generalised_energy_distance(label.cpu().numpy(), prediction.argmax(dim=2).cpu().numpy(),
#                                                                                         num_classes)

#             ged += np.sum(batch_ged)
#             sim_samples += np.sum(similarity_samples)

#             lcm = np.lcm(num_samples, label.shape[1])

#             hm_labels = label.repeat_interleave(lcm // label.shape[1], dim=1).cpu().numpy()
#             predictions = prediction.repeat_interleave(lcm // num_samples, dim=1).argmax(dim=2).cpu().numpy()

#             batch_hm_iou = batched_hungarian_matching(hm_labels, predictions, num_classes)

#             hm_iou += np.sum(batch_hm_iou)

#             cnt += len(batch_ged)



#         ged = ged / cnt
#         sim_samples = sim_samples / cnt
#         hm_iou = hm_iou / cnt

#         geds.append(ged)
#         sim_sampless.append(sim_samples)
#         hm_ious.append(hm_iou)
    
#     ged_mean = sum(geds) / 4
#     sim_samples_mean = sum(sim_sampless) / 4
#     hm_iou_mean = sum(hm_ious) / 4
#     personal = {'ged': geds, 'sim_samples': sim_samples, 'hm_iou': hm_ious}

#     return ged_mean, sim_samples_mean, hm_iou_mean, personal
@torch.no_grad()
def compute_ged(loader, model, num_samples, average_feature_cond_encoder):
    model.eval()
    if average_feature_cond_encoder:
        average_feature_cond_encoder.eval()

    
    geds_16 = []
    hm_ious = []
    sim_sampless = []
    cnts = []
    dices = []
    ious = []
    for id in range(0,4):
        ged = 0
        hm_iou = 0
        sim_samples = 0
        cnt = 0
        dice = 0
        iou = 0
        for batch in loader:
            LOGGER.info(f"{cnt}...")
            image, labels = batch
            label = labels[:,id][:, None]
            image = image.to(idist.device())
            feature_condition = None
            if average_feature_cond_encoder:
                feature_condition = average_feature_cond_encoder(image)
                feature_condition = feature_condition.repeat_interleave(num_samples, dim=0)

            image = image.repeat_interleave(num_samples, dim=0)
            prompt = torch.tensor(id).repeat_interleave(image.shape[0]).to(idist.device())
            condition_b_enc = {'hint': image, 'txt': prompt}
            x = OneHotCategoricalBCHW(logits=torch.zeros(label[:, 0].repeat_interleave(num_samples, dim=0).shape, device=labels.device)).sample().to(
                idist.device())

            prediction = model(x,  condition_b_enc, feature_condition=feature_condition)['diffusion_out']
            prediction = prediction.reshape(label.shape[0], -1, *label.shape[2:])

            label = label.to(idist.device())
            num_classes = label.shape[2]
            label = label.argmax(dim=2)

            batch_ged, _, similarity_samples = calc_batched_generalised_energy_distance(label.cpu().numpy(), prediction.argmax(dim=2).cpu().numpy(),
                                                                                        num_classes)

            ged += np.sum(batch_ged)
            sim_samples += np.sum(similarity_samples)

            lcm = np.lcm(num_samples, label.shape[1])

            hm_labels = label.repeat_interleave(lcm // label.shape[1], dim=1).cpu().numpy()
            predictions = prediction.repeat_interleave(lcm // num_samples, dim=1).argmax(dim=2).cpu().numpy()

            batch_hm_iou = batched_hungarian_matching(hm_labels, predictions, num_classes)

            hm_iou += np.sum(batch_hm_iou)

            cnt += len(batch_ged)



        ged = ged / cnt
        sim_samples = sim_samples / cnt
        hm_iou = hm_iou / cnt

        geds.append(ged)
        sim_sampless.append(sim_samples)
        hm_ious.append(hm_iou)
    
    ged_mean = sum(geds) / 4
    sim_samples_mean = sum(sim_sampless) / 4
    hm_iou_mean = sum(hm_ious) / 4
    personal = {'ged': geds, 'sim_samples': sim_samples, 'hm_iou': hm_ious}

    return ged_mean, sim_samples_mean, hm_iou_mean, personal

def compute_iou(predictions, labels, threshold=0.5):
    # 将预测结果根据阈值进行二值化
    predictions = predictions > threshold
    
    # 计算交集：两个张量都为1的元素
    intersection = torch.logical_and(labels, predictions).float().sum((2, 3))  # 对最后两维求和
    
    # 计算并集：至少有一个张量为1的元素
    union = torch.logical_or(labels, predictions).float().sum((2, 3))  # 对最后两维求和
    
    epsilon = 1e-6
    iou = intersection / (union + epsilon)
    
    # 防止除以0的情况，可以给分母加上一个很小的值
    iou[union == 0] = 1.0  # 如果并集为0，则IoU定义为1（因为没有正例存在）

    return iou

def compute_miou(loader, model, num_samples, average_feature_cond_encoder):
    model.eval()
    if average_feature_cond_encoder:
        average_feature_cond_encoder.eval()
    mious = []
    for id in range(0,4):
        iou = 0
        cnt = 0
        for batch in loader:
            LOGGER.info(f"{cnt}...")
            image, labels = batch
            label = labels[:,id][:, None]
            image = image.to(idist.device())
            feature_condition = None
            if average_feature_cond_encoder:
                feature_condition = average_feature_cond_encoder(image)
                feature_condition = feature_condition.repeat_interleave(num_samples, dim=0)

            image = image.repeat_interleave(num_samples, dim=0)
            prompt = torch.tensor(id).repeat_interleave(image.shape[0]).to(idist.device())
            condition_b_enc = {'hint': image, 'txt': prompt}
            x = OneHotCategoricalBCHW(logits=torch.zeros(label[:, 0].repeat_interleave(num_samples, dim=0).shape, device=labels.device)).sample().to(
                idist.device())

            prediction = model(x,  condition_b_enc, feature_condition=feature_condition)['diffusion_out']
            prediction = prediction.reshape(label.shape[0], -1, *label.shape[2:])

            label = label.to(idist.device())
            num_classes = label.shape[2]
            label = label.argmax(dim=2)

            # batch_ged, _, similarity_samples = calc_batched_generalised_energy_distance(label.cpu().numpy(), prediction.argmax(dim=2).cpu().numpy(),
            #                                                                             num_classes)

            # ged += np.sum(batch_ged)
            # sim_samples += np.sum(similarity_samples)

            # lcm = np.lcm(num_samples, label.shape[1])

            # hm_labels = label.repeat_interleave(lcm // label.shape[1], dim=1).cpu().numpy()
            # predictions = prediction.repeat_interleave(lcm // num_samples, dim=1).argmax(dim=2).cpu().numpy()

            # batch_hm_iou = batched_hungarian_matching(hm_labels, predictions, num_classes)
            mean_prediction = torch.log(prediction).mean(dim=(1)).argmax(dim=1)
            iou_ = compute_iou(label, mean_prediction[:,None])
            iou += iou_.mean() 
            cnt += 1

        miou = iou / cnt

        
        mious.append(miou)
    
    
    miou_mean = sum(mious) / 4
    personal = {'mious': mious}

    return miou_mean, personal

def compute_metrics(val_loader, model, num_samples):
    
    metrics_dict = {
        'Dice_each_mean': [],
        'mIoU_each_mean': [],
    }

    model.eval()
    with torch.no_grad():
        for id in range(0,4):
            dice_each = 0.0
            iou_each = 0.0
            for val_step, (image, labels) in enumerate(tqdm(val_loader)):
                label = labels[:,id][:, None]
                image = image.to(idist.device())
                feature_condition = None
                image = image.repeat_interleave(num_samples, dim=0)
                prompt = torch.tensor(id).repeat_interleave(image.shape[0]).to(idist.device())
                condition_b_enc = {'hint': image, 'txt': prompt}
                x = OneHotCategoricalBCHW(logits=torch.zeros(label[:, 0].repeat_interleave(num_samples, dim=0).shape, device=labels.device)).sample().to(
                    idist.device())

                prediction = model(x,  condition_b_enc, feature_condition=feature_condition)['diffusion_out']
                prediction = prediction.reshape(label.shape[0], -1, *label.shape[2:])

                label = label.to(idist.device())
                label = label.argmax(dim=2)

                # Dice score 
                dice_each_iter, best_predictions = batch_dice(prediction, label)
                mean_prediction = torch.log(prediction).mean(dim=(1)).argmax(dim=1)
                iou_each_iter = compute_iou(mean_prediction.unsqueeze(1), label)

                dice_each += np.mean(dice_each_iter)
                iou_each += iou_each_iter.mean().cpu().numpy()
                    
            metrics_dict['Dice_each_mean'].append(dice_each)
            metrics_dict['mIoU_each_mean'].append(iou_each)
        
    for key in metrics_dict:
        metrics_dict[key] = np.mean(metrics_dict[key]) / len(val_loader)

    return metrics_dict 



@dataclass
class Trainer:
    polyak: PolyakAverager
    optimizer: torch.optim.Optimizer
    lr_scheduler: Union[torch.optim.lr_scheduler.LambdaLR, None]
    class_weights: torch.Tensor
    save_debug_state: Callable[[Engine, Dict[str, Any]], None]
    feature_cond_polyak: PolyakAverager
    params: dict

    @property
    def flat_model(self):
        """View of the model without DataParallel wrappers."""
        return _flatten(self.model)

    @property
    def model(self):
        return self.polyak.model

    @property
    def average_model(self):
        return self.polyak.average_model

    @property
    def feature_cond_encoder(self):
        return self.feature_cond_polyak.model if self.feature_cond_polyak else None

    @property
    def average_feature_cond_encoder(self):
        return self.feature_cond_polyak.average_model if self.feature_cond_polyak else None

    @property
    def time_steps(self):
        return self.flat_model.time_steps

    @property
    def diffusion_model(self):
        return self.flat_model.diffusion

    def train_step(self, engine: Engine, batch) -> dict:
        device = idist.device()
        if self.params['stage'] == 2:
            image, prompt, x0 = batch
            image = image.to(device, non_blocking=True)
            x0 = x0.to(device, non_blocking=True)
            prompt = prompt.to(device, non_blocking=True)
            condition = {'hint':image,'txt':prompt}
        elif self.params['stage'] == 1:
            image, x0 = batch
            image = image.to(device, non_blocking=True)
            x0 = x0.to(device, non_blocking=True)
            condition = {'hint':image}
        else:
            raise ValueError("Stage must be '1' or '2' during initialization.")

        self.model.train()
        if self.params['feature_cond_encoder']['train']:
            self.feature_cond_encoder.train()

        self.shape = x0.shape[1:]

        feature_condition = self.feature_cond_encoder(image) if self.feature_cond_polyak else None
        if isinstance(feature_condition, dict):
            for name, feature in feature_condition.items():
                feature_condition[name] = feature_condition[name].contiguous()
        elif feature_condition is not None:
            feature_condition = feature_condition.contiguous()

        batch_size = x0.shape[0]

        # Sample a random step and generate gaussian noise
        t = torch.randint(1, self.time_steps + 1, size=(batch_size,), device=device)
        xt = self.diffusion_model.q_xt_given_x0(x0, t).sample()

        # Estimate the noise with the model
        condition = {key: value.contiguous() for key, value in condition.items()}
        ret = self.model(xt.contiguous(), condition, feature_condition, t)
        x0pred = ret["diffusion_out"]

        prob_xtm1_given_xt_x0 = self.diffusion_model.theta_post(xt, x0, t)
        prob_xtm1_given_xt_x0pred = self.diffusion_model.theta_post_prob(xt, x0pred, t)

        loss_diffusion = nn.functional.kl_div(
            torch.log(torch.clamp(prob_xtm1_given_xt_x0pred, min=1e-12)),
            prob_xtm1_given_xt_x0,
            reduction='none'
        )

        # Look for nan and inf in loss and save debug state if found
        self._check_loss(loss_diffusion, locals())
        mask = self.class_weights[x0.argmax(dim=1)]

        loss_diffusion = loss_diffusion.sum(dim=1) * mask
        loss = torch.sum(loss_diffusion) / batch_size

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()
        else:
            lr = self.optimizer.defaults['lr']
        self.polyak.update()
        if self.feature_cond_polyak:
            self.feature_cond_polyak.update()

        return {"num_items": batch_size,
                "loss": loss.item(),
                "lr": lr}

    def _debug_state(self, locals: dict, debug_names: Optional[List[str]] = None) -> Dict[str, Any]:

        if debug_names is None:
            debug_names = [
                "image", "t", "x0", "xt", "x0pred",
                "prob_xtm1_given_xt_x0", "prob_xtm1_given_xt_x0pred", "loss"
            ]

        to_save = self.objects_to_save(locals["engine"])
        debug_tensors = {k: locals[k] for k in debug_names}
        to_save["tensors"] = WithStateDict(**debug_tensors)
        return to_save

    def _check_loss(self, loss: Tensor, locals: dict) -> None:

        invalid_values = []

        if torch.isnan(loss).any():
            LOGGER.error("nan found in loss!!")
            invalid_values.append("nan")

        if torch.isinf(loss).any():
            LOGGER.error("inf found in loss!!")
            invalid_values.append("inf")

        if (loss.sum(dim=1) < -1e-3).any():
            LOGGER.error("negative KL divergence in loss!!")
            invalid_values.append("neg")

        if invalid_values:
            LOGGER.error("Saving debug state...")
            self.save_debug_state(locals["engine"], self._debug_state(locals))
            raise ValueError(f"Invalid value {invalid_values} found in loss. Debug state has been saved.")

    @torch.no_grad()
    def test_step(self, _: Engine, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        max_num_samples = 10

        if self.params['stage'] == 2:
            image, prompt, labels = batch 
            prompt = prompt.repeat_interleave(max_num_samples, dim=0).to(idist.device())
            image = image.repeat_interleave(max_num_samples, dim=0).to(idist.device())
            labels = labels[:,None].to(idist.device())
            condition = {'hint': image, 'txt': prompt}
        elif self.params['stage'] == 1:
            image, labels = batch 
            image = image.repeat_interleave(max_num_samples, dim=0).to(idist.device())
            labels = labels[:,None].to(idist.device())
            condition = {'hint': image}
        else:
            raise ValueError("stage must be 1 OR 2.")

        x = OneHotCategoricalBCHW(logits=torch.zeros(labels[:, 0].repeat_interleave(max_num_samples, dim=0).shape, device=labels.device)).sample().to(idist.device())

        
        # prediction = prediction.reshape(labels.shape[0], -1, *labels.shape[2:])

        #label_shape = (label.shape[0], self.flat_model.diffusion.num_classes, *label.shape[2:])
        # x = OneHotCategoricalBCHW(logits=torch.zeros(label_shape, device=label.device)).sample()
 
        # label = label.argmax(dim=1)
        prediction = self.predict(x, condition)
        prediction = prediction.reshape(labels.shape[0], -1, *labels.shape[2:])
        labels = labels.argmax(dim=2).squeeze(1)

        mean_prediction = torch.log(prediction).mean(dim=(1))
        return {'y_pred': mean_prediction, 'y': labels}
    
    @torch.no_grad()
    def predict(self, xt: Tensor, condition: dict, label_ref: Optional[Tensor] = None) -> Tensor:
        self.average_model.eval()

        feature_condition = None
        if self.feature_cond_polyak:
            self.average_feature_cond_encoder.eval()
            feature_condition = self.average_feature_cond_encoder(condition)

        ret = self.average_model(xt, condition, feature_condition=feature_condition)
        return ret["diffusion_out"]

    def objects_to_save(self, engine: Optional[Engine] = None, weights_only: bool = False) -> Dict[str, Any]:
        to_save: Dict[str, Any] = {
            "model": self.flat_model.unet,
        }

        if self.feature_cond_encoder and self.params['feature_cond_encoder']['train']:
            to_save["feature_cond_encoder"] = self.feature_cond_encoder

        to_save["average_model"] = _flatten(self.average_model).unet
        if self.average_feature_cond_encoder and self.params['feature_cond_encoder']['train']:
            to_save["average_feature_cond_encoder"] = self.average_feature_cond_encoder

        if not weights_only:
            to_save["optimizer"] = self.optimizer
            to_save["scheduler"] = self.lr_scheduler

            if engine is not None:
                to_save["engine"] = engine

        return to_save


def build_engine(trainer: Trainer,
                 output_path: str,
                 train_loader: DataLoader,
                 validation_loader: DataLoader,
                 cond_vis_fn: FunctionType,
                 num_classes: int,
                 ignore_class: 0,
                 params: dict,
                 writer) -> Engine:
    engine = Engine(trainer.train_step)
    frequency_metric = Frequency(output_transform=lambda x: x["num_items"])
    frequency_metric.attach(engine, "imgs/s", Events.ITERATION_COMPLETED)
    GpuInfo().attach(engine, "gpu")

    validation_freq = params["validation_freq"] if "validation_freq" in params else 5000
    save_freq = params["save_freq"] if "save_freq" in params else 10000
    display_freq = params['display_freq'] if "display_freq" in params else 500
    n_validation_predictions = params["n_validation_predictions"] if "n_validation_predictions" in params else 4
    n_validation_images = params["n_validation_images"] if "n_validation_images" in params else 5

    engine_test = Engine(trainer.test_step)
    cm = ConfusionMatrix(num_classes=num_classes)
    LOGGER.info(f"Ignore class {ignore_class} in IoU evaluation...")
    IoU(cm, ignore_index=ignore_class).attach(engine_test, "IoU")
    mIoU(cm, ignore_index=ignore_class).attach(engine_test, "mIoU")

    engine_train = Engine(trainer.test_step)
    cm_train = ConfusionMatrix(num_classes=num_classes)
    IoU(cm_train, ignore_index=ignore_class).attach(engine_train, "IoU")
    mIoU(cm_train, ignore_index=ignore_class).attach(engine_train, "mIoU")

    if idist.get_local_rank() == 0:
        ProgressBar(persist=True).attach(engine_test)

        if params["wandb"]:
            tb_logger = WandBLogger(project=params["wandb_project"], entity='cdm', config=params)

            tb_logger.attach_output_handler(
                engine,
                Events.ITERATION_COMPLETED(every=50),
                tag="training",
                output_transform=lambda x: x,
                metric_names=["imgs/s"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

            tb_logger.attach_output_handler(
                engine_test,
                Events.EPOCH_COMPLETED,
                tag="testing",
                metric_names=["mIoU", "IoU"],
                global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
            )

        checkpoint_handler = ModelCheckpoint(
            output_path,
            "model",
            n_saved=18,
            require_empty=False,
            score_function=None,
            score_name=None
        )

        checkpoint_best_hmiou = ModelCheckpoint(
            output_path,
            "best",
            n_saved=10,
            require_empty=False,
            score_function=lambda engine: engine.state.metrics['HMIoU'],
            score_name='HM-IoU',
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )

        checkpoint_best_ged = ModelCheckpoint(
            output_path,
            "best",
            n_saved=10,
            require_empty=False,
            score_function=lambda engine: -engine.state.metrics['GED'],
            score_name='GED',
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )
 
        # def score_function_mIoU(engine):
        #     # 确保这里的 'mIoU' 是已经被正确计算并存储在 engine.state.metrics 中的
        #     return engine.state.metrics.get('mIoU', 0)

        checkpoint_best_miou = ModelCheckpoint(
            output_path,
            "best",
            n_saved=10,
            require_empty=False,
            score_function=lambda engine: engine.state.metrics['mIoU'],
            score_name='mIoU',
            global_step_transform=global_step_from_engine(engine, Events.ITERATION_COMPLETED)
        )

    @engine.on(Events.EPOCH_COMPLETED(every=1))
    def epoch_completed_and_set_epoch(engine: Engine):
        if isinstance(engine.state.dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            engine.state.dataloader.sampler.set_epoch(engine.state.epoch)
            LOGGER.info("DDP sampler: set_epoch=%d (iter=%d) completed rank=%d ",
                        engine.state.epoch,
                        engine.state.iteration,
                        idist.get_local_rank())

    # Display some info every 100 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=display_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_info(engine: Engine):
        LOGGER.info(
            "epoch=%d, iter=%d, speed=%.2fimg/s, loss=%.4g, lr=%.6g,  gpu:0 util=%.2f%%",
            engine.state.epoch,
            engine.state.iteration,
            engine.state.metrics["imgs/s"],
            engine.state.output["loss"],
            engine.state.output["lr"],
            engine.state.metrics["gpu:0 util(%)"]
        )
        writer.add_scalar("Training/Loss", engine.state.output["loss"], engine.state.iteration)

    # Save model every save_freq iterations
    @engine.on(Events.ITERATION_COMPLETED(every=save_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_model(engine: Engine):
        checkpoint_handler(engine, trainer.objects_to_save(engine, weights_only=False))

    # Generate and save a few segmentations every validation_freq iterations
    @engine.on(Events.ITERATION_COMPLETED(every=validation_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def save_qualitative_results(_: Engine, num_images=n_validation_images, num_predictions=n_validation_predictions):
        LOGGER.info("Generating images...")
        # loader = _loader_subset(validation_loader, num_images, randomize=False)
        # grid = grid_of_predictions(_flatten(trainer.average_model), trainer.average_feature_cond_encoder, loader, num_predictions, cond_vis_fn,
        #                            params)
        # loader = _loader_subset(validation_loader, num_images, randomize=True)
        # grid_shuffle = grid_of_predictions(_flatten(trainer.average_model), trainer.average_feature_cond_encoder, loader, num_predictions,
        #                                    cond_vis_fn, params)
        # grid = torch.concat([grid, grid_shuffle], dim=0)
        # filename = os.path.join(output_path, f"images_{engine.state.iteration:06}")
        # LOGGER.info("Saving images to %s...", filename)
        # os.makedirs(output_path, exist_ok=True)
        # for id in range(0,4):
        #     img = save_image(grid[id], f"{filename}_{id}.png", nrow=(1 + grid.shape[0]//2))

    # Compute the GED score every validation_freq iterations
    @engine.on(Events.ITERATION_COMPLETED(every=validation_freq))
    @idist.one_rank_only(rank=0, with_barrier=True)
    def test(_: Engine):
        if 'lidc' in params["dataset_file"]:
            # LOGGER.info("GED computation...")
            # ged, sim_samples, hm_iou, personal = compute_ged(validation_loader, _flatten(trainer.average_model), params["samples"],
            #                                        trainer.average_feature_cond_encoder)
            # LOGGER.info("mean GED %.3f, mean sim-samples %.3f, mean hm-IoU %.2f, Personal Scores {%s}", ged, sim_samples, hm_iou, personal)
            # writer.add_scalar('Validation/GED', ged, engine.state.iteration)
            # writer.add_scalar('Validation/HMIoU', hm_iou, engine.state.iteration)
            LOGGER.info("val mIoU computation...")
            #miou, personal = compute_miou(validation_loader, _flatten(trainer.average_model), params["samples"], trainer.average_feature_cond_encoder)
            metrics_dict = compute_metrics(validation_loader, _flatten(trainer.average_model), params["samples"])
            LOGGER.info("mean iou %.3f", metrics_dict['mIoU_each_mean'])
            # engine.state.metrics['GED'] = ged
            # engine.state.metrics['HMIoU'] = hm_iou
            engine.state.metrics['mIoU'] = metrics_dict['mIoU_each_mean'].item()
            checkpoint_best_miou(engine, trainer.objects_to_save(engine, weights_only=False))
        else:
            LOGGER.info("val mIoU computation...")
            engine_test.run(validation_loader, max_epochs=1)

    # Save the best models by mIoU score (runs once every len(validation_loader))
    @engine_test.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_iou(engine_test: Engine):
        LOGGER.info("val mIoU score: %.4g", engine_test.state.metrics["mIoU"])
        writer.add_scalar('Validation/mIoU_val', engine_test.state.metrics["mIoU"], engine.state.iteration)
        checkpoint_best_miou(engine_test, trainer.objects_to_save(engine, weights_only=False))

    @engine.on(Events.ITERATION_COMPLETED(every=validation_freq))
    def train_mIoU(_: Engine):
        LOGGER.info("train mIoU computation...")
        train_loader_ss = _loader_subset(train_loader, 6, randomize=False)
        engine_train.run(train_loader_ss, max_epochs=1)

    @engine_train.on(Events.EPOCH_COMPLETED)
    @idist.one_rank_only(rank=0, with_barrier=True)
    def log_train_mIoU(engine_train: Engine):
        if 'cityscapes' in params["dataset_file"]:
            LOGGER.info("train mIoU score: %.4g", engine_train.state.metrics["mIoU"])
    return engine


# taken from torchvision utils
def save_image(tensor: Union[torch.Tensor, List[torch.Tensor]],
               fp: Union[Text, pathlib.Path, BinaryIO],
               format: Optional[str] = None,
               **kwargs) -> Image:
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    return im


def load(filename: str, trainer: Trainer, engine: Engine, weights_only: bool = False):
    LOGGER.info("Loading state from %s...", filename)
    state = torch.load(filename, map_location=idist.device())
    to_load = trainer.objects_to_save(engine, weights_only)
    ModelCheckpoint.load_objects(to_load, state)


def _build_model(params: dict, input_shapes: List[Tuple[int, int, int]], cond_encoded_shape) -> Model:
    model: Model = build_model(
        time_steps=params["time_steps"],
        schedule=params["beta_schedule"],
        schedule_params=params.get("beta_schedule_params", None),
        cond_encoded_shape=cond_encoded_shape,
        input_shapes=input_shapes,
        backbone=params["backbone"],
        backbone_params=params[params["backbone"]],
        dataset_file=params['dataset_file'],
        step_T_sample=params.get('evaluation_vote_strategy', None),
        stage=params["stage"],
        feature_cond_encoder=params['feature_cond_encoder'] if params['feature_cond_encoder']['type'] != 'none' else None
    ).to(idist.device())

    # Wrap the model in DataParallel or DistributedDataParallel for parallel processing
    if params["distributed"]:
        local_rank = idist.get_local_rank()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif params["multigpu"]:
        model = nn.DataParallel(model)

    return model


def _build_datasets(params: dict) -> Tuple[Dataset, Dataset, torch.Tensor, int, dict]:
    dataset_file: str = params['dataset_file']
    dataset_module = importlib.import_module(dataset_file)
    train_ids_to_class_names = None

    all_dataset = dataset_module.all_dataset(params['stage'])
    all_test_dataset = dataset_module.all_test_dataset(params['stage'])
    LOGGER.info("%d images in all dataset '%s'", len(all_dataset), dataset_file)
    LOGGER.info("%d images in all test dataset '%s'", len(all_test_dataset), dataset_file)

    # If there is no 'get_weights' function in the dataset module, create a tensor full of ones.
    get_weights = getattr(dataset_module, 'get_weights', lambda: torch.ones(all_dataset[0][-1].shape[0]))
    class_weights = get_weights()

    if params['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(all_dataset,
                                                                        rank=idist.get_local_rank(),
                                                                        num_replicas=params['num_gpus'])
        batch_size = params['batch_size'] // params['num_gpus']  # batch_size of each process

    else:
        train_sampler = None
        batch_size = params['batch_size']  # if single_gpu or non-DDP
    
    return all_dataset, all_test_dataset, class_weights, dataset_module.get_ignore_class(), train_ids_to_class_names


def _build_debug_checkpoint(output_path: str) -> ModelCheckpoint:
    return ModelCheckpoint(output_path, "debug_state", require_empty=False, score_function=None, score_name=None)



def run_train(local_rank: int, params: dict):
    setup_logger(name=None, format="\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s", reset=True)

    # Load the datasets
    all_dataset, all_test_dataset, class_weights, ignore_class, train_ids_to_class_names = _build_datasets(params)
    # split datasets into k folds
    kfolds = params['kfolds']
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

    LOGGER.info("Training params:\n%s", pprint.pformat(params))
    # num_gpus = torch.cuda.device_count()
    LOGGER.info("%d GPUs available", torch.cuda.device_count())

    cudnn.benchmark = params['cudnn']['benchmark']  # this set to true usually slightly accelerates training
    LOGGER.info(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***')
    LOGGER.info(f"*** cudnn.enabled {cudnn.enabled}")
    LOGGER.info(f"*** cudnn.deterministic {cudnn.deterministic}")

    for fold_idx in range(0, kfolds):
        print(f"Starting fold {fold_idx}")
        train_indices = splits[fold_idx]['train_index']
        test_indices = splits[fold_idx]['test_index']
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(all_dataset, batch_size=params['batch_size'], sampler=train_sampler)
        test_loader = DataLoader(all_test_dataset, batch_size=20, sampler=test_sampler)
        # Create output folder and archive the current code and the parameters there
        output_path = f"{params['output_path']}stage{params['stage']}_fold{fold_idx}"
        os.makedirs(output_path, exist_ok=True)
        LOGGER.info("experiment dir: %s", output_path)
        archive_code(output_path)
        writer = SummaryWriter(log_dir=os.path.join(output_path, 'tensorboard_logs'))

        # Build the model, optimizer, trainer and training engine
        input_shapes = [i.shape for i in train_loader.dataset[0] if hasattr(i, 'shape')]
        LOGGER.info("Input shapes: " + str(input_shapes))
        cond_encoded_shape = input_shapes[0]
        LOGGER.info("Encoded condition shape: " + str(cond_encoded_shape))
        num_classes = input_shapes[1][0]
        assert len(class_weights) == num_classes, f"len(class_weights) != num_classes: {len(class_weights)} != {num_classes}"

        model, average_model = [_build_model(params, input_shapes, cond_encoded_shape) for _ in range(2)]

        if params['stage'] == 2:
            pretrained_weights_path = params['stage1_weights']#f'logs/lidc_stage1_majority/lidc_mix_majority_fold_{fold_idx}/model_checkpoint.pt'
            if pretrained_weights_path:
                pretrained_weights_path = expanduservars(pretrained_weights_path)
                pretrained_state_dict = torch.load(pretrained_weights_path, map_location='cpu')['model']
                # 加载模型权重，允许部分匹配
                pretrained_state_dict = {
                    "unet." + k: v
                    for k, v in pretrained_state_dict.items()
                }
                model_state_dict = model.state_dict()
                # 筛选匹配的键，并跳过形状不匹配的键
                pretrained_dict = {}
                skipped_keys = []
                for k, v in pretrained_state_dict.items():
                    if k in model_state_dict:
                        if v.shape == model_state_dict[k].shape:
                            pretrained_dict[k] = v
                        else:
                            skipped_keys.append(k)
                    else:
                        skipped_keys.append(k)

                missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
                loaded_keys = pretrained_dict.keys()
        
                LOGGER.info(f"Loaded pretrained weights from {pretrained_weights_path}")
                LOGGER.info(f"Loaded keys: {loaded_keys}")
                LOGGER.info(f"Missing keys: {missing_keys}")
                LOGGER.info(f"Unexpected keys: {unexpected_keys}")
                LOGGER.info(f"Skipped keys due to shape mismatch: {skipped_keys}")

        average_model.load_state_dict(model.state_dict())
        
        num_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOGGER.info("unet_openai trainable params: %d", num_of_parameters)
        feature_cond_encoder, cond_vis_fn = _build_feature_cond_encoder(params)
        average_feature_cond_encoder, _ = _build_feature_cond_encoder(params)
        feature_cond_polyak = PolyakAverager(feature_cond_encoder, average_feature_cond_encoder) if feature_cond_encoder else None

        polyak = PolyakAverager(model, average_model, alpha=params["polyak_alpha"])
        optimizer_staff = build_optimizer(params, model, feature_cond_encoder, train_loader, debug=False)

        optimizer = optimizer_staff['optimizer']
        lr_scheduler = optimizer_staff['lr_scheduler']

        debug_checkpoint = _build_debug_checkpoint(output_path)
        trainer = Trainer(polyak, optimizer, lr_scheduler, class_weights.to(idist.device()), debug_checkpoint,
                        feature_cond_polyak, params)
        engine = build_engine(trainer, output_path, train_loader, test_loader, cond_vis_fn,
                            num_classes=num_classes, ignore_class=ignore_class,
                            params=params,
                            writer=writer)

        # Load a model (if requested in params.yml) to continue training from it
        load_from = params.get('load_from', None)
        if load_from is not None:
            load_from = expanduservars(load_from)
            load(load_from, trainer=trainer, engine=engine)
            optimizer.param_groups[0]['capturable'] = True

        # Run the training engine for the requested number of epochs

        engine.run(train_loader, max_epochs=params["max_epochs"])
        writer.close()
