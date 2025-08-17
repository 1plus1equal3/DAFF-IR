import torch
import torchmetrics
from .config import *

class Metrics:
    def __init__(self, name, mode, degrade_classes, device):
        self.name = name
        self.mode = mode
        self.degrade_classes = degrade_classes
        if device:
            self.device = device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mode == 'degrade':
            self.metrics = self.init_degrade_metrics()
        elif mode == 'restore':
            self.metrics = self.init_restore_metrics()
        elif mode == 'eval':
            self.metrics = self.init_eval_metrics()
        else:
            raise ValueError("Invalid mode. Choose either 'degrade' or 'restore'.")

    def init_degrade_metrics(self):
        # Degradation Prediction Metrics
        self.degrade_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(self.degrade_classes)).to(self.device)
        self.degrade_loss = torchmetrics.MeanMetric().to(self.device)
        return {
            'degrade_acc': self.degrade_acc,
            'degrade_loss': self.degrade_loss
        }

    def init_restore_metrics(self):
        # Restoration Metrics
        self.loss = torchmetrics.MeanMetric().to(self.device)
        self.psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        return {
            f'loss_{self.name}': self.loss,
            f'psnr_{self.name}': self.psnr,
            f'ssim_{self.name}': self.ssim
        }
    
    def init_eval_metrics(self):
        self.psnr_metrics = {deg_type: torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device) 
                        for deg_type in chosen_degradation}
        self.ssim_metrics = {deg_type: torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device) 
                        for deg_type in chosen_degradation}

        self.gf_psnr_metrics = {deg_type: torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device) 
                        for deg_type in chosen_degradation}
        self.gf_ssim_metrics = {deg_type: torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
                        for deg_type in chosen_degradation}
        
    def reset_metrics(self):
        for metric in self.metrics.values():
            if isinstance(metric, torchmetrics.Metric):
                metric.reset()