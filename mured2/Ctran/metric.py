from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import torch
import numpy as np

class Metric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.total_samples = 0
        self.outputs = []
        self.labels = []
        self.outputs_thresholded =[]

    def update(self, outputs, labels, k, threshold=0.5):
        
        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().astype(np.int32)
        
        outputs_thresholded = (outputs > threshold).astype(np.int32)

        self.tp += np.sum((outputs_thresholded * labels), axis=0)
        self.fp += np.sum((outputs_thresholded * (1 - labels)), axis=0)
        self.fn += np.sum(((1 - outputs_thresholded) * labels), axis=0)
        
        self.total_samples += outputs.shape[0]
        self.outputs.append(outputs)
        self.labels.append(labels)
        self.outputs_thresholded.append(outputs_thresholded)
        

    def compute(self,k):
        outputs = np.concatenate(self.outputs)
        labels = np.concatenate(self.labels)
        outputs_thresholded = np.concatenate(self.outputs_thresholded)
        
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        
        #f1 = f1_score((self.tp + self.fn).cpu().numpy().astype(int).tolist(), self.tp.cpu().numpy().astype(int).tolist(), average='macro')
        f1 = f1_score(labels, outputs_thresholded, average='macro', zero_division=1)
        mAP = average_precision_score(labels, outputs)
        AUC = roc_auc_score(labels, outputs, average='macro', multi_class='ovr')
 
        # Exclude NORMAL class
        ml_mAP = average_precision_score(labels[:, 1:], outputs[:, 1:])
        ml_AUC = roc_auc_score(labels[:, 1:], outputs[:, 1:], average='macro', multi_class='ovr')
        ml_score = (ml_mAP + ml_AUC) / 2

        # Include NORMAL class
        bin_AUC = roc_auc_score(labels[1], outputs[1])
        model_score = (ml_score + bin_AUC) / 2

        # Compute F1-score of NORMAL label
        bin_f1 = f1_score(labels[1], outputs_thresholded[1], average='binary', zero_division=1)
        #bin_f1 = f1_score((self.tp.numpy()[1] + self.fn.numpy()[1]) > 0, self.tp.numpy()[1], average='binary', zero_division=1)

        return f1, mAP, AUC, ml_mAP, ml_AUC, ml_score, bin_AUC, model_score, bin_f1
     