import numpy as np
import torch
from utils.Data_Load import Data_Load
from models.DH_model import DH_model
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from utils.random_seed import setup_seed

if __name__ == '__main__':
    """
    数据加载
    """
    setup_seed(42)
    Data_path_dict = {
        'SVI':'../data/Graph/svi_features.npy',
        'TF_inf':'../data/Graph/traffic_inf.npy',
        'svi_adj':'scripts/svi_adj.npy',
        'similarity_adj':'scripts/similarity_edge.npy',
        'label':'../data/Graph/label_inf.npy',
        'train_idx':'../data/Graph/train_inf.npy',
        'val_idx': '../data/Graph/val_inf.npy',
    }
    SVI_feature,TF_inf,svi_adj,similarity_adj,label,train_mask,val_mask = Data_Load(Data_path_dict)
    """
    模型定义
    """
    train_epoch = 5000
    train_val_times = 2
    model = DH_model().cuda()
    loss_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 7], dtype=torch.float32),label_smoothing=0.1).cuda()
    # loss_criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99),weight_decay=0.005)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=train_epoch,eta_min=0.00001)
    """
    训练和验证 计算loss和acc
    """
    best_acc = 0.0
    model_weight_path = os.path.join('model_pth', 'DH.pth')
    writer = SummaryWriter("exp/DH",flush_secs=60)
    accuracy = MulticlassAccuracy(num_classes=2).cuda()
    precision = MulticlassPrecision(num_classes=2, average='weighted').cuda()
    recall = MulticlassRecall(num_classes=2, average='weighted').cuda()
    for epoch in range(train_epoch):
        sum_train_loss = 0.0
        model.train()
        optimizer.zero_grad()
        out = model(SVI_feature,TF_inf,svi_adj,similarity_adj)
        loss = loss_criterion(out[train_mask], label[train_mask])
        predicted,y_true = out[train_mask], label[train_mask]
        accuracy_value,precision_value,recall_value = accuracy(predicted,y_true),precision(predicted,y_true),recall(predicted,y_true)
        print(' [Train epoch {}] 训练阶段平均指标======>Acc:{:.3f} Loss:{:.3f}'.format(epoch + 1, accuracy_value.item(),loss.item()))
        writer.add_scalar('训练指标/总loss', loss.item(), epoch + 1)
        writer.add_scalar('训练指标/Acc', accuracy_value.item(), epoch + 1)
        writer.add_scalar('训练指标/precision', precision_value.item(), epoch + 1)
        writer.add_scalar('训练指标/recall', recall_value.item(), epoch + 1)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if (epoch + 1) % train_val_times == 0:
            model.eval()
            optimizer.zero_grad()
            out = model(SVI_feature, TF_inf, svi_adj, similarity_adj)
            loss = loss_criterion(out[val_mask], label[val_mask])
            predicted, y_true = out[val_mask], label[val_mask]
            accuracy_value, precision_value, recall_value = accuracy(predicted, y_true), precision(predicted,y_true), recall(predicted, y_true)
            print(' [Train epoch {}] 验证阶段平均指标======>Acc:{:.3f} Loss:{:.3f}'.format(epoch + 1, accuracy_value.item(),loss.item()))
            writer.add_scalar('验证指标/总loss', loss.item(), epoch + 1)
            writer.add_scalar('验证指标/Acc', accuracy_value.item(), epoch + 1)
            writer.add_scalar('验证指标/precision', precision_value.item(), epoch + 1)
            writer.add_scalar('验证指标/recall', recall_value.item(), epoch + 1)
            if (accuracy_value.item() >= best_acc):
                best_acc = accuracy_value.item()
                torch.save(model.state_dict(), model_weight_path)
    model_weight_path = os.path.join('model_pth', 'F_DH.pth')
    torch.save(model.state_dict(), model_weight_path)
