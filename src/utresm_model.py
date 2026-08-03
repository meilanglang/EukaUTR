import pickle
import numpy as np
import sys
import torch
from torch import nn, optim
from typing import Optional
from pathlib import Path
from pytorch_lightning import seed_everything
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl

current_directory = Path(__file__).parent.absolute()
project_root = current_directory.parent.parent.parent.parent  
sys.path.append(str(project_root))
from .model import ESM2
from utils.evo.tokenization import Vocab
from utils.evo.metrics import pcc_metric,rho_metric,acc_metric,sub_location_metrics
from utils.lr_schedulers import WarmupLinearSchedule
from utils.classfier_head import ClassificationHead,get_eos_token, get_embedding
from utils.resnet import UtrResNet1D
from utils.config.utr_transformer_config import TransformerConfig,OptimizerConfig

n_class = 1
def calculate_accuracy_v1(logits, targets):
    _, preds = torch.max(logits, dim=-1)
    correct = (preds == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()


def calculate_accuracy(logits, targets):
    _, preds = torch.max(logits, dim=-1)
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    accuracy = accuracy_score(targets, preds)
    return accuracy


class UTRESM2Model(pl.LightningModule):
    def __init__(
            self,
            config,
            model_path,
            embedding_dim,
            feature_type, 
            vocab: Vocab,
    ):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.task_type = config.task_type
        self.num_classes = config.num_classes
        self.feature_type = feature_type
        self.model = self.from_pretrained(TransformerConfig,model_path)
        #self.encoder = ClassificationHead(self.config,embedding_dim,feature_type)
        self.encoder = UtrResNet1D(in_planes=embedding_dim, main_planes=256, out_planes=self.num_classes)

        self.global_step_counter = 0
        self.pad_idx = vocab.pad_idx
        self.eos_idx = vocab.eos_idx
        self.cls_idx = vocab.bos_idx
        #self.device_gpu = device_gpu
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.test_step_output = []

    def forward(self, tokens):#esm_embedding
        hidden_states = self.model(tokens)['representations'][12] # torch.Size([1, 103, 640]) 
        
        if self.feature_type == "cls":
            feature = hidden_states
            
        if self.feature_type == "eos":
            feature = get_eos_token(hidden_states,tokens, self.eos_idx)
            #feature = hidden_states
        if self.feature_type == "mean":
            feature = get_embedding(hidden_states,tokens, self.pad_idx,self.eos_idx,self.cls_idx)
        if self.feature_type == "emb":
            feature = hidden_states.permute(0, 2, 1)
            #print(hidden_states.shape) # torch.Size([1, 103, 640])
        classfier_logits = self.encoder(feature)
        return classfier_logits,hidden_states
    

    def training_step(self, batch, batch_idx):
        tokens, labels,_= batch
        #print(tokens.shape)
        #print(labels.shape)

        classfier_logits, _ = self(tokens)

        if self.task_type == 'Regression':
            #loss = nn.MSELoss(reduction='mean')(classfier_logits.squeeze(), labels)#axis=0
            loss = nn.HuberLoss(reduction='mean')(classfier_logits.squeeze(), labels)#axis=0
            
        if self.task_type=='Classification':
            loss = nn.CrossEntropyLoss(reduction='mean')(classfier_logits.squeeze(), labels.long())
        if self.task_type == 'mutilabel':
            #print(labels)
            loss = nn.BCEWithLogitsLoss(reduction='mean')(classfier_logits.squeeze(), labels.float())

        result = {"predictions": classfier_logits, "targets": labels}
        self.train_step_outputs.append(result)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        #print(len(batch))

        tokens, labels, seq_len = batch #esm_embedding
        #print(tokens)
        classfier_logits, _ = self(tokens)
        classfier_logits = classfier_logits.squeeze()
        if self.task_type == 'Regression':
            #loss = nn.MSELoss(reduction='mean')(classfier_logits.squeeze(), labels)  # axis=0
            loss = nn.HuberLoss(reduction='mean')(classfier_logits.squeeze(), labels)
        if self.task_type == 'Classification':
            loss = nn.CrossEntropyLoss(reduction='mean')(classfier_logits.squeeze(), labels.long())
        if self.task_type == 'mutilabel':
            #print("labels:",labels.shape)
            #print("classfier_logits:",classfier_logits.shape)
            loss = nn.BCEWithLogitsLoss(reduction='mean')(classfier_logits.squeeze(), labels.float())
        
        self.log("val_loss", loss)
        result = {"predictions": classfier_logits, "targets": labels}
        self.validation_step_outputs.append(result)
        return loss
    
    
    def on_train_epoch_end(self):
        # valid
        if self.task_type == 'Regression':
            valid_metrics = rho_metric(
                self.train_step_outputs,
            )
        if self.task_type == 'Classification':
            valid_metrics = acc_metric(
                self.train_step_outputs,
            )
        
        if self.task_type == 'mutilabel':
            valid_metrics = sub_location_metrics(
                self.train_step_outputs,
            )

        for key, value in valid_metrics.items():
            if isinstance(value, dict):
                pass
            else:
                key = f"train_{key}"
                self.log(key, value, prog_bar=True, on_epoch=True)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        # valid
        if self.task_type == 'Regression':
            valid_metrics = rho_metric(
                self.validation_step_outputs,
            )
        if self.task_type == 'Classification':
            valid_metrics = acc_metric(
                self.validation_step_outputs,
            )
        if self.task_type =='mutilabel':
            valid_metrics = sub_location_metrics(
                self.validation_step_outputs,
            )
        for key, value in valid_metrics.items():
            if isinstance(value, dict):
                pass
            else:
                key = f"val_{key}"
                self.log(key, value, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        tokens, labels, _ = batch
        classfier_logits, _ = self(tokens)
        classfier_logits = classfier_logits.squeeze()
        if self.task_type == 'Regression':
            loss = nn.MSELoss(reduction='mean')(classfier_logits.squeeze(), labels)  # axis=0
        if self.task_type == 'Classification':
            loss = nn.CrossEntropyLoss(reduction='mean')(classfier_logits.squeeze(), labels.long())
        if self.task_type == 'mutilabel':
            loss = nn.BCEWithLogitsLoss(reduction='mean')(classfier_logits.squeeze(), labels.float())
        
        self.log("test_loss", loss)
        result = {"predictions": classfier_logits, "targets": labels}
        self.test_step_output.append(result)
        return loss
        
    def on_test_epoch_end(self):
        # valid
        if self.task_type == 'Regression':
            valid_metrics = rho_metric(
                self.test_step_output,
            )
        if self.task_type == 'Classification':
            valid_metrics = acc_metric(
                self.test_step_output,
            )
        if self.task_type =='mutilabel':
            valid_metrics = sub_location_metrics(
                self.test_step_output,
            )
        for k1, value in valid_metrics.items():
            if isinstance(value, dict):
                for k2,v2 in value.items():
                    key = f"{k1}_{k2}"
                    self.log(key, v2, prog_bar=True, on_epoch=True)
            else:
                key = f"test_{k1}"
                print("test performance is:",f"{key}:{value}")
                self.log(key, value, prog_bar=True, on_epoch=True)
        self.test_step_output.clear()


    def configure_optimizers(self):
        if self.config.frozen:
            for param in self.model.parameters():
                param.requires_grad = False  # 冻结预训练模型的参数
            #train_layer = [10,11]
            """
            train_layer = []
            for name, param in self.model.named_parameters():
                #print(name)
                parts = name.split('.')
                if len(parts) >=2  and parts[1].isdigit():
                    #layer_type = parts[1]  # "forward_layers" 或 "backward_layers"
                        layer_idx = int(parts[1])  # 层号（0,1,2,...）
                        # 冻结前 num_freeze_layers 层
                        if layer_idx in train_layer:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                else:
                    param.requires_grad = False
            group_params = [
                 {
                    "params": self.model.parameters(),
                    "lr": self.config.pretrained_lr,
                },
                {
                    "params": self.encoder.parameters(),
                    "lr": self.config.encode_lr,
                },
            ]
        else:
            group_params = [
                {
                    "params": self.model.parameters(),
                    "lr": self.config.pretrained_lr,
                },
                {
                    "params": self.encoder.parameters(),
                    "lr": self.config.encode_lr,
                },
        """
        group_params = [
                {"params": self.encoder.parameters(),
                "lr": self.config.encode_lr,
                }
            ]
        optimizer = torch.optim.AdamW(
            group_params,
            #lr=self.optimizer_config.pretrained_lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.adam_betas
        )
            
        scheduler = {
            'scheduler':WarmupLinearSchedule(optimizer,
                                             self.config.warmup_steps,
                                             self.config.max_steps),
            'interval': 'step',
        }
        return [optimizer], [scheduler]


    def on_save_checkpoint(self, checkpoint:dict)-> dict:
        # 保存预训练模型的权重
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        return checkpoint
    
    def from_pretrained(
            self,
            model_config,
            model_path,
    ):
        model = ESM2(
            vocab=self.vocab,
            model_config=model_config,
            token_dropout=True,
        )
        #model = nn.DataParallel(model)
        """
        model.load_state_dict(torch.load(
            model_path,
            map_location="cuda")['model_state_dict'],
            strict=True
        """
        # need sava the model and para
        # for train:
        """
        state_dict = torch.load(data_config.pretrained_model_path, map_location='cpu')
        new_state_dict = state_dict["model_state_dict"]
        fin_state_dict = {k.replace('module.', ''): v for k, v in new_state_dict.items()}
        model.load_state_dict(fin_state_dict, strict=True)
        """
        # for train fintune by RNA_ESM directly
        #checkpoint = torch.load(data_config.pretrained_model_path, map_location=torch.device('cuda'))
        #model.load_state_dict(checkpoint, strict=True)
        #for name, param in model.named_parameters():
        #    print(f'Parameter name: {name}')
        #    print(f'Parameter shape: {param.shape}')
        #    print(f'Parameter value: {param.data}')

        checkpoint = torch.load(model_path,map_location=torch.device('cuda'))
        if 'model_state_dict' not in checkpoint:
            raise KeyError("checkpoint not have 'model_state_dict' key")
        
        # 添加参数加载检查
        try:
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            print("模型参数加载成功")
        except RuntimeError as e:
            print(f"模型参数加载失败: {e}")
            raise

        """
        model.load_state_dict(torch.load(
            data_config.pretrained_model_path,
            map_location="cpu")['state_dict'],
                              strict=True)
        """

        # for test
        # state_dict = torch.load(data_config.pretrained_model_path, map_location='cpu')
        # new_state_dict = state_dict["state_dict"]
        # fin_state_dict = {k.replace('pretrained_model.', ''): v for k, v in new_state_dict.items()}
        # model.load_state_dict(fin_state_dict, strict=True)

        # for test
        # model.load_state_dict(torch.load(
        ##    data_config.pretrained_model_path,
        #    map_location="cpu")['state_dict'], strict=True)

        return model

