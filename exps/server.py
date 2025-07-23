import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw
import copy
import hashlib
import os
import pickle
import dgl
import copy
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops
from models import Split_model
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Server():
    def __init__(self, model, device, shared_loader=None, pretrain_loader=None, args=None):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}
        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

        self.device = args.device  
        self.args = args  
        self.model_cache = []
        self.customized_params = {}

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_encoders = []
        self.uploaded_filters = []
        self.uploaded_feature_extractor = []

        self.clients = []
        self.selected_clients = []

        self.uploaded_model_gs = []

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.Budget = []

        self.shared_loader = shared_loader
        self.shared_preprocessed_batches = []
        self.pretrain_loader = pretrain_loader
        self.pretrain_preprocessed_batches = []

        self.current_mean = torch.zeros(args.hidden)
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)
        self.local_consensus = nn.Parameter(Variable(torch.zeros(args.hidden)))
        self.opt_local_consensus = torch.optim.SGD([self.local_consensus], lr=self.args.lr)

        self.pm_train = []
        self.lamda = 0
        self.train_samples = 0
        self.track = []
        self.tau = args.tau_weight
        self.momentum = args.momentum
        self.global_consensus = None
        self.global_logits = None    
        self.global_feat_map = None 

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def receive_models_SSP(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_SSP(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters_SSP(w, client_model)

    def add_parameters_SSP(self, w, client_model):
        w = 1 / len(self.selected_clients)
        for (server_name, server_param), (client_name, client_param) in zip(self.global_model.named_parameters(), client_model.named_parameters()):
            if 'encoder' in server_name and 'atom' not in server_name:
                server_param.data += client_param.data.clone() * w
    def send_models_SSP(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters_SSP(self.global_model)

    def local_train_pre(self, local_epoch): 
        if isinstance(self.model, Split_model):
            train_stats = train_gc_pretrain(self, self.model, local_epoch, self.args.device, self.pretrain_preprocessed_batches)
        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()
    def local_train_shared(self, local_epoch):
        logits = []  
        feat_map = []  
        if isinstance(self.model, Split_model):
            train_stats = train_gc_shared(self, self.model, local_epoch, self.args.device, self.shared_preprocessed_batches)
        self.train_stats = train_stats
        logits = train_stats['all_logits']        
        feat_map = train_stats['all_feat_maps']

        self.weightsNorm = torch.norm(flatten(self.W)).item()
        return logits, feat_map
    
    def local_train_kd(self, all_logits, all_featmaps,all_hard_sample_indices, local_epoch):
        aggregated_featmaps = []
        non_zero_counts = [] 
        for client_idx in range(len(all_featmaps)):
            feat_map = all_featmaps[client_idx]  
            hard_sample_indices = all_hard_sample_indices[client_idx]  
            total_nodes = hard_sample_indices.shape[0]  
            padded_feat_map = torch.zeros(total_nodes, feat_map.shape[1], device=feat_map.device)  
            padded_feat_map[hard_sample_indices.bool()] = feat_map  
            non_zero_count = torch.count_nonzero(padded_feat_map).item()  
            non_zero_counts.append(non_zero_count)  
            aggregated_featmaps.append(padded_feat_map)
        aggregated_shapes = [aggregated_feat_map.shape for aggregated_feat_map in aggregated_featmaps]
        #print("Processed Feature Maps Shapes:", aggregated_shapes)
        #print("Non-zero counts in each feature map:", non_zero_counts)

        aggregated_logits = torch.mean(torch.stack(all_logits), dim=0)  
        aggregated_featmap = torch.mean(torch.stack(aggregated_featmaps), dim=0)
        self.global_logits = aggregated_logits
        self.global_feat_map = aggregated_featmap
        train_step1=train_kd(self, self.model, local_epoch, self.args.device, self.shared_preprocessed_batches, aggregated_logits, aggregated_featmap)
        train_step2=train_data(self, self.model, local_epoch, self.args.device, self.shared_preprocessed_batches)
        logits = train_step2['all_logits']
        hard_sample_feat_maps = train_step2['hard_sample_feat_maps']
        hard_sample_node_indices = train_step2['hard_sample_node_indices']
        return logits, hard_sample_feat_maps, hard_sample_node_indices

def train_gc_shared(server, model, local_epoch, device, shared_preprocessed_batches):
    losses_train, accs_train = [], []
    all_logits = []  
    all_feat_maps = []  
    
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0
        
        for batch in shared_preprocessed_batches:
            e, u, g, length, label, _ = batch  
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, 
                                        betas=(0.9, 0.999), 
                                        weight_decay=5e-4)
            
            optimizer.zero_grad()
            server.current_mean.zero_()
            server.num_batches_tracked.zero_()
            
            x = g.ndata['feat'].to(device)
            e, u, g, length, label = [t.to(device) for t in [e, u, g, length, label]]
            
            rep, pred, feat_map = model(e, u, g, length, x, is_shared=True)
            
            current_mean = torch.mean(rep, dim=0).to(device)
            server.current_mean = server.current_mean.to(device)
            server.local_consensus = server.local_consensus.to(device)
            if server.num_batches_tracked is not None:
                server.num_batches_tracked.add_(1)
            server.current_mean = (1 - server.momentum) * server.current_mean + server.momentum * current_mean
            
            if server.global_consensus is not None:
                mse_loss = 0.5 * torch.mean(0.5* (server.current_mean - server.global_consensus)**2)
                pred_pgpa = server.model.head(rep + server.local_consensus)
                loss = server.model.loss(pred_pgpa, label)
                loss = loss + mse_loss * server.tau
            else:
                pred_pgpa = server.model.head(rep)
                loss = server.model.loss(pred_pgpa, label)

            current_logits = pred_pgpa.detach().clone()
            all_logits.append(current_logits)
            all_feat_maps.append(feat_map.detach())
            pred1 = torch.softmax(pred_pgpa, dim=1)
            pred_labels = torch.argmax(pred1, dim=1)
            #print("Label shape:", label.shape)
            #print("Pred_labels shape:", pred_labels.shape)
            correct_predictions = pred_labels.eq(label).sum().item()
            acc_sum += correct_predictions
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            server.opt_local_consensus.step()
            server.current_mean.detach_()
            total_loss += loss.item() * label.size(0)
            ngraphs += label.size(0)
        total_loss /= ngraphs
        acc = acc_sum / ngraphs
        '''
        print(f"Epoch [{epoch+1}/{local_epoch}] | "
              f"Loss: {total_loss:.4f} | "
              f"Accuracy: {acc*100:.2f}%")
        losses_train.append(total_loss)
        '''
        accs_train.append(acc)
        
    return {'trainingLosses': losses_train,
            'trainingAccs': accs_train,
            'all_logits': torch.cat(all_logits, dim=0),  
            'all_feat_maps': torch.cat(all_feat_maps, dim=0)  
    }
def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def train_kd(server, model, local_epoch, device, shared_preprocessed_batches, aggregated_logits, aggregated_featmap):
    losses_train, accs_train = [], []
    aggregated_logits = aggregated_logits.to(device)
    aggregated_featmap = aggregated_featmap.to(device)
    
    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.

        all_preds = []
        all_feat_maps = []
        all_labels = []
        sum_ce_loss = 0.0
        
        for batch in shared_preprocessed_batches:
            e, u, g, length, label, _ = batch  
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, 
                                        betas=(0.9, 0.999), 
                                        weight_decay=5e-4)
            
            optimizer.zero_grad()
            server.current_mean.zero_()
            server.num_batches_tracked.zero_()
            x = g.ndata['feat'].to(device)
            e, u, g, length, label = [t.to(device) for t in [e, u, g, length, label]]
            rep, pred, feat_map = model(e, u, g, length, x, is_shared=True)
            pred_pgpa = server.model.head(rep)
            all_preds.append(pred_pgpa)
            all_feat_maps.append(feat_map)
            all_labels.append(label)
            loss = server.model.loss(pred_pgpa, label) 
            sum_ce_loss += loss          
        if not all_preds:  
            continue
        
        all_labels = torch.cat(all_labels, dim=0)    
        all_preds = torch.cat(all_preds, dim=0)
        all_feat_maps = torch.cat(all_feat_maps, dim=0)
        mask = (aggregated_featmap.abs().sum(dim=1) != 0).float() 

        temperature = 1.0

        #loss_logits = nn.MSELoss()(all_preds, aggregated_logits)
        loss_logits = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(all_preds/temperature, dim=1),
                F.softmax(aggregated_logits/temperature, dim=1)
                )*temperature**2

        #loss_feat = nn.MSELoss(reduction='none')(all_feat_maps, aggregated_featmap)  
        loss_feat = nn.KLDivLoss(reduction='none')(
                F.log_softmax(all_feat_maps/temperature, dim=1),
                F.softmax(aggregated_featmap/temperature, dim=1)
        )*temperature**2
        loss_feat = loss_feat.sum(dim=1) 
        #print("loss_feat shape:", loss_feat.shape)
        #print("mask shape:", mask.shape)
        loss_feat = (loss_feat * mask).sum()/mask.sum().clamp(min=1.0) 

        
        total_loss = 0.0*loss_logits + 0.0*loss_feat + 1.0*sum_ce_loss 
            
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            preds = all_preds.argmax(dim=1)            
            correct = (preds == all_labels).sum().item()
            total = all_labels.size(0)
            acc = correct / total * 100.0  

        epoch_loss = total_loss.item()
        losses_train.append(epoch_loss)
        accs_train.append(acc)
        #print(f"Server acc: {acc:.2f}%")
        
    return {'trainingLosses': losses_train, 'dummyAccs': [0]*len(losses_train)}  

def train_data(server, model, local_epoch, device, shared_preprocessed_batches):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    all_preds = []
    all_feat_maps = []
    all_losses      = []
    all_lengths     = []
    
    with torch.no_grad(): 
        for batch in shared_preprocessed_batches:
            e, u, g, length, label, _ = batch  
            x = g.ndata['feat'].to(device)
            e, u, g, length, label = [t.to(device) for t in [e, u, g, length, label]]
            rep, pred, feat_map = model(e, u, g, length, x, is_shared=True)
            pred_pgpa = server.model.head(rep)
            all_preds.append(pred_pgpa)
            all_feat_maps.append(feat_map)
            all_lengths.append(length)
            sample_loss = criterion(pred_pgpa, label)
            all_losses.append(sample_loss)
            
    all_preds = torch.cat(all_preds, dim=0)     
    all_feat_maps = torch.cat(all_feat_maps, dim=0)
    all_losses    = torch.cat(all_losses, dim=0)    
    all_lengths = torch.cat(all_lengths, dim=0)  
    
    min_loss = all_losses.min()
    max_loss = all_losses.max()
    normalized_losses = (all_losses - min_loss) / (max_loss - min_loss)
    threshold = 0.2
    hard_sample_indices = normalized_losses > threshold  

    total_samples = hard_sample_indices.numel()
    num_hard = hard_sample_indices.sum().item()
    #print(f"Total samples: {total_samples}")
    #print(f"Number of hard samples: {num_hard}")

    node_difficulty_mask = torch.zeros(all_lengths.sum().item(), device=device) 
    start_idx = 0
    for idx, is_hard in enumerate(hard_sample_indices):
        graph_length = all_lengths[idx].item()
        end_idx = start_idx + graph_length
        if is_hard:
            node_difficulty_mask[start_idx:end_idx] = 1
        start_idx = end_idx

    #print("node_difficulty_mask shape:", node_difficulty_mask.shape)
    #print("node_difficulty_mask contents:\n", node_difficulty_mask)

    hard_sample_feat_maps = all_feat_maps[node_difficulty_mask.bool()]
    #print("Hard Sample Feature Maps Shape:", hard_sample_feat_maps.shape)
    
    
    return {
            'all_logits': all_preds,
            'hard_sample_feat_maps': hard_sample_feat_maps,
            'hard_sample_node_indices': node_difficulty_mask,
        }


def train_gc_pretrain(server, model, local_epoch, device, preprocessed_batches):
    losses_train, accs_train = [], []

    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0
        

        for batch in preprocessed_batches:
            e, u, g, length, label, _ = batch
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
            optimizer.zero_grad()
            server.current_mean.zero_()
            server.num_batches_tracked.zero_()
            x = g.ndata['feat']
            rep, pred = model(e, u, g, length, x,  is_shared=False)
            current_mean = torch.mean(rep, dim=0).to(device)
            server.current_mean = server.current_mean.to(device)
            server.local_consensus = server.local_consensus.to(device)
            if server.num_batches_tracked is not None:
                server.num_batches_tracked.add_(1)
            server.current_mean = (1 - server.momentum) * server.current_mean + server.momentum * current_mean
            if server.global_consensus is not None:
                mse_loss = torch.mean(0.5 * (server.current_mean - server.global_consensus)**2)
                pred_pgpa = server.model.head(rep + server.local_consensus)
                pred_pgpa = server.model.mlp(pred_pgpa)
                #pred_pgpa = server.model.lin(pred_pgpa)
                loss = server.model.loss(pred_pgpa, label)
                loss = loss + mse_loss * server.tau
            else:
                pred_pgpa = server.model.head(rep)
                pred_pgpa = server.model.mlp(pred_pgpa)
                #pred_pgpa = server.model.lin(pred_pgpa)
                loss = server.model.loss(pred_pgpa, label)

            pred1 = torch.softmax(pred_pgpa, dim=1)
            pred_labels = torch.argmax(pred1, dim=1)
            correct_predictions = pred_labels.eq(label).sum().item()
            acc_sum += correct_predictions
            optimizer.zero_grad()      
            loss.backward()
            optimizer.step()
            server.opt_local_consensus.step()
            server.current_mean.detach_()
            total_loss += loss.item() * label.size(0)
            ngraphs += label.size(0)
        total_loss /= ngraphs
        acc = acc_sum / ngraphs
        #acc=0
        #print("acc:", acc)

    return {acc}