import hashlib
import os
import pickle
import torch
import dgl
import copy
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops
from models import Split_model
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
def hash_batch(batch):
    hash_obj = hashlib.sha256()
    for data in batch.to_data_list():
        hash_obj.update(data.edge_index.cpu().numpy().tobytes())
        if data.x is not None:
            hash_obj.update(data.x.cpu().numpy().tobytes())
    return hash_obj.hexdigest()

def collate_pyg_to_dgl(batch):
    dir_path = os.path.join(os.path.dirname(__file__), '..', 'preprocessed_batch')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    #file_name = os.path.join(dir_path, f"{hash_batch(batch)}.pkl")
    filtered_data_list = [data for data in batch.to_data_list() if data.edge_index.size(1) > 0]
    valid_indices = [i for i, data in enumerate(batch.to_data_list()) if data.edge_index.size(1) > 0]
    max_nodes = max([data.num_nodes for data in filtered_data_list], default=0)
    E = []
    U = []
    lengths = []
    for data in filtered_data_list:
        N = data.num_nodes
        edge_index, _ = add_self_loops(data.edge_index)
        adj = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0)
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj
        e, u = torch.linalg.eigh(L)

        pad_e = e.new_zeros([max_nodes])
        pad_e[:N] = e

        pad_u = u.new_zeros([max_nodes, max_nodes])
        pad_u[:N, :N] = u

        E.append(pad_e)
        U.append(pad_u)
        lengths.append(N)

    E = torch.stack(E)
    U = torch.stack(U)
    lengths = torch.tensor(lengths)

    graphs = []
    for data in filtered_data_list:
        edge_index = data.edge_index.cpu()
        num_nodes = data.num_nodes if data.x is not None else (edge_index.max().item() + 1)
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        if data.x is not None:
            g.ndata['feat'] = data.x.cpu()
        graphs.append(g)

    g = dgl.batch(graphs)

    #with open(file_name, 'wb') as f:
        #pickle.dump((E, U, g, lengths), f)

    return E, U, g, lengths, valid_indices

class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args
        self.device = args.device

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

        self.train_preprocessed_batches = []
        self.test_preprocessed_batches = []
        self.val_preprocessed_batches = []
        self.shared_preprocessed_batches = []
        self.cross_preprocessed_batches = []
        self.pm_train = []
        self.lamda = 0
        self.train_samples = 0
        self.track = []
        self.tau = args.tau_weight
        self.momentum = args.momentum
        self.global_consensus = None

        self.current_mean = torch.zeros(args.hidden)
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)
        self.local_consensus = nn.Parameter(Variable(torch.zeros(args.hidden)))
        self.opt_local_consensus = torch.optim.SGD([self.local_consensus], lr=self.args.lr)

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        if isinstance(self.model, Split_model):
            train_stats = train_gc_SSP(self, self.model, self.dataLoader, local_epoch, self.args.device, self.train_preprocessed_batches)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()
        return train_stats

    def evaluate(self):
        return eval_gc_test_SSP(self.model, self.args.device, self)

    def cross_domain_test(self):
        return eval_corss_test(self.model, self.args.device, self)

    def set_parameters_SSP(self, global_model):
        for (new_name, new_param), (old_name, old_param) in zip(global_model.named_parameters(), self.model.named_parameters()):
            if 'encoder' in new_name and 'atom' not in new_name:
                old_param.data = new_param.data.clone()

    def local_train_shared(self, local_epoch, global_logits, global_feat_map, hard_sample_indices=None):
        if hard_sample_indices is not None:
            device = self.args.device
            mask = hard_sample_indices.to(device).bool()              
            feat_dim = global_feat_map.size(1)                               
            full_feat_map = torch.zeros((mask.size(0), feat_dim), device=device)
            full_feat_map[mask] = global_feat_map
            global_feat_map = full_feat_map
            #校验填充效果
            #non_zero_rows = (full_feat_map.abs().sum(dim=1) != 0)
            #num_non_zero_rows = non_zero_rows.sum().item()
            #print(f"有 {num_non_zero_rows} 行不全为 0")

        if isinstance(self.model, Split_model):
            train_stats = train_gc_shared(self, self.model, local_epoch, self.args.device, self.shared_preprocessed_batches, global_logits, global_feat_map, hard_sample_indices)
        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()
        return train_stats
    
    def local_train_logits_featuremaps(self, local_epoch):
        if isinstance(self.model, Split_model):
            train_stats = train_gc_data(self, self.model, local_epoch, self.args.device, self.shared_preprocessed_batches)
        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()
        logits = train_stats['all_logits']
        feat_map = train_stats['hard_sample_feat_maps']
        hard_sample_indices = train_stats['hard_sample_node_indices']
        return logits, feat_map, hard_sample_indices
def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])

def train_gc_SSP(client, model, dataloaders, local_epoch, device, train_preprocessed_batches):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']

    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0
        

        for batch in train_preprocessed_batches:
            e, u, g, length, label, _ = batch
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
            optimizer.zero_grad()
            client.current_mean.zero_()
            client.num_batches_tracked.zero_()
            x = g.ndata['feat']
            rep, pred = model(e, u, g, length, x,  is_shared=False)
            current_mean = torch.mean(rep, dim=0).to(device)
            client.current_mean = client.current_mean.to(device)
            client.local_consensus = client.local_consensus.to(device)
            if client.num_batches_tracked is not None:
                client.num_batches_tracked.add_(1)
            client.current_mean = (1 - client.momentum) * client.current_mean + client.momentum * current_mean
            if client.global_consensus is not None:
                mse_loss = torch.mean(0.5 * (client.current_mean - client.global_consensus)**2)
                pred_pgpa = client.model.head(rep + client.local_consensus)
                pred_pgpa = client.model.mlp(pred_pgpa)
                #pred_pgpa = client.model.lin(pred_pgpa)
                loss = client.model.loss(pred_pgpa, label)
                loss = loss + mse_loss * client.tau
            else:
                pred_pgpa = client.model.head(rep)
                pred_pgpa = client.model.mlp(pred_pgpa)
                #pred_pgpa = client.model.lin(pred_pgpa)
                loss = client.model.loss(pred_pgpa, label)

            pred1 = torch.softmax(pred_pgpa, dim=1)
            pred_labels = torch.argmax(pred1, dim=1)
            correct_predictions = pred_labels.eq(label).sum().item()
            acc_sum += correct_predictions
            optimizer.zero_grad()      
            loss.backward()
            optimizer.step()
            client.opt_local_consensus.step()
            client.current_mean.detach_()
            total_loss += loss.item() * label.size(0)
            ngraphs += label.size(0)
        total_loss /= ngraphs
        acc = acc_sum / ngraphs
        #print("acc:", acc)
        loss_v, acc_v = eval_gc_val_SSP(model, device, client)
        loss_tt, acc_tt = eval_gc_test_SSP(model, device, client)
        #print(f"Final Test Loss: {loss_tt}, Acc: {acc_tt}")
        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 
            'valLosses': losses_val,'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}



def eval_gc_test(model, device, client):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    test_preprocessed_batches = client.test_preprocessed_batches
    for batch in test_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x = g.ndata['feat']
        e, u, g, length, label = e.to(device), u.to(device), g.to(device), length.to(device), label.to(device)
        with torch.no_grad():
            rep, pred = client.model(e, u, g, length, x)
            print("pred:", pred)
            print("label:", label)
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs

def eval_gc_val(model, device, client):

    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    val_preprocessed_batches = client.val_preprocessed_batches
    for batch in val_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x= g.ndata['feat']
        e, u, g, length, label, x = e.to(device), u.to(device), g.to(device), length.to(device), label.to(
            device), x.to(device)
        with torch.no_grad():
            pred, rep, rep_base = client.model(e, u, g, length, x, is_rep=True, context=client.context)
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss / ngraphs, acc_sum / ngraphs

class clientAvgSSP(Client_GC):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.tau = args.tau_weight
        self.momentum = args.momentum
        self.global_consensus = None

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.current_mean = torch.zeros_like(rep[0])
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)

        self.local_consensus = nn.Parameter(Variable(torch.zeros_like(rep[0])))
        self.opt_local_consensus = torch.optim.SGD([self.local_consensus], lr=self.learning_rate)

    def train_gc_SSP(client, model, dataloaders, local_epoch, device, train_preprocessed_batches):
        losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
        train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']

        for epoch in range(local_epoch):
            model.train()
            total_loss = 0.
            ngraphs = 0
            acc_sum = 0

            for batch in train_preprocessed_batches:
                e, u, g, length, label, _ = batch
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                              weight_decay=5e-4)
                optimizer.zero_grad()
                client.current_mean.zero_()
                client.num_batches_tracked.zero_()
                x = g.ndata['feat']
                rep, pred = model(e, u, g, length, x)
                current_mean = torch.mean(rep, dim=0).to(device)
                client.current_mean = client.current_mean.to(device)
                client.local_consensus = client.local_consensus.to(device)
                if client.num_batches_tracked is not None:
                    client.num_batches_tracked.add_(1)
                client.current_mean = (1 - client.momentum) * client.current_mean + client.momentum * current_mean
                if client.global_consensus is not None:
                    mse_loss = torch.mean(0.5 * (client.current_mean - client.global_consensus) ** 2)
                    pred_pgpa = client.model.head(rep + client.local_consensus)
                    loss = client.model.loss(pred_pgpa, label)
                    loss = loss + mse_loss * client.tau
                else:
                    pred_pgpa = client.model.head(rep)
                    loss = client.model.loss(pred_pgpa, label)

                pred1 = torch.softmax(pred_pgpa, dim=1)
                pred_labels = torch.argmax(pred1, dim=1)
                correct_predictions = pred_labels.eq(label).sum().item()
                acc_sum += correct_predictions
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                client.opt_local_consensus.step()
                client.current_mean.detach_()
                total_loss += loss.item() * label.size(0)
                ngraphs += label.size(0)
            total_loss /= ngraphs
            acc = acc_sum / ngraphs
            loss_v, acc_v = eval_gc_val_SSP(model, device, client)
            loss_tt, acc_tt = eval_gc_test_SSP(model, device, client)
            losses_train.append(total_loss)
            accs_train.append(acc)
            losses_val.append(loss_v)
            accs_val.append(acc_v)
            losses_test.append(loss_tt)
            accs_test.append(acc_tt)

        return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 
                'valLosses': losses_val,'valAccs': accs_val,
                'testLosses': losses_test, 'testAccs': accs_test}


def eval_gc_test_SSP(model, device, client):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    test_preprocessed_batches = client.test_preprocessed_batches
    for batch in test_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x = g.ndata['feat']
        e, u, g, length, label = e.to(device), u.to(device), g.to(device), length.to(device), label.to(device)
        with torch.no_grad():
            _, pred = client.model(e, u, g, length, x)
            pred = client.model.mlp(pred)
            #pred = client.model.lin(pred)
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs


def eval_corss_test(model, device, client):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    cross_preprocessed_batches = client.cross_preprocessed_batches
    for batch in cross_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x = g.ndata['feat']
        e, u, g, length, label = e.to(device), u.to(device), g.to(device), length.to(device), label.to(device)
        with torch.no_grad():
            _, pred = client.model(e, u, g, length, x)
            #pred = client.model.mlp(pred)
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs


def eval_gc_val_SSP(model, device, client):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    val_preprocessed_batches = client.val_preprocessed_batches
    for batch in val_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x= g.ndata['feat']
        e, u, g, length, label, x = e.to(device), u.to(device), g.to(device), length.to(device), label.to(
            device), x.to(device)
        with torch.no_grad():
            rep, pred = client.model(e, u, g, length, x)
            pred1 = torch.softmax(pred, dim=1)
            pred1 = client.model.mlp(pred1)
            pred_labels = torch.argmax(pred1, dim=1)
            correct_predictions = pred_labels.eq(label).sum().item()
            acc_sum += correct_predictions
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss / ngraphs, acc_sum / ngraphs

def train_gc_shared(client, model, local_epoch, device, shared_preprocessed_batches, global_logits, global_feat_map, hard_sample_indices=None):

    losses_train, accs_train = [], []
    
    model.to(device)
    global_logits = global_logits.to(device)
    global_feat_map = global_feat_map.to(device)
    #print("global_logits shape:", global_logits.shape)
    #print("global_feat_map shape:", global_feat_map.shape)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)  
    
    for epoch in range(local_epoch):
        model.train()
        batch_start_idx = 0
        node_start_idx = 0
        total_loss_sum = 0.0
        total_samples = 0
        correct_sum = 0

        for batch in shared_preprocessed_batches:
            e, u, g, length, label, _ = batch
            batch_size = label.size(0)
            num_nodes_in_batch = length.sum().item()

            batch_global_logits = global_logits[batch_start_idx: batch_start_idx + batch_size]
            batch_global_feat_map = global_feat_map[node_start_idx: node_start_idx + num_nodes_in_batch]

            if hard_sample_indices is not None:
                batch_hard_sample_indices = hard_sample_indices[node_start_idx: node_start_idx + num_nodes_in_batch]

            batch_start_idx += batch_size
            node_start_idx += num_nodes_in_batch

            optimizer.zero_grad()

            x = g.ndata['feat'].to(device)
            e, u, g, length, label = [t.to(device) for t in [e, u, g, length, label]]
            #print("lenght: ", length)

            client.current_mean.zero_()
            if client.num_batches_tracked is not None:
                client.num_batches_tracked.zero_()
                client.num_batches_tracked.add_(1)
            else:
                client.num_batches_tracked = torch.tensor(1.0, device=device)

            rep, pred, feat_map = model(e, u, g, length, x, is_shared=True)
            current_mean = torch.mean(rep, dim=0).to(device)

            client.current_mean = client.current_mean.to(device)
            client.local_consensus = client.local_consensus.to(device)

            with torch.no_grad():
                client.current_mean[:] = (1 - client.momentum) * client.current_mean + client.momentum * current_mean

            if client.global_consensus is not None:
                mse_consensus = torch.mean(0.5 * (client.current_mean - client.global_consensus.to(device)) ** 2)
                pred_pgpa = client.model.head(rep + client.local_consensus)
                loss_cls = client.model.loss(pred_pgpa, label)
                loss = loss_cls + mse_consensus * client.tau
            else:
                pred_pgpa = client.model.head(rep)
                loss_cls = client.model.loss(pred_pgpa, label)
                loss = loss_cls

            temperature = 1.0

            loss_logits = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(pred_pgpa/temperature, dim=1),
                F.softmax(batch_global_logits/temperature, dim=1)
                )* temperature**2
            
            if hard_sample_indices is None:
                loss_feat = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(feat_map, dim=1), F.softmax(batch_global_feat_map, dim=1))
                #print("loss_feat:", loss_feat)
                
            else:
                mask = (batch_hard_sample_indices != 0).float()
                loss_feat = nn.KLDivLoss(reduction='none')(
                    F.log_softmax(feat_map/temperature, dim=1),
                    F.softmax(batch_global_feat_map/temperature, dim=1)
                    ) * temperature**2
                loss_feat = loss_feat.sum(dim=1) 
                loss_feat = (loss_feat * mask).sum()/mask.sum().clamp(min=1.0)
                #print("loss_feat:", loss_feat)

            total_loss = 0.0 * loss_logits + 0.0 * loss_feat + 1.0 * loss_cls

            total_loss.backward()
            optimizer.step()

            preds_class = torch.argmax(pred_pgpa, dim=1)
            correct_sum += (preds_class == label).sum().item()
            total_samples += batch_size
            total_loss_sum += total_loss.item()

        avg_loss = total_loss_sum / len(shared_preprocessed_batches)
        avg_acc = correct_sum / total_samples if total_samples > 0 else 0
        losses_train.append(avg_loss)
        accs_train.append(avg_acc)
        #print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train}


'''
def train_gc_data(client, model, local_epoch, device, shared_preprocessed_batches):
    for epoch in range(local_epoch):
        model.eval()
        
        all_preds = []
        all_feat_maps = []
        
        for batch in shared_preprocessed_batches:
            e, u, g, length, label, _ = batch  
            x = g.ndata['feat'].to(device)
            e, u, g, length, label = [t.to(device) for t in [e, u, g, length, label]]
            rep, pred, feat_map = model(e, u, g, length, x, is_shared=True)
            pred_pgpa = client.model.head(rep)
            all_preds.append(pred_pgpa)
            all_feat_maps.append(feat_map)          
            
        all_preds = torch.cat(all_preds, dim=0)
        all_feat_maps = torch.cat(all_feat_maps, dim=0)
        
    return {'all_logits': all_preds, 'all_feat_maps': all_feat_maps}'
'''

def train_gc_data(client, model, local_epoch, device, shared_preprocessed_batches):
    criterion = nn.CrossEntropyLoss(reduction='none')  
    all_preds = []
    all_feat_maps = []
    all_losses = []  
    all_lengths = []  
    all_batch_data = []  
    
    for epoch in range(local_epoch):
        model.eval()  
        
        for batch in shared_preprocessed_batches:
            e, u, g, length, label, batch_data = batch  
            x = g.ndata['feat'].to(device)
            e, u, g, length, label = [t.to(device) for t in [e, u, g, length, label]]

            rep, pred, feat_map = model(e, u, g, length, x, is_shared=True)
            pred_pgpa = client.model.head(rep)  
            
            sample_losses = criterion(pred_pgpa, label)  
            all_preds.append(pred_pgpa)
            all_feat_maps.append(feat_map)
            all_losses.append(sample_losses) 
            all_lengths.append(length)
            all_batch_data.append(batch_data)  
        
        all_preds = torch.cat(all_preds, dim=0)
        all_feat_maps = torch.cat(all_feat_maps, dim=0)
        all_lengths = torch.cat(all_lengths, dim=0)
        all_losses = torch.cat(all_losses, dim=0)
        
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

