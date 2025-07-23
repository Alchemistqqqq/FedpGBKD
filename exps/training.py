import pandas as pd
from client import collate_pyg_to_dgl
import torch
import numpy as np
import sys

def get_tensor_memory(tensor):
    # 计算张量的内存占用 (以字节为单位)
    return tensor.element_size() * tensor.numel()

def process_loader(loader, device):
    preprocessed_batches = []
    for batch in loader:
        batch.to(device)
        e, u, g, length, valid_indices = collate_pyg_to_dgl(batch)
        valid_labels = batch.y[valid_indices].to(device)
        preprocessed_batches.append((e.to(device), u.to(device), g.to(device), length.to(device), valid_labels, len(valid_indices)))
    return preprocessed_batches

def run_fedSSP(args, clients, server, COMMUNICATION_ROUNDS, local_epoch, samp=None, frac=1.0, summary_writer=None):
    device = torch.device('cuda:0')
    if samp is None:
        sampling_fn = server.randomSample_clients
        frac = 1.0
    
    # Server端预处理
    server.shared_preprocessed_batches = process_loader(server.shared_loader, device)
    server.pretrain_preprocessed_batches = process_loader(server.pretrain_loader, device)
    server.uploaded_logits = []    
    server.uploaded_feat_maps = [] 

    for client in clients:
        dataloaders = client.dataLoader
        train_loader, val_loader, test_loader, shared_loader, corss_loader = dataloaders['train'], dataloaders['val'], dataloaders['test'], dataloaders['shared'], dataloaders['cross']
        #train_loader, val_loader, test_loader, shared_loader = dataloaders['train'], dataloaders['val'], dataloaders['test'], dataloaders['shared']
        client.train_preprocessed_batches = process_loader(train_loader, device)
        client.test_preprocessed_batches = process_loader(test_loader, device)
        client.val_preprocessed_batches = process_loader(val_loader, device)
        client.shared_preprocessed_batches = process_loader(shared_loader, device)
        client.cross_preprocessed_batches = process_loader(corss_loader, device)

        server.clients = clients
        server.selected_clients = clients
        client.train_samples = len(train_loader)
    
    frame = pd.DataFrame()
    accuracies = []
    #'''
    for client in clients:
        loss, acc = client.cross_domain_test()
        accuracies.append(acc)
        frame.loc[client.name, 'before_cross_acc'] = acc
        #print(f"Client: {client.name}, Acc: {acc:.6f}")
    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        frame.loc['average', 'before_cross_avgacc'] = avg_acc
        print(f"Average Acc: {avg_acc:.6f}")
    #'''

    # Pretrain -> 发送 global_logits 和 global_feat_map
    server.local_train_pre(local_epoch *100)
    global_logits, global_feat_map = server.local_train_shared(local_epoch)
    server.global_logits = global_logits       
    server.global_feat_map = global_feat_map 
    server.global_hard_sample_indices = None
    #print("global_logits shape:", global_logits.shape)  
    #print("global_feat_map shape:", global_feat_map.shape)
    
    # 用于保存每个客户端每轮的shared_results和local_results结果
    shared_results_records = []
    local_results_records = []

    # 初始化内存跟踪
    client_memory = 0
    server_memory = 0

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if c_round % 50 == 0:
            print(f"  > round {c_round}")  

        if c_round == 1:
            selected_clients = clients
        else:
            selected_clients = sampling_fn(clients, frac)
            server.selected_clients = selected_clients
            server.clients = clients
        
        all_logits = []       
        all_featmaps = []
        all_hard_sample_indices = []  

        round_client_memory = 0
        round_server_memory = 0

        for client in selected_clients:
            hard_idx = server.global_hard_sample_indices
            hard_idx_clone = hard_idx.clone() if isinstance(hard_idx, torch.Tensor) else None
            shared_results = client.local_train_shared(local_epoch, server.global_logits.clone(), server.global_feat_map.clone(), hard_idx_clone)
            shared_results_records.append({'client': client.name, 'round': c_round, 'trainingLoss': shared_results['trainingLosses'], 'trainingAcc': shared_results['trainingAccs']})
            local_results = client.local_train(local_epoch)
            local_results_records.append({'client': client.name, 'round': c_round, 'trainingLoss': local_results['trainingLosses'], 'trainingAcc': local_results['trainingAccs'], 
            'valLoss': local_results['valLosses'], 'valAcc': local_results['valAccs'], 'testLoss': local_results['testLosses'], 'testAcc': local_results['testAccs']})               
            updated_logits, updated_featmap, hard_sample_indices = client.local_train_logits_featuremaps(local_epoch)
            round_client_memory += get_tensor_memory(updated_featmap)
            round_client_memory += get_tensor_memory(hard_sample_indices)
            all_logits.append(updated_logits.detach().clone())    
            all_featmaps.append(updated_featmap.detach().clone())
            all_hard_sample_indices.append(hard_sample_indices.detach().clone()) 
        
        logits, maps, indice = server.local_train_kd(all_logits, all_featmaps, all_hard_sample_indices, local_epoch)
        round_server_memory += get_tensor_memory(maps)
        round_server_memory += get_tensor_memory(indice)
        server.global_logits = logits
        server.global_feat_map = maps
        server.global_hard_sample_indices = indice
        client_memory += round_client_memory
        server_memory += round_server_memory
        if c_round == 200:
            print(f"Round {c_round} - Client memory: {round_client_memory / (1024 ** 2):.2f} MB, Server memory: {round_server_memory / (1024 ** 2):.2f} MB")
            print(f"Total memory usage - Client: {client_memory / (1024 ** 2):.2f} MB, Server: {server_memory / (1024 ** 2):.2f} MB")
        
        if c_round % 1 == 0:
            accs = []
            losses = []
            for idx in range(len(clients)):
                loss, acc = clients[idx].evaluate()
                accs.append(acc)
                losses.append(loss)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)

            summary_writer.add_scalar(f'Test/Acc/Mean_{args.alg}', mean_acc, c_round)
            summary_writer.add_scalar(f'Test/Acc/Std_{args.alg}', std_acc, c_round)
    
    #frame = pd.DataFrame()
    #'''
    accuracies.clear()
    for client in clients:
        loss, acc = client.cross_domain_test()
        accuracies.append(acc)
        frame.loc[client.name, 'test_acc'] = acc
        #print(f"Client: {client.name}, Acc: {acc:.6f}")
    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        frame.loc['average', 'local_avgacc'] = avg_acc
        print(f"Average Acc: {avg_acc:.6f}")
    else:
        print("No clients to evaluate.")
    #'''

    accuracies.clear()
    for client in clients:
        loss, acc = client.evaluate()
        accuracies.append(acc)
        #print(f"Client: {client.name}, Acc: {acc:.6f}")
        frame.loc[client.name, 'test_acc'] = acc
    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        frame.loc['average', 'local_avgacc'] = avg_acc
        print(f"Average Acc: {avg_acc:.6f}")
    else:
        print("No clients to evaluate.")

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    fs = frame.style.apply(highlight_max).data
    #print(fs)
    
    
    return frame
