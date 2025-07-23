import random
from random import choices
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
from models import SSP, Split_model
from server import Server
from client import Client_GC
from utils import get_stats, split_data, get_numGraphLabels, init_structure_encoding

def _randChunk(graphs, num_client, overlap, seed=None):
    random.seed(seed)
    np.random.seed(seed)

    totalNum = len(graphs)
    minSize = min(50, int(totalNum/num_client))
    graphs_chunks = []
    if not overlap:
        for i in range(num_client):
            graphs_chunks.append(graphs[i*minSize:(i+1)*minSize])
        for g in graphs[num_client*minSize:]:
            idx_chunk = np.random.randint(low=0, high=num_client, size=1)[0]
            graphs_chunks[idx_chunk].append(g)
    else:
        sizes = np.random.randint(low=50, high=150, size=num_client)
        for s in sizes:
            graphs_chunks.append(choices(graphs, k=s))
    return graphs_chunks

def remove_num_nodes(data):
    if hasattr(data, 'num_nodes'):
        try:
            del data.num_nodes
        except Exception as e:
            pass
    return data

def prepareData_multiDS(args, datapath, group='chem', batchSize=128, seed=None):
    assert group in ['chem', "biochem", 'biochemsn', "biosncv", "chemsn", "chemsncv", "chemcv"]

    if group == 'chem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1"]
    elif group == 'biochem':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "Peking_1", "OHSU", "KKI"]
    elif group == 'biochemsn':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "Peking_1", "OHSU", "KKI", "IMDB-MULTI", "IMDB-BINARY"]
    elif group == 'biosncv':
        datasets = ["Peking_1", "OHSU", "KKI", "IMDB-MULTI", "IMDB-BINARY", "Letter-high", "Letter-med", "Letter-low"]
    elif group == 'chemsn':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "IMDB-MULTI", "IMDB-BINARY"]
    elif group == 'chemsncv':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "IMDB-MULTI", "IMDB-BINARY", "Letter-high", "Letter-med", "Letter-low"]
    elif group == 'chemcv':
        datasets = ["MUTAG", "BZR", "COX2", "DHFR", "PTC_MR", "AIDS", "NCI1", "Letter-high", "Letter-med", "Letter-low"]

    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        elif "Letter" in data:
            tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True)
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)

        # 获取原始图列表并直接删除 num_nodes 属性
        graphs = [remove_num_nodes(x) for x in tudataset]
        #print("  **", data, len(graphs))

        graphs_train, graphs_valtest = split_data(graphs, test=0.2, shuffle=True, seed=seed)
        graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)
        graphs_pretrain, graphs_shared = split_data(graphs_val, train=0.8, test=0.2, shuffle=True, seed=seed)

        # 在结构编码之后也可确保删除 num_nodes
        graphs_train = [remove_num_nodes(x) for x in init_structure_encoding(args, gs=graphs_train, type_init=args.type_init)]
        graphs_val = [remove_num_nodes(x) for x in init_structure_encoding(args, gs=graphs_val, type_init=args.type_init)]
        graphs_test = [remove_num_nodes(x) for x in init_structure_encoding(args, gs=graphs_test, type_init=args.type_init)]
        graphs_pretrain = [remove_num_nodes(x) for x in init_structure_encoding(args, gs=graphs_pretrain, type_init=args.type_init)]
        graphs_shared = [remove_num_nodes(x) for x in init_structure_encoding(args, gs=graphs_shared, type_init=args.type_init)]

        dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
        dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
        dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)
        dataloader_pretrain = DataLoader(graphs_pretrain, batch_size=batchSize, shuffle=True)
        dataloader_shared = DataLoader(graphs_shared, batch_size=batchSize, shuffle=True)

        num_node_features = graphs[0].num_node_features
        num_graph_labels = get_numGraphLabels(graphs_train)

        splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test, 'pretrain': dataloader_pretrain, 'shared': dataloader_shared},
                             num_node_features, num_graph_labels, len(graphs_train))
        #splitedData[data] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test, 'pretrain': dataloader_pretrain},
                             #num_node_features, num_graph_labels, len(graphs_train))


        df = get_stats(df, data, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df
    
    splitedData = {}
    df = pd.DataFrame()
    for data in datasets:
        if data == "IMDB-BINARY":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(135, cat=False))
        elif data == "IMDB-MULTI":
            tudataset = TUDataset(f"{datapath}/TUDataset", data, pre_transform=OneHotDegree(88, cat=False))
        elif "Letter" in data:
            tudataset = TUDataset(f"{datapath}/TUDataset", data, use_node_attr=True)
        else:
            tudataset = TUDataset(f"{datapath}/TUDataset", data)

        graphs = [x for x in tudataset]
        print("  **", data, len(graphs))
        num_node_features = graphs[0].num_node_features
        graphs_chunks = _randChunk(graphs, nc_per_ds, overlap=False, seed=seed)
        for idx, chunks in enumerate(graphs_chunks):
            ds = f'{idx}-{data}'
            ds_tvt = chunks
            graphs_train, graphs_valtest = split_data(ds_tvt, train=0.8, test=0.2, shuffle=True, seed=seed)
            graphs_val, graphs_test = split_data(graphs_valtest, train=0.5, test=0.5, shuffle=True, seed=seed)

            graphs_train = init_structure_encoding(args, gs=graphs_train, type_init=args.type_init)
            graphs_val = init_structure_encoding(args, gs=graphs_val, type_init=args.type_init)
            graphs_test = init_structure_encoding(args, gs=graphs_test, type_init=args.type_init)

            dataloader_train = DataLoader(graphs_train, batch_size=batchSize, shuffle=True)
            dataloader_val = DataLoader(graphs_val, batch_size=batchSize, shuffle=True)
            dataloader_test = DataLoader(graphs_test, batch_size=batchSize, shuffle=True)
            num_graph_labels = get_numGraphLabels(graphs_train)
            splitedData[ds] = ({'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test},
                               num_node_features, num_graph_labels, len(graphs_train))
            df = get_stats(df, ds, graphs_train, graphs_val=graphs_val, graphs_test=graphs_test)

    return splitedData, df

def get_max_node_feature_dim(shared_dataset):
    max_dim = 0
    for data in shared_dataset:
        if hasattr(data, 'x') and data.x is not None:
            max_dim = max(max_dim, data.x.size(1))
    return max_dim

def get_max_edge_attr_dim(shared_dataset):
    max_dim = 0
    for data in shared_dataset:
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            max_dim = max(max_dim, data.edge_attr.size(1))
    return max_dim

def get_max_label_dim(shared_dataset):
    max_dim = 0
    for data in shared_dataset:
        if hasattr(data, 'y') and data.y is not None:
            max_dim = max(max_dim, int(data.y.max()) + 1)  
    return max_dim

def create_shared_dataset(splitedData, percentage=1.0, seed=None, stratified=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    shared_dataset = []
    selection_info = {}
    
    for ds_name, (dataloaders, _, _ , _) in splitedData.items():
        val_loader = dataloaders['shared']
        val_graphs = list(val_loader.dataset)
        
        if stratified:
            label_to_graphs = {}
            for graph in val_graphs:
                label = graph.y.item()
                label_to_graphs.setdefault(label, []).append(graph)
            
            selected_graphs = []
            for label, graphs in label_to_graphs.items():
                num_select = max(1, int(len(graphs) * percentage))
                selected_graphs.extend(random.sample(graphs, num_select))
        else:
            num_select = max(1, int(len(val_graphs) * percentage))
            selected_graphs = random.sample(val_graphs, num_select)

        for g in selected_graphs:
            g_cp = copy.deepcopy(g)
            g_cp = ensure_edge_attributes(g_cp)
            shared_dataset.append(g_cp)
        
        selection_info[ds_name] = {
            "total_val": len(val_graphs),
            "selected": len(selected_graphs)
        }
    
    #print("\nShared Dataset Stats:")
    #for ds, info in selection_info.items():
        #print(f"{ds}: Selected {info['selected']}/{info['total_val']}")
    
    max_node_dim = get_max_node_feature_dim(shared_dataset)
    max_edge_dim = get_max_edge_attr_dim(shared_dataset)
    #print(f"Max node_dim: {max_node_dim}, Max edge_dim: {max_edge_dim}")
    
    return shared_dataset

def create_pretrain_dataset(splitedData, percentage=1.0, seed=None, stratified=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    pretrain_dataset = []
    selection_info = {}
    
    for ds_name, (dataloaders, _, _, _) in splitedData.items():
        val_loader = dataloaders['pretrain']
        val_graphs = list(val_loader.dataset)
        
        if stratified:
            label_to_graphs = {}
            for graph in val_graphs:
                label = graph.y.item()
                label_to_graphs.setdefault(label, []).append(graph)
            
            selected_graphs = []
            for label, graphs in label_to_graphs.items():
                num_select = max(1, int(len(graphs) * percentage))
                selected_graphs.extend(random.sample(graphs, num_select))
        else:
            num_select = max(1, int(len(val_graphs) * percentage))
            selected_graphs = random.sample(val_graphs, num_select)

        for g in selected_graphs:
            g_cp = copy.deepcopy(g)
            g_cp = ensure_edge_attributes(g_cp)
            pretrain_dataset.append(g_cp)

        selection_info[ds_name] = {
            "total_val": len(val_graphs),
            "selected": len(selected_graphs)
        }
    
    #print("\nPretrain Dataset Stats:")
    #for ds, info in selection_info.items():
        #print(f"{ds}: Selected {info['selected']}/{info['total_val']}")
    
    max_node_dim = get_max_node_feature_dim(pretrain_dataset)
    max_edge_dim = get_max_edge_attr_dim(pretrain_dataset)
    #print(f"Pretrain - Max node_dim: {max_node_dim}, Max edge_dim: {max_edge_dim}")

    return pretrain_dataset

def create_cross_domain_test_loader(splitedData, client_data=None):
    cross_domain_test_data = []
    selection_info = {}

    # 遍历所有的数据集
    for ds_name, (dataloaders, _, _, _) in splitedData.items():
        # 如果当前数据集是客户端的本地数据集，则跳过
        if ds_name == client_data:
            continue
        
        # 获取测试集的图数据 (改为使用 'test' 数据集)
        test_loader = dataloaders['test']
        test_graphs = list(test_loader.dataset)

        # 从测试集选取图形来创建跨域测试数据集
        for g in test_graphs:
            g_cp = copy.deepcopy(g)
            g_cp = ensure_edge_attributes(g_cp)  # 确保边特征存在
            cross_domain_test_data.append(g_cp)

        selection_info[ds_name] = {
            "total_test": len(test_graphs),
        }

    # 打印跨域测试数据集的统计信息
    #print("\nCross Domain Test Dataset Stats:")
    #for ds, info in selection_info.items():
        #print(f"{ds}: Total {info['total_test']}")

    return cross_domain_test_data



def get_local_data_dimensions(dataloader_train):
    first_batch = next(iter(dataloader_train))
    node_feature_dim = first_batch.x.size(1) if first_batch.x is not None else 0
    edge_feature_dim = first_batch.edge_attr.size(1) if first_batch.edge_attr is not None else 0
    return node_feature_dim, edge_feature_dim

def ensure_edge_attributes(data):
    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        num_edges = data.edge_index.size(1) if data.edge_index is not None else 0
        data.edge_attr = torch.zeros((num_edges, 0), dtype=torch.float)
    return data

def pad_local_dataset(local_dataset, global_node_dim, global_edge_dim, global_label_dim=None):
    padded_local_dataset = []
    
    for data in local_dataset:
        # 保留原始标签
        if data.y is not None:
            data.original_y = data.y.clone()
        
        # 填充节点特征
        if data.x is not None:
            current_node_dim = data.x.size(1)
            if current_node_dim < global_node_dim:
                padding_dim = global_node_dim - current_node_dim
                padding = torch.zeros((data.x.size(0), padding_dim), dtype=data.x.dtype)
                data.x = torch.cat([data.x, padding], dim=1)
        
        # 填充边特征
        if data.edge_attr is not None:
            current_edge_dim = data.edge_attr.size(1)
            if current_edge_dim < global_edge_dim:
                padding_dim = global_edge_dim - current_edge_dim
                padding = torch.zeros((data.edge_attr.size(0), padding_dim), dtype=data.edge_attr.dtype)
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
        
        # 处理标签：转换为类别索引（如果原标签为 one‑hot，则转换为索引）
        if global_label_dim is not None and data.y is not None:
            if data.y.dim() > 0 and data.y.numel() > 1:
                label_index = data.y.argmax().long()
            else:
                label_index = data.y.long()
            data.y = label_index
        
        padded_local_dataset.append(data)
    
    return padded_local_dataset

def pad_shared_dataset(shared_dataset, global_node_dim, global_edge_dim, global_label_dim=None):
    padded_shared_dataset = []
    
    for data in shared_dataset:
        # 填充节点特征
        if data.x is not None:
            current_node_dim = data.x.size(1)
            if current_node_dim < global_node_dim:
                padding_dim = global_node_dim - current_node_dim
                padding = torch.zeros((data.x.size(0), padding_dim), dtype=data.x.dtype)
                data.x = torch.cat([data.x, padding], dim=1)
        
        # 填充边特征
        if data.edge_attr is not None:
            current_edge_dim = data.edge_attr.size(1)
            if current_edge_dim < global_edge_dim:
                padding_dim = global_edge_dim - current_edge_dim
                padding = torch.zeros((data.edge_attr.size(0), padding_dim), dtype=data.edge_attr.dtype)
                data.edge_attr = torch.cat([data.edge_attr, padding], dim=1)
        
        # 处理标签：转换为类别索引（如果原标签为 one‑hot，则转换为索引）
        if global_label_dim is not None and data.y is not None:
            if data.y.dim() > 0 and data.y.numel() > 1:
                label_index = data.y.argmax().long()
            else:
                label_index = data.y.long()
            data.y = label_index
        
        padded_shared_dataset.append(data)
    
    return padded_shared_dataset

def get_padded_dataloader(original_dataloader, global_node_dim, global_edge_dim, global_label_dim, batch_size, shuffle):
    dataset = original_dataloader.dataset
    padded_dataset = pad_local_dataset(
        copy.deepcopy(dataset),
        global_node_dim, global_edge_dim, global_label_dim=global_label_dim
    )
    return DataLoader(padded_dataset, batch_size=batch_size, shuffle=shuffle)

def process_client_dataloaders(dataloaders, global_node_dim, global_edge_dim, global_label_dim, batch_size):
    dataloaders['train'] = get_padded_dataloader(
        dataloaders['train'], global_node_dim, global_edge_dim, global_label_dim, batch_size, shuffle=True
    )
    dataloaders['val'] = get_padded_dataloader(
        dataloaders['val'], global_node_dim, global_edge_dim, global_label_dim, batch_size, shuffle=False
    )
    dataloaders['test'] = get_padded_dataloader(
        dataloaders['test'], global_node_dim, global_edge_dim, global_label_dim, batch_size, shuffle=False
    )
    return dataloaders

def create_shared_data_loader(shared_dataset, global_node_dim, global_edge_dim, global_label_dim, batch_size, shuffle=True):
    padded_shared_dataset = pad_shared_dataset(
        copy.deepcopy(shared_dataset),
        global_node_dim, global_edge_dim, global_label_dim=global_label_dim
    )
    return DataLoader(padded_shared_dataset, batch_size=batch_size, shuffle=shuffle)

def setup_devices_SSP(splitedData, args):
    idx_clients = {}
    clients = []
    client_dims = {}  

    shared_dataset = create_shared_dataset(splitedData, percentage=0.4, seed=args.seed)
    pretrain_dataset = create_pretrain_dataset(splitedData, percentage=0.6, seed=args.seed)

    all_node_dims = []
    all_edge_dims = []
    for ds in splitedData.keys():
        dataloaders, _, _, _ = splitedData[ds]
        node_dim, edge_dim = get_local_data_dimensions(dataloaders['train'])
        all_node_dims.append(node_dim)
        all_edge_dims.append(edge_dim)

    global_node_dim = max(all_node_dims) if all_node_dims else 0
    global_edge_dim = max(all_edge_dims) if all_edge_dims else 0
    global_label_dim = max([splitedData[ds][2] for ds in splitedData.keys()])
    #print(f"max_label: {global_label_dim}")

    # 创建共享数据集的DataLoader
    shared_loader = create_shared_data_loader(
        shared_dataset, global_node_dim, global_edge_dim, global_label_dim, args.batch_size, shuffle=False
    )

    # 为每个客户端创建跨域数据集
    for idx, ds in enumerate(splitedData.keys()):
        dataloaders, num_node_features, num_graph_labels, train_size = splitedData[ds]
        node_dim, edge_dim = get_local_data_dimensions(dataloaders['train'])

        # 为当前客户端构建跨域数据集
        cross_domain_test_data = create_cross_domain_test_loader(splitedData, client_data=ds)
        padded_cross_domain_test_data = pad_local_dataset(
            cross_domain_test_data, global_node_dim, global_edge_dim, global_label_dim
        )
        cross_dataloader = DataLoader(padded_cross_domain_test_data, batch_size=args.batch_size, shuffle=False)

        dataloaders['shared'] = shared_loader
        dataloaders['cross'] = cross_dataloader
        
        # 填充数据
        dataloaders = process_client_dataloaders(
            dataloaders, global_node_dim, global_edge_dim, global_label_dim, args.batch_size
        )

        node_feature_dim = [global_node_dim]
        edge_feature_dim = global_edge_dim      
        label_dim = global_label_dim
        local_label_dim = num_graph_labels

        # 初始化客户端模型
        if args.alg == "fedSSP":
            former = SSP(global_label_dim, args.nlayer, node_feature_dim, edge_feature_dim, node_feature_dim[0], args.head, args.hidden)
            head = former.fc
            former.fc = nn.Identity()
            basicModel = Split_model(former, head, global_label_dim, num_graph_labels, args.mlp_hidden_dim)
            cmodel_gc = copy.deepcopy(basicModel)
            
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), 
                                          lr=args.lr, weight_decay=args.weight_decay)
            
            clients.append(Client_GC(copy.deepcopy(cmodel_gc), idx, ds, train_size, dataloaders, optimizer, args))
        
        client_dims[idx] = (node_dim, edge_dim)

    #print("\nSummary of Client Shared Data Dimensions (after padding):")
    #for cid, (n_dim, e_dim) in client_dims.items():
        #print(f"Client {cid}: Node Dimension={global_node_dim} (local was {n_dim}), Edge Dimension={global_edge_dim} (local was {e_dim})")
    
    # 初始化服务器模型（使用全局维度）
    server_node_feature = [global_node_dim]
    server_former = SSP(global_label_dim, args.nlayer, server_node_feature, 0, server_node_feature[0], args.head, args.hidden)
    server_head = server_former.fc
    server_former.fc = nn.Identity()
    server_model = Split_model(server_former, server_head, global_label_dim, global_label_dim, args.mlp_hidden_dim).to(args.device)
    
    # 使用共享数据集构造服务器 DataLoader（与客户端一致）
    '''shared_loader = create_shared_data_loader(
        shared_dataset, global_node_dim, global_edge_dim, global_label_dim, args.batch_size, shuffle=True
    )'''
    pretrain_loader = create_shared_data_loader(
        pretrain_dataset, global_node_dim, global_edge_dim, global_label_dim, args.batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cmodel_gc.parameters()), 
                                  lr=args.lr, weight_decay=args.weight_decay)

    server = Server(server_model, args.device, shared_loader, pretrain_loader, args)
    
    return clients, server, idx_clients
