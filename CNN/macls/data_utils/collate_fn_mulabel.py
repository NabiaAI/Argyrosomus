import torch


# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的
    batch_sorted = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    freq_size = batch_sorted[0][0].size(1)
    max_freq_length = batch_sorted[0][0].size(0)
    batch_size = len(batch_sorted)
    # 以最大的长度创建0张量
    features = torch.zeros((batch_size, max_freq_length, freq_size), dtype=torch.float32)
    #add
    masks = torch.zeros((batch_size, max_freq_length, 1), dtype=torch.bool)
    input_lens, labels = [], []
    for x in range(batch_size):
        #add 12.31
        #tensor, label, mask = batch[x]  # 解包 3 个值
        tensor, label= batch[x]  # 解包 3 个值
        
        # 如果 mask 是 None，生成一个默认的全 1 掩码
        # if mask is None:
        #     mask = torch.ones((tensor.size(0), 1), dtype=torch.bool)
        
        #tensor, label = batch[x]
        seq_length = tensor.size(0)
        # 将数据插入到0张量中，实现了padding
        features[x, :seq_length, :] = tensor[:, :]

        #add
        #masks[x, :seq_length, :] = mask[:, :]
        labels.append(label)
        input_lens.append(seq_length)
    if all(label is None for label in labels):
        labels = None
    else:
        labels = torch.stack(labels).int()
    # 将标签转换为浮点型张量，并确保形状正确
    input_lens = torch.tensor(input_lens, dtype=torch.int64)
    
    #add
    # 确保 features 和 masks 在同一设备上
    device = features.device  # 获取 features 的设备
    #masks = masks.to(device)
    return features, labels, input_lens


    #return features,  masks, labels,input_lens

