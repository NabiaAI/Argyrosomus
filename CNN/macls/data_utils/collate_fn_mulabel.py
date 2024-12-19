import torch



def collate_fn(batch):

    batch_sorted = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    freq_size = batch_sorted[0][0].size(1)
    max_freq_length = batch_sorted[0][0].size(0)
    batch_size = len(batch_sorted)

    features = torch.zeros((batch_size, max_freq_length, freq_size), dtype=torch.float32)
    input_lens, labels = [], []
    for x in range(batch_size):
        tensor, label = batch[x]
        seq_length = tensor.size(0)
        features[x, :seq_length, :] = tensor[:, :]
        labels.append(label)
        input_lens.append(seq_length)
    labels = torch.stack(labels).int()
    input_lens = torch.tensor(input_lens, dtype=torch.int64)
    return features, labels, input_lens
