import sys
from os import path

sys.path.append(path.join(path.dirname(__file__), '..', ".."))

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import ContextARCDataset, MixtureDataset
from model import ContextAttention
from script.ContextAttention.tools import *
setup_seed(724)
device = 0
epochs = 50
init_epochs = 0
# 32 batch need 8 * T4 or 4 * V100, 16 batch need 4 * T4 or 2 * V100
batchSize = 32
gpunum = 8
workernum = 4
# 保存权重
subPath = Path("context-attention/")
save = Path("weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("log", subPath))

dataset_folder = "pqa-dataset"
dataset_paths = [
    (1, Path(f"{dataset_folder}/closure-filling/")),
    (2, Path(f"{dataset_folder}/continuity-connection/")),
    (3, Path(f"{dataset_folder}/proximity-identification/")),
    (4, Path(f"{dataset_folder}/shape-reconstruction/")),
    (5, Path(f"{dataset_folder}/shape-pattern-similarity/")),
    (6, Path(f"{dataset_folder}/reflection-symmetry/")),
    (7, Path(f"{dataset_folder}/rotation-symmetry/"))
]

ecnnet = ContextAttention(src_vocab=len(ContextARCDataset.WordMap), tgt_vocab=len(ContextARCDataset.WordMap), task_num=8,
                          encoder_layer_num=6, decoder_layer_num=6, d_model=512, h=8, d_ff=2048,
                          max_shape=(30, 30),
                          padding_idx=ContextARCDataset.WordMap['pad_symbol'],
                          attention_dropout=None, enable_decoder=True)

net = torch.nn.DataParallel(ecnnet, device_ids=[device + c for c in range(gpunum)]).cuda(device)

optimizer = torch.optim.Adam([{"params": net.parameters(), "initial_lr": 1e-4}], lr=1e-4, betas=(0.9, 0.999), )


# 0.65^25 = 0.00002
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=init_epochs-1)

padding = ContextARCDataset.WordMap['pad_symbol']


def train_forward(net, inputs, inputs_mask, ctx_x, ctx_x_mask, ctx_y, ctx_y_mask):
    output, task = net(src=inputs, src_mask=inputs_mask, ctx_x=ctx_x, ctx_x_mask=ctx_x_mask, ctx_y=ctx_y,
                       ctx_y_mask=ctx_y_mask,
                       enable_future_mask=False)
    return output, task


def val_forward(net, inputs, inputs_mask, ctx_x, ctx_x_mask, ctx_y, ctx_y_mask):
    output, task = net(src=inputs, src_mask=inputs_mask, ctx_x=ctx_x, ctx_x_mask=ctx_x_mask, ctx_y=ctx_y,
                       ctx_y_mask=ctx_y_mask,
                       enable_future_mask=False)
    _, res = torch.max(output, dim=-1)
    _, task = torch.max(task, dim=-1)
    return res, task


def to_same_size(pad_value, *batch):
    max_shape = [0] * len(batch[0].shape)
    for item in batch:
        for i, d in enumerate(item.shape):
            if max_shape[i] < d:
                max_shape[i] = d
    padded_tensor = []
    for item in batch:
        batch_dim = []
        for i, d in enumerate(item.shape):
            batch_dim.insert(0, max_shape[i] - d)  # 后面pad
            batch_dim.insert(0, 0)  # 前面不pad
        padded_tensor.append(
            torch.nn.functional.pad(item, batch_dim, mode='constant', value=pad_value)
        )
    return padded_tensor


step, val_step = 0, 0


for epoch in range(init_epochs, epochs):
    net.train()
    datasets = []
    for i, dataset_path in dataset_paths:
        datasets.append(ContextARCDataset(index=i, dataset_path=Path(dataset_path), method='train'))
    dataset = MixtureDataset(datasets)
    train = DataLoader(dataset, shuffle=True, num_workers=workernum, batch_size=batchSize,
                       collate_fn=ContextARCDataset.collate_fn)
    for index, batch in enumerate(train):
        if index > 3500:  # 提前退出快速进行val
            break
        inputs, ctx_input, targets, ctx_targets, task = to_same_size(padding, *to_device(device, *batch))

        start_time = time.time()
        step += len(inputs)

        answers = targets.ne(inputs).to(torch.long)
        answers[inputs.eq(padding)] = padding

        outputs, predict_task = train_forward(net, inputs, inputs.ne(padding).to(torch.float),
                                              ctx_input, ctx_input.ne(padding).to(torch.float),
                                              ctx_targets, ctx_targets.ne(padding).to(torch.float))

        loss = compute_balance_loss(outputs, targets, answers, padding)
        _, output_index = torch.max(outputs, dim=-1)
        output_index = output_index.to(outputs.dtype)
        element_accuracy = compute_element_accuracy(output_index, targets, padding)
        mask_accuracy = compute_mask_accuracy(inputs, output_index, targets, padding)
        correct_accuracy = compute_corrects_accuracy(output_index, targets, padding)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loging(writer, 'train', epoch, step, time.time() - start_time, dataset_size=len(dataset),
               batch_size=len(inputs),
               **{'loss': loss, 'element_accuracy': element_accuracy, 'correct_accuracy': correct_accuracy,
                  'mask_accuracy': mask_accuracy})
    # 更新学习率
    scheduler.step()

    with torch.no_grad():
        net.eval()
        for i, dataset_path in dataset_paths:
            dataser_name = dataset_path.parts[-1]
            dataset = ContextARCDataset(index=i, dataset_path=Path(dataset_path),
                                        method='test')  # ARCDataset(Path(dataset_path), method='test')
            val = DataLoader(dataset, shuffle=False, pin_memory=False, num_workers=workernum,
                             batch_size=batchSize, collate_fn=ContextARCDataset.collate_fn)

            start_time = used_time = time.time()
            total_element_accuracy = torch.tensor(0.)
            total_correct_accuracy = torch.tensor(0.)
            total_mask_accuracy = torch.tensor(0.)
            index = 0
            for index, batch in enumerate(val):
                if index > 100:
                    break
                inputs, ctx_input, targets, ctx_targets, task = to_same_size(padding, *to_device(device, *batch))

                start_time = time.time()
                val_step += len(inputs)

                output_index, predict_task = val_forward(net, inputs, inputs.ne(padding).to(torch.float),
                                                         ctx_input, ctx_input.ne(padding).to(torch.float),
                                                         ctx_targets, ctx_targets.ne(padding).to(torch.float))

                # answers = targets.ne(inputs).to(torch.long)
                # answers[inputs.eq(padding)] = padding
                #

                element_accuracy = compute_element_accuracy(output_index, targets, padding)
                mask_accuracy = compute_mask_accuracy(inputs, output_index, targets, padding)
                correct_accuracy = compute_corrects_accuracy(output_index, targets, padding)
                total_element_accuracy = total_element_accuracy + element_accuracy
                total_correct_accuracy = total_correct_accuracy + correct_accuracy
                total_mask_accuracy = total_mask_accuracy + mask_accuracy
                loging(None, f'{dataser_name}-val-step', epoch, val_step, time.time() - start_time,
                       dataset_size=len(val.dataset),
                       batch_size=len(inputs),
                       **{'element_accuracy': element_accuracy, 'correct_accuracy': correct_accuracy,
                          'mask_accuracy': mask_accuracy})
            index += 1
            total_element_accuracy = total_element_accuracy / index
            total_correct_accuracy = total_correct_accuracy / index
            total_mask_accuracy = total_mask_accuracy / index
            loging(writer, f'{dataser_name}-val', epoch, epoch, time.time() - start_time,
                   dataset_size=len(train.dataset),
                   batch_size=len(val.dataset),
                   **{'element_accuracy': total_element_accuracy, 'correct_accuracy': total_correct_accuracy,
                      'mask_accuracy': total_mask_accuracy})

        torch.save(
            net.module.state_dict(),
            Path(save,
                 f"epoch{epoch}.weight")
        )
