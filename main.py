import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os

from constants import *
from dataset import NerDataset
from train import train
from bilstm_crf import BiLSTM_CRF_NER

def collate_fn(samples):
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    sentences = [x[0] for x in samples]
    tags = [x[1] for x in samples]
    return sentences, tags

def padding(sents, pad_idx, device):
    lengths = [len(sent) for sent in sents]
    max_len = lengths[0]
    padded_data = []
    for s in sents:
        padded_data.append(s.tolist() + [pad_idx] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths

print("Bắt đầu chạy main.py...")
try:
    print("Tải train_dataset...")
    train_dataset = NerDataset('C:/Users/TGDD/.vscode/BiLSTM-CRF/data/train.txt')
    print("Đã tải train_dataset, kích thước:", len(train_dataset))
    print("Ví dụ câu đầu tiên:", train_dataset[0])
    print("Vocab size:", len(train_dataset.vocab))
    print("Label set:", train_dataset.label.itos)

    print("Tải val_dataset...")
    val_dataset = NerDataset('C:/Users/TGDD/.vscode/BiLSTM-CRF/data/dev.txt', train_dataset.vocab, train_dataset.label)
    print("Đã tải val_dataset, kích thước:", len(val_dataset))

    batch_size = 32 

    train_iter = DataLoader(train_dataset, batch_size, collate_fn=collate_fn)
    val_iter = DataLoader(val_dataset, batch_size, collate_fn=collate_fn)

    print("Khởi tạo mô hình...")
    model = BiLSTM_CRF_NER(train_dataset.vocab, train_dataset.label)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    # Tải checkpoint nếu có
    checkpoint_path = 'checkpoints/lstm_ner_1.pt'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Tiếp tục huấn luyện từ epoch {start_epoch}...")
    else:
        print("Không tìm thấy checkpoint, bắt đầu huấn luyện từ đầu...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Bắt đầu huấn luyện trên", device)
    writer = SummaryWriter('runs/lstm_ner')
    train(model, optimizer, writer, train_iter, val_iter, device, epochs=10, resume=start_epoch>0)

    # Đánh giá mô hình
    print("Đánh giá mô hình...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentences, tags in val_iter:
            sentences, sent_lengths = padding(sentences, model.sent_vocab.stoi[PAD], device)
            tags, _ = padding(tags, model.tag_vocab.stoi[PAD], device)
            _, predictions = model(sentences, tags, sent_lengths)  # Truyền sent_lengths
            for pred, true in zip(predictions, tags):
                for p, t in zip(pred[1:-1], true[1:-1]):  # Bỏ BOS/EOS
                    if p == t.item():
                        correct += 1
                    total += 1
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

except Exception as e:
    print("Lỗi:", e)