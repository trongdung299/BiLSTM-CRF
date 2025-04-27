import torch
from dataset import NerDataset
from bilstm_crf import BiLSTM_CRF_NER
from constants import *

def predict(model, vocab, label, sentence, device):
    model.eval()
    tokens = sentence.split()
    x = [vocab.stoi[BOS]] + [vocab.stoi.get(t, vocab.unk_index) for t in tokens] + [vocab.stoi[EOS]]
    x = torch.tensor([x]).to(device)
    sent_lengths = [len(x[0])]  # Độ dài câu
    with torch.no_grad():
        predictions = model.predict(x, sent_lengths)  # Gọi phương thức predict
    return list(zip(tokens, [label.itos[p] for p in predictions[0][1:-1]]))

def main():
    print("Tải train_dataset...")
    train_dataset = NerDataset('C:/Users/TGDD/.vscode/BiLSTM-CRF/data/train.txt')
    print("Vocab size:", len(train_dataset.vocab))
    print("Label set:", train_dataset.label.itos)

    print("Khởi tạo mô hình...")
    model = BiLSTM_CRF_NER(train_dataset.vocab, train_dataset.label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Sử dụng thiết bị:", device)

    checkpoint_path = 'checkpoints/lstm_ner_1.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(f"Đã tải checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"Không tìm thấy checkpoint: {checkpoint_path}")
        return

    sentences = [
        "Ngày 27 tháng 4, Bộ Y tế ghi nhận 20 ca mắc COVID-19 mới tại Hà Nội và TP.HCM",
        "TP.HCM bắt đầu giãn cách xã hội từ ngày 9 tháng 7 năm 2021 theo Chỉ thị 16"
        
    ]
    print("\nDự đoán nhãn NER:")
    for sentence in sentences:
        predictions = predict(model, train_dataset.vocab, train_dataset.label, sentence, device)
        print(f"\nCâu: {sentence}")
        print("Dự đoán:", predictions)

if __name__ == "__main__":
    main()