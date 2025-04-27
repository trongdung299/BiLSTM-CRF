from collections import Counter
import torch
from torch.utils.data import Dataset


from constants import PAD, UNK, BOS, EOS

class CustomVocab:
    def __init__(self, counter, specials=None, specials_first=False):
        self.stoi = {}  # String to index
        self.itos = []  # Index to string
        self.specials = specials or []

        # Thêm special tokens trước hoặc sau
        if specials_first:
            self.itos.extend(specials)
            for i, s in enumerate(specials):
                self.stoi[s] = i

        # Thêm các từ từ counter
        for word, _ in counter.items():
            if word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)

        # Nếu không thêm special tokens trước, thêm sau
        if not specials_first and specials:
            for s in specials:
                if s not in self.stoi:
                    self.stoi[s] = len(self.itos)
                    self.itos.append(s)

        # Gán chỉ số cho UNK
        self.unk_index = self.stoi.get(UNK, 0)

    def __len__(self):
        return len(self.itos)  # Trả về số lượng từ/nhãn trong vocab

class NerDataset(Dataset):
    def __init__(self, filepath, vocab=None, label=None):
        self.data = None
        self.vocab = vocab
        self.label = label
        self.read_data(filepath)

    def read_data(self, filepath):
        all_words = []
        all_ners = []

        sents = open(filepath, 'r', encoding='utf-8').read().strip().split('\n\n')
        for sent in sents:
            items = sent.split('\n')
            for item in items:
                if not item.strip():  # Bỏ qua dòng trống
                    continue
                try:
                    w, n = item.split()  # Tách bằng dấu cách (2 cột: từ và nhãn)
                    w = '_'.join(w.split())  # Chuyển dấu cách trong từ thành dấu gạch dưới
                    all_words.append(w)
                    all_ners.append(n)  # Giữ nguyên nhãn
                except ValueError:
                    print(f"Dòng không hợp lệ trong {filepath}: {item}")
                    continue

        # Tạo vocab và label nếu chưa có
        if self.vocab is None or self.label is None:
            self.vocab = CustomVocab(Counter(all_words), specials=(PAD, UNK, BOS, EOS), specials_first=False)
            self.label = CustomVocab(Counter(all_ners), specials=(BOS, EOS, PAD), specials_first=False)
            self.vocab.unk_index = self.vocab.stoi[UNK]
            self.label.unk_index = self.label.stoi['O']

        X = []
        Y = []
        for sent in sents:
            x = [self.vocab.stoi[BOS]]
            y = [self.label.stoi[BOS]]
            items = sent.split('\n')
            for item in items:
                if not item.strip():  # Bỏ qua dòng trống
                    continue
                try:
                    w, n = item.split()  # Tách bằng dấu cách
                    w = '_'.join(w.split())
                    x.append(self.vocab.stoi.get(w, self.vocab.unk_index))
                    y.append(self.label.stoi.get(n, self.label.unk_index))
                except ValueError:
                    print(f"Dòng không hợp lệ trong {filepath}: {item}")
                    continue
            x.append(self.vocab.stoi[EOS])
            y.append(self.label.stoi[EOS])
            X.append(torch.tensor(x))
            Y.append(torch.tensor(y))

        self.data = []
        for x, y in zip(X, Y):
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]