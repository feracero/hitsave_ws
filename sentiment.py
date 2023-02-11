from typing import Iterator, List, Literal, Tuple
import torch
from itertools import islice
import time
import csv
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch import nn
from torchtext.data.functional import to_map_style_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from hitsave import experiment, FileSnapshot, memo, restore

Split = Literal["test", "train"]
# digests = {
#     "test": "d4f01cbe4876739cfde1e5306d19bc73d5acde53a7b9b152219505292829ee46",
#     "train": "8f6c92d72174ba3b5be9138ba435cbd41dcd8377f7775ce981ac9c9aa2c4f141",
# }
digests = {
    "test": "/Users/fernandoaceromarchesotti1/Downloads/amazon_review_polarity_csv/test.csv",
    "train": "/Users/fernandoaceromarchesotti1/Downloads/amazon_review_polarity_csv/train_fake.csv",
}


def load_dataset(split: Split) -> Iterator[Tuple[int, str]]:
    """Given a split, produces a list of rating, review pairs.

    Label 0 is a bad review (1-2 stars), label 1 is a good review (4-5 stars).
    """
    digest = digests[split]
    # This line downloads the needed CSV file from the HitSave data catalogue.
    # This download is cached in your home's cache directory.
    # You can clear your cache with `hitsave clear-local`
    # path = restore(digest)
    path = digest
    with open(path, "rt") as f:
        # note: we limit the number of items to keep the demo fast on cpus.
        # if you have a GPU, delete the limit!
        items = islice(csv.reader(f, delimiter=","), 100000)
        for rating, title, body in items:
            yield (int(rating) - 1, title + ": " + body)


tokenizer = get_tokenizer("basic_english")


@memo
def make_vocab() -> Vocab:
    print("Building vocab")
    items = islice(load_dataset("train"), 10000)
    vocab = build_vocab_from_iterator(
        (tokenizer(text) for _, text in items),
        specials=["<unk>"],
    )
    vocab.set_default_index(vocab["<unk>"])
    print(f"Build vocab of {len(vocab)} items.")
    return vocab


vocab = make_vocab()

def text_pipeline(x):
    return vocab(tokenizer(x))


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def create_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

# @torch.compile
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


num_class = 2
vocab_size = len(vocab)
criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.NLLLoss()


def train_epoch(
    dataloader: DataLoader, model: TextClassificationModel, optimizer, epoch: int
):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # type: ignore
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            total_loss += loss.item()
    return {"acc": total_acc / total_count, "avg_loss": total_loss / total_count}


@memo
def train_model(
    *,
    lr,
    epochs,
    emsize,
    batch_size,
):
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
    total_accu = None
    train_dataset = to_map_style_dataset(load_dataset("train"))
    train_dataset, val_dataset = random_split(
        train_dataset, [0.95, 0.05], generator=torch.Generator().manual_seed(42)
    )
    train_dataloader = create_dataloader(train_dataset, batch_size)
    val_dataloader = create_dataloader(val_dataset, batch_size)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_epoch(train_dataloader, model, optimizer, epoch)
        ev = evaluate(val_dataloader, model)
        accu_val = ev["acc"]
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

    return model

@experiment
def test(
    lr=5.0,
    epochs=10,
    emsize=64,
    batch_size=64,
):
    model = train_model(lr=lr, epochs=epochs, emsize=emsize, batch_size=batch_size)
    test_dataloader = create_dataloader(
        to_map_style_dataset(load_dataset("test")), batch_size
    )

    print("Checking the results of test dataset.")
    results = evaluate(test_dataloader, model)
    return results


if __name__ == "__main__":
    # try editing these values and watch the dashboard fill with values.
    results = test(batch_size=64, lr=5.0, epochs=5)
    print("test accuracy {:8.3f}".format(results["acc"]))