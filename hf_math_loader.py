from datasets import load_dataset
from torch.utils.data import Dataset

class MathDataset(Dataset):
    def __init__(self, dataset_name="qwedsacf/competition_math", split="train"):
        self.dataset = load_dataset(dataset_name, split=split)
        self.dataset = self.dataset.select_columns(["problem", "solution"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        return {
            "question": sample["problem"],
            "answer": sample["solution"]
        }

# dataset = MathDataset()
# print(dataset[0])