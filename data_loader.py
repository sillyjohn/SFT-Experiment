from datasets import DatasetDict, load_dataset

class MathDatasetLoader:
    """Load MATH and create a reproducible train/evaluation boundary.

    The Hugging Face dataset currently exposes all 12,500 examples in a single
    ``train`` split.  We therefore reserve a deterministic subset before SFT so
    that baseline and post-SFT scores are measured on exactly the same data.
    """

    def __init__(self, dataset_name="qwedsacf/competition_math"):
        self.dataset_name = dataset_name
        self.dataset = None

    def load_data(self, eval_size=500, seed=42):
        dataset = load_dataset(self.dataset_name)

        if "test" in dataset:
            self.dataset = dataset
            return self.dataset

        if "train" not in dataset:
            raise ValueError(
                f"Expected a 'train' split, but found: {list(dataset.keys())}"
            )

        if eval_size <= 0 or eval_size >= len(dataset["train"]):
            raise ValueError(
                f"eval_size must be between 1 and {len(dataset['train']) - 1}"
            )

        split = dataset["train"].train_test_split(
            test_size=eval_size,
            seed=seed,
            shuffle=True,
        )
        self.dataset = DatasetDict(
            train=split["train"],
            test=split["test"],
        )
        return self.dataset
