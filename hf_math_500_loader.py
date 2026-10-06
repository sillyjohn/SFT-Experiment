from datasets import load_dataset


class Math500Loader:
    def __init__(self):
        self.ds = load_dataset("HuggingFaceH4/MATH-500")
        self.question = self.ds[]  # Assuming you want to use the training split


