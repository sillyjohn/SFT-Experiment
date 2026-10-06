import torch
from torch.utils.data as DataLoader,random_split, Dataset

class training_loops:
    def __init__(self, model, 
                    optimizer, 
                    dataset, 
                    batch_size,
                    training_portion,
                    validation_portion,
                    test_portion):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_set, self.validation_set, self.test_set = random_split(
            dataset,
            [training_portion, validation_portion, testing_portion],
            generator=torch.Generator().manual_seed(42) #Ensure we get the same train/validation split every time the program runs
        )
        self.train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_set, self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_set, self.batch_size, shuffle=False)


    def samepl_train(self):
        for epoch in range(10):
            total_loss = 0.0
            sample = self.dataset[0]
            for i in range(20):
                loss = self.model.train_step(
                    sample["question"],
                    sample["answer"]
                )
                total_loss += loss
                print(f"Sample: Step {i + 1}: {loss:.4f}")
            average_loss = total_loss / 20
            print(f"Sample: Average lossL:{average_loss}")
                


    def train(self, num_epochs = 20): # Default 20 epochs
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch in self.train_loader:
                loss = self.model.train_step()
                total_loss += loss
                print(f"Step{i+1}:{loss:.4f}")



        