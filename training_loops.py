import torch
import torch.utils.data as DataLoader,random_split, Dataset

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
            [training_protion, validation_portion, testing_portion],
            generator=torch.Generator().manual_seed(42) #Ensure we get the same train/validation split every time the program runs
        )
        self.train_loader = DataLoader(train_set,batch_size,shuffle=True)
        self.validation_loader = DataLoader(validation_set,batch_size,shuffle=True)
        self.test_loader = DataLoader(test_set,batch_size,shuffle=True)



    def train(self, num_epochs = 20): # Default 20 epochs
        self.model.train()
        for i in range(num_epochs):
            for batch in self.train_loader:
                question = batch["question"]
                answer = batch["answer"]
                loss = self.model.calculate_loss(question, answer)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if i%10 == 0:    
                print(f"Epoch {i+1}: Loss = {loss.item()}")



        