import math
import os
import torch
from torch.utils.data import DataLoader,random_split, Dataset
from transformers import get_linear_schedule_with_warmup

class TrainingLoops:
    def __init__(self, model, 
                    dataset,
                    training_portion=0.8, validation_portion=0.2,
                    batch_size = 4,
                    gradient_accumulation_steps=4,
                    output_dir="./checkpoints"  
                ):
        self.epochs = 20
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.output_dir = output_dir
        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # Train/validation split
        self.train_set, self.validation_set = random_split(
            dataset,
            [training_portion, validation_portion],
            generator=torch.Generator().manual_seed(42) #Ensure we get the same train/validation split every time the program runs
        )
        # DataLoader setup
        self.train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_set, self.batch_size, shuffle=False)

        # Optimizer setup
        trainable_parameters = [
            parameter
            for parameter
            in self.model.model.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(trainable_parameters, lr=1e-4)

        # Scheduler setup
        updates_per_epoch = math.ceil(
            len(self.train_loader)
            / self.gradient_accumulation_steps
        )
        total_updates = (
            updates_per_epoch
            * self.epochs
        )
        warmup_steps = int(
            0.03 * total_updates
        )
        self.scheduler = (
            get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_updates
            )
        )
        self.global_step = 0

    def sample_train(self):
        sample = self.dataset[0]
        for epoch in range(2):
            total_loss = 0.0
            for i in range(20):
                loss = self.model.train_step(
                    sample["question"],
                    sample["answer"]
                )
                total_loss += loss
                print(f"Sample: Step {i + 1}: {loss:.4f}")
            average_loss = total_loss / 20
            print(f"Sample: Average loss:{average_loss}")
                
    def validate(self):
        self.model.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in self.validation_loader:
                questions = batch["question"]
                answers = batch["answer"]
                loss = self.model.calculate_batch_loss(questions, answers)

                batch_size = len(questions)

                total_loss += loss.item() * batch_size
                
                total_samples += batch_size
        average_loss = (
            total_loss
            / total_samples
        )
        return average_loss

    def train(self): # Default 20 epochs
        for epoch in range(self.epoch):
            self.model.model.train() # Set to Training mode
            self.optimizer.zero_grad() # Reset Gradient
            total_loss = 0.0
            total_examples = 0

            for batch_idx, batch in enumerate(self.train_loader):
                questions = batch["question"]
                answers = batch["answer"]
            
                # Forward Pass
                loss = self.model.calculate_batch_loss(questions,answers)
                batch_size = len(questions)
                total_loss += (
                    loss.item()
                    * batch_size
                )
                total_examples += batch_size

                scaled_loss = loss / self.gradient_accumulation_steps

                scaled_loss.backward()

                # Update every 4 batch
                should_update = (batch_idx + 1) % self.gradient_accumulation_steps == 0
                # Last batch gradient should also be considered
                is_last_batch = batch_idx + 1 == len(self.train_loader)

                # Last batch handling
                if should_update or is_last_batch:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        max_norm=1.0
                    )
                    self.optimizer.step()

                    self.scheduler.step()

                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                    self.global_step += 1


                    if self.global_step % 10 == 0:
                        print(
                            f"Epoch "
                            f"{epoch + 1}/{self.epochs} "
                            f"| Step "
                            f"{self.global_step} "
                            f"| Loss "
                            f"{loss.item():.4f}"
                        )

                    # --------------------------
                    # Checkpoint
                    # --------------------------

                    if self.global_step % 500 == 0:
                        self.save_checkpoint(epoch)

                average_train_loss = total_loss / total_examples
                validation_loss = self.validate()
                print(
                    f"\nEpoch {epoch + 1}/{self.epochs}"
                )

                print(
                    f"Training loss: "
                    f"{average_train_loss:.4f}"
                )

                print(
                    f"Validation loss: "
                    f"{validation_loss:.4f}"
                )

    def save_checkpoint(self, epoch):

        checkpoint_dir = os.path.join(
            self.output_dir,
            f"checkpoint-{self.global_step}"
        )

        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

        self.model.model.save_pretrained(checkpoint_dir)

        # ---------------------------------
        # Save processor/tokenizer
        # ---------------------------------

        self.model.processor.save_pretrained(checkpoint_dir)

        # ---------------------------------
        # Save training state
        # ---------------------------------

        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict":
                self.optimizer.state_dict(),
            "scheduler_state_dict":
                self.scheduler.state_dict()
        }

        torch.save(
            training_state,
            os.path.join(
                checkpoint_dir,
                "training_state.pt"
            )
        )

        print(
            f"Checkpoint saved at: "
            f"{checkpoint_dir}"
        )






        