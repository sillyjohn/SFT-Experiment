from transformers import AutoProcessor, AutoModelForMultimodalLM
from peft import LoraConfig, get_peft_model
import torch
class GemmaModel:
    def __init__ (self, model_id):
        self.processor = AutoProcessor.from_pretrained(model_id) # Load the processor for the specified model
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, # The ID of the model to load
            dtype="auto", # Automatically select the appropriate data type
            device_map="auto" # Automatically map the model to available devices (CPU/GPU)
        )
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(
            self.model,
            lora_config
        )

        self.model.print_trainable_parameters()

        trainable_params = [
            p for p in self.model.parameters() 
            if p.requires_grad
        ]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-4
        )

    def generate(self, question:str, max_new_tokens = 512):
        messages = [
            {
                "role": "user",
                "content" : question
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
            add_generation_prompt = True,
        ).to(self.model.device)

        # Get the length of the input
        input_length = inputs["input_ids"].shape[-1]

        # Generate the output
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens
        )

        # Parse the generated tokens
        generated_tokens = outputs[0][input_length:] # Get the generated tokens after the input
        response = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True

            
        )
        return response

    def calculate_loss(self, question: str, answer: str):

        # -----------------------------
        # Full training conversation
        # -----------------------------
        messages = [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(self.model.device)


        # -----------------------------
        # Prompt only
        # -----------------------------
        prompt_messages = [
            {
                "role": "user",
                "content": question
            }
        ]

        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)


        # -----------------------------
        # Find where answer starts
        # -----------------------------
        prompt_length = prompt_inputs["input_ids"].shape[-1]


        # -----------------------------
        # Create labels
        # -----------------------------
        labels = inputs["input_ids"].clone()


        # -----------------------------
        # Mask prompt tokens
        # -----------------------------
        labels[:, :prompt_length] = -100


        # -----------------------------
        # Forward pass
        # -----------------------------
        outputs = self.model(
            **inputs,
            labels=labels
        )

        return outputs.loss

    def train_step(self, question: str, answer: str):
        self.model.train()

        messages = [
            {
                "role": "user",
                "content":question
            },
            {
                "role": "assistant",
                "content":answer
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
            add_generation_prompt = False 
        ).to(self.model.device)

        # Question Only 
        prompt = [
            {
                "role":"user",
                "content":question
            }
        ]

        prompt_inputs = self.processor.apply_chat_template(
                prompt,
                tokenize = True,
                return_dict = True,
                return_tensors = "pt",
                add_generation_prompt = True
            ).to(self.model.device)

        prompt_length = prompt_inputs["input_ids"].shape[-1]

        labels = inputs["input_ids"].clone()
        labels[:,:prompt_length] = -100

        self.optimizer.zero_grad() # Clear previous gradients
        outputs = self.model(
            **inputs,
            labels=labels
        )

        loss = outputs.loss
        loss.backward() # Backpropagation
        self.optimizer.step() # Update model parameters

        return loss.item()

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)