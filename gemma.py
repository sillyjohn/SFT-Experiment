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

    def prepare_batch(self, questions, answers, max_length =2048):
        # Comment usage
        Question_Answer_Pairs = []
        Questions_Only = []

        # Comment usage
        for question, answer in zip(questions,answers):
            # For Question and Answer inputs
            full_msg = [
                {
                    "role":"user",
                    "content":question
                },
                {
                    "role": "assistant",
                    "content":answer
                }
            ]

            # Comment usage
            full_text = self.processor.apply_chat_template(
                full_msg,
                tokenize = False,
                add_generation_prompt = False
            )

            Question_Answer_Pairs.append(full_text)

            # For Question only
            prompt_msg = [
                {
                    "role":"user",
                    "content":question
                }
            ]

            # Comment usage
            prompt_text = self.processor.apply_chat_template(
                prompt_msg,
                tokenize = False,
                add_generation_prompt = True
            )

            Questions_Only.append(prompt_text)
            
            # Tokenize the Q&A inputs
            qa_inputs = tokenizer(
                Question_Answer_Pairs,
                padding = True,
                truncation = True,
                max_lenght = max_lenght,
                return_tensors = "pt",
                add_special_tokens = False
            )

            # Tokenize the question only
            tokenized_questions = tokenizer(
                Questions_Only,
                padding = True,
                truncation = True,
                max_lenght = max_lenght,
                return_tensors = "pt",
                add_special_tokens = False
            )
            
            # Move tensors to model device
            qa_inputs = {
                key: value.to(self.model.device)
                for key,value in qa_inputs.items()
            }

            tokenized_questions = {
                key : value.to(self.model.device)
                for key,value in tokenized_questions.items()
            }

            labels = qa_inputs["input_ids"].clone()
            
            # Ignore padding
            labels[
                qa_inputs["attention_mask"] == 0
            ] = -100

            # Ignore prompt tokens
            prompt_lengths = tokenized_questions["attention_mask"].sum(dim=1)

            for i, prompt_length in enumerate(
                prompt_lengths
            ):
                labels[
                    i,
                    :prompt_length.item()
                ] = -100

            return qa_inputs, labels

    def calculate_batch_loss(self, questions, answers):
        inputs, labels = self.prepare_batch(questions,answers)
        output = self.model(**inputs, labels = labels)
        return output.loss



    def generate(self, question:str, max_new_tokens = 512):
        self.model.eval()
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
        with torch.no_grad():
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

    # Deprecated, use calculate_batch_loss instead
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

    # depercated, use calculate_batch_loss instead
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