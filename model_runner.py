from kaggle_dataloader import KaggleDataLoader
from gemma import GemmaModel
from transformers import AutoProcessor, AutoModelForMultimodalLM
import torch

MODEL_ID = "google/gemma-4-E2B-it"
PATH = "/Users/johntsoi/.cache/kagglehub/datasets/alpie/mathreasoning/versions/1"

def main():
    dataset = KaggleDataLoader(PATH)
    model = GemmaModel(MODEL_ID)

    example = dataset[0]
    question = example["question"]
    answer = example["answer"]

    # print(dataset.get_columns_name())
    # print(dataset.__getitem__(0))  
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    prediction = model.generate(question)
    loss = model.calculate_loss(question, answer)
    
    for i in range(20):
        loss = model.train_step(question, answer)
        print(f"Epoch {i+1}: Loss = {loss}")

    model.save_model("./gemma_math_lora")

    # ------------------------
    # AFTER TRAINING
    # ------------------------

    after = model.generate(question)

    print("\nAFTER:")
    print(after)

    print("\nCORRECT ANSWER:")
    print(answer)
q

if __name__ == "__main__":
    main()