import argparse
from torch.utils.data import Subset
from kaggle_dataloader import KaggleDataLoader
from gemma import GemmaModel
from training_loops import TrainingLoops


MODEL_ID = "google/gemma-4-E2B-it"
PATH = "/Users/johntsoi/.cache/kagglehub/datasets/alpie/mathreasoning/versions/1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type= int, default = 100)
    parser.add_argument('--verbose',action = 'store_true')
    parser.add_argument('--ep',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--gstep',type=int,default=2)
    parser.add_argument('--bsize', type=int, default = 1)
    parser.add_argument('--tp',type=float,default=0.8)
    parser.add_argument('--vp', type=float, default=0.2)
    args = parser.parse_args()

    dataset = KaggleDataLoader(PATH)
    model = GemmaModel(MODEL_ID)

    dataset = Subset(
        dataset,
        range(
            min(args.samples, len(dataset))
        )
    )

    training = TrainingLoops(
        model=model,
        dataset=dataset,
        training_portion=args.tp,
        validation_portion=args.vp,
        batch_size=args.bsize,
        gradient_accumulation_steps=args.gstep,
        epochs=args.ep,
        lr=args.lr
    )
    training.train()
    

    


if __name__ == "__main__":
    main()