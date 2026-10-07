import argparse

from kaggle_dataloader import KaggleDataLoader
from gemma import GemmaModel
from transformers import AutoProcessor, AutoModelForMultimodalLM
from training_loops import TrainingLoops
import torch

MODEL_ID = "google/gemma-4-E2B-it"
PATH = "/Users/johntsoi/.cache/kagglehub/datasets/alpie/mathreasoning/versions/1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',action = 'store_true')
    parser.add_argument('--ep',type=int,default=1)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--gstep',type=int,default=4)
    parser.add_argument('--bsize', type=int, default = 4)
    parser.add_argument('--tp',type=float,default=0.8)
    parser.add_argument('--vp', type=float, default=0.2)
    args = parser.parse_args()

    dataset = KaggleDataLoader(PATH)
    model = GemmaModel(MODEL_ID)
    
    training = TrainingLoops(model,dataset,args.tp,args.vp,args.bsize,args.gstep)
    training.train()
    

    


if __name__ == "__main__":
    main()