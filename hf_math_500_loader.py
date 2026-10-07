from datasets import load_dataset

ds = load_dataset("qwedsacf/competition_math")

for split, dataset in ds.items():
    for cache_file in dataset.cache_files:
        print(split, cache_file["filename"])