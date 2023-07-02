def load_dataset(dataset_name, splits=None):
    from .loaders import DATASET_CLASSES

    try:
        dataset = DATASET_CLASSES[dataset_name]()
    except ValueError:
        print(f"Dataset {dataset_name} does not exist. Available datasets: {DATASET_CLASSES.keys()}")

    if splits:
        for split in splits:
            dataset.load(split)
    else:
        dataset.load()

    return dataset