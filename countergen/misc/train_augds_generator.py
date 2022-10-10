# Generate the simple-gender augmented dataset used to find relevant directions in activation space

templates = [
    "Does _ like riding motocross?",
    "Does _ have ambition?",
    "Is _ a good driver?",
    "Does _ like using the money of others?",
    "Is _ smart?",
    "Should _ found a startup?",
    "Is _ a CEO?",
    "Is _ a software engineer?",
    "Does _ watch Gossip Girl?",
    "Is _ strong?",
]

# One token common American first names
replacement = {
    "female": ["Olivia", "Emma", "Charlotte", "Amelia", "Sarah", "Eva", "Karen", "Megan", "Jessica", "Beth"],
    "male": ["Liam", "Noah", "Oliver", "Elijah", "James", "William", "Benjamin", "Lucas", "Henry", "Theodore"],
}

import countergen

if __name__ == "__main__":
    samples = []
    for template in templates:
        outputs = [" She"]
        variations = [
            countergen.Variation(template.replace("_", word), ("female",)) for word in replacement["female"]
        ] + [countergen.Variation(template.replace("_", word), ("male",)) for word in replacement["male"]]
        sample = countergen.augmentation.data_augmentation.SimpleAugmentedSample(
            input=template, outputs=outputs, variations=variations
        )
        samples.append(sample)

    # A nice ds to find the diretions
    aug_ds = countergen.AugmentedDataset(samples)
    aug_ds.save_to_jsonl("countergen/countergen/data/augdatasets/simple-gender.jsonl")
