import random


ANNOTATORS = {
    "ann_1": 0.9,  # high quality
    "ann_2": 0.7,
    "ann_3": 0.5,
    "ann_4": 0.3,  # noisy
}


def load_dataset():
    base = [
        ("I love this!", "positive"),
        ("Worst ever", "negative"),
        ("It's okay", "neutral"),
        ("Absolutely amazing", "positive"),
        ("Terrible service", "negative"),
        ("Not bad", "neutral"),
        ("Fantastic product", "positive"),
        ("I hate it", "negative"),
        ("Meh", "neutral"),
        ("Very good", "positive"),
    ]

    data = []

    random.seed(42)

    for text, true_label in base:
        annotator = random.choice(list(ANNOTATORS.keys()))
        reliability = ANNOTATORS[annotator]

        # Inject noise
        if random.random() > reliability:
            given_label = random.choice(["positive", "negative", "neutral"])
        else:
            given_label = true_label

        data.append({
            "text": text,
            "true": true_label,
            "given": given_label,
            "annotator": annotator
        })

    random.shuffle(data)
    return data
