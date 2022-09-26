from countergen.tools.cli import run


if __name__ == "__main__":
    run()

    # python -m countergen augment countergen\data\datasets\tiny-test.jsonl countergen\data\augdatasets\tiny-test.jsonl gender
    # python -m countergen augment countergen\data\datasets\twitter-sentiment.jsonl countergen\data\augdatasets\twitter-sentiment.jsonl gender
    # python -m countergen augment countergen\data\datasets\doublebind.jsonl countergen\data\augdatasets\doublebind.jsonl gender
    # python -m countergen augment countergen\data\datasets\hate-test.jsonl countergen\data\augdatasets\hate-test.jsonl gender
    # python -m countergen evaluate countergen\data\augdatasets\doublebind.jsonl
    # python -m countergen evaluate countergen\data\augdatasets\doublebind.jsonl --model-name text-davinci-001
    # python -m countergen augment countergen\data\datasets\doublebind-heilman.jsonl countergen\data\augdatasets\doublebind-heilman.jsonl paraphrase gender
