from pathlib import Path


def test_min_example_1():
    import countergen

    augmented_ds = countergen.AugmentedDataset.from_default("male-stereotypes")
    api_model = countergen.api_to_generative_model("text-davinci-001")
    model_evaluator = countergen.get_generative_model_evaluator(api_model)
    countergen.evaluate_and_print(augmented_ds.samples, model_evaluator)


def test_min_example_2(tmpdir: Path):
    lines = [
        """{"input": "a", "outputs":["b", "c"]}""",
        """{"input": "XYZ", "outputs":["WVU"]}""",
    ]
    path = tmpdir / "my_data.jsonl"
    path = tmpdir / "my_data_augmented.jsonl"
    path.write_text("\n".join(lines), encoding="UTF-8")

    import countergen

    ds = countergen.Dataset.from_jsonl(str(path))
    augmenters = [countergen.SimpleAugmenter.from_default("gender")]
    augmented_ds = ds.augment(augmenters)
    augmented_ds.save_to_jsonl(str(path))


def test_min_example_3():
    import countergen as cg
    import countergentorch as cgt
    from transformers import GPT2LMHeadModel

    augmented_ds = cg.AugmentedDataset.from_default("male-stereotypes")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    layers = cgt.get_mlp_modules(model, [2, 3])
    activation_ds = cgt.ActivationsDataset.from_augmented_samples(augmented_ds.samples, model, layers)
    # INLP is an algorithm to find important directions in a dataset
    dirs = cgt.inlp(activation_ds)
    configs = cgt.get_edit_configs(layers, dirs, has_leftover=False)
    new_model = cgt.edit_model(model, configs=configs)
