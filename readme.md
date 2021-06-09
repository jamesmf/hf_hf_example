## Health Fact Veracity Task

### Background

The first part of the ML workflow is properly understanding the problem our model aims to solve, as well as the context in which it might be used. In this case, that meant identifying the [paper](https://arxiv.org/abs/2010.09926)  the dataset originates from in order to understand the motivation behind and composition of the `health_fact`/`PUBHEALTH` dataset. The paper contains two tasks on this data, claim veracity classification and summary generation. The piece we care about is the veracity classification, which means we will use the `["claim", "main_text"]` attributes to predict the multiclass `"label"` attribute. 

### Data

The second phase of ML development is to understand the data more in depth. Some important considerations include:

- According to the paper, the `claim` is between `[25, 400]` characters long. In `huggingface`'s dataset, there are a number of claims longer than that, which appear to have their attributes mixed up. For the purposes of this exercise, they will be removed 
- There are only about 11k claims, meaning pretraining is important and scale is unlikley to be a problem
- The dataset was created using heuristics, resulting in some noise. For instance, an example with `{"claim": "Person A said X", "main_text": "Person A said X"}` might be labeled `False` in one dataset because `X` is not true, whereas another source might label it `True` because `Person A` did in fact *say* `X`.
- The `main_text` is the source text that justifies the `label`. It can be as short as a paragraph or a long as a full newspaper article. This makes it potentially important to sample from, truncate, or batch up in order to avoid scaling issues with transformers.
- Some sources (like trusted news content) contain only one `label` (`True` for Reuters, AP). While the authors took some efforts to prevent trivial mappings (like removing the string `AP News`), this is obviously still a source of potential bias
- While the paper suggests the heuristics do a good job of eliminating non-health-related content, examining a sample shows there are some political examples that made it through their lexical filtering (including the first example).
- When considering the usability of this model in production, the ability to get the `"main_text"` attribute is important. To deploy a model like this, we'd need an information retrieval step from a set of trusted sources, possibly aggregating information across multiple articles. Ideally, that IR step would mimic the one(s) used by the experts who curated the justifications in the `"main_text"` attribute.
- The `main_text` attribute varies in length a good deal, with the peak of the distribution around 2000-2400 characters.
- A number of rows have the `label` of -1, which is not documented, but appears to mostly coincide with missing `main_text` attributes. For consistency, these are removed.
- The baseline accuracy for the `train` and `validation` datasets is around `0.51`, as that is the frequency of the most common class.


<img src="https://user-images.githubusercontent.com/7809188/121294327-32b38f00-c8bb-11eb-8f66-21b24ac72906.png" width="400">


### Environment

The project uses `docker` for containerization and `poetry` for dependency management. The `huggingface/datasets` package is used to access the dataset, and the `huggingface/transformers` package is the primary source of pretrained models. The experiment tracking is done via `mlflow` and the resulting artifacts can be stored alongside the code using `git lfs`.

To train the model inside the docker container, activate the conda environment with `$HOME/.poetry/bin/poetry shell` and run `python -m healthfact_example.main`

The image is based off `tensorflow-gpu` in order to come with `CUDA`, `CUDNN`, etc. This saves the complexity of installing those on the host machine.


### Model Architecture

The package is set up to use any transformer from `huggingface/transformers` as the base model, and to use a `summary_strategy` to combine the `main_text` and `claim` columns into a single field, separated by `[SEP]`. The model is then trained on the resulting `text` attribute, predicting the label. 

The paper outlines 3 methodologies for choosing what part of the `main_text` to provide to the model. The first, providing the whole text, yields decent performance. The second, providing random sentences, doesn't even manage to converge to the baseline probability, suggesting an implementation issue in the paper. The most effective method uses `S-BERT` to encode sentences in `main_text` and choose the 5 most semantically similar sentences to the claim. 

Because of the variance in performance, this package implements this as a `strategies` submodule whose behavior is controlled by input arguments, in order to play well with experiment tracking frameworks. To keep the impelementation simple, the first strategy written was simply to take the first `n` tokens from the `main_text`. Since that produced strong results, no other strategies were explored.


### Training

The model was trained using early stopping on the validation set, monitoring accuracy. Early experimentation suggests that the training is somewhat sensitive to `learning_rate`, but with `learning_rate: 3e-5`, the model begins to overfit in about 5 epochs. The length of the input (in tokens) is obviously important, as transformer layers scale poorly with respect to input length. At size `512`, the default maximum of `bert-base-uncased`, very small batch sizes are necessary without a large GPU or model-distributed training. 

No exhaustive hyperparameter search was conducted, but manual exploration produced a set of arguments and choices that led to a model that performed as expected.

Sample MLFlow output:

<img src="https://user-images.githubusercontent.com/7809188/121294573-9f2e8e00-c8bb-11eb-84ef-a5238bd52b91.png" width="600">



### Results

Accuracy, macro-F1, precision, and recall are all reported on the test set, along with a confusion matrix.

For instance, for run `d8098f9d9dec4941917f45679ccdba17`, the metrics were:

```json
{
    "accuracy": 0.69,
    "f1": 0.562,
    "precision": 0.591,
    "recall": 0.547, 
    "confusion": [
        [278,  71,  33,   5],
        [ 88,  65,  40,   8],
        [ 53,  69, 471,   6],
        [ 18,   7,   5,  15]
    ]
}
```

This compares favorably to the baseline model in the paper with respect to accuracy, and similarly on F1. 

The differences between this package's implementation and the paper are likely due to cleaning the handful (~30) of messy datapoints from the dataset.

For comparison, here is the table from `Explainable Automated Fact-Checking for Public Health Claims`, accessed via [arxiv](https://arxiv.org/pdf/2010.09926.pdf) on 2021-06-09

![Results table for comparison](https://user-images.githubusercontent.com/7809188/121294652-c2f1d400-c8bb-11eb-83ed-91fdfd4050a7.png)


### Further Exploration

While this repository stops at implementing a model that performs comparably to the paper associated with the dataset, it is also designed to make further experimentation straightforward. The paper tests multiple pretrained models (`SCIBERT`, `BIOBERT`) and as such that is a good place to begin exploring. As mentioned above, implementing other `strategies` for combining `main_text` with `claim` is also a natural question. Additionally, pre-training any model on a health-news related dataset may also help, whether by masked language modeling or with another task.



### Paper Citation

Citation:
```
@inproceedings{kotonya-toni-2020-explainable,
    title = "Explainable Automated Fact-Checking for Public Health Claims",
    author = "Kotonya, Neema and Toni, Francesca",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods
    in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.623",
    pages = "7740--7754",
}
```