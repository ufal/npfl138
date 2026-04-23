### Assignment: sentiment_analysis
#### Date: Deadline: May 06, 22:00
#### Points: 2 points

Perform sentiment analysis on Czech Facebook data using a provided pre-trained
Czech Electra model [`eleczech-lc-small`](https://huggingface.co/ufal/eleczech-lc-small).
The dataset consists of pairs of _(document, label)_ and can be (down)loaded using the
[TextClassificationDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/text_classification_dataset/)
class.

Even though this assignment is not a competition, your goal is to submit test
set annotations with at least 77% accuracy. You can evaluate your predictions
as usual using the
[TextClassificationDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/text_classification_dataset/)
class, either by running `python3 -m npfl138.datasets.text_classification_dataset --evaluate=path --dataset=dev/test`
or by calling the [TextClassificationDataset.evaluate](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/text_classification_dataset/#npfl138.datasets.text_classification_dataset.TextClassificationDataset.evaluate)
method.

Note that contrary to working with EfficientNet, you **need** to **finetune**
the Electra model in order to achieve the required accuracy.

You can start with the
[sentiment_analysis.py](https://github.com/ufal/npfl138/tree/master/labs/10/sentiment_analysis.py)
template, which among others loads the Electra Czech model and generates test
set annotations in the required format. Note that
[example_transformers.py](https://github.com/ufal/npfl138/tree/master/labs/10/example_transformers.py)
module illustrates the usage of both the Electra tokenizer and the Electra model.
