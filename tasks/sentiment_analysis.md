### Assignment: sentiment_analysis
#### Date: Deadline: May 07, 22:00
#### Points: 2 points

Perform sentiment analysis on Czech Facebook data using a provided pre-trained
Czech Electra model [`eleczech-lc-small`](https://huggingface.co/ufal/eleczech-lc-small).
The dataset consists of pairs of _(document, label)_ and can be (down)loaded using the
[text_classification_dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/docs/datasets/text_classification_dataset/)
module.

Even though this assignment is not a competition, your goal is to submit test
set annotations with at least 77% accuracy. As usual, you can evaluate your
predictions using the
[text_classification_dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/docs/datasets/text_classification_dataset/)
module, either by running `python3 -m npfl138.datasets.text_classification_dataset --evaluate=path --dataset=dev/test`
or by calling the `TextClassificationDataset.evaluate` method.

Note that contrary to working with EfficientNet, you **need** to **finetune**
the Electra model in order to achieve the required accuracy.

You can start with the
[sentiment_analysis.py](https://github.com/ufal/npfl138/tree/master/labs/10/sentiment_analysis.py)
template, which among others loads the Electra Czech model and generates test
set annotations in the required format. Note that
[example_transformers.py](https://github.com/ufal/npfl138/tree/master/labs/10/example_transformers.py)
module illustrates the usage of both the Electra tokenizer and the Electra model.
