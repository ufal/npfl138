### Assignment: reading_comprehension
#### Date: Deadline: May 06, 22:00
#### Points: 4 points+5 bonus

Implement the best possible model for reading comprehension task using
an automatically translated version of the SQuAD 1.1 dataset, utilizing a provided
Czech RoBERTa model [`ufal/robeczech-base`](https://huggingface.co/ufal/robeczech-base).

The dataset can be loaded using the
[ReadingComprehensionDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/reading_comprehension_dataset/)
class. The loaded dataset is a direct representation of the data and not yet
ready to be directly trained on. Each of the `train`, `dev` and `test` datasets
are composed of a list of paragraphs, each consisting of:
- `context`: text with various information;
- `qas`: list of questions and answers, where each item consists of:
  - `question`: text of the question;
  - `answers`: a list of answers, each answer is composed of:
    - `text`: answer text as string, exactly as appearing in the context;
    - `start`: character offset of the answer text in the context.

In the `train` and `dev` sets, each question has exactly one answer, while in
the `test` set, there might be several answers. We evaluate the reading
comprehension task using _accuracy_, where an answer is considered correct if
its text is exactly equal to some correct answer. You can evaluate your
predictions as usual with the
[ReadingComprehensionDataset](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/reading_comprehension_dataset/)
class, either by running `python3 -m npfl138.datasets.reading_comprehension_dataset --evaluate=path --dataset=dev/test`
or by calling the [ReadingComprehensionDataset.evaluate](https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/docs/datasets/reading_comprehension_dataset/#npfl138.datasets.reading_comprehension_dataset.ReadingComprehensionDataset.evaluate)
method.

The task is a [_competition_](https://ufal.mff.cuni.cz/courses/npfl138/2526-summer#competitions). Everyone who submits
a solution with at least **65%** answer accuracy gets 4 points; the remaining 5 points
are distributed depending on the relative ordering of your solutions. Note that
usually achieving **62%** on the `dev` set is enough to get 65% on the `test`
set (because of multiple references in the `test` set).

Note that contrary to working with EfficientNet, you **need** to **finetune**
the RobeCzech model in order to achieve the required accuracy.

You can start with the
[reading_comprehension.py](https://github.com/ufal/npfl138/tree/master/labs/10/reading_comprehension.py)
template, which among others (down)loads the data and the RobeCzech model, and which contains
the description of the required test set annotation format.
