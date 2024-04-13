import os
import sys
import urllib.request
import zipfile


class MorphoAnalyzer:
    """ Loads a morphological analyses in a vertical format.

    The analyzer provides only a method `get(word: str)` returning a list
    of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2324/datasets/"

    class LemmaTag:
        def __init__(self, lemma: str, tag: str) -> None:
            self.lemma = lemma
            self.tag = tag

        def __repr__(self) -> str:
            return "(lemma: {}, tag: {})".format(self.lemma, self.tag)

    def __init__(self, dataset: str) -> None:
        path = "{}.zip".format(dataset)
        if not os.path.exists(path):
            print("Downloading dataset {}...".format(dataset), file=sys.stderr)
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
            os.rename("{}.tmp".format(path), path)

        self.analyses = {}
        with zipfile.ZipFile(path, "r") as zip_file:
            with zip_file.open("{}.txt".format(dataset), "r") as analyses_file:
                for line in analyses_file:
                    line = line.decode("utf-8").rstrip("\n")
                    columns = line.split("\t")

                    analyses = []
                    for i in range(1, len(columns) - 1, 2):
                        analyses.append(self.LemmaTag(columns[i], columns[i + 1]))
                    self.analyses[columns[0]] = analyses

    def get(self, word: str) -> list[LemmaTag]:
        return self.analyses.get(word, [])
