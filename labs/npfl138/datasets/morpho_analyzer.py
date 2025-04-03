# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
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

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2425/datasets/"

    class LemmaTag:
        """A class representing a morphological analysis of a word."""
        def __init__(self, lemma: str, tag: str) -> None:
            self.lemma = lemma
            """A lemma of the word."""
            self.tag = tag
            """A tag of the word."""

        def __repr__(self) -> str:
            return "(lemma: {}, tag: {})".format(self.lemma, self.tag)

    def __init__(self, dataset: str) -> None:
        """Loads the morphological analyses from the specified dataset."""
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
        """Returns a (possibly empty) list of morphological analyses for the given word."""
        return self.analyses.get(word, [])
