# This file is part of NPFL138 <https://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import zipfile

from .downloader import download_url_to_file


class MorphoAnalyzer:
    """ Loads a morphological analyses in a vertical format.

    The analyzer provides only a method `get(word: str)` returning a list
    of analyses, each containing two fields `lemma` and `tag`.
    If an analysis of the word is not found, an empty list is returned.
    """

    URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl138/2526/datasets"

    class LemmaTag:
        """A class representing a morphological analysis of a word."""
        def __init__(self, lemma: str, tag: str) -> None:
            self.lemma = lemma
            """A lemma of the word."""
            self.tag = tag
            """A tag of the word."""

        def __repr__(self) -> str:
            return f"(lemma: {self.lemma}, tag: {self.tag})"

    def __init__(self, dataset: str) -> None:
        """Loads the morphological analyses from the specified dataset."""
        path = download_url_to_file(self.URL, f"{dataset}.zip")

        self.analyses = {}
        with zipfile.ZipFile(path, "r") as zip_file:
            with zip_file.open(f"{dataset}.txt", "r") as analyses_file:
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
