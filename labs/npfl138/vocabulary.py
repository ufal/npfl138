# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Iterable, Sequence


class Vocabulary:
    """ A class for managing mapping between strings and indices.

    The vocabulary is initialized with a list of strings. It provides:

    - `__len__`: the number of strings in the vocabulary,
    - `__iter__`: the iterator over strings in the vocabulary,
    - `string(index: int) -> str`: the string for a given vocabulary index,
    - `strings(indices: Sequence[int]) -> list[str]`: the list of strings for the given indices,
    - `index(string: str) -> int`: the index of a given string in the vocabulary,
    - `indices(strings: Sequence[str]) -> list[int]`: the list of indices for given strings.
    """
    PAD: int = 0
    """The index of the padding token."""
    UNK: int = 1
    """The index of the unknown token, if present."""

    def __init__(self, strings: Sequence[str], add_unk: bool = False) -> None:
        """Initializes the vocabulary with the given list of strings.

        The `Vocabulary.PAD` is always the first token in the vocabulary;
        `Vocabulary.UNK` is the second token but only when `add_unk=True`.
        """
        self._strings = ["[PAD]"] + (["[UNK]"] if add_unk else [])
        self._strings.extend(strings)
        self._string_map = {string: index for index, string in enumerate(self._strings)}
        if not add_unk:
            self.UNK = None

    def __len__(self) -> int:
        """Returns the number of strings in the vocabulary."""
        return len(self._strings)

    def __iter__(self) -> Iterable[str]:
        """Returns an iterator over strings in the vocabulary."""
        return iter(self._strings)

    def string(self, index: int) -> str:
        """Returns the string for a given vocabulary index."""
        return self._strings[index]

    def strings(self, indices: Sequence[int]) -> list[str]:
        """Returns the list of strings for the given indices."""
        return [self._strings[index] for index in indices]

    def index(self, string: str) -> int | None:
        """Returns the index of a given string in the vocabulary."""
        return self._string_map.get(string, self.UNK)

    def indices(self, strings: Sequence[str]) -> list[int | None]:
        """Returns the list of indices for given strings."""
        return [self._string_map.get(string, self.UNK) for string in strings]
