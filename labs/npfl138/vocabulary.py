# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Iterable


class Vocabulary:
    """A class for managing mapping between strings and indices.

    The vocabulary is initialized with a list of strings, and additionally can contain
    two special tokens:

    - a padding token [npfl138.Vocabulary.PAD_TOKEN][], which, if present, is always at index
      [npfl138.Vocabulary.PAD][]=0;
    - an unknown token [npfl138.Vocabulary.UNK_TOKEN][], which, if present, is either at index
      [npfl138.Vocabulary.UNK][] 0 or 1 (depending on whether the padding token is present);
      the index of this token is returned when looking up a string not present in the vocabulary.

    Info:
      A `Vocabulary` instance can be pickled and unpickled efficiently as a list of strings;
      the required string-to-index mapping is reconstructed upon unpickling.
    """
    PAD: int | None
    """The index of the padding token, either `None` or `0`."""
    PAD_TOKEN: str = "[PAD]"
    """The string representing the padding token."""

    UNK: int | None
    """The index of the unknown token, either `None`, `0`, or `1`."""
    UNK_TOKEN: str = "[UNK]"
    """The string representing the unknown token."""

    def __init__(self, strings: Iterable[str], add_pad: bool = False, add_unk: bool = False) -> None:
        """Initialize the vocabulary with the given list of strings.

        The strings might be prepended with special tokens for padding and unknown tokens, respectively,
        depending on the values of `add_pad` and `add_unk`.

        Note:
          If the given strings already contain special tokens on expected indices, they are recognized
          correctly and no duplicates are added even if `add_pad` and/or `add_unk` are `True`.

        Parameters:
          strings: An iterable of strings to include in the vocabulary.
          add_pad: Whether to add a padding token [npfl138.Vocabulary.PAD_TOKEN][] at index 0
            and set [npfl138.Vocabulary.PAD][]=0.
          add_unk: Whether to add an unknown token [npfl138.Vocabulary.UNK_TOKEN][] at index 0 or 1
            (depending on whether the padding token is added) and set [npfl138.Vocabulary.UNK][] accordingly.
        """
        # Get the first two strings (if any) to check for special tokens.
        it, head = iter(strings), []
        try:
            head.append(next(it))
            head.append(next(it))
        except StopIteration:
            it = None

        # Start by adding the special tokens.
        self._strings = []

        if add_pad or (head and head[0] == self.PAD_TOKEN):  # Add PAD token if required or present.
            self.PAD = 0
            self._strings.append(self.PAD_TOKEN)
            (head and head[0] == self.PAD_TOKEN) and head.pop(0)
        else:
            self.PAD = None

        if add_unk or (head and head[0] == self.UNK_TOKEN):  # Add UNK token if required or present.
            self.UNK = len(self._strings)
            (head and head[0] == self.UNK_TOKEN) or self._strings.append(self.UNK_TOKEN)
        else:
            self.UNK = None

        # Now add the remaining strings, both from `head` and from `it`.
        self._strings.extend(head)
        it is not None and self._strings.extend(it)

        self._string_map = {string: index for index, string in enumerate(self._strings)}

    def __len__(self) -> int:
        """The number of strings in the vocabulary.

        Returns:
          The size of the vocabulary.
        """
        return len(self._strings)

    def __iter__(self) -> Iterable[str]:
        """Return an iterator over strings in the vocabulary.

        Returns:
          An iterator over strings in the vocabulary.
        """
        return iter(self._strings)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_string_map"]
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._string_map = {string: index for index, string in enumerate(self._strings)}

    def add(self, string: str) -> int:
        """If not already present, add the given string to the end of the vocabulary.

        Parameters:
          string: The string to add.

        Returns:
          The index of the newly added string (or the index of the existing string if it was already present).
        """
        index = self._string_map.get(string)
        if index is None:
            index = len(self._strings)
            self._strings.append(string)
            self._string_map[string] = index
        return index

    def string(self, index: int) -> str:
        """Convert vocabulary index to string.

        Parameters:
          index: The vocabulary index.

        Returns:
          The string corresponding to the given index.
        """
        return self._strings[index]

    def strings(self, indices: Iterable[int]) -> list[str]:
        """Convert a sequence of vocabulary indices to strings.

        Parameters:
          indices: An iterable of vocabulary indices.

        Returns:
          A list of strings corresponding to the given indices.
        """
        return [self._strings[index] for index in indices]

    def index(self, string: str, add_missing: bool = False) -> int | None:
        """Convert string to vocabulary index.

        Parameters:
          string: The string to convert.
          add_missing: Whether to add the string to the vocabulary if not present.

        Returns:
          The index corresponding to the given string. If the string is not found in the vocabulary, then

            - if `add_missing` is `True`, the string is added to the end of the vocabulary and its index returned;
            - if the [npfl138.Vocabulary.UNK_TOKEN][] was added to the vocabulary, its index is returned;
            - otherwise, `None` is returned.
        """
        if add_missing:
            return self.add(string)
        else:
            return self._string_map.get(string, self.UNK)

    def indices(self, strings: Iterable[str], add_missing: bool = False) -> list[int | None]:
        """Convert a sequence of strings to vocabulary indices.

        Parameters:
          strings: An iterable of strings to convert.
            add_missing: Whether to add strings not present in the vocabulary.

        Returns:
          A list of indices corresponding to the given strings. For each string not found in the vocabulary:

            - if `add_missing` is `True`, the string is added to the end of the vocabulary and its index returned;
            - if the [npfl138.Vocabulary.UNK_TOKEN][] was added to the vocabulary, its index is returned;
            - otherwise, `None` is returned.
        """
        if add_missing:
            return [self.add(string) for string in strings]
        else:
            return [self._string_map.get(string, self.UNK) for string in strings]
