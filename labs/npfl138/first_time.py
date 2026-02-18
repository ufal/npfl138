# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

seen_tags: set[str] = set()


def first_time(tag: str) -> bool:
    """Returns `True` when first called with the given `tag`."""
    if tag in seen_tags:
        return False

    seen_tags.add(tag)
    return True
