# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

__version__ = "2425.13.0"


def require_version(required_version: str) -> None:
    """Make sure the installed version is at least `required_version`.

    If not, show a nice error message with the instructions how to upgrade.

    Parameters:
      required_version: The required version of the npfl138 package, as
        a string in the format "semester.week" or "semester.week.patch".
    """

    required = required_version.split(".")
    assert len(required) <= 3, "Expected at most 3 version components"

    required = list(map(int, required))
    current = list(map(int, __version__.split(".")))

    assert current[:len(required)] >= required, (
        f"The npfl138>={required_version} is required, but found only {__version__}.\n"
        f"Please update the npfl138 package by running either:\n"
        f"- `VENV_DIR/bin/pip install --upgrade npfl138` when using a venv, or\n"
        f"- `python3 -m pip install --user --upgrade npfl138` otherwise.")
