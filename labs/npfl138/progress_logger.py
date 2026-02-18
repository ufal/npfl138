# This file is part of NPFL138 <http://github.com/ufal/npfl138/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Callable, Iterable
import os
import sys

import tqdm

from .loggers import BaseLogger
from .type_aliases import Logs
from .utils import compute_logs


class ProgressLogger(tqdm.tqdm):
    """A slim wrapper around `tqdm.tqdm` for showing a progress bar, optionally with logs."""

    monitor_interval = 0  # Disable internal monitoring thread.

    _report_only_first = int(os.environ.get("NPFL_PROGRESS_FIRST", -1))  # Optional global limit to first N reports.

    @staticmethod
    def get_console_verbosity(console: int | None) -> int:
        if console is None and "NPFL_PROGRESS" in os.environ:
            console = int(os.environ["NPFL_PROGRESS"])
        elif console is None and ("NPFL_PROGRESS_FIRST" in os.environ or "NPFL_PROGRESS_EACH" in os.environ):
            console = 3 if ProgressLogger._report_only_first != 0 else 1
        elif console is None:
            console = 2
        return console

    def __init__(
        self,
        data: Iterable,
        description: str,
        console: int | None = None,
        logs_fn: Callable[[], Logs] | None = None,
    ) -> None:
        """Create a ProgressLogger instance.

        Parameters:
          data: Any iterable data to wrap, usually a [torch.utils.data.DataLoader][].
          description: A description string to show in front of the progress bar.
          console: Controls the console verbosity: 0 and 1 for silent, 2 for
            only-when-writing-to-console progress bar, 3 for persistent progress bar.
            The default is 2, but can be overridden by the `NPFL_PROGRESS` environment variable.
          logs_fn: An optional function returning the current logs to show alongside the progress bar.
            If given, the logs are fully computed and shown on each refresh.
        """
        console = self.get_console_verbosity(console)

        kwargs = {}
        if "NPFL_PROGRESS_EACH" in os.environ:
            kwargs["miniters"] = int(os.environ["NPFL_PROGRESS_EACH"])
            kwargs["mininterval"] = None

        self._console = console
        self._description = description
        self._logs_fn = logs_fn
        super().__init__(data, unit="batch", leave=False, disable=None if console == 2 else console < 2, **kwargs)

    def refresh(self, nolock=False, lock_args=None):
        if ProgressLogger._report_only_first > 0:
            ProgressLogger._report_only_first -= 1
        elif ProgressLogger._report_only_first == 0:
            return

        description = self._description
        if self._logs_fn is not None:
            description += (description and " ") + BaseLogger.format_metrics(compute_logs(self._logs_fn()))
        self.set_description(description, refresh=False)

        super().refresh(nolock=nolock, lock_args=lock_args)

    @staticmethod
    def log_console(message: str, end: str = "\n", progress_only: bool = False, console: int | None = None) -> None:
        """Write the given message to the console, correctly even if a progress bar is being used.

        Parameters:
          message: The message to write.
          end: The string appended after the message.
          progress_only: If `False` (the default), the message is written to standard output when current console
            verbosity is at least 1; if `True`, the message is written to standard error only when the progress bar
            is being shown (console verbosity 2 and writing to the console, or console verbosity 3).
          console: Controls the current console verbosity. The default is 2, but can be overridden by the
            `NPFL_PROGRESS` environment variable.
        """
        console = ProgressLogger.get_console_verbosity(console)
        if progress_only and ((console == 2 and sys.stderr.isatty()) or console >= 3):
            tqdm.tqdm.write(message, end=end, file=sys.stderr)
        elif (not progress_only) and console >= 1:
            tqdm.tqdm.write(message, end=end, file=sys.stdout)
