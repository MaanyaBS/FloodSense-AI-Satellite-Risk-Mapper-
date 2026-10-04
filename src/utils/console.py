"""
src/utils/console.py
Make CLI output safe on Windows consoles.

Windows consoles default to a legacy code page (cp1252 on most Western
installs), which cannot encode characters like the arrows, box-drawing
glyphs and emoji used throughout the progress messages. Any print() of
such a character raises UnicodeEncodeError and kills the process.

Reconfiguring stdout/stderr to UTF-8 with errors="replace" keeps the
output readable and never raises: unrepresentable glyphs degrade to '?'
instead of aborting a long training run.
"""

import sys


def enable_utf8_console() -> bool:
    """
    Reconfigure stdout and stderr to UTF-8 where supported.

    Returns:
        True if reconfiguration succeeded, False if the stream does not
        support it (some redirected streams lack reconfigure()).
    """
    ok = False

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
            ok = True
        except (ValueError, OSError):
            # Stream is detached or already closed; leave it alone.
            continue

    return ok