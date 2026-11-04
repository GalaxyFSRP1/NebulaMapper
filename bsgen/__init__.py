import logging
import os
import pathlib
import sys

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_orig_read_text = pathlib.Path.read_text
_orig_write_text = pathlib.Path.write_text


def _utf8_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)


def _utf8_write_text(self, data, encoding=None, errors=None, newline=None):
    return _orig_write_text(self, data, encoding=encoding or "utf-8", errors=errors, newline=newline)


pathlib.Path.read_text = _utf8_read_text
pathlib.Path.write_text = _utf8_write_text


class _IgnoreInvalidHttpFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "Invalid HTTP request received" not in msg


for _logger_name in ("uvicorn.error", "uvicorn"):
    logging.getLogger(_logger_name).addFilter(_IgnoreInvalidHttpFilter())
