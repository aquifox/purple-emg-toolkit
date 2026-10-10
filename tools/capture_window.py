from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import time
from pathlib import Path

from PIL import ImageGrab


user32 = ctypes.windll.user32


def find_window(title_fragment: str) -> int | None:
    matches: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)

    @callback_type
    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if title_fragment.casefold() in buffer.value.casefold():
            matches.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    hwnd = None
    while time.monotonic() < deadline:
        hwnd = find_window(args.title)
        if hwnd:
            break
        time.sleep(0.25)
    if not hwnd:
        raise SystemExit(f"Window not found: {args.title}")

    user32.ShowWindow(hwnd, 9)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.75)

    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError()
    if rect.right <= rect.left or rect.bottom <= rect.top:
        raise SystemExit("Window has an invalid rectangle")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    right = min(rect.right, screen_width)
    bottom = min(rect.bottom, screen_height - 44)
    image = ImageGrab.grab(
        bbox=(max(0, rect.left), max(0, rect.top), right, bottom),
        all_screens=True,
    )
    image.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

