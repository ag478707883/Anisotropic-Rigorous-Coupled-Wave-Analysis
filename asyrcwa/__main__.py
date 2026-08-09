from __future__ import annotations

from . import _backend, module_directory


def main() -> None:
    print(f"backend: {getattr(_backend, '__file__', '<unknown>')}")
    print(f"module directory: {module_directory()}")


if __name__ == "__main__":
    main()
