"""Terminal image entrypoint; implementation lives in services/image.py."""
from services.image import main

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
