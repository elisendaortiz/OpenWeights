"""Terminal audio entrypoint; implementation lives in services/audio.py."""
from services.audio import main

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
