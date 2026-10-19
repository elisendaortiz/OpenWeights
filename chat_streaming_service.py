"""Terminal chat entrypoint."""
from services.chat import main

if __name__ == "__main__":
    main(streaming=True)
