"""Terminal time-agent entrypoint."""
import asyncio
from services.agent import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
