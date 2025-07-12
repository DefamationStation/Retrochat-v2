import sys
import asyncio
from app.chat_app import ChatApp
from setup.setup_manager import SetupManager
async def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        SetupManager().setup_rchat()
    else:
        app = ChatApp()
        await app.start()

if __name__ == "__main__":
    asyncio.run(main())
