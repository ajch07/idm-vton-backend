import asyncio
from sqlalchemy import text
from app.database import engine

async def drop_all():
    async with engine.begin() as conn:
        await conn.execute(text('DROP SCHEMA public CASCADE'))
        await conn.execute(text('CREATE SCHEMA public'))
        print('✅ Database reset complete')

if __name__ == "__main__":
    asyncio.run(drop_all())
