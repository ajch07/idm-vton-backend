import asyncio
from app.database import engine, Base

async def reset_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        print('✅ All tables dropped successfully')
        await conn.run_sync(Base.metadata.create_all)
        print('✅ All tables created successfully')

if __name__ == "__main__":
    asyncio.run(reset_db())
