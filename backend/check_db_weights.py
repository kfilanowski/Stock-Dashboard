
import asyncio
from app.database import init_db, get_db, engine
from app.models import CalibrationWeights
from sqlalchemy import select

async def check_weights():
    await init_db()
    async for db in get_db():
        result = await db.execute(select(CalibrationWeights))
        weights = result.scalars().all()
        print(f"Total calibration weights found: {len(weights)}")
        for w in weights[:10]:
            print(f"Ticker: {w.ticker}, Horizon: {w.horizon}, Indicator: {w.indicator}, Weight: {w.weight}")
        break
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(check_weights())
