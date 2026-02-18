import asyncio
import pandas as pd
from sqlalchemy import select, func
from app.api.data import load_dataframe
from app.core.database import engine, Base, AsyncSessionLocal
from app.models.passenger import Passenger

async def init_db():
    # 1. Create tables (if not exist)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("Database structure checked/created")

    async with AsyncSessionLocal() as session:
        # 2. CHECK: Are there any data in the database?
        result = await session.execute(select(func.count()).select_from(Passenger))
        count = result.scalar()

        if count > 0:
            print(f"Database already contains {count} passengers. Skipping import.")
            return # Go out without doing anything
        
        # 3. If DB is empty - loading CSV
        print("Database is empty. Loading data from CSV...")
        df = load_dataframe()

        passengers = []
        for _, row in df.iterrows():
            # Processing NaN -> None
            age = row['Age'] if pd.notna(row['Age']) else None
            cabin = row['Cabin'] if pd.notna(row['Cabin']) else None
            embarked = row['Embarked'] if pd.notna(row['Embarked']) else None

            passenger = Passenger (
                PassengerId = row['PassengerId'],
                Survived = row['Survived'],
                Pclass = row['Pclass'],
                Name = row['Name'],
                Sex = row['Sex'],
                Age = age,
                SibSp = row['SibSp'],
                Parch = row['Parch'],
                Ticket = row['Ticket'],
                Fare = row['Fare'],
                Cabin = cabin,
                Embarked = embarked
            )
            passengers.append(passenger)

        session.add_all(passengers)
        await session.commit()
        print(f"Successfully imported {len(passengers)} rows!")

if __name__ == "__main__":
    try: 
        asyncio.run(init_db())
    except Exception as e:
        print(f"Error during migration: {e}")