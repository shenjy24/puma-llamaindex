import asyncio

async def fetch_data(delay: int) -> str:
    print(f"Started fetching data with {delay}s delay")
    # Simulates I/O-bound work, such as network operation
    await asyncio.sleep(delay)
    print("Finished fetching data")
    return f"Data fetched after {delay} seconds"

async def main():
    print("Starting main function")
    # Schedule multiple fetch_data calls concurrently
    tasks = [
        fetch_data(2),
        fetch_data(3),
        fetch_data(1)
    ]
    results = await asyncio.gather(*tasks)
    print("All data fetched:")
    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())