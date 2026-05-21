from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv

from data_fetchers import EtherscanFetcher


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    if not api_key:
        print("ETHERSCAN_API_KEY is missing")
        return

    addresses = [
        "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
        "0x28C6c06298d514Db089934071355E5743bf21d60",
        "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    ]

    fetcher = EtherscanFetcher(api_key)
    rows = []
    for addr in addresses:
        data = fetcher.fetch_all(addr)
        data["address"] = addr
        rows.append(data)

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "etherscan_sample_enriched.csv"
    df.to_csv(out_file, index=False)

    cols = [
        "address",
        "balance",
        "transaction_count",
        "activity_score",
        "avg_tx_value",
        "tx_frequency",
        "unique_counterparties",
    ]
    print(df[cols].to_string(index=False))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
