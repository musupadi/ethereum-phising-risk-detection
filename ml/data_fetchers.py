"""
ml/data_fetchers.py
-------------------
Fetches wallet data from external APIs:
- Etherscan: transaction counts, balance, wallet activity, transaction edges
- Forta Network: phishing/scam/exploit labels
"""

import json
import logging
import time
from typing import Optional, Dict, List
from pathlib import Path

import pandas as pd
import requests

# ── Settings ───────────────────────────────────────────────────────────────────
ETHERSCAN_API_BASE = "https://api.etherscan.io/v2/api"
FORTA_API_BASE = "https://api.forta.network"

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class EtherscanFetcher:
    """Fetch wallet data from Etherscan API."""

    def __init__(self, api_key: Optional[str] = None, chain_id: int = 1, request_delay: float = 0.4):
        """
        Initialize Etherscan fetcher.
        
        Args:
            api_key: Etherscan API key (from environment or parameter)
        """
        self.api_key = api_key or ""
        self.base_url = ETHERSCAN_API_BASE
        self.chain_id = int(chain_id)
        self.request_delay = float(request_delay)

    def get_account_info(self, address: str) -> Dict:
        """
        Fetch wallet account info: balance, transaction count, etc.
        
        Args:
            address: Ethereum wallet address
            
        Returns:
            Dictionary with keys: balance, transaction_count, created_at
        """
        time.sleep(self.request_delay)
        
        params = {
            "chainid": self.chain_id,
            "module": "account",
            "action": "balance",
            "address": address,
            "apikey": self.api_key,
        }
        
        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("status") != "1":
                log.warning("Etherscan returned non-1 status for %s: %s", address, data.get("message"))
                return {
                    "balance": 0,
                    "transaction_count": 0,
                    "created_at": None,
                    "error": data.get("message")
                }
            
            return {
                "balance": float(data.get("result", 0)) / 1e18,  # Convert Wei to ETH
                "transaction_count": None,  # Will fetch separately
                "created_at": None,
            }
        except Exception as e:
            log.error("Error fetching from Etherscan for %s: %s", address, e)
            return {"balance": 0, "transaction_count": 0, "created_at": None, "error": str(e)}

    def get_transaction_count(self, address: str) -> int:
        """
        Fetch number of transactions for a wallet.
        
        Args:
            address: Ethereum wallet address
            
        Returns:
            Number of transactions (or 0 on error)
        """
        time.sleep(self.request_delay)
        
        params = {
            "chainid": self.chain_id,
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": 10000,
            "sort": "desc",
            "apikey": self.api_key,
        }
        
        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("status") != "1":
                return 0
            
            result = data.get("result", [])
            return len(result)
        except Exception as e:
            log.error("Error fetching tx count from Etherscan for %s: %s", address, e)
            return 0

    def get_wallet_activity(self, address: str, days: int = 30) -> Dict:
        """
        Fetch wallet activity metrics.
        
        Args:
            address: Ethereum wallet address
            days: Number of days to look back (default 30)
            
        Returns:
            Dictionary with activity metrics
        """
        time.sleep(self.request_delay)
        
        try:
            # Get recent transactions
            params = {
                "chainid": self.chain_id,
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": 100,
                "sort": "desc",
                "apikey": self.api_key,
            }
            
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("status") != "1":
                return {
                    "activity_score": 0,
                    "avg_tx_value": 0,
                    "tx_frequency": 0,
                    "unique_counterparties": 0,
                }
            
            txs = data.get("result", [])
            if not txs:
                return {
                    "activity_score": 0,
                    "avg_tx_value": 0,
                    "tx_frequency": 0,
                    "unique_counterparties": 0,
                }

            # NOTE (data-quality caveat): metrics below are derived from the most
            # recent `offset` transactions only (sampling window, see params above),
            # NOT the wallet's full history. Consequently `unique_counterparties`
            # can legitimately exceed `transaction_count` (full count) for very
            # active wallets. Documented as a limitation in hasil_implementasi.md §14.
            addr_l = address.lower()
            values = [float(tx.get("value", 0)) / 1e18 for tx in txs]
            counterparties = set()
            for tx in txs:
                to_a = (tx.get("to") or "").lower()
                from_a = (tx.get("from") or "").lower()
                # Count the *other* party only — exclude the queried address itself
                # so the metric is not inflated by self-references.
                if to_a and to_a != addr_l:
                    counterparties.add(to_a)
                if from_a and from_a != addr_l:
                    counterparties.add(from_a)

            return {
                "activity_score": len(txs),
                "avg_tx_value": sum(values) / len(values) if values else 0,
                "tx_frequency": len(txs) / max(1, days),
                "unique_counterparties": len(counterparties),
            }
        except Exception as e:
            log.error("Error fetching activity for %s: %s", address, e)
            return {
                "activity_score": 0,
                "avg_tx_value": 0,
                "tx_frequency": 0,
                "unique_counterparties": 0,
            }

    def _fetch_neighbors_for_action(self, address: str, action: str, max_pages: int) -> set[str]:
        """
        Fetch counterparty addresses for a single Etherscan account action.

        Args:
            address: Ethereum wallet address (queried account)
            action: Etherscan account action — "txlist", "txlistinternal", or "tokentx"
            max_pages: Max pages to fetch (each page = up to 10,000 records)

        Returns:
            Set of unique counterparty addresses (lowercase), excluding `address`.
        """
        address_lower = address.lower()
        counterparties: set[str] = set()

        for page in range(1, max_pages + 1):
            time.sleep(self.request_delay)
            params = {
                "chainid": self.chain_id,
                "module": "account",
                "action": action,
                "address": address,
                "startblock": 0,
                "endblock": 99999999,
                "page": page,
                "offset": 10000,
                "sort": "asc",
                "apikey": self.api_key,
            }
            try:
                resp = requests.get(self.base_url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != "1":
                    break
                records = data.get("result", [])
                if not records:
                    break
                for tx in records:
                    frm = (tx.get("from") or "").lower()
                    to  = (tx.get("to")   or "").lower()
                    if frm and frm != address_lower:
                        counterparties.add(frm)
                    if to and to != address_lower:
                        counterparties.add(to)
                if len(records) < 10000:
                    break  # last page
            except Exception as e:
                log.error("Error fetching %s neighbors for %s (page %d): %s", action, address, page, e)
                break

        return counterparties

    def get_transaction_neighbors(
        self,
        address: str,
        max_pages: int = 3,
        include_internal: bool = True,
        include_tokens: bool = True,
    ) -> List[str]:
        """
        Fetch counterparty addresses for GNN edge extraction.

        Combines edges from multiple Etherscan endpoints so the resulting graph
        captures more real relationships than normal ETH transfers alone:
          - txlist          : normal (external) ETH transactions  (always)
          - txlistinternal  : internal transactions via contracts (if include_internal)
          - tokentx         : ERC-20 token transfers              (if include_tokens)

        Each source fetches up to max_pages × 10,000 records and returns the
        deduplicated set of `from`/`to` addresses, excluding the queried address.

        Args:
            address: Ethereum wallet address
            max_pages: Max pages per source (each page = up to 10,000 records). Default 3.
            include_internal: Also pull counterparties from internal transactions.
            include_tokens: Also pull counterparties from ERC-20 token transfers.

        Returns:
            List of unique counterparty address strings (lowercase).
        """
        counterparties: set[str] = self._fetch_neighbors_for_action(address, "txlist", max_pages)
        if include_internal:
            counterparties |= self._fetch_neighbors_for_action(address, "txlistinternal", max_pages)
        if include_tokens:
            counterparties |= self._fetch_neighbors_for_action(address, "tokentx", max_pages)

        return list(counterparties)

    def get_created_at_ts(self, address: str) -> int:
        """
        Fetch the wallet "creation" time = timestamp of its earliest transaction.

        Returns a unix epoch (seconds), or 0 if the address has no transactions /
        the request fails. Previously this value was hard-coded to None (stored as
        0), making `created_at_ts` a dead constant feature — fixed here so future
        enrichment runs populate a real value.
        """
        time.sleep(self.request_delay)
        params = {
            "chainid": self.chain_id,
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": 1,
            "sort": "asc",  # oldest first → first element is the earliest tx
            "apikey": self.api_key,
        }
        try:
            resp = requests.get(self.base_url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "1":
                return 0
            result = data.get("result", [])
            if not result:
                return 0
            return int(result[0].get("timeStamp", 0))
        except Exception as e:
            log.error("Error fetching created_at for %s: %s", address, e)
            return 0

    def fetch_all(self, address: str) -> Dict:
        """Fetch all available data for a wallet."""
        log.info("Fetching Etherscan data for %s", address)

        info = self.get_account_info(address)
        info["transaction_count"] = self.get_transaction_count(address)
        info["created_at"] = self.get_created_at_ts(address)
        activity = self.get_wallet_activity(address)
        info.update(activity)

        return info


class FortaNetworkFetcher:
    """Fetch risk labels from Forta Network."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Forta Network fetcher.
        
        Args:
            api_key: Forta Network API key (optional)
        """
        self.api_key = api_key or ""
        self.base_url = FORTA_API_BASE
        self.request_delay = 0.2

    def get_wallet_risk(self, address: str) -> Dict:
        """
        Fetch risk classification for a wallet.
        Returns confidence scores for: phishing, scam, exploit
        
        Args:
            address: Ethereum wallet address
            
        Returns:
            Dictionary with keys: is_phishing, is_scam, is_exploit,
                                 confidence_phishing, confidence_scam, confidence_exploit
        """
        time.sleep(self.request_delay)
        
        try:
            # Query Forta alerts API
            # Using GraphQL or REST endpoint depending on API availability
            url = f"{self.base_url}/api/v1/alerts"
            params = {
                "chainId": 1,  # Mainnet
                "address": address,
                "limit": 100,
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            }
            
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            # Parse alerts for risk indicators
            alerts = data.get("alerts", [])
            
            phishing_score = 0
            scam_score = 0
            exploit_score = 0
            
            for alert in alerts:
                severity = alert.get("severity", "").lower()
                alert_type = alert.get("type", "").lower()
                
                # Map alert types to risk categories
                if "phishing" in alert_type.lower() or "address poisoning" in alert_type.lower():
                    phishing_score += 0.5 if severity == "high" else 0.3
                elif "scam" in alert_type.lower() or "suspicious" in alert_type.lower():
                    scam_score += 0.5 if severity == "high" else 0.3
                elif "exploit" in alert_type.lower() or "attack" in alert_type.lower():
                    exploit_score += 0.5 if severity == "high" else 0.3
            
            # Normalize scores to 0-1 range
            max_score = max(phishing_score, scam_score, exploit_score, 1)
            
            return {
                "is_phishing": phishing_score > 0.5,
                "is_scam": scam_score > 0.5,
                "is_exploit": exploit_score > 0.5,
                "confidence_phishing": min(1.0, phishing_score / max_score),
                "confidence_scam": min(1.0, scam_score / max_score),
                "confidence_exploit": min(1.0, exploit_score / max_score),
            }
        except Exception as e:
            log.warning("Error fetching from Forta for %s: %s", address, e)
            return {
                "is_phishing": False,
                "is_scam": False,
                "is_exploit": False,
                "confidence_phishing": 0.0,
                "confidence_scam": 0.0,
                "confidence_exploit": 0.0,
                "error": str(e),
            }


def enrich_dataset_with_external_data(
    input_csv: Path,
    output_csv: Path,
    etherscan_api_key: Optional[str] = None,
    forta_api_key: Optional[str] = None,
    sample_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load dataset and enrich with Etherscan and Forta data.
    
    Args:
        input_csv: Input CSV file with wallet addresses
        output_csv: Output CSV file with enriched data
        etherscan_api_key: Etherscan API key
        forta_api_key: Forta Network API key
        sample_limit: Limit number of samples (for testing)
        
    Returns:
        Enriched DataFrame
    """
    log.info("Loading dataset from %s", input_csv)
    df = pd.read_csv(input_csv)
    
    # Limit samples if specified
    if sample_limit:
        df = df.head(sample_limit)
        log.info("Limited to %d samples for testing", sample_limit)
    
    # Initialize fetchers
    etherscan = EtherscanFetcher(etherscan_api_key)
    forta = FortaNetworkFetcher(forta_api_key)
    
    # Enrich data
    new_rows = []
    for idx, row in df.iterrows():
        address = row.get("address") or row.get("node_id")
        
        if not address:
            log.warning("Row %d has no address/node_id", idx)
            continue
        
        log.info("[%d/%d] Processing %s", idx + 1, len(df), address[:10])
        
        # Fetch Etherscan data
        etherscan_data = etherscan.fetch_all(address)
        
        # Fetch Forta data
        forta_data = forta.get_wallet_risk(address)
        
        # Merge with original row
        enriched_row = dict(row)
        enriched_row.update(etherscan_data)
        enriched_row.update(forta_data)
        
        new_rows.append(enriched_row)
    
    # Create output DataFrame
    result_df = pd.DataFrame(new_rows)
    result_df.to_csv(output_csv, index=False)
    log.info("Enriched dataset saved to %s", output_csv)
    
    return result_df


if __name__ == "__main__":
    import os
    
    etherscan_key = os.getenv("ETHERSCAN_API_KEY", "")
    forta_key = os.getenv("FORTA_API_KEY", "")
    
    # Example: enrich existing dataset
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    
    input_file = DATA_RAW / "addresses.csv"
    output_file = DATA_RAW / "addresses_enriched.csv"
    
    if input_file.exists():
        enrich_dataset_with_external_data(
            input_file,
            output_file,
            etherscan_key,
            forta_key,
            sample_limit=100,  # Test with 100 addresses first
        )
    else:
        log.error("Input file not found: %s", input_file)
