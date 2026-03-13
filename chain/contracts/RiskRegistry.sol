// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title RiskRegistry
 * @notice On-chain registry mapping Ethereum wallet addresses to their
 *         ML-computed phishing risk scores.
 *
 * Risk score is an integer 0–100:
 *   0–39   → LOW
 *   40–69  → MEDIUM
 *   70–100 → HIGH
 *
 * Only the contract owner (deployer) can write risk records so that
 * the demo remains controlled; in production this role can be an
 * approved set of oracle addresses.
 */
contract RiskRegistry {
    // ── Types ────────────────────────────────────────────────────────────────
    struct Record {
        uint16  riskScore;   // 0–100
        uint64  timestamp;   // unix epoch seconds
        string  modelVersion;
    }

    // ── Storage ───────────────────────────────────────────────────────────────
    address public owner;
    mapping(address => Record) private _records;

    // ── Events ────────────────────────────────────────────────────────────────
    event RiskUpdated(
        address indexed wallet,
        uint16          riskScore,
        uint64          timestamp,
        string          modelVersion
    );

    // ── Modifiers ─────────────────────────────────────────────────────────────
    modifier onlyOwner() {
        require(msg.sender == owner, "RiskRegistry: caller is not the owner");
        _;
    }

    // ── Constructor ───────────────────────────────────────────────────────────
    constructor() {
        owner = msg.sender;
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /**
     * @notice Store or update the risk record for a wallet.
     * @param wallet       The Ethereum address being assessed.
     * @param score        Risk score (0–100).
     * @param modelVersion Short version tag, e.g. "rf-v1".
     */
    function setRisk(
        address wallet,
        uint16  score,
        string calldata modelVersion
    ) external onlyOwner {
        require(score <= 100, "RiskRegistry: score must be <= 100");
        uint64 ts = uint64(block.timestamp);
        _records[wallet] = Record(score, ts, modelVersion);
        emit RiskUpdated(wallet, score, ts, modelVersion);
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /**
     * @notice Retrieve the risk record for a wallet.
     * @return riskScore    The stored score (0 if never set).
     * @return timestamp    Unix timestamp of last update (0 if never set).
     * @return modelVersion Model version tag used when score was set.
     */
    function getRisk(address wallet)
        external
        view
        returns (
            uint16 riskScore,
            uint64 timestamp,
            string memory modelVersion
        )
    {
        Record storage r = _records[wallet];
        return (r.riskScore, r.timestamp, r.modelVersion);
    }

    /**
     * @notice Transfer contract ownership.
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "RiskRegistry: zero address");
        owner = newOwner;
    }
}
