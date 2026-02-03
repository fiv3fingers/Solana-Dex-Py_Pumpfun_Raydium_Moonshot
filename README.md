# Solana DEX Python SDK — Pumpfun SDK, Moonshot SDK & Raydium SDK

> A unified Python toolkit providing a **Pumpfun SDK**, **Moonshot SDK**, and **Raydium SDK** in a single repository. Buy, sell, and monitor tokens across three of Solana's largest decentralised exchanges — all from one codebase.

[![Language](https://img.shields.io/badge/language-Python-3572A5.svg)](https://www.python.org/)
[![Network](https://img.shields.io/badge/network-Solana-9945FF.svg)](https://solana.com/)
[![Stars](https://img.shields.io/github/stars/fiv3fingers/Solana-Dex-Py_Pumpfun_Raydium_Moonshot?style=flat-square)](https://github.com/fiv3fingers/Solana-Dex-Py_Pumpfun_Raydium_Moonshot/stargazers)
[![Forks](https://img.shields.io/github/forks/fiv3fingers/Solana-Dex-Py_Pumpfun_Raydium_Moonshot?style=flat-square)](https://github.com/fiv3fingers/Solana-Dex-Py_Pumpfun_Raydium_Moonshot/forks)

---

## Table of Contents

1. [Why This SDK?](#why-this-sdk)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Pumpfun SDK](#pumpfun-sdk)
5. [Moonshot SDK](#moonshot-sdk)
6. [Raydium SDK](#raydium-sdk)
7. [Quick-Start Cheat Sheet](#quick-start-cheat-sheet)
8. [Configuration & Security](#configuration--security)
9. [FAQ](#faq)
10. [References](#references)

---

## Why This SDK?

Pump.fun, Moonshot, and Raydium are three distinct Solana programmes with different on-chain layouts, instruction sets, and bonding-curve maths. Writing a wrapper for each one individually wastes time and forces you to maintain separate configs. This repository consolidates all three into one well-organised Python project so you can:

- Switch between exchanges by changing a single import.
- Share utility helpers (balance checks, transaction confirmation, price conversions) across all three modules.
- Get up and running with automated trading on any of the three platforms in minutes.

---

## Repository Structure

```
Solana-Dex-Py_Pumpfun_Raydium_Moonshot/
├── pumpfun-py/          # Pumpfun SDK — bonding-curve buy / sell
├── moonshot-py/         # Moonshot SDK — curve trading & account derivation
├── raydium-py/          # Raydium SDK — AMM V4, CPMM, and CLMM pools
├── .gitignore
└── README.md
```

Each sub-directory is a self-contained module: its own entry-point scripts, helper functions, and (where needed) a local `config.py` or `.env` file for wallet and RPC configuration.

---

## Prerequisites

- **Python** ≥ 3.9
- A funded Solana wallet (mainnet-beta) — keep the base-58 private key handy for configuration.
- A reliable Solana RPC endpoint. Public endpoints work for testing, but a paid provider (Helius, QuickNode, etc.) is strongly recommended for production use.

Install the core Solana Python dependencies that all three modules rely on:

```bash
pip install solana==0.35.0 solders==0.21.0
```

---

## Pumpfun SDK

### What Is Pump.fun?

Pump.fun is a permissionless token-launch platform on Solana. Every token minted through it starts life on a **constant-product bonding curve** (`x × y = k`) managed by on-chain program `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`. Tokens are 6-decimal SPL tokens. Once the curve fills completely the token graduates automatically to PumpSwap, Pump.fun's native AMM.

### Module Breakdown (`pumpfun-py/`)

| Category | Function | Purpose |
|----------|----------|---------|
| **Data** | `get_virtual_reserves` | Fetch the current virtual reserve pair from the bonding curve |
| | `get_coin_data` | Pull token metadata and live market stats |
| | `get_token_balance` | Query a wallet's SPL token balance |
| | `get_token_price` | Compute the current price for a given token amount |
| **Accounts** | `derive_bonding_curve_accounts` | Derive the PDA set needed to interact with a token's curve |
| **Math** | `sol_for_tokens` | Calculate SOL cost to buy *n* tokens at the current curve position |
| | `tokens_for_sol` | Calculate how many tokens a given SOL amount will buy |
| **Transactions** | `send_transaction_jito` | Send via Jito for faster inclusion |
| | `confirm_txn` | Poll until the transaction is confirmed on-chain |
| **Trading** | `buy` | Execute a buy against the bonding curve |
| | `sell` | Execute a sell against the bonding curve |

### Example — Buy a Pump.fun Token

```python
from pumpfun_py import buy, get_coin_data, sol_for_tokens

# 1. Fetch live token data
token = get_coin_data(mint_address="<TOKEN_MINT>")

# 2. Estimate cost for 10 000 tokens
cost_sol = sol_for_tokens(mint_address=token["mint"], amount=10_000)
print(f"Estimated cost: {cost_sol} SOL")

# 3. Execute buy
txsig = buy(mint_address=token["mint"], sol_amount=0.05)
print(f"Buy tx: {txsig}")
```

---

## Moonshot SDK

### What Is Moonshot?

Moonshot (now rebranding to Moonit) is a memecoin launchpad built on Solana. It uses a **quadratic bonding curve** — price rises slowly at first, then accelerates — over a fixed 1-billion-token supply (9 decimals). Once roughly 80 % of the supply is sold the curve collects ~88 SOL in collateral, the market-cap threshold hits ~345 SOL, and all remaining tokens and collateral migrate automatically to either **Raydium** or **Meteora** depending on the creator's choice at launch.

### Module Breakdown (`moonshot-py/`)

| Function | Purpose |
|----------|---------|
| `get_token_data` | Fetch on-chain token metadata |
| `derive_curve_accounts` | Derive the curve-related PDAs for a given token |
| `get_token_account` | Resolve the associated token account for a wallet |
| `get_token_balance` | Query the wallet's token balance |
| `find_data` | Locate and parse token state data |
| `buy_moonshot_alone` | Execute a buy transaction on the Moonshot curve |
| `sell_moonshot` | Execute a sell transaction on the Moonshot curve |

### Example — Sell a Moonshot Token

```python
from moonshot_py import sell_moonshot, get_token_balance

# 1. Check how much you hold
balance = get_token_balance(mint_address="<TOKEN_MINT>")
print(f"Current balance: {balance}")

# 2. Sell half
txsig = sell_moonshot(mint_address="<TOKEN_MINT>", token_amount=balance // 2)
print(f"Sell tx: {txsig}")
```

---

## Raydium SDK

### What Is Raydium?

Raydium has been Solana's primary liquidity layer since 2021. It ships three distinct pool types, each with its own on-chain programme and trade-off profile:

| Pool Type | Invariant | Key Characteristics |
|-----------|-----------|---------------------|
| **AMM V4** | `x × y = k` | Legacy, battle-tested, originally shared idle liquidity with OpenBook |
| **CPMM** | `x × y = k` | Newest constant-product pool; supports Token-2022; no OpenBook requirement; default target for new pool creation and LaunchLab migrations |
| **CLMM** | Concentrated liquidity | LPs choose a price range; higher capital efficiency; better for active pairs |

Pool creation costs 0.15 SOL on both AMM V4 and CPMM.

### Module Breakdown (`raydium-py/`)

#### AMM V4

| Category | Functions |
|----------|-----------|
| Utilities | `sol_for_tokens`, `tokens_for_sol`, `confirm_txn`, `get_token_balance` |
| Pool Data | `fetch_amm_v4_pool_keys`, `get_amm_v4_reserves` |
| Swap | `make_amm_v4_swap_instruction` |
| Trading | `buy`, `sell` |

#### CPMM

| Category | Functions |
|----------|-----------|
| Utilities | `sol_for_tokens`, `tokens_for_sol`, `get_token_balance`, `confirm_txn` |
| Pool Data | `fetch_cpmm_pool_keys`, `get_cpmm_reserves` |
| Swap | `make_cpmm_swap_instruction` |
| Trading | `buy`, `sell` |

#### CLMM

| Category | Functions |
|----------|-----------|
| Utilities | `get_token_balance`, `confirm_txn` |
| Pool Data | `fetch_clmm_pool_keys`, `get_clmm_reserves` |
| Swap | `make_clmm_swap_instruction` |
| Trading | `buy`, `sell` |

### Example — Swap on a Raydium CPMM Pool

```python
from raydium_py.cpmm import fetch_cpmm_pool_keys, make_cpmm_swap_instruction, buy

# 1. Resolve the pool (use pool ID, not token mint — a single token can have many pools)
pool_keys = fetch_cpmm_pool_keys(pool_id="<CPMM_POOL_ID>")

# 2. Build the swap instruction
instruction = make_cpmm_swap_instruction(
    pool_keys=pool_keys,
    input_amount=0.01,       # SOL
    slippage_bps=50          # 0.50 %
)

# 3. Or use the high-level buy helper directly
txsig = buy(pool_id="<CPMM_POOL_ID>", sol_amount=0.01, slippage_bps=50)
print(f"Swap tx: {txsig}")
```

---

## Quick-Start Cheat Sheet

| Goal | Module | Entry Point |
|------|--------|-------------|
| Buy a newly launched Pump.fun token | `pumpfun-py` | `buy(mint_address, sol_amount)` |
| Sell a Pump.fun token back to the curve | `pumpfun-py` | `sell(mint_address, token_amount)` |
| Trade on a Moonshot bonding curve | `moonshot-py` | `buy_moonshot_alone` / `sell_moonshot` |
| Swap on a Raydium AMM V4 pool | `raydium-py` | `raydium_py.amm_v4.buy` / `.sell` |
| Swap on a Raydium CPMM pool | `raydium-py` | `raydium_py.cpmm.buy` / `.sell` |
| Swap on a Raydium CLMM pool | `raydium-py` | `raydium_py.clmm.buy` / `.sell` |
| Check any SPL token balance | any module | `get_token_balance(mint_address)` |

---

## Configuration & Security

Each module reads wallet and RPC details from either a local `config.py` or a `.env` file at its root. A `.env.sample` / template is provided; copy it and fill in your values:

```ini
# Solana RPC
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com   # replace with paid endpoint in production

# Wallet
WALLET_PRIVATE_KEY=<base58_secret_key>

# Trading defaults (override per-call)
DEFAULT_SLIPPAGE_BPS=50
```

> **Security:** Never commit a live private key. The `.gitignore` already excludes `.env` files. For CI/CD or server deployments use a secrets manager.

---

## FAQ

**Q — Can I use the public Solana RPC for production trading?**
A — Technically yes, but rate limits will cause frequent failures under load. A paid provider (Helius, QuickNode, Alchemy) is the standard choice.

**Q — How do I find a Raydium pool ID for a token?**
A — Use the [Raydium API](https://docs.raydium.io) or the `fetch_*_pool_keys` helpers included in `raydium-py`. Never pass a raw token mint address as a pool ID — a single token can appear in multiple pools.

**Q — What happens to a Pump.fun token after the bonding curve fills?**
A — It graduates automatically to PumpSwap, Pump.fun's native constant-product AMM. You would then need to interact with PumpSwap rather than the bonding curve to trade it.

**Q — Does the Moonshot SDK support post-migration trading?**
A — Once a Moonshot token migrates (to Raydium or Meteora), you switch to the corresponding Raydium module in this repo or interact directly with the target AMM.

**Q — Which Python packages do I actually need?**
A — At minimum `solana==0.35.0` and `solders==0.21.0`. Individual modules may pull in additional helpers; check the local requirements or imports in each sub-directory.

---

## References

| Resource | URL |
|----------|-----|
| Pump.fun | [https://pump.fun](https://pump.fun) |
| Moonshot / Moonit Docs | [https://docs.moonshot.cc](https://docs.moonshot.cc) |
| Raydium Docs | [https://docs.raydium.io](https://docs.raydium.io) |
| Solana Web3 (Python) | [https://github.com/michaelhly/solana-py](https://github.com/michaelhly/solana-py) |
| Solders | [https://github.com/michaelhly/solders](https://github.com/michaelhly/solders) |

## Contract
[Telegram](https://t.me/WebFiveFingers)
