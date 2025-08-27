# Solana Python SDK for Pumpfun, Moonshot and Raydium

## Moonshot Py
moonshot_py is a Python script designed to analyze, buy, and sell tokens on decentralized exchanges with ease. It’s built for traders and developers who want to automate and monitor token operations in real-time.

### How It Works
  - `get_token_data` → Fetch token data
  - `derive_curve_accounts` → Derive curve accounts
  - `get_token_account` → Get token account
  - `buy_moonshot_alone` → Buy tokens
  - `find_data` → Find token data
  - `get_token_balance` → Check token balance
  - `sell_moonshot` → Sell tokens

## Pumpfun Py
Pumpfun Py is a Python script for interacting with token bonding curves and performing automated token trades on Solana. It allows users to calculate prices, trade tokens, and monitor balances programmatically.

### How It Works
1. Data Retrieval
    - `get_virtual_reserves` → Fetches the current virtual reserves of a token on the bonding curve.
    - `get_coin_data` → Retrieves token metadata and market stats.
    - `get_token_balance` → Checks the balance of a token in a wallet.
    - `get_token_price` → Returns the current price for a specified token amount.
2. Account & Curve Operations
    - `derive_bonding_curve_accounts` → Calculates accounts necessary for interacting with a token’s bonding curve.
3. Price Calculations
    - `sol_for_tokens` → Calculates how much SOL is required to buy a given amount of tokens.
    - `tokens_for_sol` → Calculates how many tokens can be bought with a given SOL amount.
4. Transactions
    - `send_transaction_jito` → Sends a transaction via Jito for faster execution.
    - `confirm_txn` → Confirms that a transaction has been processed on-chain.
5. Trading Actions
    - `buy` → Buys a specified amount of a token.
    - `sell` → Sells a specified amount of a token.

## Raydium Py
raydium_py is a Python script that interfaces with Raydium’s liquidity pools on Solana. It supports AMM V4, CLMM, and CPMM pools, allowing developers to fetch pool data, query token prices, and execute trades programmatically.

### How It Works
1. AMM_V4
   - Utility Functions
     `sol_for_tokens`, `tokens_for_sol`, `confirm_txn`, `get_token_balance`
   - Core AMM Functions
     `fetch_amm_v4_pool_keys`, `get_amm_v4_reserves`, `make_amm_v4_swap_instruction`
   - Main Trading Functions
     `buy`, `sell`
2. CPMM
   - Utility Functions
     `sol_for_tokens`, `tokens_for_sol`, `get_token_balance`
   - Confirm Transaction
     `confirm_txn`
   - Fetch CPMM Pool Keys
     `fetch_cpmm_pool_keys`
   - Get CPMM Vault Balances
     `get_cpmm_reserves`
   - Make CPMM Swap Instruction
     `make_cpmm_swap_instruction`
   - Main Trading Functions
     `buy`, `sell`
3. CLMM
   - Fetch Pool Keys
     `fetch_clmm_pool_keys`
   - Get CLMM Vault Reserves
     `get_clmm_reserves`
   - Confirm Transaction
     `confirm_txn`
   - Get SPL Token Balance
     `get_token_balance`
   - Make CLMM Swap Instruction
     `make_clmm_swap_instruction`
   - Main Trading Functions
     `buy`, `sell`
