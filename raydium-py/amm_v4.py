import asyncio
import base64
import json
import os
import struct
from dataclasses import dataclass
from typing import Optional

# Third-party imports (async version)
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Processed
from solana.rpc.types import TokenAccountOpts, TxOpts

# Solder-based imports
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price  # type: ignore
from solders.keypair import Keypair  # type: ignore
from solders.message import MessageV0  # type: ignore
from solders.pubkey import Pubkey  # type: ignore
from solders.signature import Signature  # type: ignore
from solders.system_program import CreateAccountWithSeedParams, create_account_with_seed
from solders.transaction import VersionedTransaction  # type: ignore
from solders.instruction import AccountMeta, Instruction  # type: ignore

# SPL Token imports
from spl.token.client import Token
from spl.token.instructions import (
    CloseAccountParams,
    InitializeAccountParams,
    close_account,
    create_associated_token_account,
    get_associated_token_address,
    initialize_account,
)

# Local imports
from raydium.layouts.amm_v4 import LIQUIDITY_STATE_LAYOUT_V4, MARKET_STATE_LAYOUT_V3


# ================================================
# Global Async Client & Constants
# ================================================
RPC = "https://mainnet.helius-rpc.com/?api-key=eba6d019-77f5-4715-9044-54eeeefeee23"
client = AsyncClient(RPC)  # <--- Async client

RAYDIUM_AMM_V4 = Pubkey.from_string("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8")
DEFAULT_QUOTE_MINT = "So11111111111111111111111111111111111111112"
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ACCOUNT_LAYOUT_LEN = 165
WSOL = Pubkey.from_string("So11111111111111111111111111111111111111112")
SOL_DECIMAL = 1e9


@dataclass
class AmmV4PoolKeys:
    amm_id: Pubkey
    base_mint: Pubkey
    quote_mint: Pubkey
    base_decimals: int
    quote_decimals: int
    open_orders: Pubkey
    target_orders: Pubkey
    base_vault: Pubkey
    quote_vault: Pubkey
    market_id: Pubkey
    market_authority: Pubkey
    market_base_vault: Pubkey
    market_quote_vault: Pubkey
    bids: Pubkey
    asks: Pubkey
    event_queue: Pubkey
    ray_authority_v4: Pubkey
    open_book_program: Pubkey
    token_program_id: Pubkey


# ================================================
# Utility Functions
# ================================================
def sol_for_tokens(sol_amount, base_vault_balance, quote_vault_balance, swap_fee=0.25):
    """Estimate how many tokens you receive if swapping SOL -> tokens in a constant-product AMM."""
    effective_sol_used = sol_amount - (sol_amount * (swap_fee / 100))
    constant_product = base_vault_balance * quote_vault_balance
    updated_base_vault_balance = constant_product / (quote_vault_balance + effective_sol_used)
    tokens_received = base_vault_balance - updated_base_vault_balance
    return round(tokens_received, 9)


def tokens_for_sol(token_amount, base_vault_balance, quote_vault_balance, swap_fee=0.25):
    """Estimate how many SOL you receive if swapping tokens -> SOL in a constant-product AMM."""
    effective_tokens_sold = token_amount * (1 - (swap_fee / 100))
    constant_product = base_vault_balance * quote_vault_balance
    updated_quote_vault_balance = constant_product / (base_vault_balance + effective_tokens_sold)
    sol_received = quote_vault_balance - updated_quote_vault_balance
    return round(sol_received, 9)


async def confirm_txn(txn_sig: Signature, max_retries: int = 20, retry_interval: float = 0.1) -> bool:
    """Poll the Solana RPC to confirm a transaction."""
    retries = 1
    while retries < max_retries:
        try:
            txn_res = await client.get_transaction(
                txn_sig,
                encoding="json",
                commitment=Confirmed,
                max_supported_transaction_version=0,
            )
            if txn_res.value is None:
                # If no transaction found yet, keep retrying
                print("Awaiting confirmation... try count:", retries)
                retries += 1
                await asyncio.sleep(retry_interval)
                continue

            txn_json = json.loads(txn_res.value.transaction.meta.to_json())

            if txn_json["err"] is None:
                print("Transaction confirmed... try count:", retries)
                return True

            print("Error: Transaction not confirmed. Retrying...")
            if txn_json["err"]:
                print("Transaction failed.")
                return False

        except Exception as e:
            print("Awaiting confirmation... try count:", retries)
            retries += 1
            await asyncio.sleep(retry_interval)

    print("Max retries reached. Transaction confirmation failed.")
    return False


async def get_token_balance(mint_str: str, pub_key_str: str) -> float | None:
    """Retrieve the SPL token balance for a given mint and owner."""
    mint = Pubkey.from_string(mint_str)
    owner_pubkey = Pubkey.from_string(pub_key_str)

    response = await client.get_token_accounts_by_owner_json_parsed(
        owner_pubkey,
        TokenAccountOpts(mint=mint),
        commitment=Processed,
    )

    if response.value:
        accounts = response.value
        if accounts:
            token_amount = accounts[0].account.data.parsed["info"]["tokenAmount"]["uiAmount"]
            if token_amount:
                return float(token_amount)
    return None


# ================================================
# Core AMM Functions
# ================================================
async def fetch_amm_v4_pool_keys(pair_address: str) -> Optional[AmmV4PoolKeys]:
    """Fetch and parse on-chain data to build AmmV4PoolKeys."""

    def bytes_of(value):
        if not (0 <= value < 2**64):
            raise ValueError("Value must be in the range of a u64 (0 to 2^64 - 1).")
        return struct.pack("<Q", value)

    try:
        amm_id = Pubkey.from_string(pair_address)

        # Fetch Raydium AMM account info (V4 liquidity state)
        amm_data_resp = await client.get_account_info_json_parsed(amm_id, commitment=Processed)
        amm_data = amm_data_resp.value.data
        amm_data_decoded = LIQUIDITY_STATE_LAYOUT_V4.parse(amm_data)

        # Fetch Serum/OpenBook market info
        market_id = Pubkey.from_bytes(amm_data_decoded.serumMarket)
        market_info_resp = await client.get_account_info_json_parsed(market_id, commitment=Processed)
        market_info = market_info_resp.value.data
        market_decoded = MARKET_STATE_LAYOUT_V3.parse(market_info)

        vault_signer_nonce = market_decoded.vault_signer_nonce
        ray_authority_v4 = Pubkey.from_string("5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1")
        open_book_program = Pubkey.from_string("srmqPvymJeFKQ4zGQed1GFppgkRHL9kaELCbyksJtPX")
        token_program_id = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

        pool_keys = AmmV4PoolKeys(
            amm_id=amm_id,
            base_mint=Pubkey.from_bytes(market_decoded.base_mint),
            quote_mint=Pubkey.from_bytes(market_decoded.quote_mint),
            base_decimals=amm_data_decoded.coinDecimals,
            quote_decimals=amm_data_decoded.pcDecimals,
            open_orders=Pubkey.from_bytes(amm_data_decoded.ammOpenOrders),
            target_orders=Pubkey.from_bytes(amm_data_decoded.ammTargetOrders),
            base_vault=Pubkey.from_bytes(amm_data_decoded.poolCoinTokenAccount),
            quote_vault=Pubkey.from_bytes(amm_data_decoded.poolPcTokenAccount),
            market_id=market_id,
            market_authority=Pubkey.create_program_address(
                seeds=[bytes(market_id), bytes_of(vault_signer_nonce)], program_id=open_book_program
            ),
            market_base_vault=Pubkey.from_bytes(market_decoded.base_vault),
            market_quote_vault=Pubkey.from_bytes(market_decoded.quote_vault),
            bids=Pubkey.from_bytes(market_decoded.bids),
            asks=Pubkey.from_bytes(market_decoded.asks),
            event_queue=Pubkey.from_bytes(market_decoded.event_queue),
            ray_authority_v4=ray_authority_v4,
            open_book_program=open_book_program,
            token_program_id=token_program_id,
        )

        return pool_keys
    except Exception as e:
        print(f"Error fetching pool keys: {e}")
        return None


async def get_amm_v4_reserves(pool_keys: AmmV4PoolKeys) -> tuple:
    """Retrieve current base and quote vault balances from on-chain token accounts."""
    try:
        quote_vault = pool_keys.quote_vault
        quote_decimal = pool_keys.quote_decimals
        quote_mint = pool_keys.quote_mint

        base_vault = pool_keys.base_vault
        base_decimal = pool_keys.base_decimals
        base_mint = pool_keys.base_mint

        # Fetch vault balances
        balances_response = await client.get_multiple_accounts_json_parsed(
            [quote_vault, base_vault],
            Processed,
        )
        balances = balances_response.value

        if len(balances) < 2:
            print("Error: Could not retrieve vault balances.")
            return None, None, None

        quote_account = balances[0]
        base_account = balances[1]

        quote_account_balance = quote_account.data.parsed["info"]["tokenAmount"]["uiAmount"]
        base_account_balance = base_account.data.parsed["info"]["tokenAmount"]["uiAmount"]

        if quote_account_balance is None or base_account_balance is None:
            print("Error: One of the account balances is None.")
            return None, None, None

        # Determine which side is SOL and which is the custom token
        if base_mint == WSOL:
            # If the base_mint is WSOL, SOL is the 'base', so the 'quote_account_balance' is the base reserve
            base_reserve = quote_account_balance
            quote_reserve = base_account_balance
            token_decimal = quote_decimal
        else:
            base_reserve = base_account_balance
            quote_reserve = quote_account_balance
            token_decimal = base_decimal

        print(f"Base Mint: {base_mint} | Quote Mint: {quote_mint}")
        print(f"Base Reserve: {base_reserve} | Quote Reserve: {quote_reserve} | Token Decimal: {token_decimal}")
        return base_reserve, quote_reserve, token_decimal

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None


async def make_amm_v4_swap_instruction(
    amount_in: int,
    minimum_amount_out: int,
    token_account_in: Pubkey,
    token_account_out: Pubkey,
    accounts: AmmV4PoolKeys,
    owner: Pubkey,
) -> Instruction:
    """
    Construct a Raydium AMM v4 swap instruction (async version).
    Note that there is no actual I/O here; it's simply marked async
    for convenience if you need to await it in an async context.
    """
    try:
        keys = [
            AccountMeta(pubkey=accounts.token_program_id, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.amm_id, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.ray_authority_v4, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.open_orders, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.target_orders, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.base_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.quote_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.open_book_program, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.market_id, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.bids, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.asks, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.event_queue, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_base_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_quote_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.market_authority, is_signer=False, is_writable=False),
            AccountMeta(pubkey=token_account_in, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_account_out, is_signer=False, is_writable=True),
            AccountMeta(pubkey=owner, is_signer=True, is_writable=False),
        ]

        data = bytearray()
        # Discriminator for the AMM V4 swap
        discriminator = 9
        data.extend(struct.pack("<B", discriminator))
        data.extend(struct.pack("<Q", amount_in))
        data.extend(struct.pack("<Q", minimum_amount_out))

        swap_instruction = Instruction(RAYDIUM_AMM_V4, bytes(data), keys)
        return swap_instruction
    except Exception as e:
        print(f"Error occurred in make_amm_v4_swap_instruction: {e}")
        return None
# ================================================
# Main Trading Functions (Asynchronous)
# ================================================
async def buy(
    priv_key: str,
    pair_address: str,
    sol_in: float,
    slippage: int,
    UNIT_BUDGET: int,
    UNIT_PRICE: int,
):
    """Buy tokens from a Raydium AMM v4 pool, paying with SOL."""
    payer_keypair = Keypair.from_base58_string(priv_key)
    try:
        # Fetch pool keys
        pool_keys = await fetch_amm_v4_pool_keys(pair_address)
        if pool_keys is None:
            print("No pool keys found (DB or network).")
            return False
        print("Pool keys acquired (from DB or network).")

        # Decide which mint is the token being bought
        mint = pool_keys.base_mint if pool_keys.base_mint != WSOL else pool_keys.quote_mint

        # Convert SOL amount to lamports
        amount_in = int(sol_in * SOL_DECIMAL)

        # Get current reserves
        base_reserve, quote_reserve, token_decimal = await get_amm_v4_reserves(pool_keys)
        # Estimate how many tokens you'll get
        amount_out = sol_for_tokens(sol_in, base_reserve, quote_reserve)
        slippage_adjustment = 1 - (slippage / 100)
        amount_out_with_slippage = amount_out * slippage_adjustment
        minimum_amount_out = int(amount_out_with_slippage * (10**token_decimal))

        # Check for existing token account; if none, create it
        token_account_check = await client.get_token_accounts_by_owner(
            payer_keypair.pubkey(), TokenAccountOpts(mint), Processed
        )
        if token_account_check.value:
            token_account = token_account_check.value[0].pubkey
            create_token_account_instruction = None
        else:
            token_account = get_associated_token_address(payer_keypair.pubkey(), mint)
            create_token_account_instruction = create_associated_token_account(
                payer_keypair.pubkey(), payer_keypair.pubkey(), mint
            )

        # Create a temporary WSOL account
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        wsol_token_account = Pubkey.create_with_seed(
            payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID
        )

        # 1) Get rent-exemption minimum
        balance_resp = await client.get_minimum_balance_for_rent_exemption(
            ACCOUNT_LAYOUT_LEN, commitment=Processed
        )
        balance_needed = balance_resp.value

        # 2) We'll deposit that rent + the amount_in lamports of SOL
        lamports_for_wsol = balance_needed + amount_in

        create_wsol_account_instruction = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=wsol_token_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=lamports_for_wsol,
                space=ACCOUNT_LAYOUT_LEN,
                owner=TOKEN_PROGRAM_ID,
            )
        )

        init_wsol_account_instruction = initialize_account(
            InitializeAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_token_account,
                mint=WSOL,
                owner=payer_keypair.pubkey(),
            )
        )

        # Construct swap instruction
        swap_instruction = await make_amm_v4_swap_instruction(
            amount_in=amount_in,
            minimum_amount_out=minimum_amount_out,
            token_account_in=wsol_token_account,
            token_account_out=token_account,
            accounts=pool_keys,
            owner=payer_keypair.pubkey(),
        )

        # Close the WSOL account afterwards
        close_wsol_account_instruction = close_account(
            CloseAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_token_account,
                dest=payer_keypair.pubkey(),
                owner=payer_keypair.pubkey(),
            )
        )

        instructions = [
            set_compute_unit_limit(UNIT_BUDGET),
            set_compute_unit_price(UNIT_PRICE),
            create_wsol_account_instruction,
            init_wsol_account_instruction,
        ]

        if create_token_account_instruction:
            instructions.append(create_token_account_instruction)

        instructions.append(swap_instruction)
        instructions.append(close_wsol_account_instruction)

        # Compile transaction
        latest_blockhash = (await client.get_latest_blockhash()).value.blockhash
        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(), instructions, [], latest_blockhash
        )

        # Send transaction
        txn_sig = (await client.send_transaction(
            txn=VersionedTransaction(compiled_message, [payer_keypair]),
            opts=TxOpts(skip_preflight=True),
        )).value

        print("Transaction Signature:", txn_sig)
        return txn_sig

    except Exception as e:
        print("Error occurred during transaction:", e)
        return False


async def sell(
    priv_key: str,
    pair_address: str,
    token_amount_to_sell: float,
    percentage: int,
    slippage: int,
    UNIT_BUDGET: int,
    UNIT_PRICE: int,
):
    payer_keypair = Keypair.from_base58_string(priv_key)
    pub_key = payer_keypair.pubkey()
    token_balance = token_amount_to_sell
    print("token balance from bot" , token_balance)

    try:
        if not (1 <= percentage <= 100):
            print("Percentage must be between 1 and 100.")
            return False

        # Fetch pool keys
        pool_keys = await fetch_amm_v4_pool_keys(pair_address)
        if pool_keys is None:
            print("No pool keys found (DB or network).")
            return False

        # Determine which mint is the token being sold
        mint = pool_keys.base_mint if pool_keys.base_mint != WSOL else pool_keys.quote_mint

        # # Retrieve how many tokens the user has
        token_balance_test = await get_token_balance(str(mint), str(pub_key))
        if token_balance_test is None or token_balance_test == 0:
            print("No token balance available to sell.")
            return False

        token_balance_ttt = token_balance_test * (percentage / 100)
        print(f"Selling {percentage}% of the token balance, adjusted balance: {token_balance_ttt}")

        # Compute how many SOL we expect to get
        base_reserve, quote_reserve, token_decimal = await get_amm_v4_reserves(pool_keys)
        amount_out = tokens_for_sol(token_balance, base_reserve, quote_reserve)
        slippage_adjustment = 1 - (slippage / 100)
        amount_out_with_slippage = amount_out * slippage_adjustment
        minimum_amount_out = int(amount_out_with_slippage * SOL_DECIMAL)

        amount_in = int(token_balance * (10**token_decimal))
        token_account = get_associated_token_address(payer_keypair.pubkey(), mint)

        # Create a temporary WSOL account to receive SOL
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        wsol_token_account = Pubkey.create_with_seed(
            payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID
        )
        # 1) Get rent-exemption minimum
        balance_resp = await client.get_minimum_balance_for_rent_exemption(
            ACCOUNT_LAYOUT_LEN, commitment=Processed
        )
        balance_needed = balance_resp.value

        create_wsol_account_instruction = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=wsol_token_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(balance_needed),
                space=ACCOUNT_LAYOUT_LEN,
                owner=TOKEN_PROGRAM_ID,
            )
        )
        init_wsol_account_instruction = initialize_account(
            InitializeAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_token_account,
                mint=WSOL,
                owner=payer_keypair.pubkey(),
            )
        )

        # Create swap instruction (token -> WSOL)
        swap_instructions = await make_amm_v4_swap_instruction(
            amount_in=amount_in,
            minimum_amount_out=minimum_amount_out,
            token_account_in=token_account,
            token_account_out=wsol_token_account,
            accounts=pool_keys,
            owner=payer_keypair.pubkey(),
        )

        # Close WSOL account after swap
        close_wsol_account_instruction = close_account(
            CloseAccountParams(
                program_id=TOKEN_PROGRAM_ID,
                account=wsol_token_account,
                dest=payer_keypair.pubkey(),
                owner=payer_keypair.pubkey(),
            )
        )

        instructions = [
            set_compute_unit_limit(UNIT_BUDGET),
            set_compute_unit_price(UNIT_PRICE),
            create_wsol_account_instruction,
            init_wsol_account_instruction,
            swap_instructions,
            close_wsol_account_instruction,
        ]

        # If selling 100%, close the token account afterward
        if percentage == 100:
            close_token_account_instruction = close_account(
                CloseAccountParams(
                    program_id=TOKEN_PROGRAM_ID,
                    account=token_account,
                    dest=payer_keypair.pubkey(),
                    owner=payer_keypair.pubkey(),
                )
            )
            instructions.append(close_token_account_instruction)

        # Compile transaction
        latest_blockhash = (await client.get_latest_blockhash()).value.blockhash
        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(), instructions, [], latest_blockhash
        )

        # Send transaction
        txn_sig = (await client.send_transaction(
            txn=VersionedTransaction(compiled_message, [payer_keypair]),
            opts=TxOpts(skip_preflight=True),
        )).value

        print("Transaction Signature:", txn_sig)
        return txn_sig

    except Exception as e:
        print("Error occurred during transaction:", e)
        return False
