import asyncio
import base64
import json
import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Asynchronous Solana client
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

# SPL Token
from spl.token.client import Token
from spl.token.instructions import (
    CloseAccountParams,
    InitializeAccountParams,
    close_account,
    create_associated_token_account,
    get_associated_token_address,
    initialize_account,
)

from raydium.layouts.clmm import CLMM_POOL_STATE_LAYOUT

# Raydium-specific constants
from raydium.constants import (
    ACCOUNT_LAYOUT_LEN,
    SOL_DECIMAL,
    TOKEN_PROGRAM_ID,
    WSOL
)

# Create a global async client
RPC = "https://mainnet.helius-rpc.com/?api-key=eba6d019-77f5-4715-9044-54eeeefeee23"
async_client = AsyncClient(RPC)

RAYDIUM_CLMM = Pubkey.from_string("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK")
DEFAULT_QUOTE_MINT = "So11111111111111111111111111111111111111112"
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ACCOUNT_LAYOUT_LEN = 165

WSOL = Pubkey.from_string("So11111111111111111111111111111111111111112")
SOL_DECIMAL = 1e9

@dataclass
class ClmmPoolKeys:
    pool_state: Pubkey
    amm_config: Pubkey
    owner: Pubkey
    token_mint_0: Pubkey
    token_mint_1: Pubkey
    token_vault_0: Pubkey
    token_vault_1: Pubkey
    observation_key: Pubkey
    current_tick_array: Pubkey
    prev_tick_array: Pubkey
    additional_tick_array: Pubkey
    bitmap_extension: Pubkey
    mint_decimals_0: int
    mint_decimals_1: int
    tick_spacing: int
    liquidity: int
    sqrt_price_x64: int
    tick_current: int
    observation_index: int
    observation_update_duration: int
    fee_growth_global_0_x64: int
    fee_growth_global_1_x64: int
    protocol_fees_token_0: int
    protocol_fees_token_1: int
    swap_in_amount_token_0: int
    swap_out_amount_token_1: int
    swap_in_amount_token_1: int
    swap_out_amount_token_0: int
    status: int
    total_fees_token_0: int
    total_fees_claimed_token_0: int
    total_fees_token_1: int
    total_fees_claimed_token_1: int
    fund_fees_token_0: int
    fund_fees_token_1: int


class DIRECTION(Enum):
    BUY = 0
    SELL = 1


# --------------------------------------------------------------------------------
# Helper Conversions
# --------------------------------------------------------------------------------
def sqrt_price_x64_to_price(sqrt_price_x64: int, mint_decimals_0: int, mint_decimals_1: int) -> float:
    Q64 = 2 ** 64
    sqrt_price = sqrt_price_x64 / Q64
    price = (sqrt_price ** 2) * (10 ** (mint_decimals_0 - mint_decimals_1))
    return price


def sol_for_tokens(sol_in: float, sqrt_price_x64: int, mint_decimals_0: int, mint_decimals_1: int) -> float:
    """Estimate how many tokens you'll get when you input `sol_in` SOL, 
    based on the sqrt price."""
    token_price = 1 / sqrt_price_x64_to_price(sqrt_price_x64, mint_decimals_0, mint_decimals_1)
    tokens_out = sol_in / token_price
    return round(tokens_out, 9)


def tokens_for_sol(tokens_in: float, sqrt_price_x64: int, mint_decimals_0: int, mint_decimals_1: int) -> float:
    """Estimate how many SOL you'll get by selling `tokens_in` tokens."""
    token_price = 1 / sqrt_price_x64_to_price(sqrt_price_x64, mint_decimals_0, mint_decimals_1)
    sol_out = tokens_in * token_price
    return round(sol_out, 9)


# --------------------------------------------------------------------------------
# Fetch Pool Keys (CLMM)
# --------------------------------------------------------------------------------
async def fetch_clmm_pool_keys(pair_address: str, zero_for_one: bool = True) -> Optional[ClmmPoolKeys]:
    """Load the pool's on-chain data asynchronously and build ClmmPoolKeys."""

    def calculate_start_index(tick_current: int, tick_spacing: int, tick_array_size: int = 60) -> int:
        return (tick_current // (tick_spacing * tick_array_size)) * (tick_spacing * tick_array_size)

    def get_pda_tick_array_address(pool_id: Pubkey, start_index: int):
        tick_array, _ = Pubkey.find_program_address(
            [b"tick_array", bytes(pool_id), struct.pack(">i", start_index)],
            Pubkey.from_string("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"),
        )
        return tick_array

    def get_pda_tick_array_bitmap_extension(pool_id: Pubkey):
        bitmap_extension, _ = Pubkey.find_program_address(
            [b"pool_tick_array_bitmap_extension", bytes(pool_id)],
            Pubkey.from_string("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"),
        )
        return bitmap_extension

    try:
        pool_state = Pubkey.from_string(pair_address)

        # Get the pool state via async
        resp = await async_client.get_account_info_json_parsed(pool_state, commitment=Processed)
        if resp.value is None:
            print("Unable to fetch pool data. Account does not exist?")
            return None

        pool_state_data = resp.value.data
        parsed_data = CLMM_POOL_STATE_LAYOUT.parse(pool_state_data)

        tick_spacing = int(parsed_data.tick_spacing)
        tick_current = int(parsed_data.tick_current)
        array_size = 60

        start_index = calculate_start_index(tick_current, tick_spacing)
        if zero_for_one:
            prev_index = start_index - (tick_spacing * array_size)
            additional_index = prev_index - (tick_spacing * array_size)
        else:
            prev_index = start_index + (tick_spacing * array_size)
            additional_index = prev_index + (tick_spacing * array_size)

        current_tick_array = get_pda_tick_array_address(pool_state, start_index)
        prev_tick_array = get_pda_tick_array_address(pool_state, prev_index)
        additional_tick_array = get_pda_tick_array_address(pool_state, additional_index)
        bitmap_extension = get_pda_tick_array_bitmap_extension(pool_state)

        pool_keys = ClmmPoolKeys(
            pool_state=pool_state,
            amm_config=Pubkey.from_bytes(parsed_data.amm_config),
            owner=Pubkey.from_bytes(parsed_data.owner),
            token_mint_0=Pubkey.from_bytes(parsed_data.token_mint_0),
            token_mint_1=Pubkey.from_bytes(parsed_data.token_mint_1),
            token_vault_0=Pubkey.from_bytes(parsed_data.token_vault_0),
            token_vault_1=Pubkey.from_bytes(parsed_data.token_vault_1),
            observation_key=Pubkey.from_bytes(parsed_data.observation_key),
            current_tick_array=current_tick_array,
            prev_tick_array=prev_tick_array,
            additional_tick_array=additional_tick_array,
            bitmap_extension=bitmap_extension,
            mint_decimals_0=parsed_data.mint_decimals_0,
            mint_decimals_1=parsed_data.mint_decimals_1,
            tick_spacing=parsed_data.tick_spacing,
            liquidity=parsed_data.liquidity,
            sqrt_price_x64=parsed_data.sqrt_price_x64,
            tick_current=parsed_data.tick_current,
            observation_index=parsed_data.observation_index,
            observation_update_duration=parsed_data.observation_update_duration,
            fee_growth_global_0_x64=parsed_data.fee_growth_global_0_x64,
            fee_growth_global_1_x64=parsed_data.fee_growth_global_1_x64,
            protocol_fees_token_0=parsed_data.protocol_fees_token_0,
            protocol_fees_token_1=parsed_data.protocol_fees_token_1,
            swap_in_amount_token_0=parsed_data.swap_in_amount_token_0,
            swap_out_amount_token_1=parsed_data.swap_out_amount_token_1,
            swap_in_amount_token_1=parsed_data.swap_in_amount_token_1,
            swap_out_amount_token_0=parsed_data.swap_out_amount_token_0,
            status=parsed_data.status,
            total_fees_token_0=parsed_data.total_fees_token_0,
            total_fees_claimed_token_0=parsed_data.total_fees_claimed_token_0,
            total_fees_token_1=parsed_data.total_fees_token_1,
            total_fees_claimed_token_1=parsed_data.total_fees_claimed_token_1,
            fund_fees_token_0=parsed_data.fund_fees_token_0,
            fund_fees_token_1=parsed_data.fund_fees_token_1
        )
        return pool_keys

    except Exception as e:
        print(f"Error fetching pool keys: {e}")
        return None


# --------------------------------------------------------------------------------
# Get CLMM Vault Reserves (Synchronous Example?)
# If needed asynchronously, you'd do: 
#    await async_client.get_multiple_accounts_json_parsed([...])
# --------------------------------------------------------------------------------
def get_clmm_reserves(pool_keys: ClmmPoolKeys):
    """
    NOTE: If you want to be fully async, you should also convert
    this function to async and call `await async_client.get_multiple_accounts_json_parsed(...)`.
    Currently it's using a synchronous call for demonstration.
    """
    pass  # Omitted or changed as needed


# --------------------------------------------------------------------------------
# Confirm Transaction (Async)
# --------------------------------------------------------------------------------
async def confirm_txn(txn_sig: Signature, max_retries: int = 20, retry_interval: float = 0.1) -> bool:
    """Poll the Solana RPC to confirm a transaction."""
    retries = 1
    while retries < max_retries:
        try:
            txn_res = await async_client.get_transaction(
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


# --------------------------------------------------------------------------------
# Get SPL Token Balance (Async)
# --------------------------------------------------------------------------------
async def get_token_balance(mint_str: str, pub_key_str: str) -> float | None:
    """Retrieve the SPL token balance for a given mint and owner."""
    mint = Pubkey.from_string(mint_str)
    owner_pubkey = Pubkey.from_string(pub_key_str)

    response = await async_client.get_token_accounts_by_owner_json_parsed(
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


# --------------------------------------------------------------------------------
# Make CLMM Swap Instruction
# --------------------------------------------------------------------------------
async def make_clmm_swap_instruction(
    amount: int,
    token_account_in: Pubkey,
    token_account_out: Pubkey,
    accounts: ClmmPoolKeys,
    owner: Pubkey,
    action: DIRECTION
) -> Instruction:
    """
    Build the Raydium CLMM swap instruction data + accounts (async version).
    Same logic, but marked 'async' for consistency in an async environment.
    """
    try:
        # Decide which vaults to use based on buy or sell direction
        if action == DIRECTION.BUY:
            input_vault = accounts.token_vault_0
            output_vault = accounts.token_vault_1
        else:  # SELL
            input_vault = accounts.token_vault_1
            output_vault = accounts.token_vault_0

        keys = [
            AccountMeta(pubkey=owner, is_signer=True, is_writable=True),
            AccountMeta(pubkey=accounts.amm_config, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.pool_state, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_account_in, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_account_out, is_signer=False, is_writable=True),
            AccountMeta(pubkey=input_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=output_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.observation_key, is_signer=False, is_writable=True),
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.current_tick_array, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.bitmap_extension, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.prev_tick_array, is_signer=False, is_writable=True),
            AccountMeta(pubkey=accounts.additional_tick_array, is_signer=False, is_writable=True),
        ]

        data = bytearray()
        # 8-byte discriminator known for Raydium CLMM swap
        data.extend(bytes.fromhex("f8c69e91e17587c8"))
        # amount_in
        data.extend(struct.pack("<Q", amount))
        # min_amount_out (hard-coded to 0 in this example)
        data.extend(struct.pack("<Q", 0))
        # Possibly a placeholder or extra data
        data.extend((0).to_bytes(16, byteorder="little"))
        # Some boolean (True)
        data.extend(struct.pack("<?", True))

        swap_instruction = Instruction(RAYDIUM_CLMM, bytes(data), keys)
        return swap_instruction
    except Exception as e:
        print(f"Error occurred creating CLMM swap instruction: {e}")
        return None


# --------------------------------------------------------------------------------
# BUY (Async)
# --------------------------------------------------------------------------------
async def buy(
    priv_key: str,
    pair_address: str,
    sol_in: float,
    UNIT_BUDGET: int,
    UNIT_PRICE: int
):
    """
    Buy tokens from CLMM pool, paying with SOL.
    """
    try:
        payer_keypair = Keypair.from_base58_string(priv_key)
        print(f"Starting buy transaction for pair address: {pair_address}")

        print("Fetching pool keys (async)...")
        pool_keys: Optional[ClmmPoolKeys] = await fetch_clmm_pool_keys(pair_address, zero_for_one=True)
        if pool_keys is None:
            print("No pool keys found...")
            return False
        print("Pool keys fetched successfully.")

        # Decide which mint is the token being bought
        if pool_keys.token_mint_0 == WSOL:
            mint = pool_keys.token_mint_1
        else:
            mint = pool_keys.token_mint_0

        # Convert SOL to lamports
        amount = int(sol_in * SOL_DECIMAL)

        # Estimate tokens out (using the sqrt_price_x64 approach)
        tokens_out = sol_for_tokens(
            sol_in, 
            pool_keys.sqrt_price_x64, 
            pool_keys.mint_decimals_0, 
            pool_keys.mint_decimals_1
        )
        print(f"Amount In: {sol_in} SOL | Estimated Tokens Out: {tokens_out}")

        # Check for existing token account
        print("Checking for existing token account (async)...")
        token_account_resp = await async_client.get_token_accounts_by_owner(
            payer_keypair.pubkey(), 
            TokenAccountOpts(mint=mint),
            Processed
        )
        if token_account_resp.value:
            token_account = token_account_resp.value[0].pubkey
            token_account_instruction = None
            print("Token account found.")
        else:
            token_account = get_associated_token_address(payer_keypair.pubkey(), mint)
            token_account_instruction = create_associated_token_account(
                payer_keypair.pubkey(), 
                payer_keypair.pubkey(), 
                mint
            )
            print("No existing token account found; creating associated token account.")

        # Create WSOL account
        print("Generating seed for WSOL account...")
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        wsol_token_account = Pubkey.create_with_seed(
            payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID
        )

        print("Fetching rent-exemption (async)...")
        balance_resp = await async_client.get_minimum_balance_for_rent_exemption(
            ACCOUNT_LAYOUT_LEN,
            commitment=Processed
        )
        # The actual integer value is in `balance_resp.value`
        balance_needed = balance_resp.value

        print("Creating and initializing WSOL account...")
        create_wsol_account_instruction = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=wsol_token_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(balance_needed + amount),
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

        print("Creating swap instructions...")
        swap_instruction = await make_clmm_swap_instruction(
            amount=amount,
            token_account_in=wsol_token_account,
            token_account_out=token_account,
            accounts=pool_keys,
            owner=payer_keypair.pubkey(),
            action=DIRECTION.BUY,
        )

        print("Preparing to close WSOL account after swap...")
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

        if token_account_instruction:
            instructions.append(token_account_instruction)

        instructions.append(swap_instruction)
        instructions.append(close_wsol_account_instruction)

        print("Compiling transaction message...")
        latest_blockhash = (await async_client.get_latest_blockhash()).value.blockhash
        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],
            latest_blockhash,
        )

        print("Sending transaction...")
        txn_sig = (await async_client.send_transaction(
            txn=VersionedTransaction(compiled_message, [payer_keypair]),
            opts=TxOpts(skip_preflight=True),
        )).value
        print("Transaction Signature:", txn_sig)

        return txn_sig

    except Exception as e:
        print("Error occurred during transaction:", e)
        return False


# --------------------------------------------------------------------------------
# SELL (Async)
# --------------------------------------------------------------------------------
async def sell(
    priv_key: str, 
    pair_address: str, 
    token_amount_to_sell: float,
    percentage: int, 
    UNIT_BUDGET: int, 
    UNIT_PRICE: int
):
    """
    Sell tokens (base/quote) in a Raydium CLMM pool, receiving SOL.
    """
    try:
        payer_keypair = Keypair.from_base58_string(priv_key)
        pub_key = payer_keypair.pubkey()
        token_balance = token_amount_to_sell

        print("Fetching pool keys (async)...")
        pool_keys: Optional[ClmmPoolKeys] = await fetch_clmm_pool_keys(pair_address, zero_for_one=True)
        if pool_keys is None:
            print("No pool keys found...")
            return False
        print("Pool keys fetched successfully.")

        if not (1 <= percentage <= 100):
            print("Percentage must be between 1 and 100.")
            return False

        # Figure out which mint is the token being sold
        if pool_keys.token_mint_0 == WSOL:
            mint = pool_keys.token_mint_1
            token_decimal = pool_keys.mint_decimals_1
        else:
            mint = pool_keys.token_mint_0
            token_decimal = pool_keys.mint_decimals_0

        # print("Retrieving token balance (async)...")
        # token_balance = await get_token_balance(str(mint), str(pub_key))
        # print("Token Balance:", token_balance)

        # if token_balance is None or token_balance == 0:
        #     print("No token balance available to sell.")
        #     return False

        # # Adjust by percentage
        # token_balance = token_balance * (percentage / 100)
        # print(f"Selling {percentage}% of the token balance, adjusted balance: {token_balance}")

        # Estimate how many SOL we get
        sol_out = tokens_for_sol(
            token_balance, 
            pool_keys.sqrt_price_x64, 
            pool_keys.mint_decimals_0, 
            pool_keys.mint_decimals_1
        )
        print(f"Amount In: {token_balance} tokens | Estimated SOL Out: {sol_out}")

        # Convert tokens_in to integer
        amount_in = int(token_balance * (10 ** token_decimal))
        token_account = get_associated_token_address(payer_keypair.pubkey(), mint)

        # Create a WSOL account to receive SOL
        print("Generating seed for WSOL account...")
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        wsol_token_account = Pubkey.create_with_seed(
            payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID
        )

        print("Fetching rent-exemption (async) for WSOL account...")
        balance_resp = await async_client.get_minimum_balance_for_rent_exemption(
            ACCOUNT_LAYOUT_LEN,
            commitment=Processed
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

        print("Creating swap instructions (token -> WSOL)...")
        swap_instruction = await make_clmm_swap_instruction(
            amount=amount_in,
            token_account_in=token_account,
            token_account_out=wsol_token_account,
            accounts=pool_keys,
            owner=payer_keypair.pubkey(),
            action=DIRECTION.SELL,
        )

        print("Preparing to close WSOL account after swap...")
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
            swap_instruction,
            close_wsol_account_instruction,
        ]

        # If selling 100%, optionally close the user’s token account
        if percentage == 100:
            print("Preparing to close token account after swap...")
            close_token_account_instruction = close_account(
                CloseAccountParams(
                    program_id=TOKEN_PROGRAM_ID,
                    account=token_account,
                    dest=payer_keypair.pubkey(),
                    owner=payer_keypair.pubkey(),
                )
            )
            instructions.append(close_token_account_instruction)

        print("Compiling transaction message...")
        latest_blockhash = (await async_client.get_latest_blockhash()).value.blockhash
        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],
            latest_blockhash,
        )

        print("Sending transaction...")
        txn_sig = (await async_client.send_transaction(
            txn=VersionedTransaction(compiled_message, [payer_keypair]),
            opts=TxOpts(skip_preflight=True),
        )).value
        print("Transaction Signature:", txn_sig)

        return txn_sig

    except Exception as e:
        print("Error occurred during transaction:", e)
        return False
