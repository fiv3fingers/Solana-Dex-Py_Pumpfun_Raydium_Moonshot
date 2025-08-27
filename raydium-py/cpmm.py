import asyncio
import base64
import os
import struct
import json
from enum import Enum
from dataclasses import dataclass
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
from solders.system_program import (
    CreateAccountWithSeedParams,
    create_account_with_seed,
)
from solders.signature import Signature  # type: ignore
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

# Local or layout imports
from raydium.layouts.cpmm import CPMM_POOL_STATE_LAYOUT

# Constants
RPC = "https://mainnet.helius-rpc.com/?api-key=eba6d019-77f5-4715-9044-54eeeefeee23"
async_client = AsyncClient(RPC)  # <-- Async client

RAYDIUM_CPMM = Pubkey.from_string("CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C")
RAYDIUM_VAULT_AUTH_2 = Pubkey.from_string("GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbL")

ACCOUNT_LAYOUT_LEN = 165
WSOL = Pubkey.from_string("So11111111111111111111111111111111111111112")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
SOL_DECIMAL = 1e9


@dataclass
class CpmmPoolKeys:
    pool_state: Pubkey
    raydium_vault_auth_2: Pubkey
    amm_config: Pubkey
    pool_creator: Pubkey
    token_0_vault: Pubkey
    token_1_vault: Pubkey
    lp_mint: Pubkey
    token_0_mint: Pubkey
    token_1_mint: Pubkey
    token_0_program: Pubkey
    token_1_program: Pubkey
    observation_key: Pubkey
    auth_bump: int
    status: int
    lp_mint_decimals: int
    mint_0_decimals: int
    mint_1_decimals: int
    lp_supply: int
    protocol_fees_token_0: int
    protocol_fees_token_1: int
    fund_fees_token_0: int
    fund_fees_token_1: int
    open_time: int


class DIRECTION(Enum):
    BUY = 0
    SELL = 1


# ------------------------------------------------
# Utility: Convert SOL <-> Tokens with a constant-product formula
# ------------------------------------------------
def sol_for_tokens(sol_amount, base_vault_balance, quote_vault_balance, swap_fee=0.25):
    """Estimate how many tokens you receive for a given SOL input."""
    effective_sol_used = sol_amount - (sol_amount * (swap_fee / 100))
    constant_product = base_vault_balance * quote_vault_balance
    updated_base_vault_balance = constant_product / (quote_vault_balance + effective_sol_used)
    tokens_received = base_vault_balance - updated_base_vault_balance
    return round(tokens_received, 9)


def tokens_for_sol(token_amount, base_vault_balance, quote_vault_balance, swap_fee=0.25):
    """Estimate how many SOL you receive for a given token input."""
    effective_tokens_sold = token_amount * (1 - (swap_fee / 100))
    constant_product = base_vault_balance * quote_vault_balance
    updated_quote_vault_balance = constant_product / (base_vault_balance + effective_tokens_sold)
    sol_received = quote_vault_balance - updated_quote_vault_balance
    return round(sol_received, 9)


# ------------------------------------------------
# Confirm Transaction (Async)
# ------------------------------------------------
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


# ------------------------------------------------
# Async: Get SPL Token Balance
# ------------------------------------------------
async def get_token_balance(mint_str: str, pub_key_str: str) -> float | None:
    """Retrieve the SPL token balance for a given mint and owner."""
    mint = Pubkey.from_string(mint_str)
    owner_pubkey = Pubkey.from_string(pub_key_str)

    resp = await async_client.get_token_accounts_by_owner_json_parsed(
        owner_pubkey,
        TokenAccountOpts(mint=mint),
        commitment=Processed,
    )
    if resp.value:
        accounts = resp.value
        if accounts:
            token_amount = accounts[0].account.data.parsed["info"]["tokenAmount"]["uiAmount"]
            if token_amount:
                return float(token_amount)
    return None


# ------------------------------------------------
# Fetch CPMM Pool Keys (Async)
# ------------------------------------------------
async def fetch_cpmm_pool_keys(pair_address: str) -> Optional[CpmmPoolKeys]:
    """Parse CPMM pool layout data from the on-chain account (async)."""
    try:
        pool_state = Pubkey.from_string(pair_address)

        resp = await async_client.get_account_info_json_parsed(pool_state, commitment=Processed)
        if resp.value is None:
            print("No account data found for the given pool address.")
            return None

        pool_state_data = resp.value.data
        parsed_data = CPMM_POOL_STATE_LAYOUT.parse(pool_state_data)

        pool_keys = CpmmPoolKeys(
            pool_state=pool_state,
            raydium_vault_auth_2=Pubkey.from_string("GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbL"),
            amm_config=Pubkey.from_bytes(parsed_data.amm_config),
            pool_creator=Pubkey.from_bytes(parsed_data.pool_creator),
            token_0_vault=Pubkey.from_bytes(parsed_data.token_0_vault),
            token_1_vault=Pubkey.from_bytes(parsed_data.token_1_vault),
            lp_mint=Pubkey.from_bytes(parsed_data.lp_mint),
            token_0_mint=Pubkey.from_bytes(parsed_data.token_0_mint),
            token_1_mint=Pubkey.from_bytes(parsed_data.token_1_mint),
            token_0_program=Pubkey.from_bytes(parsed_data.token_0_program),
            token_1_program=Pubkey.from_bytes(parsed_data.token_1_program),
            observation_key=Pubkey.from_bytes(parsed_data.observation_key),
            auth_bump=parsed_data.auth_bump,
            status=parsed_data.status,
            lp_mint_decimals=parsed_data.lp_mint_decimals,
            mint_0_decimals=parsed_data.mint_0_decimals,
            mint_1_decimals=parsed_data.mint_1_decimals,
            lp_supply=parsed_data.lp_supply,
            protocol_fees_token_0=parsed_data.protocol_fees_token_0,
            protocol_fees_token_1=parsed_data.protocol_fees_token_1,
            fund_fees_token_0=parsed_data.fund_fees_token_0,
            fund_fees_token_1=parsed_data.fund_fees_token_1,
            open_time=parsed_data.open_time,
        )
        return pool_keys
    except Exception as e:
        print(f"Error fetching pool keys: {e}")
        return None


# ------------------------------------------------
# Get CPMM Vault Balances (Async)
# ------------------------------------------------
async def get_cpmm_reserves(pool_keys: CpmmPoolKeys) -> tuple:
    """
    Returns (base_reserve, quote_reserve, token_decimal).

    If the base mint is WSOL, the base is effectively SOL side, and quote is the other token.
    This function also subtracts out the fees from the vault balances.
    """
    try:
        quote_vault = pool_keys.token_0_vault
        quote_decimal = pool_keys.mint_0_decimals
        quote_mint = pool_keys.token_0_mint

        base_vault = pool_keys.token_1_vault
        base_decimal = pool_keys.mint_1_decimals
        base_mint = pool_keys.token_1_mint

        # Convert the pool's recorded fees from integer to float
        protocol_fees_token_0 = pool_keys.protocol_fees_token_0 / (10 ** quote_decimal)
        fund_fees_token_0 = pool_keys.fund_fees_token_0 / (10 ** quote_decimal)

        protocol_fees_token_1 = pool_keys.protocol_fees_token_1 / (10 ** base_decimal)
        fund_fees_token_1 = pool_keys.fund_fees_token_1 / (10 ** base_decimal)

        # Fetch both vaults
        resp = await async_client.get_multiple_accounts_json_parsed(
            [quote_vault, base_vault],
            commitment=Processed
        )
        accounts = resp.value
        if len(accounts) < 2 or not accounts[0] or not accounts[1]:
            print("Error: Could not retrieve vault balances.")
            return None, None, None

        quote_account = accounts[0]
        base_account = accounts[1]

        quote_account_balance = quote_account.data.parsed["info"]["tokenAmount"]["uiAmount"]
        base_account_balance = base_account.data.parsed["info"]["tokenAmount"]["uiAmount"]
        if quote_account_balance is None or base_account_balance is None:
            print("Error: One of the account balances is None.")
            return None, None, None

        # Subtract out fees from vault balances
        if base_mint == WSOL:
            # If base is WSOL, that means base is effectively 'SOL'
            base_reserve = quote_account_balance - (protocol_fees_token_0 + fund_fees_token_0)
            quote_reserve = base_account_balance - (protocol_fees_token_1 + fund_fees_token_1)
            token_decimal = quote_decimal
        else:
            base_reserve = base_account_balance - (protocol_fees_token_1 + fund_fees_token_1)
            quote_reserve = quote_account_balance - (protocol_fees_token_0 + fund_fees_token_0)
            token_decimal = base_decimal

        print(f"Base Mint: {base_mint} | Quote Mint: {quote_mint}")
        print(f"Base Reserve: {base_reserve} | Quote Reserve: {quote_reserve} | Token Decimal: {token_decimal}")
        return base_reserve, quote_reserve, token_decimal

    except Exception as e:
        print(f"Error retrieving CPMM reserves: {e}")
        return None, None, None


# ------------------------------------------------
# Make CPMM Swap Instruction
# ------------------------------------------------
def make_cpmm_swap_instruction(
    amount_in: int,
    minimum_amount_out: int,
    token_account_in: Pubkey,
    token_account_out: Pubkey,
    accounts: CpmmPoolKeys,
    owner: Pubkey,
    action: DIRECTION,
) -> Instruction:
    """Construct the Raydium CPMM swap instruction."""
    try:
        if action == DIRECTION.BUY:
            input_vault = accounts.token_0_vault
            output_vault = accounts.token_1_vault
            input_token_program = accounts.token_0_program
            output_token_program = accounts.token_1_program
            input_token_mint = accounts.token_0_mint
            output_token_mint = accounts.token_1_mint
        else:  # SELL
            input_vault = accounts.token_1_vault
            output_vault = accounts.token_0_vault
            input_token_program = accounts.token_1_program
            output_token_program = accounts.token_0_program
            input_token_mint = accounts.token_1_mint
            output_token_mint = accounts.token_0_mint

        keys = [
            AccountMeta(pubkey=owner, is_signer=True, is_writable=True),
            AccountMeta(pubkey=accounts.raydium_vault_auth_2, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.amm_config, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.pool_state, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_account_in, is_signer=False, is_writable=True),
            AccountMeta(pubkey=token_account_out, is_signer=False, is_writable=True),
            AccountMeta(pubkey=input_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=output_vault, is_signer=False, is_writable=True),
            AccountMeta(pubkey=input_token_program, is_signer=False, is_writable=False),
            AccountMeta(pubkey=output_token_program, is_signer=False, is_writable=False),
            AccountMeta(pubkey=input_token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=output_token_mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=accounts.observation_key, is_signer=False, is_writable=True),
        ]

        # The known discriminator for Raydium CPMM swap
        data = bytearray()
        data.extend(bytes.fromhex("8fbe5adac41e33de"))
        data.extend(struct.pack("<Q", amount_in))
        data.extend(struct.pack("<Q", minimum_amount_out))

        swap_instruction = Instruction(RAYDIUM_CPMM, bytes(data), keys)
        return swap_instruction
    except Exception as e:
        print(f"Error occurred creating CPMM swap instruction: {e}")
        return None


# ------------------------------------------------
# BUY (Async)
# ------------------------------------------------
async def buy(
    priv_key: str,
    pair_address: str,
    sol_in: float,
    slippage: int,
    UNIT_BUDGET: int,
    UNIT_PRICE: int,
):
    """
    Asynchronously buy tokens from a Raydium CPMM pool with SOL input.
    """
    try:
        payer_keypair = Keypair.from_base58_string(priv_key)
        print(f"Starting buy transaction for pair address: {pair_address}")

        print("Fetching pool keys (async)...")
        pool_keys = await fetch_cpmm_pool_keys(pair_address)
        if pool_keys is None:
            print("No pool keys found...")
            return False
        print("Pool keys fetched successfully.")

        # Decide which mint is the 'other token'
        if pool_keys.token_0_mint == WSOL:
            mint = pool_keys.token_1_mint
            token_program = pool_keys.token_1_program
        else:
            mint = pool_keys.token_0_mint
            token_program = pool_keys.token_0_program

        print("Calculating transaction amounts...")
        amount_in = int(sol_in * SOL_DECIMAL)

        base_reserve, quote_reserve, token_decimal = await get_cpmm_reserves(pool_keys)
        if base_reserve is None or quote_reserve is None or token_decimal is None:
            print("Error: Could not retrieve pool reserves.")
            return False

        amount_out = sol_for_tokens(sol_in, base_reserve, quote_reserve)
        print(f"Estimated Amount Out: {amount_out}")

        slippage_adjustment = 1 - (slippage / 100)
        amount_out_with_slippage = amount_out * slippage_adjustment
        minimum_amount_out = int(amount_out_with_slippage * (10 ** token_decimal))
        print(f"Amount In: {amount_in} | Minimum Amount Out: {minimum_amount_out}")

        print("Checking for existing token account (async)...")
        token_account_resp = await async_client.get_token_accounts_by_owner(
            payer_keypair.pubkey(), TokenAccountOpts(mint=mint), Processed
        )
        if token_account_resp.value:
            token_account = token_account_resp.value[0].pubkey
            token_account_instruction = None
            print("Token account found.")
        else:
            token_account = get_associated_token_address(
                payer_keypair.pubkey(), mint, token_program
            )
            token_account_instruction = create_associated_token_account(
                payer_keypair.pubkey(), payer_keypair.pubkey(), mint, token_program
            )
            print("No existing token account found; creating associated token account.")

        print("Generating seed for WSOL account...")
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        wsol_token_account = Pubkey.create_with_seed(
            payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID
        )

        print("Fetching min rent-exemption (async)...")
        min_balance_resp = await async_client.get_minimum_balance_for_rent_exemption(ACCOUNT_LAYOUT_LEN, commitment=Processed)
        balance_needed = min_balance_resp.value

        print("Creating and initializing WSOL account...")
        create_wsol_account_instruction = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer_keypair.pubkey(),
                to_pubkey=wsol_token_account,
                base=payer_keypair.pubkey(),
                seed=seed,
                lamports=int(balance_needed + amount_in),
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
        swap_instruction = make_cpmm_swap_instruction(
            amount_in=amount_in,
            minimum_amount_out=minimum_amount_out,
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
        print("Error occurred during buy transaction:", e)
        return False


# ------------------------------------------------
# SELL (Async)
# ------------------------------------------------
async def sell(
    priv_key: str,
    pair_address: str,
    token_amount_to_sell: float,
    percentage: int,
    slippage: int,
    UNIT_BUDGET: int,
    UNIT_PRICE: int,
):
    """
    Asynchronously sell tokens in a Raydium CPMM pool, receiving SOL.
    """
    try:
        payer_keypair = Keypair.from_base58_string(priv_key)
        pub_key = payer_keypair.pubkey()
        token_balance = token_amount_to_sell

        print("Fetching pool keys (async)...")
        pool_keys = await fetch_cpmm_pool_keys(pair_address)
        if pool_keys is None:
            print("No pool keys found...")
            return False
        print("Pool keys fetched successfully.")

        if not (1 <= percentage <= 100):
            print("Percentage must be between 1 and 100.")
            return False

        # Figure out which side is the token being sold
        if pool_keys.token_0_mint == WSOL:
            mint = pool_keys.token_1_mint
            token_program_id = pool_keys.token_1_program
            token_decimal = pool_keys.mint_1_decimals
        else:
            mint = pool_keys.token_0_mint
            token_program_id = pool_keys.token_0_program
            token_decimal = pool_keys.mint_0_decimals

        # print("Retrieving token balance (async)...")
        # current_balance = await get_token_balance(str(mint), str(pub_key))
        # print("Token Balance:", current_balance)

        # if current_balance is None or current_balance == 0:
        #     print("No token balance available to sell.")
        #     return False

        # # Adjust the balance by the user-specified percentage
        # token_balance = current_balance * (percentage / 100)
        # print(f"Selling {percentage}% of the token balance, adjusted balance: {token_balance}")

        # Figure out how much SOL we get
        base_reserve, quote_reserve, base_decimal = await get_cpmm_reserves(pool_keys)
        if base_reserve is None or quote_reserve is None or base_decimal is None:
            print("Error: Could not retrieve pool reserves.")
            return False

        sol_estimate = tokens_for_sol(token_balance, base_reserve, quote_reserve)
        print(f"Estimated Amount Out: {sol_estimate} SOL")

        slippage_adjustment = 1 - (slippage / 100)
        sol_out_with_slippage = sol_estimate * slippage_adjustment
        minimum_amount_out = int(sol_out_with_slippage * SOL_DECIMAL)

        # Convert token balance to integer
        amount_in = int(token_balance * (10 ** token_decimal))
        print(f"Amount In: {amount_in} | Minimum SOL Out (lamports): {minimum_amount_out}")

        # Get user's token account
        token_account = get_associated_token_address(
            payer_keypair.pubkey(), mint, token_program_id
        )

        print("Generating seed for WSOL account...")
        seed = base64.urlsafe_b64encode(os.urandom(24)).decode("utf-8")
        wsol_token_account = Pubkey.create_with_seed(
            payer_keypair.pubkey(), seed, TOKEN_PROGRAM_ID
        )

        print("Fetching min rent-exemption (async)...")
        min_balance_resp = await async_client.get_minimum_balance_for_rent_exemption(
            ACCOUNT_LAYOUT_LEN,
            commitment=Processed
        )
        balance_needed = min_balance_resp.value

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

        print("Creating swap instructions...")
        swap_instruction = make_cpmm_swap_instruction(
            amount_in=amount_in,
            minimum_amount_out=minimum_amount_out,
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
        print("Error occurred during sell transaction:", e)
        return False

