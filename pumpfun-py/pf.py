import asyncio
import base64
import aiohttp
import base58
import json
import struct
import time
from dataclasses import dataclass
from typing import Optional

# Solana / Solder / SPL Imports
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Processed
from solana.rpc.types import TokenAccountOpts, TxOpts
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price  # type: ignore
from solders.instruction import AccountMeta, Instruction  # type: ignore
from solders.keypair import Keypair  # type: ignore
from solders.message import MessageV0  # type: ignore
from solders.pubkey import Pubkey  # type: ignore
from solders.signature import Signature  # type: ignore
from solders.system_program import CreateAccountWithSeedParams, create_account_with_seed
from solders.transaction import VersionedTransaction  # type: ignore

# SPL Token
from spl.token.instructions import (
    CloseAccountParams,
    close_account,
    create_associated_token_account,
    get_associated_token_address,
)

# Construct library
from construct import Flag, Int64ul, Padding, Struct

# Constants
RPC = "https://mainnet.helius-rpc.com/?api-key=eba6d019-77f5-4715-9044-54eeeefeee23"
async_client = AsyncClient(RPC)  # <--- ASYNC client

# Program and account addresses
GLOBAL = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
FEE_RECIPIENT = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOC_TOKEN_ACC_PROG = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
RENT = Pubkey.from_string("SysvarRent111111111111111111111111111111111")
EVENT_AUTHORITY = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")
PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")


@dataclass
class CoinData:
    mint: Pubkey
    bonding_curve: Pubkey
    associated_bonding_curve: Pubkey
    virtual_token_reserves: int
    virtual_sol_reserves: int
    token_total_supply: int
    complete: bool


# --------------------------------------------------------------------------------
# Utility: Fetch Virtual Reserves
# --------------------------------------------------------------------------------
async def get_virtual_reserves(bonding_curve: Pubkey):
    """Fetch and parse bonding curve account data."""
    bonding_curve_struct = Struct(
        Padding(8),
        "virtualTokenReserves" / Int64ul,
        "virtualSolReserves" / Int64ul,
        "realTokenReserves" / Int64ul,
        "realSolReserves" / Int64ul,
        "tokenTotalSupply" / Int64ul,
        "complete" / Flag
    )
    try:
        account_info = await async_client.get_account_info(bonding_curve)
        if account_info.value is None:
            return None
        data = account_info.value.data
        parsed_data = bonding_curve_struct.parse(data)
        return parsed_data
    except Exception:
        return None


def derive_bonding_curve_accounts(mint_str: str):
    """Derive the bonding_curve and associated_bonding_curve addresses."""
    try:
        mint = Pubkey.from_string(mint_str)
        bonding_curve, _ = Pubkey.find_program_address(
            ["bonding-curve".encode(), bytes(mint)],
            PUMP_FUN_PROGRAM
        )
        associated_bonding_curve = get_associated_token_address(bonding_curve, mint)
        return bonding_curve, associated_bonding_curve
    except Exception:
        return None, None


# --------------------------------------------------------------------------------
# Build and Return CoinData
# --------------------------------------------------------------------------------
async def get_coin_data(mint_str: str) -> Optional[CoinData]:
    """Get coin data by deriving bonding curve addresses and fetching virtual reserves."""
    bonding_curve, associated_bonding_curve = derive_bonding_curve_accounts(mint_str)
    if bonding_curve is None or associated_bonding_curve is None:
        return None

    virtual_reserves = await get_virtual_reserves(bonding_curve)
    if virtual_reserves is None:
        return None

    try:
        return CoinData(
            mint=Pubkey.from_string(mint_str),
            bonding_curve=bonding_curve,
            associated_bonding_curve=associated_bonding_curve,
            virtual_token_reserves=int(virtual_reserves.virtualTokenReserves),
            virtual_sol_reserves=int(virtual_reserves.virtualSolReserves),
            token_total_supply=int(virtual_reserves.tokenTotalSupply),
            complete=bool(virtual_reserves.complete),
        )
    except Exception as e:
        print(e)
        return None


# --------------------------------------------------------------------------------
# Pricing Formulas
# --------------------------------------------------------------------------------
def sol_for_tokens(sol_spent, sol_reserves, token_reserves):
    """Constant-product estimate of tokens for a given SOL input."""
    new_sol_reserves = sol_reserves + sol_spent
    new_token_reserves = (sol_reserves * token_reserves) / new_sol_reserves
    token_received = token_reserves - new_token_reserves
    return round(token_received)


def tokens_for_sol(tokens_to_sell, sol_reserves, token_reserves):
    """Constant-product estimate of SOL for a given token input."""
    new_token_reserves = token_reserves + tokens_to_sell
    new_sol_reserves = (sol_reserves * token_reserves) / new_token_reserves
    sol_received = sol_reserves - new_sol_reserves
    return sol_received


# --------------------------------------------------------------------------------
# Async: Get SPL Token Balance
# --------------------------------------------------------------------------------
async def get_token_balance(mint_str: str, pub_key_str: str) -> float | None:
    """Retrieve the SPL token balance for a given mint and owner."""
    try:
        mint = Pubkey.from_string(mint_str)
        owner_pubkey = Pubkey.from_string(pub_key_str)

        resp = await async_client.get_token_accounts_by_owner_json_parsed(
            owner_pubkey,
            TokenAccountOpts(mint=mint),
            commitment=Processed
        )
        if resp.value:
            accounts = resp.value
            if accounts:
                token_amount = accounts[0].account.data.parsed["info"]["tokenAmount"]["uiAmount"]
                return float(token_amount)
        return None
    except Exception as e:
        print(f"Error fetching token balance: {e}")
        return None


# --------------------------------------------------------------------------------
# Async: Confirm Transaction
# --------------------------------------------------------------------------------
async def confirm_txn(txn_sig: Signature, max_retries: int = 50, retry_interval: int = 0.5) -> bool:
    """Poll the Solana RPC to confirm a transaction, waiting up to max_retries."""
    retries = 1
    while retries < max_retries:
        try:
            txn_res = await async_client.get_transaction(
                txn_sig,
                encoding="json",
                commitment=Confirmed,
                max_supported_transaction_version=0
            )
            if txn_res.value is None:
                # Not yet found
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
# Async: Get token price
# --------------------------------------------------------------------------------
async def get_token_price(mint_str: str) -> float:
    """Calculate a token's price in SOL from the bonding curve virtual reserves."""
    try:
        coin_data = await get_coin_data(mint_str)
        if not coin_data:
            print("Failed to retrieve coin data...")
            return None

        virtual_sol_reserves = coin_data.virtual_sol_reserves / 10**9
        virtual_token_reserves = coin_data.virtual_token_reserves / 10**6

        token_price = virtual_sol_reserves / virtual_token_reserves
        print(f"Token Price: {token_price:.20f} SOL")
        return token_price
    except Exception as e:
        print(f"Error calculating token price: {e}")
        return None

async def send_transaction_jito(versioned_txn):
    try:
        # Serialize the transaction
        serialized_transaction = base58.b58encode(bytes(versioned_txn)).decode('ascii')
        
        # Jito API endpoint with the UUID as a query parameter
        url = "https://frankfurt.mainnet.block-engine.jito.wtf/api/v1/transactions?uuid=ef022ae0-8ce9-11ef-9749-3f78db976128"
        
        # Headers including the x-jito-auth header with the UUID
        headers = {
            "Content-Type": "application/json",
            "x-jito-auth": "ef022ae0-8ce9-11ef-9749-3f78db976128"
        }
        
        # Payload for the request
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [serialized_transaction]
        })
        
        # Send the POST request
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                response_json = await response.json()
                print(f"Jito transaction response: {response_json}")
                
                # Extract and return the transaction signature if available
                txn_sig = response_json.get('result')
                return txn_sig
    except Exception as e:
        print(f"Exception occurred while sending Jito transaction: {e}")
        return None


# --------------------------------------------------------------------------------
# BUY (Async)
# --------------------------------------------------------------------------------
async def buy(
    priv_key: str,
    mint_str: str,
    sol_in: float,
    slippage: int,
    UNIT_BUDGET: int,
    UNIT_PRICE: int,
):
    """Buy from the bonding curve with SOL."""
    payer_keypair = Keypair.from_base58_string(priv_key)
    try:
        print(f"Starting buy transaction for mint: {mint_str}")

        coin_data = await get_coin_data(mint_str)
        if not coin_data:
            print("Failed to retrieve coin data.")
            return False

        if coin_data.complete:
            print("Warning: This token has bonded and is only tradable on Raydium.")
            return False

        MINT = coin_data.mint
        BONDING_CURVE = coin_data.bonding_curve
        ASSOCIATED_BONDING_CURVE = coin_data.associated_bonding_curve
        USER = payer_keypair.pubkey()

        print("Fetching or creating associated token account...")
        try:
            resp = await async_client.get_token_accounts_by_owner(
                USER, TokenAccountOpts(MINT), commitment=Processed
            )
            if resp.value:
                ASSOCIATED_USER = resp.value[0].pubkey
                token_account_instruction = None
                print(f"Token account found: {ASSOCIATED_USER}")
            else:
                ASSOCIATED_USER = get_associated_token_address(USER, MINT)
                token_account_instruction = create_associated_token_account(USER, USER, MINT)
                print(f"Creating token account : {ASSOCIATED_USER}")
        except:
            ASSOCIATED_USER = get_associated_token_address(USER, MINT)
            token_account_instruction = create_associated_token_account(USER, USER, MINT)
            print(f"Creating token account : {ASSOCIATED_USER}")

        print("Calculating transaction amounts...")
        sol_dec = 1e9
        token_dec = 1e6
        virtual_sol_reserves = coin_data.virtual_sol_reserves / sol_dec
        virtual_token_reserves = coin_data.virtual_token_reserves / token_dec
        amount = sol_for_tokens(sol_in, virtual_sol_reserves, virtual_token_reserves)
        amount = int(amount * token_dec)

        slippage_adjustment = 1 + (slippage / 100)
        max_sol_cost = int((sol_in * slippage_adjustment) * sol_dec)
        print(f"Amount: {amount}, Max Sol Cost: {max_sol_cost}")

        print("Creating swap instructions...")
        keys = [
            AccountMeta(pubkey=GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(pubkey=FEE_RECIPIENT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=MINT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=BONDING_CURVE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_BONDING_CURVE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_USER, is_signer=False, is_writable=True),
            AccountMeta(pubkey=USER, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=RENT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FUN_PROGRAM, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("66063d1201daebea"))
        data.extend(struct.pack("<Q", amount))
        data.extend(struct.pack("<Q", max_sol_cost))

        swap_instruction = Instruction(PUMP_FUN_PROGRAM, bytes(data), keys)

        instructions = [
            set_compute_unit_limit(UNIT_BUDGET),
            set_compute_unit_price(UNIT_PRICE),
        ]
        if token_account_instruction:
            instructions.append(token_account_instruction)
        instructions.append(swap_instruction)

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

        print(f"Transaction Signature: {txn_sig}")

        return txn_sig

    except Exception as e:
        print(f"Error occurred during transaction: {e}")
        return False


# --------------------------------------------------------------------------------
# SELL (Async)
# --------------------------------------------------------------------------------
async def sell(
    priv_key: str,
    mint_str: str,
    token_balance: float,
    percentage: int,
    slippage: int,
    UNIT_BUDGET: int,
    UNIT_PRICE: int
):
    payer_keypair = Keypair.from_base58_string(priv_key)
    pub_key = payer_keypair.pubkey()
    print(priv_key, mint_str, token_balance, percentage, slippage, UNIT_BUDGET, UNIT_PRICE)

    try:
        if not (1 <= percentage <= 100):
            print("Percentage must be between 1 and 100.")
            return False

        coin_data = await get_coin_data(mint_str)
        if not coin_data:
            print("Failed to retrieve coin data.")
            return False

        if coin_data.complete:
            print("Warning: This token has bonded and is only tradable on Raydium.")
            return False

        MINT = coin_data.mint
        BONDING_CURVE = coin_data.bonding_curve
        ASSOCIATED_BONDING_CURVE = coin_data.associated_bonding_curve
        USER = payer_keypair.pubkey()
        ASSOCIATED_USER = get_associated_token_address(USER, MINT)

        # print("Retrieving token balance (async)...")
        # token_balance = await get_token_balance(mint_str, str(pub_key))
        # if token_balance is None or token_balance == 0:
        #     print("Token balance is zero. Nothing to sell.")
        #     return False
        print(f"Token Balance: {token_balance}")

        print("Calculating transaction amounts...")
        sol_dec = 1e9
        token_dec = 1e6
        amount = int(token_balance * token_dec)

        virtual_sol_reserves = coin_data.virtual_sol_reserves / sol_dec
        virtual_token_reserves = coin_data.virtual_token_reserves / token_dec
        sol_out = tokens_for_sol(token_balance, virtual_sol_reserves, virtual_token_reserves)

        slippage_adjustment = 1 - (slippage / 100)
        min_sol_output = int((sol_out * slippage_adjustment) * sol_dec)
        print(f"Amount: {amount}, Minimum Sol Out: {min_sol_output}")

        print("Creating swap instructions...")
        keys = [
            AccountMeta(pubkey=GLOBAL, is_signer=False, is_writable=False),
            AccountMeta(pubkey=FEE_RECIPIENT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=MINT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=BONDING_CURVE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_BONDING_CURVE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOCIATED_USER, is_signer=False, is_writable=True),
            AccountMeta(pubkey=USER, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),
            AccountMeta(pubkey=EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=PUMP_FUN_PROGRAM, is_signer=False, is_writable=False),
        ]

        data = bytearray()
        data.extend(bytes.fromhex("33e685a4017f83ad"))
        data.extend(struct.pack("<Q", amount))
        data.extend(struct.pack("<Q", min_sol_output))
        swap_instruction = Instruction(PUMP_FUN_PROGRAM, bytes(data), keys)

        instructions = [
            set_compute_unit_limit(UNIT_BUDGET),
            set_compute_unit_price(UNIT_PRICE),
            swap_instruction,
        ]

        if percentage == 100:
            print("Preparing to close token account after swap...")
            close_account_instruction = close_account(
                CloseAccountParams(
                    program_id=TOKEN_PROGRAM,
                    account=ASSOCIATED_USER,
                    dest=USER,
                    owner=USER
                )
            )
            instructions.append(close_account_instruction)

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
            opts=TxOpts(skip_preflight=False),
        )).value
        print(f"Transaction Signature: {txn_sig}")

        return txn_sig

    except Exception as e:
        print(f"Error occurred during transaction: {e}")
        return False


