import asyncio
import struct
from solana.rpc.types import TokenAccountOpts, TxOpts
from solders.pubkey import Pubkey  # type: ignore
from solders.keypair import Keypair  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore
from solders.message import MessageV0  # type: ignore
from solders.instruction import Instruction  # type: ignore
from spl.token.instructions import create_associated_token_account, get_associated_token_address
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price  # type: ignore
from solders.signature import Signature  # type: ignore
from solders.transaction import VersionedTransaction  # type: ignore
from solders.instruction import AccountMeta, Instruction  # type: ignore
import json
import requests
import aiohttp
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment

#moonshot
DEX_FEE = Pubkey.from_string("3udvfL24waJcLhskRAsStNMoNUvtyXdxrWQz4hgi953N")
HELIO_FEE = Pubkey.from_string("5K5RtTWzzLp4P8Npi84ocf7F1vBsAu29N1irG4iiUnzt")

ASSOC_TOKEN_ACC_PROG = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
CONFIG_ACCOUNT = Pubkey.from_string("36Eru7v11oU5Pfrojyn5oY3nETA1a1iqsw2WUu6afkM9")
MOONSHOT_PROGRAM = Pubkey.from_string("MoonCVVNZFSYkqNXP6bxHLPL6QQJiMagDL3qcqUQTrG")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

RPC = "https://mainnet.helius-rpc.com/?api-key=eba6d019-77f5-4715-9044-54eeeefeee23"
client = AsyncClient(RPC)

#moonshot

def get_token_data(token_address):
    url = f"https://api.moonshot.cc/token/v1/solana/{token_address}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        token_data = response.json() 
        
        return token_data
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return None
   

def derive_curve_accounts(mint: Pubkey):
    try:
        MOONSHOT_PROGRAM = Pubkey.from_string("MoonCVVNZFSYkqNXP6bxHLPL6QQJiMagDL3qcqUQTrG")
        SEED = "token".encode()

        curve_account, _ = Pubkey.find_program_address(
            [SEED, bytes(mint)],
            MOONSHOT_PROGRAM
        )

        curve_token_account = get_associated_token_address(curve_account, mint)
        return curve_account, curve_token_account
    except Exception:
        return None, None

async def get_token_account(owner: Pubkey, mint: Pubkey):
    try:
        account_data = await client.get_token_accounts_by_owner(
            owner, 
            opts=TokenAccountOpts(mint), 
            commitment=Commitment("processed")
        )
        
        if account_data.value:
            token_account = account_data.value[0].pubkey
            token_account_instructions = None
        else:
            raise Exception("No token account found")
    except Exception as e:
        print(f"Error while checking or creating associated token account: {e}")
        token_account = get_associated_token_address(owner, mint)
        token_account_instructions = create_associated_token_account(owner, owner, mint)
    
    return token_account, token_account_instructions

async def buy_moonshot_alone(priv_key: str, mint_str: str, sol_in: float, slippage, unit_price, unit_budget):
    payer_keypair = Keypair.from_base58_string(priv_key)
    slippage_bps = slippage*100
    try:
        token_data = get_token_data(mint_str)
        if token_data == None:
            return
 
        sol_decimal = 10**9
        token_decimal = 10**9
        token_price = token_data['priceNative']
        tokens_out = float(sol_in) / float(token_price)
        token_amount = int(tokens_out * token_decimal)
        collateral_amount = int(sol_in * sol_decimal)
        print(f"Collateral Amount: {collateral_amount}, Token Amount: {token_amount}, Slippage (bps): {slippage_bps}")

        SENDER = payer_keypair.pubkey()
        MINT = Pubkey.from_string(mint_str)
            
        # Get token account and token account instructions
        token_account, token_account_instructions = await get_token_account(SENDER, MINT)

        CURVE_ACCOUNT, CURVE_TOKEN_ACCOUNT = derive_curve_accounts(MINT)
        SENDER_TOKEN_ACCOUNT = token_account

        # Build account key list 
        keys = [
            AccountMeta(pubkey=SENDER, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SENDER_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=CURVE_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=CURVE_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=DEX_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=HELIO_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=MINT, is_signer=False, is_writable=True), 
            AccountMeta(pubkey=CONFIG_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False)
        ]


        # Construct the swap instruction
        data = bytearray()
        data.extend(bytes.fromhex("66063d1201daebea"))
        data.extend(struct.pack('<Q', token_amount))
        data.extend(struct.pack('<Q', collateral_amount))
        data.extend(struct.pack('<B', 0))
        data.extend(struct.pack('<Q', slippage_bps))
        data = bytes(data)
        swap_instruction = Instruction(MOONSHOT_PROGRAM, data, keys)

        instructions = []
        instructions.append(set_compute_unit_price(unit_price))
        instructions.append(set_compute_unit_limit(unit_budget))
        
        # Add token account creation instruction if needed
        if token_account_instructions:
            instructions.append(token_account_instructions)

        # Add the swap instruction
        instructions.append(swap_instruction)

        latest_blockhash = await client.get_latest_blockhash()

        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],
            latest_blockhash.value.blockhash,
        )

        transaction = VersionedTransaction(compiled_message, [payer_keypair])
        txn_sig = await client.send_transaction(transaction, opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed"))
        txn_sig = txn_sig.value
        print("Transaction Signature:", txn_sig)
        return txn_sig
    except Exception as e:
        print(e)
        return None

async def find_data(data, field):
    if isinstance(data, dict):
        if field in data:
            return data[field]
        else:
            for value in data.values():
                result = await find_data(value, field)
                if result is not None:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = await find_data(item, field)
            if result is not None:
                return result
    return None

async def get_token_balance(pub_key, token_address):
    try:
        headers = {
            "accept": "application/json",
            "content-type": "application/json"
        }
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "getTokenAccountsByOwner",
            "params": [
                pub_key,
                {"mint": token_address},
                {"commitment": "processed", "encoding": "jsonParsed"}
            ],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(RPC, json=payload, headers=headers) as response:
                response_json = await response.json()
                # print(f"Json: {response_json}")
                ui_amount = await find_data(response_json, "uiAmount")
                return float(ui_amount) if ui_amount is not None else 0.0
    except Exception as e:
        print(f"Error retrieving token balance for {pub_key}: {str(e)}")
        return 0.0

async def sell_moonshot(priv_key: str, mint_str: str, percentage, slippage, unit_price, unit_budget):
    payer_keypair = Keypair.from_base58_string(priv_key)
    pub_key = payer_keypair.pubkey()
    slippage_bps = slippage*100
    try:

        token_data = get_token_data(mint_str)
        if token_data == None:
            return
        
        token_balance = await get_token_balance(str(pub_key), mint_str)
        if token_balance is None or token_balance == 0:
            print("No token balance available to sell.")
            return False

        token_balance = token_balance * (percentage / 100)
        print(f"Selling {percentage}% of the token balance, adjusted balance: {token_balance}")



        sol_decimal = 10**9
        token_decimal = 10**9
        token_price = token_data['priceNative']
        token_value = float(token_balance) * float(token_price)
        collateral_amount = int(token_value * sol_decimal)
        token_amount = int(token_balance * token_decimal)
        print(f"Sell Collateral Amount: {collateral_amount}, Token Amount: {token_amount}, Slippage (bps): {slippage_bps}")
        
        MINT = Pubkey.from_string(mint_str)
        CURVE_ACCOUNT, CURVE_TOKEN_ACCOUNT = derive_curve_accounts(MINT)
        SENDER = payer_keypair.pubkey()
        SENDER_TOKEN_ACCOUNT = get_associated_token_address(SENDER, MINT)

        # Build account key list 
        keys = [
            AccountMeta(pubkey=SENDER, is_signer=True, is_writable=True),
            AccountMeta(pubkey=SENDER_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=CURVE_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=CURVE_TOKEN_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=DEX_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=HELIO_FEE, is_signer=False, is_writable=True),
            AccountMeta(pubkey=MINT, is_signer=False, is_writable=True), 
            AccountMeta(pubkey=CONFIG_ACCOUNT, is_signer=False, is_writable=True),
            AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=True),
            AccountMeta(pubkey=ASSOC_TOKEN_ACC_PROG, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False)
        ]

        # Construct the swap instruction
        data = bytearray()
        data.extend(bytes.fromhex("33e685a4017f83ad"))
        data.extend(struct.pack('<Q', token_amount))
        data.extend(struct.pack('<Q', collateral_amount))
        data.extend(struct.pack('<B', 0))
        data.extend(struct.pack('<Q', slippage_bps))
        data = bytes(data)
        swap_instruction = Instruction(MOONSHOT_PROGRAM, data, keys)
        
        
        instructions = []
        instructions.append(set_compute_unit_price(unit_price))
        instructions.append(set_compute_unit_limit(unit_budget))
        instructions.append(swap_instruction)


        # Compile message
        latest_blockhash = await client.get_latest_blockhash()
        compiled_message = MessageV0.try_compile(
            payer_keypair.pubkey(),
            instructions,
            [],  
            latest_blockhash.value.blockhash,
        )

        transaction = VersionedTransaction(compiled_message, [payer_keypair])
        txn_sig = await client.send_transaction(transaction, opts=TxOpts(skip_preflight=True, preflight_commitment="confirmed"))
        txn_sig = txn_sig.value
        print("Transaction Signature:", txn_sig)
        return txn_sig
    except Exception as e:
        print(e)
        return None