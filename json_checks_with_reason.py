# Code Author: Kayroze Shroff, Gaurav More
# Updated Date : 20-04-2026

import json
import glob
import os
import pandas as pd
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
import shutil
import re
import sys
import traceback
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def update_json_with_ifsc_info(json_data, csv_path="csv/IFSC.csv"):
    """
    Update JSON with IFSC/MICR information from CSV based on the specified logic
    """
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}, skipping IFSC/MICR lookup")
            return json_data
            
        # Read the CSV file
        df = pd.read_csv(csv_path, dtype={'MICR': str, 'IFSC': str})
        df = df.fillna('')
        
        # Get customer information
        customer_info = json_data.get("customerInformation", {})
        
        ifsc = customer_info.get("ifsCode", "")
        micr = customer_info.get("micr", "")
        branch_name = customer_info.get("branchName", "")
        bank_name = customer_info.get("bankName", "")
        
        # Print original values for debugging
        print(f"Original values - IFSC: {ifsc}, MICR: {micr}, Branch: {branch_name}, Bank: {bank_name}")
        
        matched_row = None
        match_found = False
        
        # Bank name mapping for logic 3
        bank_name_mapping = {
            'sbi': 'state bank of india',
            'osbi': 'state bank of india', 
            'statebank': 'state bank of india',
            'statebankofindia': 'state bank of india',
            'state bank': 'state bank of india',
            'st bank of india': 'state bank of india',
            's.b.i': 'state bank of india',
        }
        
        # Clean and validate values
        def is_valid_value(value):
            if not value or pd.isna(value):
                return False
            value_str = str(value).strip().lower()
            return value_str not in ['', 'null', 'not available', 'nan', 'none']
        
        # Logic 1: IFSC is available
        if is_valid_value(ifsc):
            ifsc_clean = str(ifsc).strip().upper()
            matched_rows = df[df['IFSC'].str.upper() == ifsc_clean]
            if not matched_rows.empty:
                matched_row = matched_rows.iloc[0]
                match_found = True
                print(f" Found match by IFSC: {ifsc_clean}")
        
        # Logic 2: IFSC not available but MICR is available
        if not match_found and is_valid_value(micr):
            micr_clean = str(micr).strip()
            matched_rows = df[df['MICR'] == micr_clean]
            if not matched_rows.empty:
                matched_row = matched_rows.iloc[0]
                match_found = True
                print(f" Found match by MICR: {micr_clean}")
        
        # Logic 3: Neither IFSC nor MICR available, but branch and bank names are
        if not match_found and is_valid_value(branch_name) and is_valid_value(bank_name):
            # Apply bank name mapping
            bank_name_lower = str(bank_name).lower()
            mapped_bank_name = bank_name
            for alias, standard in bank_name_mapping.items():
                if alias in bank_name_lower:
                    mapped_bank_name = standard
                    # print(f"Mapping bank name from '{bank_name}' to '{mapped_bank_name}'")
                    break
            
            branch_name_clean = str(branch_name).strip()
            
            # Try to find a match with both bank and branch name
            matched_rows = df[
                (df['BANK'].str.contains(mapped_bank_name, case=False, na=False)) & 
                (df['BRANCH'].str.contains(branch_name_clean, case=False, na=False))
            ]
            
            if not matched_rows.empty:
                matched_row = matched_rows.iloc[0]
                match_found = True
                print(f"Found match by bank and branch: {mapped_bank_name}, {branch_name_clean}")
        
        # If we found a match, update the JSON
        if match_found and matched_row is not None and not matched_row.empty:
            updates_made = []
            
            # Always update MICR with value from CSV
            if matched_row.get('MICR') and str(matched_row['MICR']).strip():
                micr_value = str(matched_row['MICR']).strip()
                # Remove .0 suffix if present
                if micr_value.endswith('.0'):
                    micr_value = micr_value[:-2]
                if micr_value != str(customer_info.get("micr", "")).strip():
                    customer_info["micr"] = micr_value
                    updates_made.append(f"MICR → {micr_value}")
                    print(f"Updating MICR to '{micr_value}'")
            
            # Always update IFSC with value from CSV
            if matched_row.get('IFSC') and str(matched_row['IFSC']).strip():
                new_ifsc = str(matched_row['IFSC']).strip().upper()
                if new_ifsc != str(customer_info.get("ifsCode", "")).strip().upper():
                    customer_info["ifsCode"] = new_ifsc
                    updates_made.append(f"IFSC → {new_ifsc}")
                    print(f"Updating IFSC to '{new_ifsc}'")
            
            # Always update branch name with value from CSV
            if matched_row.get('BRANCH') and str(matched_row['BRANCH']).strip():
                new_branch = str(matched_row['BRANCH']).strip()
                if new_branch != str(customer_info.get("branchName", "")).strip():
                    customer_info["branchName"] = new_branch
                    updates_made.append(f"Branch → {new_branch}")
                    print(f"Updating branch name to '{new_branch}'")
            
            # Always update bank name with value from CSV
            if matched_row.get('BANK') and str(matched_row['BANK']).strip():
                new_bank = str(matched_row['BANK']).strip()
                if new_bank != str(customer_info.get("bankName", "")).strip():
                    customer_info["bankName"] = new_bank
                    updates_made.append(f"Bank → {new_bank}")
                    print(f"Updating bank name to '{new_bank}'")
            
            # Extract branch code from IFSC if available
            if customer_info.get("ifsCode") and is_valid_value(customer_info["ifsCode"]):
                ifsc_code = customer_info["ifsCode"]
                # Extract branch code (last part after zeros)
                if len(ifsc_code) == 11:
                    # The first 4 characters are alphabets, the 5th is always zero
                    # The branch code is the last 6 characters with leading zeros removed
                    branch_code = ifsc_code[5:].lstrip('0')
                    current_branch_code = str(customer_info.get("branchCode", "")).strip()
                    if branch_code != current_branch_code:
                        customer_info["branchCode"] = branch_code
                        updates_made.append(f"Branch Code → {branch_code}")
                        print(f"Extracted branch code: {branch_code}")
            
            if updates_made:
                print(f"Updates applied: {', '.join(updates_made)}")
            else:
                print("No updates needed - values already correct")
                
        else:
            print("No matching IFSC/MICR/bank-branch found in CSV")
        
        return json_data
        
    except Exception as e:
        print(f"Error updating JSON with IFSC info: {e}")
        # Return original JSON if any error occurs
        return json_data

def date_missing_check(df):
    """
    Check if any transaction dates are missing.
    
    Args:
        df (DataFrame): Transactions DataFrame
        
    Returns:
        str: "Date(s) missing" if dates are missing, "Pass" otherwise
    """
    if df['xnsdate'].isna().any():
        return "Date(s) missing"
    return "Pass"

def credit_debit_balance(df):
    """
    Validate that calculated balances match the recorded balances.
    
    Args:
        df (DataFrame): Transactions DataFrame
        
    Returns:
        str: "Balance mismatch" if balances don't match, "Pass" otherwise
    """
    df2 = df.copy()

    # Handle None values by replacing with 0 for calculations
    df2['credit'] = df2['credit'].fillna(0)
    df2['debit'] = df2['debit'].fillna(0)
    df2['balance'] = df2['balance'].fillna(0)  # Fix: Handle None in balance column
    
    df2['balan'] = 0.0

    for i in range(len(df2)):
        if i == 0:
            df2.at[i, 'balan'] = df2.at[i, 'balance']
        else:
            prev_bal = df2.at[i - 1, 'balan']
            if df2.at[i, 'credit'] != 0:
                df2.at[i, 'balan'] = prev_bal + df2.at[i, 'credit']
            elif df2.at[i, 'debit'] != 0:
                df2.at[i, 'balan'] = prev_bal + df2.at[i, 'debit']
            else:
                df2.at[i, 'balan'] = prev_bal
    
    # Fix: Check if the last balance is None before rounding
    last_calculated = df2['balan'].iloc[-1]
    last_actual = df2['balance'].iloc[-1]
    
    # If either value is None, return balance mismatch
    if last_calculated is None or last_actual is None:
        return "Balance mismatch"
    
    # Now safely round and compare
    if round(last_calculated, 2) != round(last_actual, 2):
        return "Balance mismatch"
    return "Pass"

def empty_credit_debit_check(df):
    """
    Check if any transactions have both credit and debit fields empty.
    
    Args:
        df (DataFrame): Transactions DataFrame
        
    Returns:
        str: "Failed: empty credit debit Found" if empty fields found, "Pass" otherwise
    """
    if not ((df['credit'].isna()) & (df['debit'].isna())).any():
        return "Pass"
    return "Failed: empty credit debit Found"

def narration_missing_check(df):
    """
    Check if any transactions are missing narration.
    
    Args:
        df (DataFrame): Transactions DataFrame
        
    Returns:
        str: "Missing narration" if narration is missing, "Pass" otherwise
    """
    if df['narration'].isna().any():
        return "Missing narration"
    return "Pass"

def classify_row(prev, curr, next_):
    """
    Classify the order of dates in transactions.
    
    Args:
        prev: Previous date
        curr: Current date
        next_: Next date
        
    Returns:
        str: Classification of date order
    """
    dates = [d for d in [prev, curr, next_] if pd.notna(d)]
    if len(dates) < 2 or all(d == dates[0] for d in dates):
        return 'same'
    elif prev < curr < next_:
        return 'ascending'
    elif prev > curr:
        return 'descending'
    return 'mixed'

def date_order_check(df):
    """
    Check if transaction dates are in correct order.
    This is the original function that uses classify_row.
    
    Args:
        df (DataFrame): Transactions DataFrame
        
    Returns:
        str: "Dates are out of order" if dates are descending, "Pass" otherwise
    """
    df = df.dropna(subset=['xnsdate'])
    df['xnsdate'] = pd.to_datetime(df['xnsdate'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['xnsdate'])
    
    if len(df) < 2:
        return "Pass"  # Not enough transactions to check order
    
    df['prev'] = df['xnsdate'].shift(1)
    df['next'] = df['xnsdate'].shift(-1)
    df['order_status'] = df.apply(lambda row: classify_row(row['prev'], row['xnsdate'], row['next']), axis=1)
    if (df['order_status'] == 'descending').any():
        return "Dates are out of order"
    return "Pass"

def customer_information_check(df):
    """
    Validate customer information for completeness.
    
    Args:
        df (DataFrame): Customer information DataFrame
        
    Returns:
        str: Status message
    """
    msg = []
    if pd.isna(df['accountName'].iloc[0]) or df['accountName'].iloc[0] == "":
        msg.append("Missing account name")
    if pd.isna(df['ifsCode'].iloc[0]) or df['ifsCode'].iloc[0] == "":
        msg.append("Missing IFSC code")
    if pd.isna(df['statementPeriod'].iloc[0]) or df['statementPeriod'].iloc[0] == "":
        msg.append("Missing statement period")
    if msg:
        return "; ".join(msg)
    return "Pass"

def account_name_check(df):
    """
    Check if account name is not null and is text (not only numbers).
    
    Args:
        df (DataFrame): Customer information DataFrame
        
    Returns:
        str: "Account name invalid" if invalid, "Pass" otherwise
    """
    account_name = df['accountName'].iloc[0]
    if pd.isna(account_name) or account_name == "":
        return "Account name is null or empty"
    
    # Check if account name contains only numbers
    if isinstance(account_name, (int, float)) or (isinstance(account_name, str) and account_name.replace('.', '').isdigit()):
        return "Account name contains only numbers"
    
    return "Pass"

def account_number_check(df):
    """
    Check if account number is not null and length is between 9-17 digits.
    
    Args:
        df (DataFrame): Customer information DataFrame
        
    Returns:
        str: "Account number invalid" if invalid, "Pass" otherwise
    """
    account_number = df['accountNumber'].iloc[0]
    if pd.isna(account_number) or account_number == "":
        return "Account number is null or empty"
    
    # Convert to string and remove any non-digit characters
    acc_num_str = str(account_number).strip()
    # Remove any spaces or special characters keeps only X or x and digits
    acc_num_clean = re.sub(r'[^0-9xX]', '', acc_num_str)
    
    # Check length
    if len(acc_num_clean) < 9 or len(acc_num_clean) > 17:
        return f"Account number length invalid: {len(acc_num_clean)} digits (should be 9-17)"
    
    return "Pass"

def cif_mobile_swap_check(df):
    """
    Check and swap cifNumberCustomerId and mobileNumber if needed.
    
    Args:
        df (DataFrame): Customer information DataFrame
        
    Returns:
        tuple: (updated_df, swap_status, swap_performed)
    """
    cif = df['cifNumberCustomerId'].iloc[0] if 'cifNumberCustomerId' in df.columns else None
    mobile = df['mobileNumber'].iloc[0] if 'mobileNumber' in df.columns else None
    
    # Helper function to check if value is valid
    def is_valid_value(value):
        if value is None or pd.isna(value):
            return False
        value_str = str(value).strip().lower()
        return value_str not in ['', 'null', 'not available', 'nan', 'none']
    
    # Check if cif is null/empty and mobile is not null/empty
    if not is_valid_value(cif) and is_valid_value(mobile):
        # Store original values for logging
        original_cif = cif
        original_mobile = mobile
        
        # Swap the values
        df.at[df.index[0], 'cifNumberCustomerId'] = mobile
        df.at[df.index[0], 'mobileNumber'] = cif if is_valid_value(cif) else "Not Available"
        
        return df, f"CIF and mobile swapped (CIF: {original_cif} -> {mobile}, Mobile: {original_mobile} -> {cif if is_valid_value(cif) else 'Not Available'})", True
    
    return df, "Pass", False

def balance_check(df):
    """
    Check if openingBalance and closingBalance are not null.
    
    Args:
        df (DataFrame): Customer information DataFrame
        
    Returns:
        str: "Balance missing" if balances are missing, "Pass" otherwise
    """
    opening_balance = df['openingBalance'].iloc[0]
    closing_balance = df['closingBalance'].iloc[0]
    
    if pd.isna(opening_balance) or pd.isna(closing_balance):
        return "Opening or closing balance missing"
    
    return "Pass"

def balance_null_check(df):
    """
    Check if any transaction balance is null.
    
    Args:
        df (DataFrame): Transactions DataFrame
        
    Returns:
        str: "Null balance found" if any balance is null, "Pass" otherwise
    """
    if df['balance'].isna().any():
        return "Null balance found"
    return "Pass"

def statement_period_check(df):
    """
    Check if statementPeriod is in format "from date to date".
    
    Args:
        df (DataFrame): Customer information DataFrame
        
    Returns:
        str: "Statement period format invalid" if invalid, "Pass" otherwise
    """
    statement_period = df['statementPeriod'].iloc[0]
    if pd.isna(statement_period) or statement_period == "":
        return "Statement period is null or empty"
    
    # Check format using regex
    pattern = r'from\s+\d{1,2}/\d{1,2}/\d{4}\s+to\s+\d{1,2}/\d{1,2}/\d{4}'
    if not re.search(pattern, statement_period, re.IGNORECASE):
        return "Statement period format invalid"
    
    return "Pass"

def transaction_dates_not_null_check(df):
    """
    Check if all transaction dates are not null.
    
    Args:
        df (DataFrame): Transactions DataFrame
        
    Returns:
        str: "Transaction dates missing" if dates are missing, "Pass" otherwise
    """
    if df['xnsdate'].isna().any():
        return "Transaction dates missing"
    return "Pass"

def credit_debit_exclusivity_check(df):
    """
    Check that for each transaction, either credit or debit has value (not both, not none).
    
    Args:
        df (DataFrame): Transactions DataFrame
        
    Returns:
        str: "Credit/debit exclusivity issue" if issue found, "Pass" otherwise
    """
    for _, row in df.iterrows():
        credit_null = pd.isna(row['credit']) or row['credit'] == 0
        debit_null = pd.isna(row['debit']) or row['debit'] == 0
        
        # Both null or both have values
        if (credit_null and debit_null) or (not credit_null and not debit_null):
            return "Credit/debit exclusivity issue"
    
    return "Pass"

def pincode_check(df):
    """
    Check if customerAddress contains a valid 5-6 digit pincode or indicates "not available".
    
    Args:
        df (DataFrame): Customer information DataFrame
        
    Returns:
        str: "Pincode missing" if no valid pincode found and address doesn't indicate unavailability, "Pass" otherwise
    """
    customer_address = df['customerAddress'].iloc[0] if 'customerAddress' in df.columns else None
    
    if pd.isna(customer_address) or customer_address == "":
        return "Pincode missing - address is null or empty"
    
    # Convert to string and lowercase for consistent checking
    address_str = str(customer_address).lower()
    
    # Check for "not available" variations (case insensitive, spacing insensitive)
    not_available_patterns = [
        r'not\s*available',
        r'notavailable',
        r'n/a',
        r'na',
        r'not\s*avail',
        r'unavailable'
    ]
    
    for pattern in not_available_patterns:
        if re.search(pattern, address_str, re.IGNORECASE):
            # print(f"Address indicates pincode not available: '{customer_address}'")
            return "Pass"
    
    # Remove all non-digit characters (including spaces) to check digit sequences
    digits_only = re.sub(r'\D', '', address_str)
    
    # Look for 5 or 6 consecutive digits in the original string
    # This regex pattern looks for exactly 5 or 6 digits, with optional non-digit characters between them
    # but ensures the total digit count is exactly 5 or 6
    pincode_pattern_5 = r'(?<!\d)(\d\s*?\d\s*?\d\s*?\d\s*?\d)(?!\d)'  # 5 digits
    pincode_pattern_6 = r'(?<!\d)(\d\s*?\d\s*?\d\s*?\d\s*?\d\s*?\d)(?!\d)'  # 6 digits
    
    # Check for 6-digit pincode first
    match_6 = re.search(pincode_pattern_6, address_str)
    if match_6:
        # Extract the matched pincode and remove any non-digit characters to verify it's exactly 6 digits
        pincode_clean = re.sub(r'\D', '', match_6.group(1))
        if len(pincode_clean) == 6:
            # print(f"Found 6-digit pincode: {pincode_clean} in address")
            return "Pass"
    
    # Check for 5-digit pincode
    match_5 = re.search(pincode_pattern_5, address_str)
    if match_5:
        # Extract the matched pincode and remove any non-digit characters to verify it's exactly 5 digits
        pincode_clean = re.sub(r'\D', '', match_5.group(1))
        if len(pincode_clean) == 5:
            print(f"Found 5-digit pincode: {pincode_clean} in address")
            return "Pass"
    
    # Also check the digits-only string for 5 or 6 consecutive digits
    if len(digits_only) >= 5:
        # Look for sequences of exactly 5 or 6 digits within the digit string
        if re.search(r'^\d{5}$', digits_only) or re.search(r'^\d{6}$', digits_only):
            pincode = digits_only[:6] if len(digits_only) >= 6 else digits_only[:5]
            print(f"Found {len(pincode)}-digit pincode: {pincode} in address")
            return "Pass"
        # Check if there's a 5 or 6 digit sequence within longer digit strings
        match_5_in_long = re.search(r'(?<!\d)(\d{5})(?!\d)', digits_only)
        match_6_in_long = re.search(r'(?<!\d)(\d{6})(?!\d)', digits_only)
        
        if match_6_in_long:
            print(f"Found 6-digit pincode: {match_6_in_long.group(1)} in address")
            return "Pass"
        elif match_5_in_long:
            print(f"Found 5-digit pincode: {match_5_in_long.group(1)} in address")
            return "Pass"
    
    print(f"No valid 5/6-digit pincode found in address: {customer_address}")
    return "Pincode missing"

def ckyc_check(df):
    """
    Check if CKYC number is present in any customer information field.
    
    Args:
        df (DataFrame): Customer information DataFrame
        
    Returns:
        str: "CKYC found" if CKYC number is detected, "Pass" otherwise
    """
    # print("\nRunning CKYC check...")
    
    # CKYC keywords to search for
    ckyc_keywords = ['ckyc no', 'ckyc', 'ckyc no', 'ckyc']
    
    # Check all customer information fields for CKYC keywords
    for column in df.columns:
        value = df[column].iloc[0] if not df.empty else None
        
        if pd.isna(value) or value == "":
            continue
            
        value_str = str(value).lower()
        
        # Check for CKYC keywords
        for keyword in ckyc_keywords:
            if keyword in value_str:
                print(f"CKYC keyword '{keyword}' found in field '{column}': {value}")
                return "CKYC found"
    
    # If no keywords found, check for 14-digit patterns with x's and numbers
    for column in df.columns:
        value = df[column].iloc[0] if not df.empty else None
        
        if pd.isna(value) or value == "":
            continue
            
        value_str = str(value)
        
        # Look for 14-character sequences that might be CKYC numbers
        # Pattern 1: Exactly 14 digits
        pattern_14_digits = r'\b\d{14}\b'
        if re.search(pattern_14_digits, value_str):
            print(f"14-digit number found in field '{column}': {value}")
            return "CKYC found"
        
        # Pattern 2: Mixed x's and digits (like xxxxxxxxxx4567)
        # This matches sequences of exactly 14 characters containing only x/X and digits
        pattern_mixed = r'\b[xX\d]{14}\b'
        matches = re.findall(pattern_mixed, value_str)
        
        for match in matches:
            # Count x's and digits
            x_count = match.lower().count('x')
            digit_count = sum(c.isdigit() for c in match)
            
            # If it has many x's with few numbers (like the example xxxxxxxxxx4567)
            if x_count >= 10 and digit_count <= 4 and len(match) == 14:
                print(f"CKYC-like pattern found in field '{column}': {match}")
                return "CKYC found"
            
            # Also check if it's mostly digits with some x's
            if digit_count >= 10 and x_count <= 4 and len(match) == 14:
                print(f"CKYC-like pattern found in field '{column}': {match}")
                return "CKYC found"
    
    print("No CKYC detected")
    return "Pass"

        
def save_updated_json_with_updates(json_data, file_path):
    """
    Save updated JSON data to the specified path with proper formatting.
    
    Args:
        json_data (dict or str): Updated JSON data or string content
        file_path (Path): Output file path
        
    Returns:
        str: Path where file was saved
    """
    try:
        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(json_data, str):
            # If it's already a string (invalid JSON case), write it directly
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
        else:
            # If it's a dict, use json.dump
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
                
        # print(f"Updated JSON saved to: {file_path}")
        return str(file_path)
    except Exception as e:
        print(f"Error saving updated JSON: {e}")
        return None


def update_validation_csv(result, csv_path):
    """
    Append or create a CSV file with validation results.
    
    Args:
        result (dict): Result dictionary from process_single_json
        csv_path (str): Path to validation CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        result_df = pd.DataFrame([result])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        if os.path.exists(csv_path):
            # Append without header
            result_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # Write with header
            result_df.to_csv(csv_path, mode='w', header=True, index=False)
        return True
        
    except Exception as e:
        print(f"Error updating validation CSV: {e}")
        return False

def get_checks_config():
    """
    Function to define which checks should be enabled/disabled.
    This function must be explicitly defined and will be called by the main function.
    No default configuration - must be explicitly set by the user.
    
    Returns:
        dict: Configuration dictionary with check names as keys and boolean values
    """
    # USER MUST EXPLICITLY DEFINE THIS CONFIGURATION
    # Example configuration - user must modify this based on their needs
    checks_config = {
        'account_name_check': True,
        'account_number_check': True,
        'balance_check': True,
        'statement_period_check': True,
        'customer_info_check': False,
        'pincode_check': False,
        'ckyc_check': False,
        'date_missing_check': True,
        'credit_debit_balance_check': True,
        'narration_missing_check': True,
        'empty_credit_debit_check': True,
        'date_order_check': True,
        'transaction_dates_not_null': True,
        'credit_debit_exclusivity': True,
        'balance_null_check': True,
        'cif_mobile_swap': False
    }
    
    return checks_config


def get_exception_config():
    """
    Define which failed checks can be treated as exceptions (ignored)
    when credit_debit_balance_check passes.
    
    Returns:
        dict: Exception configuration with check names as keys and boolean values
    """
    exception_config = {
        'account_name_check': False,
        'account_number_check': False,
        'balance_check': False,
        'statement_period_check': False,
        'customer_info_check': False,
        'pincode_check': True,           
        'ckyc_check': False,
        'date_missing_check': True,
        'narration_missing_check': True,      
        'empty_credit_debit_check': False,
        'date_order_check': True,             
        'transaction_dates_not_null': True,
        'credit_debit_exclusivity': False,
        'balance_null_check': False,
        'cif_mobile_swap': False
    }
    return exception_config

def json_check_main(json_file_path, output_folder=None, ifsc_csv_path="IFSC/IFSC.csv", apply_ifsc_updates=False, log_to_csv=True):
    """
    MAIN FUNCTION: Process a single JSON file and determine its status.
    Always calls get_checks_config() to get the check configuration.
    
    Args:
        json_file_path (str): Path to the JSON file
        output_folder (str): Output folder path for storing processed JSON
        ifsc_csv_path (str): Path to IFSC CSV file
        apply_ifsc_updates (bool): Whether to apply IFSC/MICR updates (False by default)
        log_to_csv (bool): Whether to save logs to csv (True by default)
        
    Returns:
        tuple: (result_dict, updated_json_data, output_json_path)
    """
    # Always get checks configuration from the function
    checks_config = get_checks_config()
    
    json_file = Path(json_file_path)
    
    # Determine output path: overwrite original if output_folder is None
    if output_folder is None:
        output_path = json_file
    else:
        output_path = Path(output_folder) / json_file.name
        print(f"Output folder specified, processed JSON will be saved to: {output_path}")
    
    # Fixed validation csv path in json_check_log folder
    csv_path = Path("json_check_log") / "json_checks_result.csv"
    
    try:
        # Read the entire file content as text first
        with open(json_file, 'r', encoding='utf-8') as file:
            file_content = file.read().strip()
        
        # Check if file is empty or has no content
        if not file_content:
            print(f"File is empty: {json_file.name}")
            # Create minimal JSON with checks field
            minimal_json = {
                "checks": "File is empty",
                "status": "Failed",
                "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement.",
                'checks': 'File is empty'
            }
            
            result = {
                'json_file': json_file.name,
                'status': 'Failed',
                'reason': 'File is empty'
            }
            
            save_updated_json_with_updates(minimal_json, output_path)
            
            if log_to_csv:
                update_validation_csv(result, str(csv_path))          
            
            return result, minimal_json, output_path
        
        # Try to parse the JSON - if it fails, we'll use the original content as-is
        try:
            original_data = json.loads(file_content)
            
                       # Check if status is already Failed
            if 'status' in original_data and original_data['status'] == 'Failed':
                payload_check = original_data.get('data', {})
                customer_info_check = payload_check.get('customerInformation')
                transactions_check = payload_check.get('xnsTransactions')
                
                # Structural failure: missing or null critical data – keep as Failed.
                if customer_info_check is None or not isinstance(customer_info_check, dict) or \
                   transactions_check is None or not isinstance(transactions_check, list):
                    print(f"File already Failed : {json_file.name}")
                    original_data['checks'] = 'Already Failed'
                    original_data['exception'] = None
                    save_updated_json_with_updates(original_data, output_path)
                    if log_to_csv:
                        result_for_csv = {
                            'json_file': json_file.name,
                            'status': 'Failed',
                            'reason': 'Already Failed',
                            'exception': ''
                        }
                        update_validation_csv(result_for_csv, str(csv_path))
                    return original_data, original_data, output_path
                
                # Structurally valid – decide whether to re‑evaluate based on exception config
                exception_config = get_exception_config()
                re_evaluate = any(
                    checks_config.get(chk, False) and exception_config.get(chk, False)
                    for chk in exception_config
                )
                if not re_evaluate:
                    print(f"File already Failed and no exception‑eligible checks enabled, skipping: {json_file.name}")
                    original_data['checks'] = 'Already Failed'
                    original_data['exception'] = None
                    save_updated_json_with_updates(original_data, output_path)
                    if log_to_csv:
                        result_for_csv = {
                            'json_file': json_file.name,
                            'status': 'Failed',
                            'reason': 'Already Failed',
                            'exception': ''
                        }
                        update_validation_csv(result_for_csv, str(csv_path))
                    return original_data, original_data, output_path
                else:
                   # print(f"⚠️ File was Failed but exception logic enabled, re‑evaluating: {json_file.name}")
                   pass
            # Extract data payload
            payload = original_data.get('data', {})
            
            # Conditionally apply IFSC/MICR updates based on flag (False by default)
            if apply_ifsc_updates:
                print("\nApplying IFSC/MICR updates...")
                updated_payload = update_json_with_ifsc_info(payload, ifsc_csv_path)
            else:
                # print("\nSkipping IFSC/MICR updates (default behavior)")
                updated_payload = payload
            
            # Convert to DataFrames for CIF swap (based on configuration)
            customer_info_df = pd.json_normalize(updated_payload.get('customerInformation', {}))
            
            # Apply CIF-mobile swap if needed (based on configuration)
            swap_performed = False
            swap_message = "Pass"
            if checks_config.get('cif_mobile_swap', False):
                # print("\nChecking CIF-Mobile swap...")
                customer_info_df, swap_message, swap_performed = cif_mobile_swap_check(customer_info_df)
                
                # Update the payload with the swapped data
                if swap_performed:
                    updated_payload['customerInformation'] = customer_info_df.to_dict('records')[0]
                    print(f" 🔄 {swap_message}")
            else:
                # print("\nSkipping CIF-Mobile swap (disabled in config)")
                swap_message = "Disabled"
            
            # Check bank name and apply mapping - Handle None values
            bank_name_raw = updated_payload.get('customerInformation', {}).get('bankName')

            if bank_name_raw is None:
                print(" Bank name is None - proceeding with validation")
                bank_name = ""
            else:
                bank_name = str(bank_name_raw)
                print(f"Bank name: {bank_name_raw}")
            
            # Convert updated payload to DataFrames for validation checks
            # print("\nConverting to DataFrames for validation...")
            customer_info_df_for_checks = pd.json_normalize(updated_payload.get('customerInformation', {}))
            transactions_df = pd.json_normalize(updated_payload.get('xnsTransactions', []))
            
            # Run all validation checks on UPDATED data based on configuration
            # print("\nRunning validation checks on UPDATED data...")
            checks = {}
            
            # Customer information checks (only if enabled in config)
            if checks_config.get('account_name_check', False):
                checks['account_name_check'] = account_name_check(customer_info_df_for_checks)
            else:
                checks['account_name_check'] = "Disabled"
            
            if checks_config.get('account_number_check', False):
                checks['account_number_check'] = account_number_check(customer_info_df_for_checks)
            else:
                checks['account_number_check'] = "Disabled"
            
            if checks_config.get('balance_check', False):
                checks['balance_check'] = balance_check(customer_info_df_for_checks)
            else:
                checks['balance_check'] = "Disabled"
            
            if checks_config.get('statement_period_check', False):
                checks['statement_period_check'] = statement_period_check(customer_info_df_for_checks)
            else:
                checks['statement_period_check'] = "Disabled"
            
            if checks_config.get('customer_info_check', False):
                checks['customer_info_check'] = customer_information_check(customer_info_df_for_checks)
            else:
                checks['customer_info_check'] = "Disabled"
            
            if checks_config.get('pincode_check', False):
                checks['pincode_check'] = pincode_check(customer_info_df_for_checks)
            else:
                checks['pincode_check'] = "Disabled"
            
            if checks_config.get('ckyc_check', False):
                checks['ckyc_check'] = ckyc_check(customer_info_df_for_checks)
            else:
                checks['ckyc_check'] = "Disabled"
            
            # Transaction checks (only if there are transactions and check is enabled)
            if not transactions_df.empty:
                if checks_config.get('date_missing_check', False):
                    checks['date_missing_check'] = date_missing_check(transactions_df)
                else:
                    checks['date_missing_check'] = "Disabled"
                
                if checks_config.get('credit_debit_balance_check', False):
                    checks['credit_debit_balance_check'] = credit_debit_balance(transactions_df)
                else:
                    checks['credit_debit_balance_check'] = "Disabled"
                
                if checks_config.get('narration_missing_check', False):
                    checks['narration_missing_check'] = narration_missing_check(transactions_df)
                else:
                    checks['narration_missing_check'] = "Disabled"
                
                if checks_config.get('empty_credit_debit_check', False):
                    checks['empty_credit_debit_check'] = empty_credit_debit_check(transactions_df)
                else:
                    checks['empty_credit_debit_check'] = "Disabled"
                
                if checks_config.get('date_order_check', False):
                    checks['date_order_check'] = date_order_check(transactions_df)
                else:
                    checks['date_order_check'] = "Disabled"
                
                if checks_config.get('transaction_dates_not_null', False):
                    checks['transaction_dates_not_null'] = transaction_dates_not_null_check(transactions_df)
                else:
                    checks['transaction_dates_not_null'] = "Disabled"
                
                if checks_config.get('credit_debit_exclusivity', False):
                    checks['credit_debit_exclusivity'] = credit_debit_exclusivity_check(transactions_df)
                else:
                    checks['credit_debit_exclusivity'] = "Disabled"
                
                if checks_config.get('balance_null_check', False):
                    checks['balance_null_check'] = balance_null_check(transactions_df)
                else:
                    checks['balance_null_check'] = "Disabled"
            else:
                # If no transactions, mark transaction checks as "No transactions"
                transaction_checks = [
                    'date_missing_check', 'credit_debit_balance_check', 'narration_missing_check',
                    'empty_credit_debit_check', 'date_order_check', 'transaction_dates_not_null',
                    'credit_debit_exclusivity', 'balance_null_check'
                ]
                for check_name in transaction_checks:
                    checks[check_name] = "No transactions"
            
            # ========== EXCEPTION LOGIC (CORRECTED WITH DEBUG) ==========
            exception_applied = False
            exception_checks_list = []

            # Retrieve credit_debit_balance_check result
            cdb_result = checks.get('credit_debit_balance_check')
            cdb_enabled = checks_config.get('credit_debit_balance_check', False)
            
            # print(f"\n[DEBUG] credit_debit_balance_check result: {cdb_result}, enabled: {cdb_enabled}")

            # Only apply exception logic if credit_debit_balance_check passed AND is enabled
            if cdb_result == "Pass" and cdb_enabled:
                exception_config = get_exception_config()
                
                # Identify failed checks (exclude "Pass", "Disabled", "No transactions")
                failed_checks = [
                    chk for chk, res in checks.items()
                    if res not in ("Pass", "Disabled", "No transactions")
                ]
               # print(f"[DEBUG] Failed checks before exception: {failed_checks}")

                hard_failures = []
                for chk in failed_checks:
                    if exception_config.get(chk, False):
                        exception_checks_list.append(chk)
                    else:
                        hard_failures.append(chk)

               # print(f"[DEBUG] Exception‑eligible checks: {exception_checks_list}")
                #print(f"[DEBUG] Hard failures: {hard_failures}")

                # If no hard failures and at least one exception, mark as success
                if not hard_failures and exception_checks_list:
                    exception_applied = True
                   # print("[DEBUG] Exception applied – file will be marked Success")
                else:
                    #print("[DEBUG] Exception NOT applied (hard failures exist or no exceptions)")
                    pass
            else:
                # print("[DEBUG] Exception logic skipped – credit_debit_balance_check not Pass or not enabled")
                pass

            # Determine final status and reason
            if exception_applied:
                overall_status = 'Success'
                reason_value = None
            else:
                enabled_checks_passed = all(
                    value in ("Pass", "Disabled", "No transactions") for value in checks.values()
                )
                overall_status = 'Success' if enabled_checks_passed else 'Failed'
                reason_value = None if overall_status == 'Success' else (
                    'Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement.'
                )
            request_id = original_data.get('request_id', '')
            message = original_data.get('message', '')
            # Prepare result dictionary for csv log
            base_result = {
                'request_id': request_id,
                'message': message,
                'json_file': json_file.name,
                'cif_mobile_swap': swap_message,
                'ifsc_updates_applied': apply_ifsc_updates,
                'status': overall_status,
                'exception': ', '.join(exception_checks_list) if exception_applied else '',
                'reason': reason_value if overall_status == 'Failed' else ''
            }

            # Add individual check results
            for check_name, check_result in checks.items():
                base_result[check_name] = check_result

            result = base_result
            
            # Update the main JSON data with processed payload
            updated_data = original_data.copy()
            updated_data['data'] = updated_payload
            
                        # --- UPDATE JSON WITH CHECKS, STATUS, AND EXCEPTION ---
                        # Always compute the failed checks list (for both Failed status and Exception case)
            failed_checks_list = [
                k for k, v in checks.items()
                if v not in ("Pass", "Disabled", "No transactions")
            ]
            failed_checks_str = ", ".join(failed_checks_list)

            if exception_applied:
                # Keep original failed checks in 'checks', add 'exception' field
                updated_data['checks'] = failed_checks_str if failed_checks_str else "Success"
                #updated_data['exception'] = ', '.join(exception_checks_list)
                updated_data['alert'] = exception_checks_list   # <-- list directly
                updated_data.pop('exception', None)             # <-- remove old key if exists
                updated_data['status'] = 'Success'
                updated_data['reason'] = None
                print(f"\nFile passed with exceptions: {', '.join(exception_checks_list)}")
                print(f"   (Failed checks recorded in 'checks': {failed_checks_str})")
            else:
                # Always include exception field, set to None when not applied
                updated_data['alert'] = None
                if overall_status == 'Success':
                    updated_data['checks'] = "Success"
                    updated_data['status'] = 'Success'
                    updated_data['reason'] = None
                    print(f"\nAll enabled checks passed for {json_file.name}")
                else:
                    updated_data['checks'] = failed_checks_str
                    updated_data['status'] = 'Failed'
                    updated_data['reason'] = reason_value
                    print(f"\n Some checks failed for {json_file.name}")
                    print(f"Failed checks: {failed_checks_str}")

            if swap_performed and not exception_applied:
                print("(CIF-mobile swap was performed)")

            # Save to output path
            save_updated_json_with_updates(updated_data, output_path)
            
            if log_to_csv:
               update_validation_csv(result, str(csv_path))
            
            return result, updated_data, output_path
            
        except json.JSONDecodeError as e:
            print(f"JSON structure is invalid in {json_file.name}")
            print(f"Error: {str(e)}")
            
            # Fix the comma placement issue and add checks, status, reason
            modified_content = file_content.rstrip()
            
            # Check if the last character is a closing brace
            if modified_content.endswith('}'):
                # Remove the last closing brace
                content_without_final_brace = modified_content[:-1].rstrip()
                
                # Check if the last character before the brace is a comma
                if content_without_final_brace.endswith(','):
                    # If there's already a comma, just add the fields and closing brace
                    modified_content = content_without_final_brace + '\n    "checks": "Re-review",\n    "status": "Failed",\n    "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement."\n}'
                else:
                    # If no comma, add comma, then fields, then closing brace
                    modified_content = content_without_final_brace + ',\n    "checks": "Re-review",\n    "status": "Failed",\n    "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement."\n}'
            else:
                # If no closing brace, just append the fields and closing brace
                modified_content = modified_content + ',\n    "checks": "Re-review",\n    "status": "Failed",\n    "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement."\n}'
            
            # Parse the modified content to ensure it's valid JSON for saving
            try:
                # Try to parse to ensure we have valid JSON
                parsed_data = json.loads(modified_content)
                output_data = parsed_data
            except:
                # If still invalid, just use the modified content as string and let json.dump handle it
                output_data = modified_content
            
            result = {
                'json_file': json_file.name,
                'status': 'Re-review',
                'reason': f'Invalid JSON structure: {str(e)}',
                'ifsc_updates_applied': apply_ifsc_updates
            }
            
            save_updated_json_with_updates(output_data, output_path)
            
            if log_to_csv:
                update_validation_csv(result, str(csv_path))
            
            return result, output_data, output_path
            
    except Exception as e:
        print(f"Error processing {json_file.name}: {str(e)}")
        traceback.print_exc()
        
        # For any other errors, try to read the file and add checks field
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                file_content = file.read().strip()
            
            # Check if file is empty
            if not file_content:
                minimal_json = {
                    "checks": "Re-review",
                    "status": "Failed",
                    "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement."
                }
                result = {
                    'json_file': json_file.name,
                    'status': 'Re-review',
                    'reason': f'File is empty - processing error: {str(e)}',
                    'ifsc_updates_applied': apply_ifsc_updates
                }
                save_updated_json_with_updates(minimal_json, output_path)
                
                if log_to_csv:
                    update_validation_csv(result, str(csv_path))
                
                return result, minimal_json, output_path
            
            # Fix the comma placement issue and add checks, status, reason
            modified_content = file_content.rstrip()
            
            # Check if the last character is a closing brace
            if modified_content.endswith('}'):
                # Remove the last closing brace
                content_without_final_brace = modified_content[:-1].rstrip()
                
                # Check if the last character before the brace is a comma
                if content_without_final_brace.endswith(','):
                    # If there's already a comma, just add the fields and closing brace
                    modified_content = content_without_final_brace + '\n    "checks": "Re-review",\n    "status": "Failed",\n    "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement."\n}'
                else:
                    # If no comma, add comma, then fields, then closing brace
                    modified_content = content_without_final_brace + ',\n    "checks": "Re-review",\n    "status": "Failed",\n    "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement."\n}'
            else:
                # If no closing brace, just append the fields and closing brace
                modified_content = modified_content + ',\n    "checks": "Re-review",\n    "status": "Failed",\n    "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement."\n}'
            
            # Try to parse, otherwise use as string
            try:
                output_data = json.loads(modified_content)
            except:
                output_data = modified_content
            
            result = {
                'json_file': json_file.name,
                'status': 'Re-review',
                'reason': f'Processing error: {str(e)}',
                'ifsc_updates_applied': apply_ifsc_updates
            }
            
            save_updated_json_with_updates(output_data, output_path)
            
            if log_to_csv:
                update_validation_csv(result, str(csv_path))
            
            return result, output_data, output_path
            
        except Exception as file_error:
            print(f"Error reading file: {file_error}")
            # Absolute last resort - create minimal JSON with curly brackets
            minimal_json = {
                "checks": "Re-review",
                "status": "Failed",
                "reason": "Document parsing failed. The file may be incomplete, rotated, poor quality, or not a valid bank statement."
            }
            result = {
                'json_file': json_file.name,
                'status': 'Re-review',
                'reason': f'File processing error: {str(file_error)}',
                'ifsc_updates_applied': apply_ifsc_updates
            }
            save_updated_json_with_updates(minimal_json, output_path)
            
            if log_to_csv:
                update_validation_csv(result, str(csv_path))
                
            return result, minimal_json, output_path

# if __name__ == "__main__":
#     folder_path = r"./output"
    
#     # 1. Get the list of files first so tqdm knows the total count
#     json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
#     print(f"Found {len(json_files)} files to process.")

#     # 2. Wrap the list in tqdm() to create the progress bar
#     for file_path in tqdm(json_files, desc="Processing Files", unit="file"):
#         try:
#             json_check_main(file_path)
#         except Exception as e:
#             # tqdm.write allows printing without breaking the progress bar layout
#             tqdm.write(f"Error processing {os.path.basename(file_path)}: {e}")    