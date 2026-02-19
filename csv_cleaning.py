import pandas as pd
import re
from fuzzywuzzy import fuzz, process
import numpy as np
from difflib import SequenceMatcher
from date_cleaning import process_all_dates  
import os
from datetime import datetime
from rapidfuzz import fuzz as rfuzz

# Create check_date directory at module load
CHECK_DATE_DIR = "check_date"
if not os.path.exists(CHECK_DATE_DIR):
	os.makedirs(CHECK_DATE_DIR)
	print(f"Created log directory: {CHECK_DATE_DIR}")

HEADER_REGEX = [
	r'\bcr(edit)?\b',
	r'\bdr(ebit)?\b',
	r'\bdebits?\b',
	r'\bcredits?\b',r'\bdeposits?\b',r'\bwithdrawals?\b',
	r'\bamt|amount|value\b',r'\bpayment\s*n\s*a\s*r\s*r\s*a\s*t\s*i\s*o\s*n\b',r'\b(?:txn|tran|transaction)\s*d\s*a\s*t\s*e\s*(?:&|and)\s*t\s*i\s*m\s*e\b',
	r'\btype\b',r'\bcredit\s*amount\b',r'\bdebit\s*amount\b',
	r'\btxn|transaction\b',r'\btran\s*date\b',
	r'\b(narration|naration|description|escription|remark|remarks|particulars?|details?)\b',
	r'\bdate|post\s*date|value\s*date\b',r'\btran(?:saction)?\s*date\b',r'\brunning\s*bal(?:ance)?\b',
	r'\bcheque|chq|ref(erence)?\b',r'\bbook\s*bal(?:ance)?\b',r'\bdat\s*value\b',
	r'\bdate\s*(?:&|and)\s*time\b',r'\btxn\s*date\s*(?:&|and)\s*time\b',r'\btransaction\s*details\s*comment.*payment\s*method\b',
	r'\bdate\s*day\s*/\s*night\b',r'\btransaction\s*date\b',r'\btransaction\s*remarks\b',r'\bdeposit\s*amt.*inr\b',r'\bwithdrawal\s*amt\s*\(?inr\)?\b'


]


# FUZZY + REGEX MATCHER
def fuzzy_regex_match(text, patterns, threshold=0.75):
	"""
	Returns True if text matches regex OR is fuzzily similar
	"""
	text = text.lower().strip()

	for pat in patterns:
		# Direct regex match
		if re.search(pat, text, flags=re.I):
			# print(re.search(pat, text, flags=re.I))
			return True

		# Fuzzy similarity fallback
		core = re.sub(r'[\\b\\(\\)\\?\\|]', '', pat)
		ratio = SequenceMatcher(None, text, core).ratio()
		if ratio >= threshold:
			return True

	return False


# HEADER ROW DETECTOR
def detect_header_row(df_raw):
	"""
	Detects header row index usin fuzzy + regex logic.
	Returns row index or None.
	"""
	for i, row in df_raw.iterrows():

		# Normalize row values
		row_values = [
			str(cell).strip().lower()
			for cell in row
			if str(cell).strip() not in ["", "nan", "none"]
		]

		# Skip very small rows
		if len(row_values) < 3:
			continue

		# Remove numeric-only cells
		non_numeric = [
			v for v in row_values
			if not re.fullmatch(r'-?\d+(\.\d+)?', v.replace(',', ''))
		]

		# Must have enough non-numeric cells
		if len(non_numeric) < len(row) // 2:
			continue

		# Count fuzzy header matches
		header_hits = sum(
			fuzzy_regex_match(val, HEADER_REGEX)
			for val in non_numeric
		)
		# print("Header Hits")
		# print(header_hits)

		# Header confidence threshold
		confidence = header_hits / max(len(non_numeric), 1)

		if header_hits >= 2 and confidence >= 0.4:
			return i

	return None


def is_partial_row(row):
	"""
	Check if a row is a partial row (only contains narration/description)
	"""
	

	has_amount = any(
		any(k in col.lower() for k in ['credit', 'debit', 'amount', 'balance'])
		and pd.notnull(row[col]) and str(row[col]).strip()
		for col in row.index
	)

	has_narration = any(
		any(k in col.lower() for k in ['narration', 'description', 'details'])
		and pd.notnull(row[col]) and str(row[col]).strip()
		for col in row.index
	)

	return  not has_amount and has_narration


def remove_duplicate_column(df):
	"""
	Remove duplicate columns from dataframe
	"""
	df = df.loc[:, ~df.columns.duplicated()]
	normalized_headers = [re.sub(r'\s+', '', str(col)).strip().lower() for col in df.columns]

	def row_has_any_header_value(row):
		for cell in row:
			norm_cell = re.sub(r'\s+', '', str(cell)).strip().lower()
			for header in normalized_headers:
				if header == norm_cell:
					return True
		return False

	df = df[~df.apply(row_has_any_header_value, axis=1)]
	# df.to_csv("debug_after_remove_duplicate.csv", index=False) # Debugging output
	return df


def extract_amount(value):
	if pd.isna(value):
		return ""

	text = str(value).lower().strip()
	
	is_negative = '-' in text


	# remove cr / dr
	text = re.sub(r'(cr|dr)', '', text, flags=re.I)

	# remove commas and spaces
	text = text.replace(',', '').replace(' ', '')

	nums = re.findall(r'[\d.]+', text)
	if not nums:
		return ""

	raw = nums[0]
	parts = raw.split('.')

	# multiple dot handling
	if len(parts) > 2:
		# 4.54.554 -> 454554
		if len(parts[-1]) == 3:
			raw = ''.join(parts)
		# 5.328.28 -> 5328.28
		else:
			raw = ''.join(parts[:-1]) + '.' + parts[-1]
			
	#2.444467 -> 24444.67
	elif len(parts) == 2 and len(parts[1]) > 2:
		digits = parts[0] + parts[1]
		raw = digits[:-2] + '.' + digits[-2:]

	try:
		amount = float(raw)
		return -amount if is_negative else amount

	except:
		return ""
	

def parse_balance(value):
	if pd.isna(value):
		return ""

	text = str(value).strip().upper()

	if text in ["", "NAN", "NONE"]:
		return ""

	# Detect DR / CR
	is_dr = "DR" in text
	is_cr = "CR" in text

	# Remove DR / CR only
	text = re.sub(r'(CR|DR)', '', text, flags=re.I)

	# Remove commas and spaces
	text = text.replace(',', '').replace(' ', '')

	#  IMPORTANT: Keep minus sign
	nums = re.findall(r'-?[\d.]+', text)
	if not nums:
		return ""

	raw = nums[0]
	parts = raw.split('.')

	# Multiple dot handling
	if len(parts) > 2:
		if len(parts[-1]) == 3:
			raw = ''.join(parts)
		else:
			raw = ''.join(parts[:-1]) + '.' + parts[-1]

	elif len(parts) == 2 and len(parts[1]) > 2:
		digits = parts[0] + parts[1]
		raw = digits[:-2] + '.' + digits[-2:]

	try:
		amount = float(raw)

		# If already negative, keep it
		if amount < 0:
			return amount

		# Apply DR logic only if positive
		if is_dr:
			return -amount
		else:
			return amount

	except:
		return ""


def extract_amount_new(value):
	"""
	NEW FUNCTION: Extract amount with OCR error handling.
	NEW FIXED LOGIC:
	1. Clean the input string (remove non-numeric except dots, replace colons/semicolons with dots)
	2. Extract only digits (0-9) for counting after decimal
	3. If 1 digit → remove ALL dots, divide by 10
	4. If 2+ digits → remove ALL dots, divide by 100
	5. No dots → take as is
	"""
	if pd.isna(value) or value == "" or str(value).strip() == "":
		return np.nan
	
	# Convert to string and clean
	text = str(value).strip()
	
	# Handle negative
	is_negative = False
	if text.startswith('-') or '(' in text:
		is_negative = True
		text = re.sub(r'[-()]', '', text)
	
	# Remove CR/DR - but preserve for counting digits after decimal
	# First extract the numeric part only for digit counting
	numeric_part = re.sub(r'[^0-9.]', '', text)
	
	# Keep original text for full processing
	clean_text = text
	
	# Remove CR/DR from the text to process
	clean_text = re.sub(r'\b(cr|dr)\b', '', clean_text, flags=re.IGNORECASE)
	
	# Convert colons/semicolons to dots
	clean_text = re.sub(r'[:;]', '.', clean_text)
	
	# Remove commas and spaces
	clean_text = clean_text.replace(',', '').replace(' ', '')
	
	# Keep only digits and dots
	clean_text = re.sub(r'[^\d.]', '', clean_text)
	
	if not clean_text:
		return np.nan
	
	# Check for dots in the NUMERIC part (not in the text with CR/DR)
	if '.' in numeric_part:
		# Get the part after the last dot in the NUMERIC part
		last_dot_index = numeric_part.rfind('.')
		after_last_dot = numeric_part[last_dot_index + 1:]
		
		# Count only DIGITS (0-9) after last dot
		digits_after_last_dot = len(re.sub(r'[^0-9]', '', after_last_dot))
		
		# Remove ALL dots from clean_text for calculation
		digits_only = clean_text.replace('.', '')
		
		try:
			if digits_only:  # Check if not empty
				if digits_after_last_dot == 1:
					# One digit after last dot → divide by 10
					result = float(digits_only) / 10.0
				elif digits_after_last_dot >= 2:
					# Two or more digits after last dot → divide by 100
					result = float(digits_only) / 100.0
				else:
					# No digits after dot (edge case)
					result = float(digits_only)
				
				result = round(result, 2)
				return -result if is_negative else result
			else:
				return np.nan
		except:
			return np.nan
	else:
		# No dots - take as is
		try:
			if clean_text:  # Check if not empty
				result = float(clean_text)
				result = round(result, 2)
				return -result if is_negative else result
			else:
				return np.nan
		except:
			return np.nan


def clean_debit_credit(df):
	"""
	Decides whether to apply DR/CR-based split or single-column split.
	Ensures only one strategy is used to prevent duplication.
	"""
	# Lowercase column names for flexible matching
	# cols_lower = [col.lower() for col in df.columns]


	has_drcr = any(
		re.fullmatch(r'(?:dr[/|]cr|cr[/|]dr|dricr|dr_cr|drcr|dr\.|cr\.|cr/dr|Debit[/|]Credit|Credit[/|]Debit|Debit\s*/\s*Credit|Credit\s*/\s*Debit)', col, flags=re.IGNORECASE)
		for col in df.columns
	) or any(col.lower() in ['type', 'txn type', 'transaction type', 'cr/dr'] for col in df.columns)



	has_mixed_amount = any(
		re.search(r'withdrawal\s*\(?\s*dr\s*\)?\s*[/|\\-]\s*deposit\s*\(?\s*cr\s*\)?|debit.*credit|dr.*cr|amount', col, re.IGNORECASE) for col in df.columns
	)

	if has_drcr:
		df = parse_debit_credit_split_safe(df)
	elif has_mixed_amount:
		df = split_drcr_from_amount_column(df)
		pass
	
	return df


def split_drcr_from_amount_column(df):
	"""
	Handles columns like:
	Withdrawal(Dr)/ Deposit(Cr)
	"""

	if "Debits" in df.columns or "Credits" in df.columns:
		return df

	# Detect unified amount column
	amount_col = next(
		(c for c in df.columns if re.search(r'withdrawal\s*\(dr\)\s*/\s*deposit\s*\(cr\)|amount|amt', c, re.I)),
		None
	)
	if not amount_col:
		return df

	def split_val(x):
		if pd.isna(x):
			return "", ""

		text = str(x)

		amt = extract_amount(text)

		# STRICT detection – no guessing
		if re.search(r'\(\s*dr\s*\)|\bdr\b', text, re.I):
			return amt, ""
		elif re.search(r'\(\s*cr\s*\)|\bcr\b', text, re.I):
			return "", amt
		else:
			# neither DR nor CR → leave blank
			return "", ""

	df[['Debits', 'Credits']] = df[amount_col].apply(
		lambda x: pd.Series(split_val(x))
	)

	df['Debits'] = pd.to_numeric(df['Debits'], errors='coerce')
	df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce')
 
	# df.to_csv("debug_after_split_drcr_amount.csv", index=False) # Debugging output

	return df


def normalize_drcr_value(val):
	val = str(val).strip().upper()
	val = val.replace('0R', 'CR')  # Fix misread "0R" instead of "CR"
	val = val.replace('Dr', 'DR').replace('Cr', 'CR')
	val = re.sub(r'\s+', '', val)  # Remove all whitespace
	val = str(val).strip().lower()
	mapping = {
			'credit': 'CR', 'cr': 'CR', 'cr.': 'CR', 'c': 'CR','(Cr)': 'CR','(Cr': 'CR', 'Cr)':'CR',
			'debit': 'DR', 'dr': 'DR', 'dr.': 'DR', 'd': 'DR','(Dr)': 'Dr', '(Dr': 'DR', 'Cr)':'DR'
	}
	return mapping.get(val, '')


def parse_debit_credit_split_safe(df):
	"""
	Split debit/credit ONLY if DR/CR or Type column exists.
	Clean amount using extract_amount().
	"""

	# Detect DR/CR columns
	drcr_cols = [
	col for col in df.columns
		if re.fullmatch(
			r'(?:'
			r'DR[/|]CR|CR[/|]DR|'
			r'DR[_]?CR|DRCR|'
			r'DR\.?|CR\.?|'
			r'Debit[/|]Credit|Credit[/|]Debit|'
			r'Debit\s*/\s*Credit|Credit\s*/\s*Debit'
			r')',
			col.strip(),
			flags=re.IGNORECASE
		)
	]


	# Detect Type columns
	type_cols = [
		col for col in df.columns
		if col.strip().lower() in [
			'type', 'txn type', 'transaction type', 'cr/dr'
		]
	]


	# If neither exists → DO NOTHING
	if not drcr_cols and not type_cols:
		return df

	# Detect Amount column
	amount_patterns = [
		r'amount', r'amt', r'transaction.*amount',
		r'amount\s*\(.*\)'
	]

	amount_col = None
	for col in df.columns:
		if any(re.search(pat, col.lower()) for pat in amount_patterns):
			amount_col = col
			break

	if not amount_col:
		return df  # No amount column → exit

	# Standardize DR/CR column
	if drcr_cols:
		df = df.rename(columns={drcr_cols[0]: 'DR/CR'})
	else:
		df = df.rename(columns={type_cols[0]: 'DR/CR'})

	df['DR/CR'] = df['DR/CR'].apply(normalize_drcr_value)

	# Clean amount using extract_amount
	df['Amount'] = df[amount_col].apply(extract_amount)

	# Split debit / credit
	df['Debits'] = df.apply(
		lambda r: r['Amount'] if r['DR/CR'] == 'DR' else "",
		axis=1
	)

	df['Credits'] = df.apply(
		lambda r: r['Amount'] if r['DR/CR'] == 'CR' else "",
		axis=1
	)
	
	

	# Ensure numeric
	df['Debits'] = pd.to_numeric(df['Debits'], errors='coerce')
	df['Credits'] = pd.to_numeric(df['Credits'], errors='coerce')
	

	return df
# def normalize_headers(df):
# 	"""
# 	Normalize column headers to standard names
# 	"""
# 	headers = {
# 		"XN Date": {"TxnPostedDate","Date(ValueDate)","Tran Date","Date Day/Night","TransactionDate","TxnDate& Time","DAT VALUE", "Date(Value Date","Date& Time", "Post Date", "PostDate","Txn Posted Date", "TRANSACTION DATE", "Tran Date", "TranDate","XN Date", "Date", "Transaction date", "Txn Date", "Post date","ate", "DATE", "Transaction Date", "TRAN DATE", "TRANDATE","Transactio n Date"},
# 		"Value Date": {"Value Date", "VAL DATE", "Val Date", "VALUE DT","ValueDate", "VALDATE"},
# 		"Cheque No": {"Cheque/Refer enceNo","Cheque.No./Ref.No", "Cheq No ue", "CHQ/REFNO.","CHEQUE/REFERENCE#", "ChequeNo.", "Chq.No", "Cheque No","Chq./ref.no", "Ref No", "Cheque number", "Ref no./cheque no.","Chq.no", "Chq No", "CHQ.NO.", "CHQ NO", "Cheque No.","Cheque Number", "Chq./Ref.No", "Chq.No."," Ref No./Cheque No.", "CHQ.NO", "Cnq.No.","Chq/Ref number", "Chq/Ref No"},
# 		"Narration": {"RANSACTIONDETAILS","TransactionRemarks","TransactionDetails CommentÂ·PlaceÂ·PaymentMethod","TransactionDescription","Transaction Description", "TRANSACTIONDETAILS", "Narration","Description", "Details", "Remarks", "Particulars","Transaction Particulars", "Partculars","TRANSACTION DETAILS", "DETAILS", "NARRATION","PARTICULARS", "Transaction Remarks","PARTICULARS CHO.NO.", "Transactio nRemarks"},
# 		"Credits": {"CR","Cr Amount","Credl","CreditAmount","Deposits (in Rs.)","DepositAmtï¼ˆINR)", "Deposits (INR)", "CREDIT()","Credit","Deposits (INR)", "Cr Amt", "Deposit amt."," Credit(INR)", "CREDIT", "DEPOSIT(CR)", "DEPOSITS","Deposit Amt.", "Deposits", "Credit Amount"," Deposit Amount(INR)", "DEPOSIT (CR)"},
# 		"Debits": {"DR","Dr Amount","Debit Amount", "DebitAmount","DEBIT(R)","WithdrawalAmt(INR)", "Withdrawal (Dr)","Debit","Withdrawal(INR)", "Dr Amt", "Withdrawalamt"," Debit(INR)", "DEBIT", " WITHDRAWAL(DR)", "WITHDRAWALS", "Withdrawal Amt.", "Withdrawals", "WITHDRAWAL (DR)","Witndrawals"},
# 		"Balance": {"BALANCE()","BOOKBAL", "BALANCER","RunningBalance", "Closing balance","Available balance", "Balance (Rs.)", "Balance"," Balance(INR)", "BALANCE", "Closing Balance"," Available Balance(INR)", "BALANCE(INR)", "Balance(IN R)", "Balance (INR)", "Available Balance(INR", "NetBalance"}
# 	}


# 	#problem in this parts of code
# 	HEADER_REGEX = {
# 		"XN Date": [
# 			r'\btxn\s*d-ate\b',
# 			r'\btran\s*date\b',
# 			r'\bpost\s*date\b',
# 			r'\btransaction\s*date\b',
# 			r'\bdate\s*value\s*date\b',
# 			r'\bdate\s*(?:&|and)\s*time\b',r'\bdat\s*value\b',r'\btransaction\s*date\b',r'\btxn\s*date\s*(?:&|and)\s*time\b',r'\bdate\s*day\s*/\s*night\b',



# 		],
# 		"Value Date": [
# 			r'\bvalue\s*date\b',
# 			r'\bval\s*date\b'
# 		],
# 		"Cheque No": [r'\bcheq\b', r'\bchq\b', r'\bref\b'],
# 		"Narration": [r'\bnarr\b',r'\btransaction\s*remarks\b',r'\btransaction\s*description\b', r'\bparticulars?\b', r'\bremarks?\b', r'\bdetails?\b', r'\bdescription\b'],
# 		"Credits": [r'\bcredit\b', r'\bdeposits?\b',r'\bdeposit\b',r'\bcredit\s*amount\b'],
# 		"Debits": [r'\bdebit\b', r'\bwithdrawals?\b', r'\bwithdraw\b',r'\bdebit\s*amount\b',r'\bwithdrawal\s*amt\s*\(?inr\)?\b'],
# 		"Balance": [r'\bbalance\b',r'\bbalance\s*\(inr\)\b',r'\bclosing\b', r'\btransaction\s*details\s*comment.*payment\s*method\b',
# 					r'\bavailable\b',r'\bbook\s*bal(?:ance)?\b',r'\brunning\s*bal(?:ance)?\b']
# 	}

# 	reverse_mapping = {
# 		variant.lower(): std
# 		for std, variants in headers.items()
# 		for variant in variants
# 	}
# 	all_possible_headers = list(reverse_mapping.keys())

# 	normalized_cols = []

# 	for col in df.columns:

# 		clean_col = str(col).strip().lower()
# 		clean_col = re.sub(r'[^a-z0-9 ]', ' ', clean_col)
# 		clean_col = re.sub(r'\s+', ' ', clean_col)

# 		mapped = None

# 		# ----------  HEADER DICT FIRST (ADDED) ----------
# 		for std, variants in headers.items():
# 			for v in variants:
# 				v_clean = v.lower()
# 				v_clean = re.sub(r'[^a-z0-9 ]', ' ', v_clean)
# 				v_clean = re.sub(r'\s+', ' ', v_clean)

# 				if clean_col == v_clean:
# 					mapped = std
# 					break
# 			if mapped:
# 				break

# 		# ----------  REGEX SECOND (UNCHANGED) ----------
# 		if not mapped:
# 			for std, patterns in HEADER_REGEX.items():
# 				for pat in patterns:
# 					if re.search(pat, clean_col):
# 						mapped = std
# 						break
# 				if mapped:
# 					break

# 		if not mapped:
# 			mapped = col.strip()
			
# 		normalized_cols.append(mapped)

# 		# ----------  FUZZY LAST (UNCHANGED) ------------
# 		# if not mapped:
# 		# 	match, score = process.extractOne(
# 		# 		clean_col, all_possible_headers, scorer=fuzz.token_sort_ratio
# 		# 	)
# 		# 	if score >= 75:
# 		# 		mapped = reverse_mapping[match]
# 		# 	else:
# 		# 		mapped = col.strip()

		

# 	df.columns = normalized_cols
	
# 	df.to_csv("C:\\metis\\excel_cleaning\\SBI_OUTPUT\\debug_after_normalize_headers2.csv", index=False) # Debugging output
	
# 	# If duplicate columns exist, merge them instead of dropping
	
# 	# If duplicate columns exist, merge them instead of dropping
# 	df = df.loc[:, ~df.columns.duplicated()]


# 	if "Balance" in df.columns:
# 		df["Balance"] = df["Balance"].apply(parse_balance)

# 	if 'Cheque No' in df.columns:
# 		df['Cheque No'] = (
# 			df['Cheque No']
# 			.replace(r'\.0$', '', regex=True)
# 			.replace(['0', 0], np.nan)
# 			.fillna('')
# 			.astype(str)
# 		)
# 	else:
# 		df['Cheque No'] = ""

# 	print("============")
# 	# print(df['Credits'])
# 	df.to_csv("debug_after_normalize_headers_final.csv", index=False) # Debugging output
# 	return df

def normalize_headers(df):
	"""
	Normalize column headers to standard names
	"""

	headers = {
		"XN Date": {"Date(ValueDate)","TransactionDate &Time","Tran Date","Date Day/Night","TransactionDate","TxnDate& Time","DAT VALUE", "Date(Value Date","Date& Time", "Post Date", "PostDate","Txn Posted Date", "TRANSACTION DATE", "Tran Date", "TranDate","XN Date", "Date", "Transaction date", "Txn Date", "Post date","ate", "DATE", "Transaction Date", "TRAN DATE", "TRANDATE","Transactio n Date"},
		"Value Date": {"Value Date", "VAL DATE", "Val Date", "VALUE DT","ValueDate", "VALDATE"},
		"Cheque No": {"Cheque/Refer enceNo","Cheque.No./Ref.No", "Cheq No ue", "CHQ/REFNO.","CHEQUE/REFERENCE#", "ChequeNo.", "Chq.No", "Cheque No","Chq./ref.no","Cheque number", "Ref no./cheque no.","Chq.no", "Chq No", "CHQ.NO.", "CHQ NO", "Cheque No.","Cheque Number", "Chq./Ref.No", "Chq.No."," Ref No./Cheque No.", "CHQ.NO", "Cnq.No.","Chq/Ref number", "Chq/Ref No"},
		"Narration": {"TransactionParticulars","RANSACTIONDETAILS","Payment Narration","TransactionRemarks","TransactionDetails CommentÂ·PlaceÂ·PaymentMethod","TransactionDescription","Transaction Description", "TRANSACTIONDETAILS", "Narration","Description", "Details", "Remarks", "Particulars","Transaction Particulars", "Partculars","TRANSACTION DETAILS", "DETAILS", "NARRATION","PARTICULARS", "Transaction Remarks","PARTICULARS CHO.NO.", "Transactio nRemarks"},
		"Credits": {"Credl","CreditAmount","Deposits (in Rs.)","DepositAmtï¼ˆINR)", "Deposits (INR)", "CREDIT()","Credit","Deposits (INR)", "Cr", "Cr Amt", "Deposit amt."," Credit(INR)", "CREDIT", "DEPOSIT(CR)", "DEPOSITS","Deposit Amt.", "Deposits", "Credit Amount"," Deposit Amount(INR)", "DEPOSIT (CR)", "CR"},
		"Debits": {"Debit Amount", "DebitAmount","DEBIT(R)","WithdrawalAmt(INR)", "Withdrawal (Dr)","Debit","Withdrawal(INR)", "Dr", "Dr Amt", "Withdrawalamt"," Debit(INR)", "DEBIT", " WITHDRAWAL(DR)", "WITHDRAWALS", "Withdrawal Amt.", "Withdrawals"," Transaction Amount(INR)", "WITHDRAWAL (DR)","Witndrawals", "DR"},
		"Balance": {"BALANCE()","TotalAmount","BOOKBAL", "BALANCER","RunningBalance", "Closing balance","Available balance", "Balance (Rs.)", "Balance"," Balance(INR)", "BALANCE", "Closing Balance"," Available Balance(INR)", "BALANCE(INR)", "Balance(IN R)", "Balance (INR)", "Available Balance(INR", "NetBalance"}
	}

	HEADER_REGEX = {
		"XN Date": [
			r'\btxn\s*d-ate\b',
			r'\btran\s*date\b',
			r'\bpost\s*date\b',
			r'\btransaction\s*date\b',
			r'\bdate\s*value\s*date\b',r'\b(?:txn|tran|transaction)\s*d\s*a\s*t\s*e\s*(?:&|and)\s*t\s*i\s*m\s*e\b',
			r'\bdate\s*(?:&|and)\s*time\b',r'\bdat\s*value\b',r'\btransaction\s*date\b',r'\btxn\s*date\s*(?:&|and)\s*time\b',r'\bdate\s*day\s*/\s*night\b',



		],
		"Value Date": [
			r'\bvalue\s*date\b',
			r'\bval\s*date\b'
		],
		"Cheque No": [r'\bcheq\b', r'\bchq\b'],
		"Narration": [r'\bnarr\b',r'\bpayment\s*n\s*a\s*r\s*r\s*a\s*t\s*i\s*o\s*n\b',r'\btransaction\s*remarks\b',r'\btransaction\s*description\b', r'\bparticulars?\b', r'\bremarks?\b', r'\bdetails?\b', r'\bdescription\b'],
		"Credits": [r'\bcredit\b', r'\bcr\b', r'\bdeposits?\b',r'\bdeposit\b',r'\bcredit\s*amount\b'],
		"Debits": [r'\bdebit\b', r'\bwithdrawals?\b',r'\bdr\b', r'\bwithdraw\b',r'\bdebit\s*amount\b',r'\bwithdrawal\s*amt\s*\(?inr\)?\b',r'\bdeposit\s*amt.*inr\b'],
		"Balance": [r'\bbalance\b',r'\btotal\s*a\s*m\s*o\s*u\s*n\s*t\b',r'\bbalance\s*\(inr\)\b' r'\bclosing\b', r'\btransaction\s*details\s*comment.*payment\s*method\b',
					r'\bavailable\b',r'\bbook\s*bal(?:ance)?\b',r'\brunning\s*bal(?:ance)?\b']
	}



	# 🔥 IMPORTANT CHECK
	drcr_already_split = {"Debits", "Credits"}.issubset(df.columns)

	normalized_cols = []

	for col in df.columns:

		original_col = str(col).strip()

		# If already standardized → skip
		if original_col in {"XN Date","Value Date","Cheque No","Narration","Debits","Credits","Balance"}:
			normalized_cols.append(original_col)
			continue

		clean_col = original_col.lower()
		clean_col = re.sub(r'[^a-z0-9 ]', ' ', clean_col)
		clean_col = re.sub(r'\s+', ' ', clean_col)

		mapped = None

		# ---------- HEADER DICT ----------
		for std, variants in headers.items():

			# 🚫 Skip DR/CR mapping if already split
			if drcr_already_split and std in {"Debits","Credits"}:
				continue

			for v in variants:
				v_clean = re.sub(r'[^a-z0-9 ]', ' ', v.lower())
				v_clean = re.sub(r'\s+', ' ', v_clean)

				if clean_col == v_clean:
					mapped = std
					break

			if mapped:
				break

		# ---------- REGEX ----------
		if not mapped:
			for std, patterns in HEADER_REGEX.items():

				# 🚫 Skip DR/CR regex if already split
				if drcr_already_split and std in {"Debits","Credits"}:
					continue

				for pat in patterns:
					if re.search(pat, clean_col):
						mapped = std
						break

				if mapped:
					break

		if not mapped:
			mapped = original_col

		normalized_cols.append(mapped)

	df.columns = normalized_cols

	# Remove duplicates safely
	df = df.loc[:, ~df.columns.duplicated()]

	if "Balance" in df.columns:
		df["Balance"] = df["Balance"].apply(parse_balance)

	if "Cheque No" in df.columns:
		df["Cheque No"] = (
			df["Cheque No"]
			.replace(r'\.0$', '', regex=True)
			.replace(['0', 0], np.nan)
			.fillna('')
			.astype(str)
		)
	else:
		df["Cheque No"] = ""


	return df



def create_ocr_corrected_columns(df):
	"""
	Create corrected columns for Debit, Credit and Balance
	with OCR error handling and decimal correction.
	"""
	print(">>> Creating OCR corrected columns...")
	
	# Create new columns with corrected values
	# Use the raw string values that were saved before extraction
	# If raw columns exist, use them. Otherwise, use current columns (as strings)
	
	if 'Debits_Raw' in df.columns:
		df['Debits_Original'] = df['Debits_Raw']
		df['Debits_Corrected'] = df['Debits_Raw'].apply(extract_amount_new)
	elif 'Debits' in df.columns:
		df['Debits_Original'] = df['Debits'].astype(str)
		df['Debits_Corrected'] = df['Debits'].astype(str).apply(extract_amount_new)
	else:
		df['Debits_Original'] = ""
		df['Debits_Corrected'] = ""
	
	if 'Credits_Raw' in df.columns:
		df['Credits_Original'] = df['Credits_Raw']
		df['Credits_Corrected'] = df['Credits_Raw'].apply(extract_amount_new)
	elif 'Credits' in df.columns:
		df['Credits_Original'] = df['Credits'].astype(str)
		df['Credits_Corrected'] = df['Credits'].astype(str).apply(extract_amount_new)
	else:
		df['Credits_Original'] = ""
		df['Credits_Corrected'] = ""
	
	if 'Balance_Raw' in df.columns:
		df['Balance_Original'] = df['Balance_Raw']
		df['Balance_Corrected'] = df['Balance_Raw'].apply(extract_amount_new)
	elif 'Balance' in df.columns:
		df['Balance_Original'] = df['Balance'].astype(str)
		df['Balance_Corrected'] = df['Balance'].astype(str).apply(extract_amount_new)
	else:
		df['Balance_Original'] = ""
		df['Balance_Corrected'] = ""
	
	# Ensure numeric types
	for col in ['Debits_Corrected', 'Credits_Corrected', 'Balance_Corrected']:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce')
	
	print("<<< OCR corrected columns created")
	return df


def calculate_difference_and_verify(df):
	"""
	Calculate difference using formula:
	BALANCE(i-1) + CREDIT(i) + DEBIT(i) - BALANCE(i)
	and check if difference is 0 for all rows.
	
	If differences are only in decimals (between -0.99 and 0.99),
	adjust ALL balances to make difference 0 for ALL rows.
	"""
	print(">>> Calculating differences for OCR verification...")
	
	# Create a copy and reset index to ensure continuous integer indices
	df_diff = df.copy().reset_index(drop=True)
	
	# Debug: Print DataFrame info
	# print(f"DataFrame shape after reset: {df_diff.shape}")
	# print(f"Available columns: {list(df_diff.columns)}")
	
	# Initialize difference column
	df_diff['Difference'] = 0.0
	df_diff['Balance_Adjusted'] = df_diff['Balance_Corrected'].copy() if 'Balance_Corrected' in df_diff.columns else df_diff['Balance'].copy()
	
	# Check if we have the corrected columns
	has_corrected_debits = 'Debits_Corrected' in df_diff.columns
	has_corrected_credits = 'Credits_Corrected' in df_diff.columns
	has_corrected_balance = 'Balance_Corrected' in df_diff.columns
	
	# print(f"Has corrected columns - Debits: {has_corrected_debits}, Credits: {has_corrected_credits}, Balance: {has_corrected_balance}")
	
	# Use corrected columns if available, otherwise use original
	debit_col = 'Debits_Corrected' if has_corrected_debits else 'Debits'
	credit_col = 'Credits_Corrected' if has_corrected_credits else 'Credits'
	balance_col = 'Balance_Adjusted'  # Use adjusted balance column for calculation
	
	# Check if columns exist, if not try to find alternatives
	if debit_col not in df_diff.columns:
		# print(f"WARNING: {debit_col} not in DataFrame columns!")
		# Try to find alternative debit column
		for col in df_diff.columns:
			if 'debit' in col.lower():
				debit_col = col
				# print(f"Found alternative debit column: {debit_col}")
				break
	
	if credit_col not in df_diff.columns:
		# print(f"WARNING: {credit_col} not in DataFrame columns!")
		# Try to find alternative credit column
		for col in df_diff.columns:
			if 'credit' in col.lower():
				credit_col = col
				# print(f"Found alternative credit column: {credit_col}")
				break
	
	if balance_col not in df_diff.columns:
		# print(f"WARNING: {balance_col} not in DataFrame columns!")
		# Try to find alternative balance column
		for col in df_diff.columns:
			if 'balance' in col.lower():
				balance_col = col
				# print(f"Found alternative balance column: {balance_col}")
				break
	
	# print(f"Using columns - Debit: {debit_col}, Credit: {credit_col}, Balance: {balance_col}")
	
	# Check if required columns exist
	required_cols_missing = []
	if debit_col not in df_diff.columns:
		required_cols_missing.append(debit_col)
	if credit_col not in df_diff.columns:
		required_cols_missing.append(credit_col)
	if balance_col not in df_diff.columns:
		required_cols_missing.append(balance_col)
	
	if required_cols_missing:
		# print(f"ERROR: Missing required columns: {required_cols_missing}")
		# print("Cannot calculate differences. Returning original DataFrame.")
		return df_diff, False, False
	
	# Helper function to round to 2 decimal places
	def round_to_2(val):
		if pd.isna(val) or str(val).strip().lower() in ["", "nan", "none"]:
			return 0.0
		try: 
			return round(float(val), 2)
		except:
			return 0.0
	
	# Round all numeric columns to 2 decimal places for consistent calculations
	for col in [debit_col, credit_col, balance_col]:
		if col in df_diff.columns:
			df_diff[col] = df_diff[col].apply(round_to_2)
	
	# Calculate difference for each row starting from row 1
	# print(f"DataFrame has {len(df_diff)} rows, will process rows 1 to {len(df_diff)-1}")
	
	# CRITICAL FIX: Use iloc instead of at to avoid index issues
	for i in range(1, len(df_diff)):
		try:
			# Get previous balance (rounded) - using iloc for position-based access
			prev_balance = 0
			if i > 0 and balance_col in df_diff.columns:
				prev_balance = round_to_2(df_diff.iloc[i-1][balance_col])
			
			# Get current values (rounded) - using iloc for position-based access
			curr_debit = 0
			if debit_col in df_diff.columns:
				curr_debit = round_to_2(df_diff.iloc[i][debit_col])
			
			curr_credit = 0
			if credit_col in df_diff.columns:
				curr_credit = round_to_2(df_diff.iloc[i][credit_col])
			
			curr_balance = 0
			if balance_col in df_diff.columns:
				curr_balance = round_to_2(df_diff.iloc[i][balance_col])
			
			# Calculate difference using the formula: Balance(i-1) + Credit(i) + Debit(i) - Balance(i)
			# Round to 2 decimal places to avoid floating-point errors
			difference = round(prev_balance + curr_credit + curr_debit - curr_balance, 2)
			
			# Use iloc to set the value safely
			df_diff.iloc[i, df_diff.columns.get_loc('Difference')] = difference
			
		except IndexError as e:
			# print(f"IndexError at position {i}: {e}")
			# print(f"DataFrame has {len(df_diff)} rows, trying to access row {i}")
			break
		except KeyError as e:
			# print(f"KeyError at position {i}: {e}")
			# print(f"Trying to access column that doesn't exist")
			break
		except Exception as e:
			# print(f"Unexpected error at position {i}: {e}")
			break
	
	# REQUIREMENT 2: Check if all differences are in the acceptable decimal range (-0.99 to 0.99)
	differences = df_diff['Difference']
	
	# Check conditions for adjustment:
	# 1. All differences must be between -0.99 and 0.99 (exclusive of -1 and 1)
	all_differences_in_range = ((differences > -1.0) & (differences < 1.0)).all()
	
	if all_differences_in_range and len(df_diff) > 1:
		# print("✓ All differences are within -0.99 to 0.99. Adjusting balances...")
		
		# FIXED: We need to adjust ALL balances consistently
		# Start with first balance as reference
		adjusted_balances = []
		
		# Keep the first balance as is (rounded to 2 decimal places)
		if balance_col in df_diff.columns and not pd.isna(df_diff.iloc[0][balance_col]):
			adjusted_balances.append(round_to_2(df_diff.iloc[0][balance_col]))
		else:
			adjusted_balances.append(0.0)
		
		# Calculate adjusted balances for all subsequent rows
		for i in range(1, len(df_diff)):
			# Get the adjusted previous balance
			prev_adjusted_balance = adjusted_balances[-1]
			
			# Get current debit and credit (rounded)
			curr_debit = 0
			if debit_col in df_diff.columns:
				curr_debit = round_to_2(df_diff.iloc[i][debit_col])
			
			curr_credit = 0
			if credit_col in df_diff.columns:
				curr_credit = round_to_2(df_diff.iloc[i][credit_col])
			
			# Calculate what the current balance SHOULD be based on previous adjusted balance
			# Formula: Current Balance = Previous Balance + Credit + Debit
			# Round to 2 decimal places to avoid floating-point errors
			should_be_balance = round(prev_adjusted_balance + curr_credit + curr_debit, 2)
			
			# Store this as the adjusted balance
			adjusted_balances.append(should_be_balance)
			
			# Update the Balance_Adjusted column using iloc
			df_diff.iloc[i, df_diff.columns.get_loc('Balance_Adjusted')] = should_be_balance
		
		# Update the first row's Balance_Adjusted if it exists
		if len(adjusted_balances) > 0 and balance_col in df_diff.columns:
			df_diff.iloc[0, df_diff.columns.get_loc('Balance_Adjusted')] = adjusted_balances[0]
		
		# Recalculate differences after adjustment (with rounding)
		for i in range(1, len(df_diff)):
			# Get previous adjusted balance (rounded)
			prev_balance = 0
			if i > 0 and 'Balance_Adjusted' in df_diff.columns:
				prev_balance = round_to_2(df_diff.iloc[i-1]['Balance_Adjusted'])
			
			# Get current values (rounded)
			curr_debit = 0
			if debit_col in df_diff.columns:
				curr_debit = round_to_2(df_diff.iloc[i][debit_col])
			
			curr_credit = 0
			if credit_col in df_diff.columns:
				curr_credit = round_to_2(df_diff.iloc[i][credit_col])
			
			curr_balance = 0  
			if 'Balance_Adjusted' in df_diff.columns:
				curr_balance = round_to_2(df_diff.iloc[i]['Balance_Adjusted'])
			
			# Recalculate difference with rounding
			difference = round(prev_balance + curr_credit + curr_debit - curr_balance, 2)
			df_diff.iloc[i, df_diff.columns.get_loc('Difference')] = difference
		
		adjusted = True
		# print("✓ All balances have been adjusted to synchronize with transactions.")
	
	else:
		adjusted = False
		if len(df_diff) > 1:
			# print("✗ Differences are not all within the decimal range. No adjustment made.")
			pass
			# Show the differences that are out of range
			out_of_range = df_diff[~((df_diff['Difference'] > -1.0) & (df_diff['Difference'] < 1.0))]
			if not out_of_range.empty:
				# print(f"  Rows with differences out of range: {list(out_of_range.index)}")
				pass
				for idx in out_of_range.index[:3]:
					# print(f"    Row {idx}: Difference = {df_diff.at[idx, 'Difference']}")
					pass
		else:
			# print("✗ Not enough rows for adjustment.")
			pass
	
	# Check if all differences are close to 0 (within tolerance)
	tolerance = 0.001  # Allow very small floating point errors due to rounding
	
	# Calculate statistics
	differences_abs = df_diff['Difference'].abs()
	non_zero_diffs = df_diff[differences_abs > tolerance]
	
	if len(non_zero_diffs) == 0:
		# print("✓ All differences are 0 - OCR values appear correct!")
		all_correct = True
	else:
		# print(f"✗ Differences found - {len(non_zero_diffs)} rows have non-zero differences")
		# print(f"  Max difference: {differences_abs.max()}")
		# print(f"  Rows with differences: {list(non_zero_diffs.index)}")
		pass
		# Show some examples of problematic rows
		for idx in non_zero_diffs.index[:5]:  # Show first 5 problematic rows
			# print(f"\n  Row {idx}:")
			# print(f"    Prev Balance: {round_to_2(df_diff.at[idx-1, 'Balance_Adjusted']) if idx > 0 and 'Balance_Adjusted' in df_diff.columns else 'N/A'}")
			# print(f"    Debit: {round_to_2(df_diff.at[idx, debit_col])}")
			# print(f"    Credit: {round_to_2(df_diff.at[idx, credit_col])}")
			# print(f"    Current Balance: {round_to_2(df_diff.at[idx, 'Balance_Adjusted']) if 'Balance_Adjusted' in df_diff.columns else 'N/A'}")
			# print(f"    Difference: {df_diff.at[idx, 'Difference']}")
			
			# Also show original values for debugging
			if 'Debits_Original' in df_diff.columns:
				# print(f"    Debit Original: {df_diff.at[idx, 'Debits_Original']}")
				pass
			if 'Balance_Original' in df_diff.columns:
				# print(f"    Balance Original: {df_diff.at[idx, 'Balance_Original']}")
				pass
		
		all_correct = False
	
	# print("<<< Difference calculation completed")
	return df_diff, all_correct, adjusted

# def resolve_debit_credit_using_balance(df):
#     """
#     If both Debits and Credits exist in the same row,
#     resolve correct one using balance movement.
#     """

#     if not {"Debits", "Credits", "Balance"}.issubset(df.columns):
#         return df

#     # Ensure numeric comparison
#     df["Debits"] = pd.to_numeric(df["Debits"], errors="coerce")
#     df["Credits"] = pd.to_numeric(df["Credits"], errors="coerce")
#     df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")

#     for i in range(1, len(df)):
#         prev_bal = df.at[i - 1, "Balance"]
#         curr_bal = df.at[i, "Balance"]

#         debit = df.at[i, "Debits"]
#         credit = df.at[i, "Credits"]

#         # Only when BOTH are present
#         if pd.notna(debit) and pd.notna(credit):
#             if pd.notna(prev_bal) and pd.notna(curr_bal):

#                 # Balance decreased → Debit
#                 if curr_bal < prev_bal:
#                     df.at[i, "Credits"] = np.nan

#                 # Balance increased → Credit
#                 elif curr_bal > prev_bal:
#                     df.at[i, "Debits"] = np.nan
					
#         #  CASE 2: BOTH missing (NEW logic added)
#         elif pd.isna(debit) and pd.isna(credit):

#             diff = curr_bal - prev_bal

#             # Balance increased → Credit
#             if diff > 0:
#                 df.at[i, "Credits"] = abs(diff)

#             # Balance decreased → Debit
#             elif diff < 0:
#                 df.at[i, "Debits"] = abs(diff)

#     return df


def run_step(step_name, func, df):
	"""
	Helper function to run each step with error handling
	"""
	try:
		# print(f"\n>>> Running step: {step_name}")
		df = func(df)
		print(f"<<< Step {step_name} completed")
	except Exception as e:
		print(f"{step_name}: FAILED with error: {e}\n")
	return df


def clean_bank_statement(df, file_path=None, logging=True):
	"""
	Main cleaning pipeline for bank statements
	"""
	df.columns = [str(col).strip().capitalize() for col in df.columns]

	def step_clean_debit_credit(df): return clean_debit_credit(df)

	def step_remove_duplicate_column(df): return remove_duplicate_column(df)

	def step_normalize_headers(df): return normalize_headers(df)

	def step_merge_partial_rows(df):
		# Fix the applymap deprecation
		for col in df.columns:
			df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
		
		df = df.replace('', pd.NA)
		df = df.dropna(how='all')
		df.reset_index(drop=True, inplace=True)
		rows_to_drop = []
		for i in range(1, len(df)):
			row = df.iloc[i]
			if is_partial_row(row):
				for col in df.columns:
					if any(k in col.lower() for k in ['narration', 'description']):
						df.at[i - 1, col] = str(
							df.iloc[i - 1][col]) + ' ' + str(df.iloc[i][col])
				rows_to_drop.append(i)
		df.drop(index=df.index[rows_to_drop], inplace=True)
		df.reset_index(drop=True, inplace=True)
		return df
	
	# def step_resolve_drcr_balance(df):
	#     return resolve_debit_credit_using_balance(df)

	def step_remove_metadata_rows(df):
		"""
		Remove ANY row where ANY cell fuzzy-matches a metadata phrase.
		Uses RapidFuzz (threshold ≥70) on all columns.
		"""
		# Comprehensive metadata phrases (including common OCR errors)
		METADATA_PHRASES = [
			"BROUGHT FORWARD", "BROUGHTFORWARD", "BROUGHT FWD", "B/F", "BF",
			"CARRIED FORWARD", "CARRIEDFORWARD", "CARRIED FWD", "C/F", "CF",
			"CLOSING BALANCE", "CLOSINGBALANCE", "CLOSING BAL", "CL BAL",
			"OPENING BALANCE", "OPENINGBALANCE", "OPENING BAL", "OP BAL",
			"BALANCE", "TOTAL", "TOT", "SUB TOTAL", "SUBTOTAL",
			"TOTAL AMOUNT", "TOTAL AMT", "GRAND TOTAL", "SUMMARY",
			"TRANSACTION TOTAL", "TRANSACTIONTOTAL",
			"TOTAL DEBIT", "TOTAL CREDIT", "TOTALDEBIT", "TOTALCREDIT",
			"YOUR OPENING", "BALANCE ON","PageTotal",
			"LOSINGBALANCE", "RROUGHTFOROWARD", "BROOGHTFORWARD",
			"TRANSACTIONTOTAI", "TRANSACTION TOTAL DRICR",
			"BALANCE CARRIED", "BALANCE BROUGHT"
		]

		# Clean and lowercase once
		phrases_clean = [p.lower().strip() for p in METADATA_PHRASES]

		def is_metadata_row(row):
			"""Return True if ANY cell in the row fuzzy-matches ANY metadata phrase."""
			for col in row.index:
				cell = str(row[col]).strip()
				if not cell or cell.lower() in ["nan", "none", ""]:
					continue
				cell_lower = cell.lower()
				for phrase in phrases_clean:
					# Skip very short phrases unless they are abbreviations
					if len(phrase) < 4 and phrase not in ["b/f", "c/f", "bf", "cf", "tot"]:
						continue
					ratio = rfuzz.ratio(cell_lower, phrase)
					if ratio >= 70:   # Catches OCR errors (e.g., "Totol Amont")
						return True
			return False

		# Apply filter – remove rows where condition is True
		mask = df.apply(is_metadata_row, axis=1)
		removed_count = mask.sum()
		if removed_count > 0:
			print(f" Removed {removed_count} metadata row(s) (fuzzy match in all column)")
		df = df[~mask]

		return df

	# def step_parse_amounts(df):
	#     # Store original string values BEFORE processing
	#     if 'Debits' in df.columns:
	#         df['Debits_Raw'] = df['Debits'].astype(str)
	#     if 'Credits' in df.columns:
	#         df['Credits_Raw'] = df['Credits'].astype(str)
	#     if 'Balance' in df.columns:
	#         df['Balance_Raw'] = df['Balance'].astype(str)
		
	#     for col in df.columns:
	#         if any(key in col.lower() for key in ['credit', 'debit', 'amount', 'withdrawalamt', 'deposit amt.']):
	#             # Skip raw columns we just created
	#             if col.endswith('_Raw'):
	#                 continue
	#             df[col] = df[col].apply(extract_amount)
	#             if df[col].isnull().all():
	#                 df.drop(columns=[col], inplace=True)
	#     return df


	def step_parse_amounts(df):
		for col in df.columns:
			if any(key in col.lower() for key in ['credit', 'debit', 'amount', 'withdrawalamt', 'deposit amt.']):
				# Skip if column already contains numeric values
				if pd.api.types.is_numeric_dtype(df[col]):
					continue
					
				df[col] = df[col].apply(extract_amount)
				if df[col].isnull().all():
					df.drop(columns=[col], inplace=True)
		return df

	def step_process_all_dates(df):
		return process_all_dates(df, file_path, logging)

	def step_cleanup_columns(df):
		df.dropna(axis=1, how='all', inplace=True)
		df = df.loc[:, df.apply(lambda col: col.astype(str).str.strip()).ne('').any()]
		df.reset_index(drop=True, inplace=True)
		return df

	def step_create_ocr_corrected_columns(df):
		"""NEW STEP: Create OCR corrected columns using RAW string values"""
		return create_ocr_corrected_columns(df)


	def step_ensure_required_columns(df):
		"""
		Ensure all required columns exist and apply final formatting
		"""
		required_columns = ["XN Date", "Cheque No", "Narration", "Debits", "Credits", "Balance"]
		# df.to_csv("debug_before_ensure_required.csv", index=False) # Debugging output
		# Helper function to round to 2 decimal places
		def round_to_2_if_numeric(val):
			if pd.isna(val) or str(val).strip().lower() in ["", "nan", "none"]:
				return ""
			try:
				return round(float(val), 2)
			except:
				return val

		# Use corrected values if available
		if 'Debits_Corrected' in df.columns:
			df['Debits'] = df['Debits_Corrected'].apply(
				lambda x: round(-1 * float(x), 2) if pd.notna(x) else ""
			)
		elif "Debits" in df.columns:
			df["Debits"] = df["Debits"].apply(
				lambda x: round(-1 * float(x), 2)
				if str(x).strip() not in ["", "nan", "None"]
				else ""
			)

		# Credits → positive, empty stays empty
		if 'Credits_Corrected' in df.columns:
			df['Credits'] = df['Credits_Corrected'].apply(
				lambda x: round(float(x), 2) if pd.notna(x) else ""
			)
	
 
 
		elif "Credits" in df.columns:
			df["Credits"] = df["Credits"].apply(
				lambda x: round(float(x), 2)
				if str(x).strip() not in ["", "nan", "None"]
				else ""
			)
   

		# Balance - use corrected if available
		if 'Balance_Corrected' in df.columns:
			df['Balance'] = df['Balance_Corrected'].apply(round_to_2_if_numeric)
		elif "Balance" in df.columns:
			df["Balance"] = df["Balance"].apply(
				lambda x: round(float(x), 2)
				if str(x).strip() not in ["", "nan", "None"]
				else ""
			)

		# Narration cleanup and metadata removal
		def _is_empty_amount(val):
			return (
				pd.isna(val) or
				str(val).strip().lower() in ["", "nan", "none", "0", "0.0", "0.00"]
			)
		
		if all(col in df.columns for col in ["Narration", "Debits", "Credits", "Balance"]):
			def _remove_metadata_row(row):
				narration = str(row["Narration"]).strip()
				debit = row["Debits"]
				credit = row["Credits"]
				balance = row["Balance"]

				if narration and _is_empty_amount(debit) and _is_empty_amount(credit) and _is_empty_amount(balance):
					return True
				return False

			df = df[~df.apply(_remove_metadata_row, axis=1)]

		for col in required_columns:
			if col not in df.columns:
				df[col] = ""

		df = df[[col for col in required_columns if col in df.columns]]
		return df

	# Run all cleaning steps in sequence
	
	df = run_step("clean_debit_credit", step_clean_debit_credit, df)
	df = run_step("remove_duplicate_column", step_remove_duplicate_column, df)
	
	df = run_step("normalize_headers", step_normalize_headers, df)
	# df = run_step("parse_amounts", step_parse_amounts, df)
	df = run_step("merge_partial_rows", step_merge_partial_rows, df)
	# Parse amounts FIRST (but save raw values)
	
	# df = run_step("resolve_debit_credit_using_balance", step_resolve_drcr_balance, df)
	df = run_step("remove_metadata_rows", step_remove_metadata_rows, df)
	df = run_step("parse_amounts", step_parse_amounts, df)
	
	# Add the new step for OCR correction (uses raw values saved in parse_amounts)
	df = run_step("create_ocr_corrected_columns", step_create_ocr_corrected_columns, df)
	
	# print(f"\n>>> Running step: process_all_dates")
	df = step_process_all_dates(df)
	# print(f"<<< Step process_all_dates completed")
	
	df = run_step("cleanup_columns", step_cleanup_columns, df)
	df = run_step("ensure_required_columns", step_ensure_required_columns, df)
	

	return df


def clean_main(file_path, output_path, logging=True, debug=True):
	"""
	Main function to process bank statement files
	"""
	try:
		print(f"\nProcessing file: {file_path}")
		print(f"Logging enabled: {logging}")
		print(f"Debug mode: {debug}")
		
		df_raw = pd.read_csv(file_path, header=None)
		
		header_row = detect_header_row(df_raw)
		# print("############# Header Row #############")
		# print(header_row)
		
		if header_row is not None:
			df_raw = df_raw.iloc[header_row:]
			df_raw.columns = df_raw.iloc[0].str.strip()  # Set headers (strip spaces)
			df_raw = df_raw.loc[:, df_raw.columns.notna()]  # Remove NaN headers
			df = df_raw.reset_index(drop=True)  # Set actual header
			
			# Remove unwanted rows
			# unwanted_keywords = ['column']
			# unwanted_exact_labels = [
			#     "City", "State DL", "Phoneno. OD Limit", "Emall Cust ID",
			#     "AccountNo A/C OpenDate", "Account Status RTGS/NEFT IFS", 'HCY', 'INT', 'BKNG', 'CNCL', 'ISSUE', 'AMEND', 'OWRTN', 'Ln', 'CLG', "BRN-Branch",
			#     "LDG-Lodge",
			#     "INB -Internet Banking",
			#     "RLZ-Realise",
			#     "DLK-Delink",
			#     "DHR-Dishonour",
			#     "REC -Recovery",
			#     "LN-Loan",
			#     "HCY-Home Currency A dvance",
			#     "ISSUE-Issuance AMEND-Amendment",
			#     "AMEND-Amendment PUR-Purchase",
			#     "AMEND-Amendment",
			# ]
			# try:
			#     df = df[~df.apply(lambda row: any(
			#         any(keyword.lower() in str(cell).lower() for keyword in unwanted_keywords)
			#         or str(cell).strip() in unwanted_exact_labels
			#         for cell in row
			#     ), axis=1)]
			# except:
			#     pass

			cleaned_df = clean_bank_statement(df, file_path, logging)

			if cleaned_df is not None and not cleaned_df.empty:
				# NEW: Calculate differences and verify OCR accuracy
				# print("\n" + "="*60)
				# print("OCR VERIFICATION AND DIFFERENCE CALCULATION")
				# print("="*60)
				
				# Create dataframe with corrected columns and differences
				df_with_diff, all_correct, adjusted = calculate_difference_and_verify(cleaned_df)
				
				# REQUIREMENT 2: Update cleaned_df with adjusted balances if adjustment was made
				if adjusted and 'Balance_Adjusted' in df_with_diff.columns:
					# Update the Balance column in cleaned_df with adjusted values
					if 'Balance_Corrected' in cleaned_df.columns:
						cleaned_df['Balance_Corrected'] = df_with_diff['Balance_Adjusted']
					# Also update the main Balance column
					cleaned_df['Balance'] = df_with_diff['Balance_Adjusted']
					# print("✓ Updated cleaned file with adjusted balances.")
				
				# Save the main cleaned file (without debug columns)
				required_cols = ['XN Date', 'Cheque No', 'Narration', 'Debits', 'Credits', 'Balance']
				available_cols = [col for col in required_cols if col in cleaned_df.columns]
				main_cleaned_df = cleaned_df[available_cols]
				main_cleaned_df.to_csv(output_path, index=False)
				
				# print(f"\n✓ MAIN FILE: Cleaned file saved to: {output_path}")
				
				# Save debug file with all new columns as Excel ONLY if debug=True
				if debug:
					debug_output_path = output_path.replace('.csv', '_debug.xlsx')
					
					# Prepare debug dataframe with all columns
					debug_cols = []
					
					# Add basic columns
					basic_cols = ['XN Date', 'Cheque No', 'Narration']
					for col in basic_cols:
						if col in df_with_diff.columns:
							debug_cols.append(col)
					
					# Add original amount columns
					for col in ['Debits', 'Credits', 'Balance']:
						if col in df_with_diff.columns:
							debug_cols.append(col + '_Original')
							# Store original values
							df_with_diff[col + '_Original'] = df_with_diff[col]
					
					# Add corrected columns
					for col in ['Debits_Corrected', 'Credits_Corrected', 'Balance_Corrected']:
						if col in df_with_diff.columns:
							debug_cols.append(col)
					
					# Add adjusted balance column if available
					if 'Balance_Adjusted' in df_with_diff.columns:
						debug_cols.append('Balance_Adjusted')
					
					# Add difference column
					if 'Difference' in df_with_diff.columns:
						debug_cols.append('Difference')
					
					# Create debug dataframe
					debug_df = df_with_diff[debug_cols].copy()
					
					# Save to Excel
					debug_df.to_excel(debug_output_path, index=False)
					# print(f"✓ DEBUG FILE: Full analysis saved to: {debug_output_path}")
				else:
					# print("✓ Debug file not created (debug=False)")
					pass
				
				
				# Check if corrected values differ from original
				corrections_made = 0
				
				if 'Debits_Corrected' in df_with_diff.columns and 'Debits_Original' in df_with_diff.columns:
					# Compare non-null values
					mask = ~pd.isna(df_with_diff['Debits_Corrected'])
					if mask.any():
						# Get the original extracted values (from extract_amount, not string)
						# We need to convert the original string to number for comparison
						original_values = df_with_diff.loc[mask, 'Debits_Original'].apply(
							lambda x: extract_amount(x) if isinstance(x, str) and x.strip() not in ['', 'nan', 'None'] else np.nan
						)
						corrected_values = df_with_diff.loc[mask, 'Debits_Corrected']
						
						# Compare with tolerance for floating point
						diff_mask = ~np.isclose(original_values, corrected_values, rtol=1e-9, atol=1e-9)
						debit_changes = diff_mask.sum()
						corrections_made += debit_changes
						print(f"Debits corrected: {debit_changes} rows")
				
				if 'Credits_Corrected' in df_with_diff.columns and 'Credits_Original' in df_with_diff.columns:
					mask = ~pd.isna(df_with_diff['Credits_Corrected'])
					if mask.any():
						original_values = df_with_diff.loc[mask, 'Credits_Original'].apply(
							lambda x: extract_amount(x) if isinstance(x, str) and x.strip() not in ['', 'nan', 'None'] else np.nan
						)
						corrected_values = df_with_diff.loc[mask, 'Credits_Corrected']
						
						diff_mask = ~np.isclose(original_values, corrected_values, rtol=1e-9, atol=1e-9)
						credit_changes = diff_mask.sum()
						corrections_made += credit_changes
						#print(f"Credits corrected: {credit_changes} rows")
				
				if 'Balance_Corrected' in df_with_diff.columns and 'Balance_Original' in df_with_diff.columns:
					mask = ~pd.isna(df_with_diff['Balance_Corrected'])
					if mask.any():
						original_values = df_with_diff.loc[mask, 'Balance_Original'].apply(
							lambda x: extract_amount(x) if isinstance(x, str) and x.strip() not in ['', 'nan', 'None'] else np.nan
						)
						corrected_values = df_with_diff.loc[mask, 'Balance_Corrected']
						
						diff_mask = ~np.isclose(original_values, corrected_values, rtol=1e-9, atol=1e-9)
						balance_changes = diff_mask.sum()
						corrections_made += balance_changes
						#print(f"Balance corrected: {balance_changes} rows")
				
				if adjusted:
					# print("\n BALANCES ADJUSTED: Decimal differences have been synchronized.")
					# print("   Updated balances saved to cleaned file.")
					pass
				elif all_correct:
					# print("\n VERIFICATION PASSED: All differences are 0")
					# print("   The corrected values are mathematically consistent.")
					pass
				else:
					# print("\n  VERIFICATION WARNING: Some differences found")
					# if debug:
					#     print("   Check the debug file for details.")
					# else:
					#     print("   Run with debug=True to see details.")
					pass
				
				if corrections_made > 0:
					# print(f"\n TOTAL CORRECTIONS: {corrections_made} values were corrected for OCR errors")
					pass
			else:
				print("ERROR: Cleaned data is empty.")

		else:
			print("ERROR: No header detected in the CSV file.")

	except Exception as e:
		print(f"\n!!! CRITICAL ERROR: {e}")
		import traceback
		traceback.print_exc()


if __name__ == "__main__":
	input_csv = r"download 392 P VIJAYAKUMAR.csv"
	output_csv = r"rdownload 392 P VIJAYAKUMAR.csv"
	clean_main(input_csv, output_csv, logging=False, debug=True)	