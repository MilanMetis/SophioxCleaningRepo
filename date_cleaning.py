#Code Author: Kayroze Shroff
#Updated Date: 13-Jan-2026

import re
import gc
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz, process
from difflib import get_close_matches
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
import os

# Custom map for non-standard month abbreviations
custom_month_map = {
    'JAN': '01', 'AN': '01', 'JA': '01', 'JANUARY': '01',
    'FEB': '02', 'FE': '02', 'FB': '02', 'FEBRUARY': '02',
    'MAR': '03', 'MR': '03', 'RCH': '03', 'MARCH': '03',
    'APR': '04', 'AP': '04', 'APRIL': '04',
    'MAY': '05', 'MY': '05',
    'JUN': '06', 'JN': '06', 'UN': '06', 'JUNE': '06',
    'JUL': '07', 'JL': '07', 'UL': '07', 'JULY': '07', 'JU': '07',
    'AUG': '08', 'AU': '08', 'A0G': '08', 'AUGUST': '08',
    'SEP': '09', 'SE': '09', 'SEPT': '09', 'SEPTEMBER': '09',
    'OCT': '10', 'OC': '10', '0CT': '10', '0ct': '10', 'OCTOBER': '10',
    'NOV': '11', 'NO': '11', 'NOVEMBER': '11',
    'DEC': '12', 'DE': '12', 'DECEMBER': '12'
}

# Extended month map with all variations
full_month_names = {
    'january': '01', 'jan': '01',
    'february': '02', 'feb': '02',
    'march': '03', 'mar': '03',
    'april': '04', 'apr': '04',
    'may': '05',
    'june': '06', 'jun': '06',
    'july': '07', 'jul': '07',
    'august': '08', 'aug': '08',
    'september': '09', 'sep': '09', 'sept': '09',
    'october': '10', 'oct': '10',
    'november': '11', 'nov': '11',
    'december': '12', 'dec': '12'
}

# Create extended map with all case variations
extended_month_map = {}
for k, v in {**custom_month_map, **full_month_names}.items():
    extended_month_map[k.upper()] = v
    extended_month_map[k.lower()] = v
    extended_month_map[k.capitalize()] = v

def normalize_month(token: str) -> Optional[str]:
    """Fuzzy match token to a valid month."""
    try:
        if not token:
            return None
        
        token = str(token).upper().strip()
        
        # Direct match
        if token in extended_month_map:
            return extended_month_map[token]
        
        # Try removing non-alphabetic characters
        clean_token = re.sub(r'[^A-Z]', '', token)
        if clean_token and clean_token in extended_month_map:
            return extended_month_map[clean_token]
        
        # Try matching the first 3-4 characters
        for length in [4, 3]:
            if len(token) >= length:
                substr = token[:length]
                if substr in extended_month_map:
                    return extended_month_map[substr]
        
        # Fuzzy matching
        matches = get_close_matches(token, extended_month_map.keys(), n=1, cutoff=0.6)
        if matches:
            return extended_month_map[matches[0]]
        
        # Try partial matching
        for month_name, month_num in extended_month_map.items():
            if month_name in token or token in month_name:
                return month_num
        
        return None
    except:
        return None

def parse_custom_date(x) -> Optional[str]:
    """Enhanced date parser that handles month names, abbreviations, mixed strings, and various formats."""
    try:
        if x is None or str(x).strip() == "":
            return None

        original_x = str(x).strip()
        
        # Check for invalid day "00" or "00:" patterns - return None immediately
        if re.search(r'(^|\D)00($|\D)', original_x):
            return None
        
        # Handle mixed strings with prefix numbers like "25 28-11-2025" FIRST
        # BEFORE any cleaning that might remove spaces
        mixed_pattern1 = re.compile(r'^\s*(\d+)\s+(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2,4})\s*$', re.IGNORECASE)
        match = mixed_pattern1.match(original_x)
        if match:
            prefix, day, month, year = match.groups()
            if len(year) == 2:
                year = '20' + year if int(year) <= 30 else '19' + year
            return f"{int(day):02d}/{int(month):02d}/{year}"
        
        mixed_pattern2 = re.compile(r'^\s*(\d+)\s+(\d{1,2})[-/\.\s]+([A-Za-z]{3,9})[-/\.\s]+(\d{2,4})\s*$', re.IGNORECASE)
        match = mixed_pattern2.match(original_x)
        if match:
            prefix, day, month_str, year = match.groups()
            month_num = normalize_month(month_str.upper())
            if month_num:
                if len(year) == 2:
                    year = '20' + year if int(year) <= 30 else '19' + year
                return f"{int(day):02d}/{month_num}/{year}"
        
        # Now proceed with the original cleaning
        x_upper = original_x.upper()
        
        # Clean common OCR errors and special characters
        x_upper = re.sub(r'\(.*?\)', '', x_upper)
        x_upper = re.sub(r'\s+', ' ', x_upper)
        x_upper = x_upper.replace('0CT', 'OCT').replace('A0G', 'AUG')
        x_upper = x_upper.replace('`', '').replace("'", '')
        x_upper = re.sub(r'[\[\]{}]', '', x_upper)
        
        # Handle specific month OCR errors
        month_corrections = {
            'JULY': 'JUL', 'JANUARY': 'JAN', 'FEBRUARY': 'FEB',
            'MARCH': 'MAR', 'APRIL': 'APR', 'JUNE': 'JUN',
            'AUGUST': 'AUG', 'SEPTEMBER': 'SEP', 'SEPT': 'SEP',
            'OCTOBER': 'OCT', 'NOVEMBER': 'NOV', 'DECEMBER': 'DEC'
        }
        
        for wrong, correct in month_corrections.items():
            x_upper = x_upper.replace(wrong, correct)
        
        x_upper = x_upper.strip()
        
        # Strategy 1: Try standard date formats with day, month, year (complete dates)
        match = re.match(r'(\d{1,2})[-/\s\.]+([A-Z]{3,9})[-/\s\.]+(\d{2,4})', x_upper, re.IGNORECASE)
        if match:
            day, month_str, year = match.groups()
            month_num = normalize_month(month_str)
            if month_num:
                if len(year) == 2:
                    year = '20' + year if int(year) <= 30 else '19' + year
                return f"{int(day):02d}/{month_num}/{year}"
        
        match = re.match(r'([A-Z]{3,9})[-/\s\.]+(\d{1,2})[-/\s\.]+(\d{2,4})', x_upper, re.IGNORECASE)
        if match:
            month_str, day, year = match.groups()
            month_num = normalize_month(month_str)
            if month_num:
                if len(year) == 2:
                    year = '20' + year if int(year) <= 30 else '19' + year
                return f"{int(day):02d}/{month_num}/{year}"
        
        match = re.match(r'(\d{4})[-/\s\.]+(\d{1,2})[-/\s\.]+(\d{1,2})', x_upper)
        if match:
            year, month, day = match.groups()
            return f"{int(day):02d}/{int(month):02d}/{year}"
        
        match = re.match(r'(\d{1,2})[-/\s\.]+(\d{1,2})[-/\s\.]+(\d{4})', x_upper)
        if match:
            day, month, year = match.groups()
            return f"{int(day):02d}/{int(month):02d}/{year}"
        
        match = re.match(r'(\d{4})(\d{2})(\d{2})', x_upper)
        if match:
            year, month, day = match.groups()
            return f"{int(day):02d}/{int(month):02d}/{year}"
        
        # Strategy 2: Handle compact formats with month names
        x_no_space = re.sub(r'\s+', '', x_upper)
        match = re.match(r'(\d{1,2})([A-Z]{3,9})(\d{2,4})', x_no_space, re.IGNORECASE)
        if match:
            day, month_str, year = match.groups()
            month_num = normalize_month(month_str)
            if month_num:
                if len(year) == 2:
                    year = '20' + year if int(year) <= 30 else '19' + year
                return f"{int(day):02d}/{month_num}/{year}"
        
        match = re.match(r'([A-Z]{3,9})(\d{1,2})(\d{2,4})', x_no_space, re.IGNORECASE)
        if match:
            month_str, day, year = match.groups()
            month_num = normalize_month(month_str)
            if month_num:
                if len(year) == 2:
                    year = '20' + year if int(year) <= 30 else '19' + year
                return f"{int(day):02d}/{month_num}/{year}"
        
        # Strategy 3: Handle year-only
        if re.match(r'^\d{4}$', x_upper):
            return f"{x_upper}"
        
        # Strategy 4: Try pandas to_datetime as last resort
        try:
            parsed = pd.to_datetime(original_x, dayfirst=True, errors='coerce')
            if pd.notnull(parsed):
                date_str = parsed.strftime('%d/%m/%Y')
                # Check if day is "00" in the parsed date
                if date_str.startswith('00/'):
                    return None
                return date_str
            
            parsed = pd.to_datetime(original_x, errors='coerce')
            if pd.notnull(parsed):
                date_str = parsed.strftime('%d/%m/%Y')
                # Check if day is "00" in the parsed date
                if date_str.startswith('00/'):
                    return None
                return date_str
        except:
            pass
        
        return None
    except:
        return None

def is_valid_date(date_str: str) -> bool:
    """Check if a date string is valid in dd/mm/yyyy format."""
    try:
        if not date_str or str(date_str).strip() == "" or str(date_str).lower() == 'nan' or str(date_str) == 'None':
            return False
        
        date_str = str(date_str).strip()
        
        if '/' in date_str and date_str.count('/') == 2:
            try:
                day, month, year = map(int, date_str.split('/'))
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                    if month in [4, 6, 9, 11] and day > 30:
                        return False
                    elif month == 2:
                        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                            if day > 29:
                                return False
                        elif day > 28:
                            return False
                    return True
            except:
                return False
        
        return False
    except:
        return False

def extract_date_components(date_str: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract day, month, year components from a date string."""
    try:
        if not date_str:
            return None, None, None
        
        date_str = str(date_str).strip()
        
        if is_valid_date(date_str):
            try:
                day, month, year = date_str.split('/')
                return int(day), int(month), int(year)
            except:
                pass
        
        if re.match(r'^\d{4}$', date_str):
            year = int(date_str)
            return None, None, year
        
        if '/' in date_str and date_str.count('/') == 1:
            parts = date_str.split('/')
            month_num = normalize_month(parts[0])
            if month_num:
                try:
                    year = int(parts[1])
                    if year < 100:
                        year = 2000 + year if year >= 23 else 1900 + year
                    return None, int(month_num), year
                except:
                    pass
            
            try:
                day = int(parts[0])
                month_num = normalize_month(parts[1])
                if month_num and 1 <= day <= 31:
                    return day, int(month_num), None
            except:
                pass
        
        numbers = re.findall(r'\d+', date_str)
        
        if len(numbers) >= 3:
            try:
                day = int(numbers[0])
                month = int(numbers[1])
                year = int(numbers[2])
                
                if year < 100:
                    year = 2000 + year if year >= 23 else 1900 + year
                
                return day, month, year
            except:
                pass
        
        if len(numbers) == 2:
            try:
                num1, num2 = int(numbers[0]), int(numbers[1])
            except:
                pass
        
        if len(numbers) == 1:
            try:
                num = int(numbers[0])
            except:
                pass
        
        month_num = normalize_month(date_str)
        if month_num:
            return None, int(month_num), None
        
        return None, None, None
    except:
        return None, None, None

def get_date_order(df: pd.DataFrame, date_col: str = 'XN Date') -> str:
    """Determine if dates are in ascending or descending order by analyzing UNIQUE valid dates."""
    try:
        valid_dates_set = set()
        valid_dates_info = []
        
        for idx, val in enumerate(df[date_col]):
            if is_valid_date(val):
                if val not in valid_dates_set:
                    valid_dates_set.add(val)
                    try:
                        day, month, year = map(int, val.split('/'))
                        valid_dates_info.append((idx, day, month, year, val))
                    except:
                        pass
        
        if len(valid_dates_info) < 2:
            return 'ascending'
        
        asc_matches = 0
        desc_matches = 0
        
        for i in range(1, len(valid_dates_info)):
            prev_idx, prev_day, prev_month, prev_year, prev_str = valid_dates_info[i-1]
            curr_idx, curr_day, curr_month, curr_year, curr_str = valid_dates_info[i]
            
            try:
                prev_date = datetime(prev_year, prev_month, prev_day)
                curr_date = datetime(curr_year, curr_month, curr_day)
                
                if prev_date <= curr_date:
                    asc_matches += 1
                else:
                    desc_matches += 1
            except:
                continue
        
        first_idx, first_day, first_month, first_year, first_str = valid_dates_info[0]
        last_idx, last_day, last_month, last_year, last_str = valid_dates_info[-1]
        
        try:
            first_date = datetime(first_year, first_month, first_day)
            last_date = datetime(last_year, last_month, last_day)
            
            if first_date <= last_date:
                asc_matches += 1
            else:
                desc_matches += 1
        except:
            pass
        
        if asc_matches >= desc_matches:
            return 'ascending'
        else:
            return 'descending'
    except Exception as e:
        return 'ascending'

def find_nearest_valid_date(df: pd.DataFrame, current_idx: int, direction: str, date_col: str = 'XN Date') -> Tuple[Optional[str], Optional[int]]:
    """Find the nearest valid date in the specified direction."""
    try:
        if direction == 'above':
            step = -1
            start = current_idx - 1
            end = -1
        else:
            step = 1
            start = current_idx + 1
            end = len(df)
        
        for idx in range(start, end, step):
            date_str = str(df.at[idx, date_col]).strip() if pd.notna(df.at[idx, date_col]) else ""
            if is_valid_date(date_str):
                return date_str, idx
        
        return None, None
    except:
        return None, None

def create_date_list_through_backtracking(df: pd.DataFrame, missing_indices: List[int], date_order: str, value_date_col: Optional[str] = None) -> List[Tuple[int, str]]:
    """Create a list of dates by backtracking from boundary dates."""
    results = []
    
    try:
        if not missing_indices:
            return results
        
        # Get boundary dates for chronological check
        before_date = None
        before_idx = missing_indices[0] - 1
        while before_idx >= 0:
            date_str = str(df.at[before_idx, 'XN Date']).strip() if pd.notna(df.at[before_idx, 'XN Date']) else ""
            if is_valid_date(date_str):
                before_date = date_str
                break
            before_idx -= 1
        
        after_date = None
        after_idx = missing_indices[-1] + 1
        while after_idx < len(df):
            date_str = str(df.at[after_idx, 'XN Date']).strip() if pd.notna(df.at[after_idx, 'XN Date']) else ""
            if is_valid_date(date_str):
                after_date = date_str
                break
            after_idx += 1
        
        # Check ValueDate for each missing index WITH chronological validation
        value_dates_found = []
        for idx in missing_indices:
            if value_date_col and value_date_col in df.columns:
                value_date_str = str(df.at[idx, value_date_col]).strip() if pd.notna(df.at[idx, value_date_col]) else ""
                if value_date_str and is_valid_date(value_date_str):
                    # Check chronological order BEFORE using ValueDate
                    if is_date_in_order(value_date_str, before_date, after_date, date_order):
                        results.append((idx, value_date_str))
                        value_dates_found.append(idx)
        
        # If ValueDate provided chronologically valid dates for all, return them
        if len(results) == len(missing_indices):
            return results
        
        # Otherwise, proceed with original backtracking logic
        remaining_indices = [idx for idx in missing_indices if idx not in value_dates_found]
        
        if not remaining_indices:
            return results
        
        if before_date and after_date and is_valid_date(before_date) and is_valid_date(after_date):
            try:
                b_day, b_month, b_year = map(int, before_date.split('/'))
                a_day, a_month, a_year = map(int, after_date.split('/'))
                
                before_dt = datetime(b_year, b_month, b_day)
                after_dt = datetime(a_year, a_month, a_day)
                
                if date_order == 'ascending':
                    total_days = (after_dt - before_dt).days
                    if total_days > 0:
                        step_size = total_days / (len(remaining_indices) + 1)
                        
                        for i, idx in enumerate(remaining_indices):
                            days_to_add = int((i + 1) * step_size)
                            current_dt = before_dt + timedelta(days=days_to_add)
                            new_date = current_dt.strftime('%d/%m/%Y')
                            if is_valid_date(new_date):
                                results.append((idx, new_date))
                    else:
                        current_dt = before_dt
                        for i, idx in enumerate(remaining_indices):
                            current_dt = current_dt + timedelta(days=1)
                            new_date = current_dt.strftime('%d/%m/%Y')
                            if is_valid_date(new_date):
                                results.append((idx, new_date))
                else:
                    total_days = (before_dt - after_dt).days
                    if total_days > 0:
                        step_size = total_days / (len(remaining_indices) + 1)
                        
                        for i, idx in enumerate(remaining_indices):
                            days_to_subtract = int((i + 1) * step_size)
                            current_dt = before_dt - timedelta(days=days_to_subtract)
                            new_date = current_dt.strftime('%d/%m/%Y')
                            if is_valid_date(new_date):
                                results.append((idx, new_date))
                    else:
                        current_dt = before_dt
                        for i, idx in enumerate(remaining_indices):
                            current_dt = current_dt - timedelta(days=1)
                            new_date = current_dt.strftime('%d/%m/%Y')
                            if is_valid_date(new_date):
                                results.append((idx, new_date))
            except Exception as e:
                if before_date and is_valid_date(before_date):
                    try:
                        b_day, b_month, b_year = map(int, before_date.split('/'))
                        current_dt = datetime(b_year, b_month, b_day)
                        
                        for i, idx in enumerate(remaining_indices):
                            if date_order == 'ascending':
                                current_dt = current_dt + timedelta(days=1)
                            else:
                                current_dt = current_dt - timedelta(days=1)
                            
                            new_date = current_dt.strftime('%d/%m/%Y')
                            if is_valid_date(new_date):
                                results.append((idx, new_date))
                    except:
                        pass
        
        elif before_date and is_valid_date(before_date):
            try:
                b_day, b_month, b_year = map(int, before_date.split('/'))
                current_dt = datetime(b_year, b_month, b_day)
                
                for i, idx in enumerate(remaining_indices):
                    if date_order == 'ascending':
                        current_dt = current_dt + timedelta(days=1)
                    else:
                        current_dt = current_dt - timedelta(days=1)
                    
                    new_date = current_dt.strftime('%d/%m/%Y')
                    if is_valid_date(new_date):
                        results.append((idx, new_date))
            except Exception as e:
                pass
        
        elif after_date and is_valid_date(after_date):
            try:
                a_day, a_month, a_year = map(int, after_date.split('/'))
                current_dt = datetime(a_year, a_month, a_day)
                
                for i, idx in enumerate(reversed(remaining_indices)):
                    if date_order == 'ascending':
                        current_dt = current_dt - timedelta(days=1)
                    else:
                        current_dt = current_dt + timedelta(days=1)
                    
                    new_date = current_dt.strftime('%d/%m/%Y')
                    if is_valid_date(new_date):
                        results.append((idx, new_date))
            except Exception as e:
                pass
        
        else:
            any_valid_date = None
            any_valid_idx = -1
            for idx in range(len(df)):
                date_str = str(df.at[idx, 'XN Date']).strip() if pd.notna(df.at[idx, 'XN Date']) else ""
                if is_valid_date(date_str):
                    any_valid_date = date_str
                    any_valid_idx = idx
                    break
            
            if any_valid_date:
                try:
                    day, month, year = map(int, any_valid_date.split('/'))
                    current_dt = datetime(year, month, day)
                    
                    if any_valid_idx < remaining_indices[0]:
                        for i, idx in enumerate(remaining_indices):
                            if date_order == 'ascending':
                                current_dt = current_dt + timedelta(days=(idx - any_valid_idx))
                            else:
                                current_dt = current_dt - timedelta(days=(idx - any_valid_idx))
                            
                            new_date = current_dt.strftime('%d/%m/%Y')
                            if is_valid_date(new_date):
                                results.append((idx, new_date))
                    else:
                        for i, idx in enumerate(reversed(remaining_indices)):
                            if date_order == 'ascending':
                                current_dt = current_dt - timedelta(days=(any_valid_idx - idx))
                            else:
                                current_dt = current_dt + timedelta(days=(any_valid_idx - idx))
                            
                            new_date = current_dt.strftime('%d/%m/%Y')
                            if is_valid_date(new_date):
                                results.append((idx, new_date))
                except Exception as e:
                    pass
        
        return results
    except Exception as e:
        return []

def get_valid_date_context(df: pd.DataFrame, current_idx: int, date_col: str = 'XN Date') -> Dict[str, Any]:
    """Get the context (previous and next valid dates) for a given row."""
    context = {'prev_date': None, 'prev_idx': None, 'next_date': None, 'next_idx': None}
    
    try:
        for idx in range(current_idx - 1, -1, -1):
            date_str = str(df.at[idx, date_col]).strip() if pd.notna(df.at[idx, date_col]) else ""
            if is_valid_date(date_str):
                context['prev_date'] = date_str
                context['prev_idx'] = idx
                break
        
        for idx in range(current_idx + 1, len(df)):
            date_str = str(df.at[idx, date_col]).strip() if pd.notna(df.at[idx, date_col]) else ""
            if is_valid_date(date_str):
                context['next_date'] = date_str
                context['next_idx'] = idx
                break
        
        return context
    except:
        return context

def is_date_in_order(date_str: str, prev_date: Optional[str], next_date: Optional[str], date_order: str) -> bool:
    """Check if a date is in correct order relative to previous and next dates."""
    try:
        if not is_valid_date(date_str):
            return False
        
        curr_day, curr_month, curr_year = map(int, date_str.split('/'))
        curr_dt = datetime(curr_year, curr_month, curr_day)
        
        if prev_date and is_valid_date(prev_date):
            prev_day, prev_month, prev_year = map(int, prev_date.split('/'))
            prev_dt = datetime(prev_year, prev_month, prev_day)
            
            if date_order == 'ascending' and curr_dt < prev_dt:
                return False
            elif date_order == 'descending' and curr_dt > prev_dt:
                return False
        
        if next_date and is_valid_date(next_date):
            next_day, next_month, next_year = map(int, next_date.split('/'))
            next_dt = datetime(next_year, next_month, next_day)
            
            if date_order == 'ascending' and curr_dt > next_dt:
                return False
            elif date_order == 'descending' and curr_dt < next_dt:
                return False
        
        return True
    except:
        return False

def is_empty_row(row: pd.Series) -> bool:
    """Check if a row is completely empty (page break indicator)."""
    try:
        for val in row.values:
            if pd.notna(val) and str(val).strip() != '':
                return False
        return True
    except:
        return True

def date_correction(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Enhanced date correction with systematic approach for missing dates and strict chronological ordering."""
    try:
        if 'XN Date' not in df.columns:
            return df, 'ascending'
        
        df = df.reset_index(drop=True)
        
        # Find ValueDate column if it exists
        value_date_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'value' in col_lower and 'date' in col_lower and col != 'XN Date':
                value_date_col = col
                break
        
        # Step 1: Parse all XN Dates first
        parsed_dates = []
        for idx in range(len(df)):
            if is_empty_row(df.iloc[idx]):
                parsed_dates.append("")
                continue
            
            xn_val = str(df.at[idx, 'XN Date']).strip() if pd.notna(df.at[idx, 'XN Date']) else ""
            parsed = parse_custom_date(xn_val)
            
            if parsed and is_valid_date(parsed):
                parsed_dates.append(parsed)
            elif parsed:
                parsed_dates.append(parsed)
            else:
                parsed_dates.append("")
        
        df['XN Date'] = parsed_dates
        
        # Step 2: Determine date order
        date_order = get_date_order(df, 'XN Date')
        
        # Step 3: Iterative chronological ordering fix
        # We'll iterate multiple times to ensure all dates are in order
        max_iterations = 5
        for iteration in range(max_iterations):
            made_correction = False
            corrected_dates = df['XN Date'].tolist()
            
            for idx in range(len(df)):
                if is_empty_row(df.iloc[idx]):
                    continue
                
                current_date = corrected_dates[idx]
                
                # Skip if empty
                if not current_date:
                    continue
                
                # Get previous valid date
                prev_date = None
                prev_idx = -1
                for i in range(idx-1, -1, -1):
                    if is_valid_date(corrected_dates[i]):
                        prev_date = corrected_dates[i]
                        prev_idx = i
                        break
                
                # Get next valid date
                next_date = None
                next_idx = -1
                for i in range(idx+1, len(corrected_dates)):
                    if is_valid_date(corrected_dates[i]):
                        next_date = corrected_dates[i]
                        next_idx = i
                        break
                
                # Check if current date is in chronological order
                if is_valid_date(current_date):
                    if not is_date_in_order(current_date, prev_date, next_date, date_order):
                        # Date is valid but out of order - need to correct
                        made_correction = True
                        
                        # STRICT RULE: If above and below dates are the same, use that date
                        if (prev_date and next_date and 
                            is_valid_date(prev_date) and is_valid_date(next_date) and
                            prev_date == next_date):
                            candidate = prev_date
                        else:
                            # Original logic for different above/below dates
                            candidate = None
                            
                            # Strategy 1: Try ValueDate first
                            if value_date_col and value_date_col in df.columns:
                                value_date_str = str(df.at[idx, value_date_col]).strip() if pd.notna(df.at[idx, value_date_col]) else ""
                                if value_date_str:
                                    parsed_value = parse_custom_date(value_date_str)
                                    if parsed_value and is_valid_date(parsed_value):
                                        if is_date_in_order(parsed_value, prev_date, next_date, date_order):
                                            candidate = parsed_value
                            
                            # Strategy 2: Predict from context
                            if not candidate:
                                if date_order == 'ascending':
                                    if prev_date and is_valid_date(prev_date):
                                        try:
                                            p_day, p_month, p_year = map(int, prev_date.split('/'))
                                            prev_dt = datetime(p_year, p_month, p_day)
                                            
                                            if next_date and is_valid_date(next_date):
                                                n_day, n_month, n_year = map(int, next_date.split('/'))
                                                next_dt = datetime(n_year, n_month, n_day)
                                                
                                                # Current date should be between prev and next
                                                if prev_dt < next_dt:
                                                    # Try to keep date close to original if possible
                                                    c_day, c_month, c_year = map(int, current_date.split('/'))
                                                    current_dt = datetime(c_year, c_month, c_day)
                                                    
                                                    # If current date is before previous date, set to day after previous
                                                    if current_dt < prev_dt:
                                                        candidate_dt = prev_dt + timedelta(days=1)
                                                        candidate = candidate_dt.strftime('%d/%m/%Y')
                                                    # If current date is after next date, set to day before next
                                                    elif current_dt > next_dt:
                                                        candidate_dt = next_dt - timedelta(days=1)
                                                        candidate = candidate_dt.strftime('%d/%m/%Y')
                                                    # If current date is same as a previous date (duplicate), increment by 1 day
                                                    elif current_dt == prev_dt:
                                                        # Only use +1 if above and below are NOT the same
                                                        if prev_date != next_date:
                                                            candidate_dt = prev_dt + timedelta(days=1)
                                                            candidate = candidate_dt.strftime('%d/%m/%Y')
                                                        else:
                                                            candidate = prev_date
                                                    else:
                                                        # Current date is between, but might be out of sequence with other dates
                                                        # Use mid-point approach
                                                        total_days = (next_dt - prev_dt).days
                                                        if 0 < total_days <= 100:
                                                            days_to_add = total_days // 2
                                                            candidate_dt = prev_dt + timedelta(days=days_to_add)
                                                            candidate = candidate_dt.strftime('%d/%m/%Y')
                                                        else:
                                                            # Only use +1 if above and below are NOT the same
                                                            if prev_date != next_date:
                                                                candidate_dt = prev_dt + timedelta(days=1)
                                                                candidate = candidate_dt.strftime('%d/%m/%Y')
                                                            else:
                                                                candidate = prev_date
                                                else:
                                                    # prev_dt >= next_dt, which shouldn't happen if dates are valid
                                                    # Only use +1 if above and below are NOT the same
                                                    if prev_date != next_date:
                                                        candidate_dt = prev_dt + timedelta(days=1)
                                                        candidate = candidate_dt.strftime('%d/%m/%Y')
                                                    else:
                                                        candidate = prev_date
                                            else:
                                                # Only previous date available
                                                # Only use +1 if we don't have same above/below constraint
                                                candidate_dt = prev_dt + timedelta(days=1)
                                                candidate = candidate_dt.strftime('%d/%m/%Y')
                                        except:
                                            # If any parsing error, use default logic
                                            pass
                                    
                                    elif next_date and is_valid_date(next_date):
                                        # Only next date available
                                        try:
                                            n_day, n_month, n_year = map(int, next_date.split('/'))
                                            next_dt = datetime(n_year, n_month, n_day)
                                            candidate_dt = next_dt - timedelta(days=1)
                                            candidate = candidate_dt.strftime('%d/%m/%Y')
                                        except:
                                            pass
                                
                                else:  # descending order
                                    if next_date and is_valid_date(next_date):
                                        try:
                                            n_day, n_month, n_year = map(int, next_date.split('/'))
                                            next_dt = datetime(n_year, n_month, n_day)
                                            
                                            if prev_date and is_valid_date(prev_date):
                                                p_day, p_month, p_year = map(int, prev_date.split('/'))
                                                prev_dt = datetime(p_year, p_month, p_day)
                                                
                                                if next_dt < prev_dt:
                                                    # Try to keep date close to original if possible
                                                    c_day, c_month, c_year = map(int, current_date.split('/'))
                                                    current_dt = datetime(c_year, c_month, c_day)
                                                    
                                                    # If current date is after previous date (should be before for descending)
                                                    if current_dt > prev_dt:
                                                        candidate_dt = prev_dt - timedelta(days=1)
                                                        candidate = candidate_dt.strftime('%d/%m/%Y')
                                                    # If current date is before next date (should be after for descending)
                                                    elif current_dt < next_dt:
                                                        candidate_dt = next_dt + timedelta(days=1)
                                                        candidate = candidate_dt.strftime('%d/%m/%Y')
                                                    # If duplicate
                                                    elif current_dt == prev_dt:
                                                        # Only use -1 if above and below are NOT the same
                                                        if prev_date != next_date:
                                                            candidate_dt = prev_dt - timedelta(days=1)
                                                            candidate = candidate_dt.strftime('%d/%m/%Y')
                                                        else:
                                                            candidate = prev_date
                                                    else:
                                                        total_days = (prev_dt - next_dt).days
                                                        if 0 < total_days <= 100:
                                                            days_to_add = total_days // 2
                                                            candidate_dt = next_dt + timedelta(days=days_to_add)
                                                            candidate = candidate_dt.strftime('%d/%m/%Y')
                                                        else:
                                                            # Only use +1 if above and below are NOT the same
                                                            if prev_date != next_date:
                                                                candidate_dt = next_dt + timedelta(days=1)
                                                                candidate = candidate_dt.strftime('%d/%m/%Y')
                                                            else:
                                                                candidate = next_date
                                                else:
                                                    # Only use +1 if above and below are NOT the same
                                                    if prev_date != next_date:
                                                        candidate_dt = next_dt + timedelta(days=1)
                                                        candidate = candidate_dt.strftime('%d/%m/%Y')
                                                    else:
                                                        candidate = next_date
                                            else:
                                                # Only next date available
                                                candidate_dt = next_dt + timedelta(days=1)
                                                candidate = candidate_dt.strftime('%d/%m/%Y')
                                        except:
                                            pass
                                    
                                    elif prev_date and is_valid_date(prev_date):
                                        try:
                                            p_day, p_month, p_year = map(int, prev_date.split('/'))
                                            prev_dt = datetime(p_year, p_month, p_day)
                                            candidate_dt = prev_dt - timedelta(days=1)
                                            candidate = candidate_dt.strftime('%d/%m/%Y')
                                        except:
                                            pass
                        
                        # Apply candidate if valid
                        if candidate and is_valid_date(candidate):
                            if is_date_in_order(candidate, prev_date, next_date, date_order):
                                corrected_dates[idx] = candidate
            
            # Update df with corrected dates
            df['XN Date'] = corrected_dates
            
            # If no corrections were made, break the loop
            if not made_correction:
                break
        
        # Step 4: Handle consecutive missing dates with backtracking - WITH SAME LOGIC
        remaining_invalid = []
        for idx in range(len(df)):
            if is_empty_row(df.iloc[idx]):
                continue
            
            if not is_valid_date(corrected_dates[idx]):
                remaining_invalid.append(idx)
        
        if remaining_invalid:
            backtrack_groups = []
            current_group = []
            
            for idx in remaining_invalid:
                if not current_group or idx == current_group[-1] + 1:
                    current_group.append(idx)
                else:
                    if current_group:
                        backtrack_groups.append(current_group.copy())
                    current_group = [idx]
            
            if current_group:
                backtrack_groups.append(current_group)
            
            for group in backtrack_groups:
                # Try to get dates from ValueDate column first WITH SAME LOGIC
                date_list = []
                for idx in group:
                    if value_date_col and value_date_col in df.columns:
                        value_date_str = str(df.at[idx, value_date_col]).strip() if pd.notna(df.at[idx, value_date_col]) else ""
                        if value_date_str:
                            parsed_value = parse_custom_date(value_date_str)
                            if parsed_value and is_valid_date(parsed_value):
                                # Get context for this date
                                prev_date = None
                                for i in range(idx-1, -1, -1):
                                    if is_valid_date(corrected_dates[i]):
                                        prev_date = corrected_dates[i]
                                        break
                                
                                next_date = None
                                for i in range(idx+1, len(corrected_dates)):
                                    if is_valid_date(corrected_dates[i]):
                                        next_date = corrected_dates[i]
                                        break
                                
                                # Apply SAME LOGIC: if above and below are same, use that date
                                if (prev_date and next_date and 
                                    is_valid_date(prev_date) and is_valid_date(next_date) and
                                    prev_date == next_date):
                                    date_list.append((idx, prev_date))
                                elif is_date_in_order(parsed_value, prev_date, next_date, date_order):
                                    date_list.append((idx, parsed_value))
                
                # If we found some ValueDates, apply them
                for row_idx, new_date in date_list:
                    corrected_dates[row_idx] = new_date
                
                # Update group with remaining indices
                remaining_in_group = [idx for idx in group if not is_valid_date(corrected_dates[idx])]
                
                if remaining_in_group:
                    # APPLY SAME LOGIC BEFORE BACKTRACKING: Check for same above/below dates
                    for idx in remaining_in_group[:]:  # Create a copy for safe removal
                        # Get above and below dates
                        prev_date = None
                        for i in range(idx-1, -1, -1):
                            if is_valid_date(corrected_dates[i]):
                                prev_date = corrected_dates[i]
                                break
                        
                        next_date = None
                        for i in range(idx+1, len(corrected_dates)):
                            if is_valid_date(corrected_dates[i]):
                                next_date = corrected_dates[i]
                                break
                        
                        # SAME LOGIC: If above and below are same, use that date
                        if (prev_date and next_date and 
                            is_valid_date(prev_date) and is_valid_date(next_date) and
                            prev_date == next_date):
                            corrected_dates[idx] = prev_date
                            remaining_in_group.remove(idx)
                    
                    # Use backtracking for remaining indices
                    if remaining_in_group:
                        backtrack_dates = create_date_list_through_backtracking(df, remaining_in_group, date_order, value_date_col)
                        
                        # APPLY SAME LOGIC TO BACKTRACKING RESULTS
                        for row_idx, new_date in backtrack_dates:
                            if is_valid_date(new_date):
                                # Get context for this date from corrected_dates
                                prev_date = None
                                for i in range(row_idx-1, -1, -1):
                                    if is_valid_date(corrected_dates[i]):
                                        prev_date = corrected_dates[i]
                                        break
                                
                                next_date = None
                                for i in range(row_idx+1, len(corrected_dates)):
                                    if is_valid_date(corrected_dates[i]):
                                        next_date = corrected_dates[i]
                                        break
                                
                                # SAME LOGIC: If above and below are same, use that date
                                if (prev_date and next_date and 
                                    is_valid_date(prev_date) and is_valid_date(next_date) and
                                    prev_date == next_date):
                                    corrected_dates[row_idx] = prev_date
                                else:
                                    corrected_dates[row_idx] = new_date
        
        df['XN Date'] = corrected_dates
        
        # Final verification pass
        corrected_dates = df['XN Date'].tolist()
        for idx in range(len(df)):
            if is_empty_row(df.iloc[idx]):
                continue
            
            current_date = corrected_dates[idx]
            if not current_date or not is_valid_date(current_date):
                continue
            
            # Get previous valid date
            prev_date = None
            for i in range(idx-1, -1, -1):
                if is_valid_date(corrected_dates[i]):
                    prev_date = corrected_dates[i]
                    break
            
            # Get next valid date
            next_date = None
            for i in range(idx+1, len(corrected_dates)):
                if is_valid_date(corrected_dates[i]):
                    next_date = corrected_dates[i]
                    break
            
            # Final check - if still out of order, use appropriate logic WITH SAME RULE
            if not is_date_in_order(current_date, prev_date, next_date, date_order):
                # STRICT RULE: If above and below dates are the same, use that date
                if (prev_date and next_date and 
                    is_valid_date(prev_date) and is_valid_date(next_date) and
                    prev_date == next_date):
                    corrected_dates[idx] = prev_date
                else:
                    # Use simple increment/decrement only when above and below are NOT the same
                    if date_order == 'ascending':
                        if prev_date and is_valid_date(prev_date):
                            try:
                                p_day, p_month, p_year = map(int, prev_date.split('/'))
                                prev_dt = datetime(p_year, p_month, p_day)
                                candidate_dt = prev_dt + timedelta(days=1)
                                corrected_dates[idx] = candidate_dt.strftime('%d/%m/%Y')
                            except:
                                pass
                    else:
                        if next_date and is_valid_date(next_date):
                            try:
                                n_day, n_month, n_year = map(int, next_date.split('/'))
                                next_dt = datetime(n_year, n_month, n_day)
                                candidate_dt = next_dt + timedelta(days=1)
                                corrected_dates[idx] = candidate_dt.strftime('%d/%m/%Y')
                            except:
                                pass
        
        df['XN Date'] = corrected_dates
        
        # Final order verification
        final_order = get_date_order(df, 'XN Date')
        
        gc.collect()
        return df, final_order
        
    except Exception as e:
        return df, 'ascending'
def fix_chronological_year_issues(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Fix year issues by ensuring chronological order in date sequences."""
    try:
        corrected_dates = df['XN Date'].copy().tolist()
        corrections_made = 0
        
        # Find ValueDate column for year correction
        value_date_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'value' in col_lower and 'date' in col_lower and col != 'XN Date':
                value_date_col = col
                break
        
        # Get chronological pattern
        valid_dates_info = []
        for i, date_str in enumerate(corrected_dates):
            if is_valid_date(date_str):
                try:
                    day, month, year = map(int, date_str.split('/'))
                    valid_dates_info.append((i, day, month, year, date_str))
                except:
                    pass
        
        if len(valid_dates_info) < 2:
            return df, 0
        
        # Determine chronological pattern
        date_order = 'ascending'
        asc_count = 0
        desc_count = 0
        
        for i in range(1, len(valid_dates_info)):
            _, day1, month1, year1, _ = valid_dates_info[i-1]
            _, day2, month2, year2, _ = valid_dates_info[i]
            
            try:
                date1 = datetime(year1, month1, day1)
                date2 = datetime(year2, month2, day2)
                
                if date1 <= date2:
                    asc_count += 1
                else:
                    desc_count += 1
            except:
                continue
        
        # Check first and last dates
        if len(valid_dates_info) >= 2:
            _, fd, fm, fy, _ = valid_dates_info[0]
            _, ld, lm, ly, _ = valid_dates_info[-1]
            try:
                first_date = datetime(fy, fm, fd)
                last_date = datetime(ly, lm, ld)
                if first_date <= last_date:
                    asc_count += 1
                else:
                    desc_count += 1
            except:
                pass
        
        date_order = 'ascending' if asc_count >= desc_count else 'descending'
        
        # Fix year anomalies with chronological validation
        for i in range(len(corrected_dates)):
            if not is_valid_date(corrected_dates[i]):
                continue
            
            try:
                day, month, year = map(int, corrected_dates[i].split('/'))
            except:
                continue
            
            # Get chronological context
            context = {'prev_date': None, 'next_date': None}
            
            # Find previous valid date
            for j in range(i-1, -1, -1):
                if is_valid_date(corrected_dates[j]):
                    context['prev_date'] = corrected_dates[j]
                    break
            
            # Find next valid date
            for j in range(i+1, len(corrected_dates)):
                if is_valid_date(corrected_dates[j]):
                    context['next_date'] = corrected_dates[j]
                    break
            
            # Check ValueDate for correct year WITH chronological validation
            if value_date_col and value_date_col in df.columns:
                value_date_str = str(df.at[i, value_date_col]).strip() if pd.notna(df.at[i, value_date_col]) else ""
                if value_date_str and is_valid_date(value_date_str):
                    try:
                        v_day, v_month, v_year = map(int, value_date_str.split('/'))
                        
                        # Only use ValueDate if year is different AND maintains chronological order
                        if v_year != year:
                            test_date = f"{day:02d}/{month:02d}/{v_year}"
                            
                            # Check chronological order BEFORE using ValueDate year
                            if is_date_in_order(test_date, context['prev_date'], context['next_date'], date_order):
                                corrected_dates[i] = test_date
                                corrections_made += 1
                                continue
                    except:
                        pass
            
            # Check for year anomalies using chronological logic
            if date_order == 'ascending' and context['prev_date']:
                try:
                    p_day, p_month, p_year = map(int, context['prev_date'].split('/'))
                    if year < p_year and abs(year - p_year) <= 5:
                        # Try to correct to previous year
                        test_date = f"{day:02d}/{month:02d}/{p_year}"
                        if is_date_in_order(test_date, context['prev_date'], context['next_date'], date_order):
                            corrected_dates[i] = test_date
                            corrections_made += 1
                except:
                    pass
        
        df['XN Date'] = corrected_dates
        
        gc.collect()
        return df, corrections_made
        
    except Exception as e:
        return df, 0

def rotate_if_descending(df: pd.DataFrame, date_order: str) -> Tuple[pd.DataFrame, str]:
    """Rotate dataframe if order is descending (to convert to ascending)."""
    try:
        if date_order == 'descending':
            df = df.iloc[::-1].reset_index(drop=True)
            new_date_order = get_date_order(df, 'XN Date')
            gc.collect()
            return df, 'ascending'
        
        return df, date_order
        
    except Exception as e:
        return df, date_order

# LOGGING FUNCTION - CSV format (NOT Excel)
def log_date_changes_to_csv(file_path, original_dates, processed_dates):
    """
    Log date changes to check_date directory CSV file.
    Appends new entries without replacing existing ones.
    """
    try:
        # Create check_date directory if it doesn't exist
        check_date_dir = "check_date"
        if not os.path.exists(check_date_dir):
            os.makedirs(check_date_dir)
        
        # Get base file name for logging
        base_filename = os.path.basename(file_path)
        log_file = os.path.join(check_date_dir, "date_log.csv")  # CSV, not Excel
        
        # Prepare log data
        log_data = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure both have the same length
        min_length = min(len(original_dates), len(processed_dates))
        
        for idx in range(min_length):
            # Get old and new dates
            old_val = original_dates.iloc[idx] if idx < len(original_dates) else ""
            new_val = processed_dates.iloc[idx] if idx < len(processed_dates) else ""
            
            # Convert to strings
            old_str = str(old_val) if pd.notna(old_val) and str(old_val).strip() not in ["", "nan", "None"] else ""
            new_str = str(new_val) if pd.notna(new_val) and str(new_val).strip() not in ["", "nan", "None"] else ""
            
            # Only log if there's a change or at least one has value
            if old_str != new_str or old_str or new_str:
                log_data.append({
                    'File_Name': base_filename,
                    'Row_Index': idx + 1,  # 1-based index for readability
                    'Old_Date': old_str,
                    'New_Date': new_str,
                    'Timestamp': timestamp
                })
        
        if not log_data:
            return
        
        # Create DataFrame from log data
        log_df = pd.DataFrame(log_data)
        
        # Check if CSV file exists
        if os.path.exists(log_file):
            try:
                # Read existing CSV
                existing_df = pd.read_csv(log_file)
                # Append new entries (DO NOT remove old ones)
                combined_df = pd.concat([existing_df, log_df], ignore_index=True)
            except Exception as e:
                combined_df = log_df
        else:
            combined_df = log_df
        
        # Save to CSV (NOT Excel)
        combined_df.to_csv(log_file, index=False)
        
    except Exception as e:
        pass

# MAIN FUNCTION - Modified to include logging parameter
def process_all_dates(df: pd.DataFrame, file_path=None, logging=True):
    """
    Process all dates in the dataframe with optional logging.
    
    Args:
        df: Input dataframe
        file_path: Source file name for logging (optional)
        logging: Boolean to enable/disable logging (default: True)
    
    Returns:
        Processed dataframe
    """
    try:
        original_dates = df['XN Date'].copy() if 'XN Date' in df.columns else pd.Series()
        
        # Step 1: Parse all date columns
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if not date_cols:
            return df
        
        for col in date_cols:
            df[col] = df[col].apply(parse_custom_date)
        
        # Step 2: Perform comprehensive date correction
        df, date_order = date_correction(df)
        
        # Store dates after correction but BEFORE rotation
        dates_after_correction = df['XN Date'].copy() if 'XN Date' in df.columns else pd.Series()
        
        # Step 3: Rotate dataframe if descending
        df, date_order = rotate_if_descending(df, date_order)
        
        # Step 4: Fix chronological year issues
        df, corrections_made = fix_chronological_year_issues(df)
        
        # Remove ValueDate column if it exists (after all processing is done)
        value_date_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'value' in col_lower and 'date' in col_lower and col != 'XN Date':
                value_date_col = col
                break
            
        
        if value_date_col and value_date_col in df.columns:
            df = df.drop(columns=[value_date_col])
        
        # Log date changes
        if file_path and logging and 'XN Date' in df.columns:
            try:
                if not original_dates.empty and not dates_after_correction.empty:
                    log_date_changes_to_csv(file_path, original_dates, dates_after_correction)
            except Exception as e:
                pass
        
        gc.collect()
        
        return df
        
    except Exception as e:
        return df