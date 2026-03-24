#!/usr/bin/env python
"""
Submit predictions to Kaggle competition.

This script handles submission to Kaggle and can be used standalone or by other scripts.

Usage:
    # Submit with default message
    python submit_to_kaggle.py submission.csv
    
    # Submit with custom message
    python submit_to_kaggle.py submission.csv --message "UrbanFloodNet v2 with autoregressive"
    
    # Dry run (verify without submitting)
    python submit_to_kaggle.py submission.csv --dry-run
    
    # Check recent submissions
    python submit_to_kaggle.py --check-submissions
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


def check_kaggle_api(args=None):
    """Check if kaggle API is installed and configured.

    Auth is considered configured if any of:
    1) KAGGLE_API_TOKEN env var is set (new preferred method), or
    2) KAGGLE_USERNAME and KAGGLE_KEY are both set (legacy env vars), or
    3) ~/.kaggle/access_token file exists (new token file), or
    4) kaggle.json exists in KAGGLE_CONFIG_DIR or ~/.kaggle (legacy).
    """
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            return False, "kaggle command not found"

        if args is not None:
            if getattr(args, 'kaggle_username', None):
                os.environ['KAGGLE_USERNAME'] = args.kaggle_username
            if getattr(args, 'kaggle_key', None):
                os.environ['KAGGLE_KEY'] = args.kaggle_key
            if getattr(args, 'kaggle_config_dir', None):
                os.environ['KAGGLE_CONFIG_DIR'] = args.kaggle_config_dir

        # Option A: new single-token env var
        if os.environ.get('KAGGLE_API_TOKEN'):
            return True, None

        # Option B: legacy env vars
        if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
            return True, None

        config_dir = os.environ.get('KAGGLE_CONFIG_DIR')
        kaggle_dir = Path(config_dir) if config_dir else (Path.home() / '.kaggle')

        # Option C: new access_token file
        if (kaggle_dir / 'access_token').exists():
            return True, None

        # Option D: legacy kaggle.json
        if (kaggle_dir / 'kaggle.json').exists():
            return True, None

        return False, (
            "Kaggle credentials not found. Options:\n"
            "  1) export KAGGLE_API_TOKEN=<token>  (from kaggle.com/settings → API)\n"
            "  2) Save token to ~/.kaggle/access_token\n"
            "  3) Legacy: save kaggle.json to ~/.kaggle/kaggle.json"
        )

    except FileNotFoundError:
        return False, "kaggle CLI not installed. Run: pip install kaggle"


def validate_submission(csv_path):
    """Validate submission CSV format."""
    import pandas as pd
    
    if not os.path.exists(csv_path):
        return False, f"File not found: {csv_path}"
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"Failed to read CSV: {e}"
    
    # Check required columns
    required_cols = {'row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level'}
    actual_cols = set(df.columns)
    
    if not required_cols.issubset(actual_cols):
        missing = required_cols - actual_cols
        return False, f"Missing columns: {missing}"
    
    # Check for NaN values
    if df['water_level'].isna().any():
        nan_count = df['water_level'].isna().sum()
        return False, f"Found {nan_count} NaN values in water_level"
    
    # Check data types
    try:
        df['row_id'].astype(int)
        df['model_id'].astype(int)
        df['event_id'].astype(int)
        df['node_id'].astype(int)
        df['water_level'].astype(float)
    except Exception as e:
        return False, f"Invalid data types: {e}"
    
    return True, None


def submit_to_kaggle(csv_path, message, competition='urban-flood-modelling', args=None):
    """Submit to Kaggle using kaggle API."""
    
    print(f"\n{'='*70}")
    print("KAGGLE SUBMISSION")
    print(f"{'='*70}")
    
    # Check API
    print(f"\n[1/5] Checking Kaggle API...")
    api_ok, api_error = check_kaggle_api(args)
    if not api_ok:
        print(f"[ERROR] {api_error}")
        return False
    print(f"[OK] Kaggle API is configured")
    
    # Validate submission
    print(f"\n[2/5] Validating submission file...")
    valid, error = validate_submission(csv_path)
    if not valid:
        print(f"[ERROR] {error}")
        return False
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"[OK] Submission valid")
    print(f"  Rows: {len(df)}")
    print(f"  Models: {sorted(df['model_id'].unique())}")
    print(f"  Events: {df['event_id'].nunique()}")
    print(f"  Water level range: [{df['water_level'].min():.6f}, {df['water_level'].max():.6f}]")
    
    # Confirm submission
    print(f"\n[3/5] Submission details:")
    print(f"  File: {csv_path}")
    print(f"  Competition: {competition}")
    print(f"  Message: {message}")
    print(f"  Size: {os.path.getsize(csv_path) / 1024 / 1024:.2f} MB")
    
    auto_yes = args is not None and getattr(args, 'yes', False)
    if auto_yes:
        print("\n  [INFO] --yes flag set, proceeding automatically")
    else:
        response = input(f"\n  Proceed with submission? (y/n): ").strip().lower()
        if response != 'y':
            print("[INFO] Submission cancelled")
            return False
    
    # Submit
    print(f"\n[4/5] Uploading to Kaggle...")
    try:
        cmd = [
            'kaggle', 'competitions', 'submit',
            '-c', competition,
            '-f', csv_path,
            '-m', message
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            print(f"[ERROR] Submission failed!")
            if result.stdout:
                print(f"[STDOUT] {result.stdout}")
            if result.stderr:
                print(f"[STDERR] {result.stderr}")
            return False
        
        print(f"[OK] Submission uploaded")
        
        # Parse submission output
        if "Successfully submitted" in result.stdout:
            print(f"[SUCCESS] Submission successful!")
        
        print(result.stdout)
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Submission timed out (>300s)")
        return False
    except Exception as e:
        print(f"[ERROR] Submission failed: {e}")
        return False
    
    # Check status
    print(f"\n[5/5] Checking submission status...")
    try:
        # Get recent submissions
        status_cmd = [
            'kaggle', 'competitions', 'submissions',
            '-c', competition,
            '-q'
        ]
        status_result = subprocess.run(status_cmd, capture_output=True, text=True, timeout=30)
        
        if status_result.returncode == 0:
            lines = status_result.stdout.strip().split('\n')
            if len(lines) >= 2:
                print(f"\n[RECENT SUBMISSIONS]")
                print(lines[0])  # Header
                for line in lines[1:4]:  # Show top 3
                    if line.strip():
                        print(line)
            else:
                print(f"[INFO] Could not parse submission status")
        
    except Exception as e:
        print(f"[WARN] Failed to check status: {e}")
    
    print(f"\n{'='*70}")
    print("SUBMISSION COMPLETE")
    print(f"{'='*70}\n")
    
    return True


def check_recent_submissions(competition='urban-flood-modelling'):
    """Check recent submissions for competition."""
    try:
        cmd = [
            'kaggle', 'competitions', 'submissions',
            '-c', competition
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"[ERROR] Failed to check submissions: {result.stderr}")
            return False
        
        lines = result.stdout.strip().split('\n')
        
        print(f"\n{'='*70}")
        print(f"RECENT SUBMISSIONS - {competition}")
        print(f"{'='*70}\n")
        
        if len(lines) > 0:
            print(lines[0])  # Header
            for line in lines[1:]:
                if line.strip():
                    print(line)
        else:
            print("[INFO] No submissions found")
        
        print(f"\n{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to check submissions: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Submit predictions to Kaggle competition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit with default message
  python submit_to_kaggle.py submission.csv
  
  # Submit with custom message
  python submit_to_kaggle.py submission.csv --message "v2 with dropout"
  
  # Dry run (validate without submitting)
  python submit_to_kaggle.py submission.csv --dry-run
  
  # Check recent submissions
  python submit_to_kaggle.py --check-submissions
        """
    )
    
    parser.add_argument('file', nargs='?', default=None,
                        help='Path to submission CSV file')
    parser.add_argument('--message', type=str, default='UrbanFloodNet submission',
                        help='Submission message (default: "UrbanFloodNet submission")')
    parser.add_argument('--competition', type=str, default='urban-flood-modelling',
                        help='Kaggle competition name')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate without submitting')
    parser.add_argument('--check-submissions', action='store_true',
                        help='Check recent submissions instead of submitting')
    parser.add_argument('--kaggle-username', type=str, default=None,
                        help='Kaggle username (alternative to kaggle.json)')
    parser.add_argument('--kaggle-key', type=str, default=None,
                        help='Kaggle API key (alternative to kaggle.json)')
    parser.add_argument('--kaggle-config-dir', type=str, default=None,
                        help='Directory containing kaggle.json (sets KAGGLE_CONFIG_DIR)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt (for use in pipelines)')
    
    args = parser.parse_args()
    
    if args.check_submissions:
        check_recent_submissions(args.competition)
        return
    
    if args.file is None:
        parser.print_help()
        print("\n[ERROR] Please provide a submission file path")
        sys.exit(1)
    
    if args.dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN - VALIDATION ONLY")
        print(f"{'='*70}\n")
        
        api_ok, api_error = check_kaggle_api(args)
        if not api_ok:
            print(f"[ERROR] {api_error}")
            print("[INFO] For env-based auth, set KAGGLE_USERNAME and KAGGLE_KEY.")
            sys.exit(1)

        valid, error = validate_submission(args.file)
        
        if valid:
            import pandas as pd
            df = pd.read_csv(args.file)
            print(f"[OK] Submission is valid")
            print(f"  File: {args.file}")
            print(f"  Rows: {len(df)}")
            print(f"  Models: {sorted(df['model_id'].unique())}")
            print(f"  Events: {df['event_id'].nunique()}")
            print(f"  Water level range: [{df['water_level'].min():.6f}, {df['water_level'].max():.6f}]")
            print(f"  Size: {os.path.getsize(args.file) / 1024 / 1024:.2f} MB")
        else:
            print(f"[ERROR] {error}")
            sys.exit(1)
        
        print(f"\n{'='*70}\n")
        return
    
    # Submit
    success = submit_to_kaggle(args.file, args.message, args.competition, args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
