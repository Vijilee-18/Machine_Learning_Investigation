import os
import zipfile
import tarfile
import re
from email import policy
from email.parser import BytesParser
import pandas as pd
import numpy as np

def extract_email_text(raw_bytes):
    """
    Parses RFC 822 / MIME email bytes to extract subject and body text.
    Handles multipart emails, HTML-to-text fallback, and encoding edge cases.
    """
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        subject = msg['subject'] or ""
        
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition', ''))
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    try:
                        content = part.get_content()
                        if isinstance(content, str):
                            body_parts.append(content)
                    except Exception:
                        pass
                elif ctype == 'text/html' and 'attachment' not in cdispo and not body_parts:
                    try:
                        content = part.get_content()
                        if isinstance(content, str):
                            body_parts.append(content)
                    except Exception:
                        pass
        else:
            try:
                content = msg.get_content()
                if isinstance(content, str):
                    body_parts.append(content)
            except Exception:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_parts.append(payload.decode('utf-8', errors='ignore'))
                    
        body = " ".join(body_parts)
        full_text = f"{subject} {body}".strip()
        return full_text if full_text else ""
    except Exception:
        # Fallback raw decoding if parser fails
        return raw_bytes.decode('utf-8', errors='ignore')

def load_enron_dataset(zip_path="enron_spam_data.zip"):
    """
    Loads Enron spam dataset from zip archive.
    Returns DataFrame with columns ['text', 'label', 'dataset_name']
    label: 1 for spam, 0 for ham.
    """
    print(f"Loading Enron dataset from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('enron_spam_data.csv') as f:
            df = pd.read_csv(f)
            
    df['Subject'] = df['Subject'].fillna('')
    df['Message'] = df['Message'].fillna('')
    df['text'] = df['Subject'] + " " + df['Message']
    df['label'] = (df['Spam/Ham'].str.lower().str.strip() == 'spam').astype(int)
    df['dataset_name'] = 'Enron'
    
    # Filter empty texts
    df = df[df['text'].str.strip() != ''].copy()
    print(f"Enron loaded: {len(df)} samples ({df['label'].value_counts().to_dict()})")
    return df[['text', 'label', 'dataset_name']]

def load_spamassassin_dataset(ham_tar="20030228_easy_ham_2.tar.bz2", spam_tar="20030228_spam_2.tar.bz2"):
    """
    Extracts and parses SpamAssassin easy_ham_2 and spam_2 archives.
    Returns DataFrame with columns ['text', 'label', 'dataset_name']
    label: 1 for spam, 0 for ham.
    """
    print(f"Loading SpamAssassin dataset from {ham_tar} and {spam_tar}...")
    records = []
    
    # Process easy_ham_2 (label = 0)
    with tarfile.open(ham_tar, 'r:bz2') as t:
        for member in t.getmembers():
            if member.isfile():
                f = t.extractfile(member)
                if f:
                    content_bytes = f.read()
                    text = extract_email_text(content_bytes)
                    if text.strip():
                        records.append({
                            'text': text,
                            'label': 0,
                            'dataset_name': 'SpamAssassin_EasyHam2'
                        })
                        
    # Process spam_2 (label = 1)
    with tarfile.open(spam_tar, 'r:bz2') as t:
        for member in t.getmembers():
            if member.isfile():
                f = t.extractfile(member)
                if f:
                    content_bytes = f.read()
                    text = extract_email_text(content_bytes)
                    if text.strip():
                        records.append({
                            'text': text,
                            'label': 1,
                            'dataset_name': 'SpamAssassin_Spam2'
                        })
                        
    df = pd.DataFrame(records)
    df['dataset_name_combined'] = 'SpamAssassin'
    print(f"SpamAssassin loaded: {len(df)} samples ({df['label'].value_counts().to_dict()})")
    return df

if __name__ == '__main__':
    enron_df = load_enron_dataset()
    sa_df = load_spamassassin_dataset()
    print("DataLoader test completed successfully.")
