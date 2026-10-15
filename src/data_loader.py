import pandas as pd
from typing import Tuple

def load_shipments(file_path: str) -> pd.DataFrame:
    """
    Loads and validates shipment records from CSV.
    Derives route, week_of (Monday of Mon-Sun week), and tonne_km.
    """
    df = pd.read_csv(file_path)
    required_cols = [
        'shipment_id', 'origin', 'destination', 'route_type',
        'material', 'quantity_tonnes', 'distance_km',
        'freight_cost_inr', 'shipment_date', 'transporter'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"shipment_records missing required columns: {missing}")
        
    df['shipment_date'] = pd.to_datetime(df['shipment_date'])
    df['route'] = df['origin'].str.strip() + '-' + df['destination'].str.strip()
    df['route_type'] = df['route_type'].str.strip()
    
    # Monday of Mon-Sun week:
    # weekday() is 0 for Monday, 6 for Sunday
    df['week_of'] = df['shipment_date'].apply(lambda d: d - pd.Timedelta(days=d.weekday()))
    
    # Work done: quantity in tonnes * distance in km
    df['tonne_km'] = df['quantity_tonnes'] * df['distance_km']
    
    return df

def load_context_notes(file_path: str) -> pd.DataFrame:
    """
    Loads and validates context notes from CSV.
    """
    df = pd.read_csv(file_path)
    required_cols = ['note_id', 'date', 'applies_to', 'note']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"context_notes missing required columns: {missing}")
        
    df['date'] = pd.to_datetime(df['date'])
    df['note_id'] = df['note_id'].str.strip()
    df['applies_to'] = df['applies_to'].str.strip()
    df['note'] = df['note'].str.strip()
    
    return df
