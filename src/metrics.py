import pandas as pd

def compute_weekly_route_metrics(shipments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups shipment records by origin-destination route, route_type, and week_of.
    Computes cost_per_tonne_km using ratio-of-sums:
        sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)
    
    Why Ratio-of-Sums:
    Ratio-of-sums represents the true network-level weighted economic unit cost,
    weighting each trip by its actual freight work (tonne-km) rather than treating
    a 1-tonne short haul identically to a 25-tonne long haul. It also avoids
    distortion from high-rate minimum-charge shipments and matches sample_output_format_v2.csv.
    """
    weekly = shipments_df.groupby(['route', 'route_type', 'week_of'], as_index=False).agg(
        total_freight_cost=('freight_cost_inr', 'sum'),
        total_tonne_km=('tonne_km', 'sum'),
        total_shipments=('shipment_id', 'count')
    )
    
    weekly['cost_per_tonne_km'] = weekly['total_freight_cost'] / weekly['total_tonne_km']
    weekly = weekly.sort_values(by=['route', 'week_of']).reset_index(drop=True)
    return weekly
