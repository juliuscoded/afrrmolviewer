# afrrenergybids.py
# Streamlit app for aFRR Energy and Capacity Bids Merit Order List Visualization
# Usage: pip install -r requirements.txt && streamlit run afrrenergybids.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import io
import tempfile
import time

# ----------------------
# Download Functions
# ----------------------
def download_afrr_data(date=None, market='ENERGY'):
    """
    Download aFRR data from regelleistung.net
    
    Parameters:
    -----------
    date : str, optional
        Date in format 'YYYY-MM-DD', default is today's date
    market : str
        Market type: 'ENERGY' or 'CAPACITY'
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Construct URL based on market type
    if market == 'ENERGY':
        url = f"https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/anonymousresults?&productTypes=aFRR&market=ENERGY&exportFormat=xlsx&date={date}&countryCodeA2=DE"
    elif market == 'CAPACITY':
        url = f"https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/anonymousresults?&productTypes=aFRR&market=CAPACITY&exportFormat=xlsx&date={date}&countryCodeA2=DE"
    else:
        st.error(f"Invalid market type: {market}")
        return None
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content))
        df.columns = [col.strip() for col in df.columns]
        
        # Add market type to dataframe for later processing
        df['MARKET_TYPE'] = market
        df['DOWNLOAD_DATE'] = date
        
        return df
    except Exception as e:
        st.error(f"Error downloading {market} data for {date}: {e}")
        return None

# ----------------------
# Batch Download Functions for Long-term Analysis
# ----------------------
def download_afrr_data_batch(start_date, end_date, market='ENERGY', progress_bar=None):
    """
    Download aFRR data for a date range with progress tracking.
    
    Parameters:
    -----------
    start_date : datetime.date
        Start date for the range
    end_date : datetime.date
        End date for the range
    market : str
        Market type: 'ENERGY' or 'CAPACITY'
    progress_bar : streamlit.progress_bar, optional
        Progress bar to update
        
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame for the date range
    """
    all_data = []
    current_date = start_date
    total_days = (end_date - start_date).days + 1
    successful_downloads = 0
    failed_downloads = 0
    
    st.write(f"🔄 Starting batch download for {market} market from {start_date} to {end_date}")
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        if progress_bar:
            progress_bar.progress((current_date - start_date).days / total_days, 
                                text=f"Downloading {market} data for {date_str}...")
        
        # Add debug info
        st.write(f"📥 Attempting download for {date_str}...")
        
        try:
            df = download_afrr_data(date_str, market)
            if df is not None:
                all_data.append(df)
                successful_downloads += 1
                st.write(f"✅ Successfully downloaded {len(df)} records for {date_str}")
            else:
                failed_downloads += 1
                st.write(f"❌ No data returned for {date_str}")
        except Exception as e:
            failed_downloads += 1
            st.write(f"❌ Error downloading {date_str}: {e}")
        
        current_date += timedelta(days=1)
        time.sleep(0.5)  # Small delay to be respectful to the API
    
    st.write(f"📊 Download summary: {successful_downloads} successful, {failed_downloads} failed")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        st.write(f"🎉 Combined {len(combined_df)} total records from {len(all_data)} days")
        return combined_df
    else:
        st.error("❌ No data was successfully downloaded")
        return None

def get_date_range_suggestion(days_back):
    """
    Get a suggested date range for analysis.
    
    Parameters:
    -----------
    days_back : int
        Number of days to go back from today
        
    Returns:
    --------
    tuple
        (start_date, end_date)
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date

# ----------------------
# Load Local Excel File
# ----------------------
def load_local_excel(file_path, market_type='ENERGY'):
    """
    Load aFRR data from a local Excel file.
    
    Parameters:
    -----------
    file_path : str or file object
        Path to the Excel file or file object
    market_type : str
        Market type: 'ENERGY' or 'CAPACITY'
    """
    try:
        df = pd.read_excel(file_path)
        df.columns = [col.strip() for col in df.columns]
        df['MARKET_TYPE'] = market_type
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ----------------------
# Process Data
# ----------------------
def process_data(df):
    """
    Process the aFRR data to prepare it for visualization.
    Handles both ENERGY and CAPACITY markets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the aFRR data
        
    Returns:
    --------
    tuple
        (processed_df, unique_products)
    """
    if df is None:
        return None, None
    
    processed_df = df.copy()
    
    # Process based on market type
    if 'MARKET_TYPE' not in processed_df.columns:
        # Try to infer market type from columns
        if 'ENERGY_PRICE_[EUR/MWh]' in processed_df.columns:
            processed_df['MARKET_TYPE'] = 'ENERGY'
        elif 'CAPACITY_PRICE_[(EUR/MW)/h]' in processed_df.columns:
            processed_df['MARKET_TYPE'] = 'CAPACITY'
        else:
            st.error("Cannot determine market type from data columns")
            return None, None
    
    # Process ENERGY market data
    if processed_df['MARKET_TYPE'].iloc[0] == 'ENERGY':
        # Add signed price column based on ENERGY_PRICE_PAYMENT_DIRECTION and product type
        processed_df['SIGNED_PRICE'] = processed_df.apply(
            lambda row: row['ENERGY_PRICE_[EUR/MWh]'] if ('NEG' in row['PRODUCT'] and row['ENERGY_PRICE_PAYMENT_DIRECTION'] == 'PROVIDER_TO_GRID') or 
                                                       ('POS' in row['PRODUCT'] and row['ENERGY_PRICE_PAYMENT_DIRECTION'] == 'GRID_TO_PROVIDER')
                       else -row['ENERGY_PRICE_[EUR/MWh]'], 
            axis=1
        )
        processed_df['CAPACITY_COL'] = processed_df['ALLOCATED_CAPACITY_[MW]']
        processed_df['PRICE_UNIT'] = '€/MWh'
        processed_df['CAPACITY_UNIT'] = 'MW'
    
    # Process CAPACITY market data
    elif processed_df['MARKET_TYPE'].iloc[0] == 'CAPACITY':
        # For capacity, price is already signed (positive for POS, negative for NEG)
        processed_df['SIGNED_PRICE'] = processed_df['CAPACITY_PRICE_[(EUR/MW)/h]']
        processed_df['CAPACITY_COL'] = processed_df['ALLOCATED_CAPACITY_[MW]']
        processed_df['PRICE_UNIT'] = '€/MW/h'
        processed_df['CAPACITY_UNIT'] = 'MW'
    
    # Get unique products
    unique_products = processed_df['PRODUCT'].unique()
    
    return processed_df, unique_products

# ----------------------
# Plot Merit Order List for a Product
# ----------------------
def plot_merit_order_list(df, product):
    """
    Plot the Merit Order List for a specific product.
    Works for both ENERGY and CAPACITY markets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed DataFrame containing the aFRR data
    product : str
        The product to plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if df is None:
        return None
    
    # Filter data for the selected product
    filtered_df = df[df['PRODUCT'] == product]
    
    if len(filtered_df) == 0:
        st.warning(f"No data available for product {product}")
        return None
    
    # Get market type and units
    market_type = filtered_df['MARKET_TYPE'].iloc[0]
    price_unit = filtered_df['PRICE_UNIT'].iloc[0]
    capacity_unit = filtered_df['CAPACITY_UNIT'].iloc[0]
    
    # Determine sorting direction based on product type
    is_negative = 'NEG' in product
    
    # Corrected sorting logic based on user feedback
    if market_type == 'CAPACITY' and is_negative:
        ascending = True  # For NEG Capacity, sort ascending to have highest price on the right
    else:
        ascending = not is_negative  # Default: POS ascending, NEG descending
    
    # Sort by price
    sorted_df = filtered_df.sort_values('SIGNED_PRICE', ascending=ascending)
    sorted_df['CUMULATIVE_CAPACITY'] = sorted_df['CAPACITY_COL'].cumsum()
    
    # Create the figure
    fig = go.Figure()
    
    # Use a color from Plotly's qualitative palette
    color = px.colors.qualitative.Plotly[0] if not is_negative else px.colors.qualitative.Plotly[1]
    
    # Add step line for merit order
    x_values = [0] + list(sorted_df['CUMULATIVE_CAPACITY'])
    y_values = [sorted_df['SIGNED_PRICE'].iloc[0]] + list(sorted_df['SIGNED_PRICE'])
    
    # Speed optimization: Use a filled area chart instead of looping to draw shapes
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        line=dict(color='black', width=2, shape='hv'),
        fill='tozeroy',
        fillcolor=color,
        opacity=0.7,
        name='Merit Order',
        hoverinfo='text',
        hovertext=[f"Price: {y:.2f} {price_unit}<br>Capacity: {x:.2f} {capacity_unit}" for x, y in zip(x_values, y_values)]
    ))
    
    # Add points for individual bids
    fig.add_trace(go.Scatter(
        x=sorted_df['CUMULATIVE_CAPACITY'],
        y=sorted_df['SIGNED_PRICE'],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Bids',
        hoverinfo='text',
        hovertext=[f"Price: {row['SIGNED_PRICE']:.2f} {price_unit}<br>Capacity: {row['CAPACITY_COL']:.2f} {capacity_unit}" for _, row in sorted_df.iterrows()]
    ))
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=0,
        x1=sorted_df['CUMULATIVE_CAPACITY'].max(),
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Set layout
    market_type_full = "Negative" if is_negative else "Positive"
    fig.update_layout(
        title=f"Merit Order List for {product} - {market_type_full} aFRR {market_type} Market",
        xaxis_title=f"Cumulative Capacity ({capacity_unit})",
        yaxis_title=f"Price ({price_unit})",
        height=600,
        width=1000,
        hovermode="closest",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white"
    )
    
    return fig

# ----------------------
# Plot All Merit Order Curves for a Day
# ----------------------
def plot_daily_merit_order_curves(processed_df, market_type='NEG', data_market='ENERGY'):
    """
    Plot all merit order curves for a given day and market type.
    
    Parameters:
    -----------
    processed_df : pandas.DataFrame
        Processed DataFrame containing the aFRR data
    market_type : str
        Market type: 'NEG' or 'POS'
    data_market : str
        Data market: 'ENERGY' or 'CAPACITY'
    """
    if processed_df is None:
        return None
    
    # Get all unique products and sort them
    products = sorted([p for p in processed_df['PRODUCT'].unique() if market_type in p])
    
    if len(products) == 0:
        st.warning(f"No {market_type} products found for {data_market} market")
        return None
    
    fig = go.Figure()
    
    # Use a color scale with more range
    color_scale = px.colors.sequential.Viridis
    n_colors = len(color_scale)
    
    # Get units from data
    price_unit = processed_df['PRICE_UNIT'].iloc[0]
    capacity_unit = processed_df['CAPACITY_UNIT'].iloc[0]
    
    for i, product in enumerate(products):
        product_df = processed_df[processed_df['PRODUCT'] == product]
        if len(product_df) == 0:
            continue
        
        # Special case: For CAPACITY NEG, always sort descending
        if data_market == 'CAPACITY' and market_type == 'NEG':
            ascending = True
        else:
            ascending = (market_type == 'POS')
        sorted_df = product_df.sort_values('SIGNED_PRICE', ascending=ascending)
        sorted_df['CUMULATIVE_CAPACITY'] = sorted_df['CAPACITY_COL'].cumsum()
        
        x_values = [0] + list(sorted_df['CUMULATIVE_CAPACITY'])
        y_values = [sorted_df['SIGNED_PRICE'].iloc[0]] + list(sorted_df['SIGNED_PRICE'])
        color = color_scale[i % n_colors]
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color=color, width=1.5, shape='hv'),
            name=f"Period {i+1}",
            hovertext=[f"Period: {i+1}<br>Price: {y:.2f} {price_unit}<br>Capacity: {x:.2f} {capacity_unit}" for x, y in zip(x_values, y_values)],
            hoverinfo='text',
            showlegend=False
        ))
    
    # Add a dummy scatter for colorbar legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=color_scale,
            cmin=1,
            cmax=len(products),
            colorbar=dict(
                title="Period",
                tickvals=[1, len(products)//4, len(products)//2, 3*len(products)//4, len(products)],
                ticktext=["00:00", "06:00", "12:00", "18:00", "24:00"],
                len=0.75,
                y=0.5,
                yanchor='middle',
                x=1.02,
                xanchor='left',
            ),
            size=0.1,
            color=[i+1 for i in range(len(products))],
            showscale=True
        ),
        hoverinfo='none',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"{'Negative' if market_type=='NEG' else 'Positive'} aFRR {data_market} Merit Order Curves (All Periods)",
        xaxis_title=f"Cumulative Capacity ({capacity_unit})",
        yaxis_title=f"Price ({price_unit})",
        height=700,
        width=1000,
        template="plotly_white"
    )
    
    return fig

# ----------------------
# Long-term Analysis Functions
# ----------------------
def analyze_long_term_trends(df, market_type='NEG', data_market='ENERGY'):
    """
    Analyze long-term trends in the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Combined DataFrame for the date range
    market_type : str
        Market type: 'NEG' or 'POS'
    data_market : str
        Data market: 'ENERGY' or 'CAPACITY'
        
    Returns:
    --------
    dict
        Dictionary containing trend analysis results
    """
    if df is None or len(df) == 0:
        return None
    
    # Filter for the specified market type
    filtered_df = df[df['PRODUCT'].str.contains(market_type, na=False)].copy()
    
    if len(filtered_df) == 0:
        return None
    
    # Process the data
    processed_df, _ = process_data(filtered_df)
    if processed_df is None:
        return None
    
    # Extract date from DOWNLOAD_DATE
    processed_df['DATE'] = pd.to_datetime(processed_df['DOWNLOAD_DATE'])
    
    # Daily aggregations
    daily_stats = processed_df.groupby('DATE').agg({
        'SIGNED_PRICE': ['mean', 'std', 'min', 'max', 'count'],
        'CAPACITY_COL': ['sum', 'mean'],
        'PRODUCT': 'nunique'
    }).round(2)
    
    # Flatten column names
    daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns]
    daily_stats = daily_stats.reset_index()
    
    # Overall statistics
    overall_stats = {
        'total_records': len(processed_df),
        'date_range': f"{processed_df['DATE'].min().strftime('%Y-%m-%d')} to {processed_df['DATE'].max().strftime('%Y-%m-%d')}",
        'avg_price': processed_df['SIGNED_PRICE'].mean(),
        'price_std': processed_df['SIGNED_PRICE'].std(),
        'min_price': processed_df['SIGNED_PRICE'].min(),
        'max_price': processed_df['SIGNED_PRICE'].max(),
        'total_capacity': processed_df['CAPACITY_COL'].sum(),
        'avg_capacity_per_day': processed_df.groupby('DATE')['CAPACITY_COL'].sum().mean(),
        'unique_products': processed_df['PRODUCT'].nunique()
    }
    
    return {
        'daily_stats': daily_stats,
        'overall_stats': overall_stats,
        'processed_df': processed_df
    }

def plot_price_trends(daily_stats, market_type='NEG', data_market='ENERGY'):
    """
    Plot price trends over time.
    
    Parameters:
    -----------
    daily_stats : pandas.DataFrame
        Daily statistics DataFrame
    market_type : str
        Market type: 'NEG' or 'POS'
    data_market : str
        Data market: 'ENERGY' or 'CAPACITY'
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Price unit based on market
    price_unit = '€/MWh' if data_market == 'ENERGY' else '€/MW/h'
    
    # Add mean price line
    fig.add_trace(go.Scatter(
        x=daily_stats['DATE'],
        y=daily_stats['SIGNED_PRICE_mean'],
        mode='lines+markers',
        name='Mean Price',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add price range (min to max)
    fig.add_trace(go.Scatter(
        x=daily_stats['DATE'].tolist() + daily_stats['DATE'].tolist()[::-1],
        y=daily_stats['SIGNED_PRICE_max'].tolist() + daily_stats['SIGNED_PRICE_min'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Price Range',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f"{'Negative' if market_type=='NEG' else 'Positive'} aFRR {data_market} Price Trends",
        xaxis_title="Date",
        yaxis_title=f"Price ({price_unit})",
        height=500,
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def plot_capacity_trends(daily_stats, market_type='NEG', data_market='ENERGY'):
    """
    Plot capacity trends over time.
    
    Parameters:
    -----------
    daily_stats : pandas.DataFrame
        Daily statistics DataFrame
    market_type : str
        Market type: 'NEG' or 'POS'
    data_market : str
        Data market: 'ENERGY' or 'CAPACITY'
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Add total capacity line
    fig.add_trace(go.Scatter(
        x=daily_stats['DATE'],
        y=daily_stats['CAPACITY_COL_sum'],
        mode='lines+markers',
        name='Total Capacity',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"{'Negative' if market_type=='NEG' else 'Positive'} aFRR {data_market} Capacity Trends",
        xaxis_title="Date",
        yaxis_title="Total Capacity (MW)",
        height=500,
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig

def plot_average_merit_order(processed_df, market_type='NEG', data_market='ENERGY'):
    """
    Plot an average merit order curve based on aggregated data.
    
    Parameters:
    -----------
    processed_df : pandas.DataFrame
        Processed DataFrame
    market_type : str
        Market type: 'NEG' or 'POS'
    data_market : str
        Data market: 'ENERGY' or 'CAPACITY'
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if processed_df is None or len(processed_df) == 0:
        return None
    
    # Filter for the specified market type
    filtered_df = processed_df[processed_df['PRODUCT'].str.contains(market_type, na=False)]
    
    if len(filtered_df) == 0:
        return None
    
    # Get units
    price_unit = filtered_df['PRICE_UNIT'].iloc[0]
    capacity_unit = filtered_df['CAPACITY_UNIT'].iloc[0]
    
    # Sort by price
    ascending = (market_type == 'POS')
    sorted_df = filtered_df.sort_values('SIGNED_PRICE', ascending=ascending)
    sorted_df['CUMULATIVE_CAPACITY'] = sorted_df['CAPACITY_COL'].cumsum()
    
    # Create the figure
    fig = go.Figure()
    
    # Use a color from Plotly's qualitative palette
    color = px.colors.qualitative.Plotly[0] if not (market_type == 'NEG') else px.colors.qualitative.Plotly[1]
    
    # Add step line for merit order
    x_values = [0] + list(sorted_df['CUMULATIVE_CAPACITY'])
    y_values = [sorted_df['SIGNED_PRICE'].iloc[0]] + list(sorted_df['SIGNED_PRICE'])
    
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        line=dict(color='black', width=2, shape='hv'),
        fill='tozeroy',
        fillcolor=color,
        opacity=0.7,
        name='Average Merit Order',
        hoverinfo='text',
        hovertext=[f"Price: {y:.2f} {price_unit}<br>Capacity: {x:.2f} {capacity_unit}" for x, y in zip(x_values, y_values)]
    ))
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=0,
        x1=sorted_df['CUMULATIVE_CAPACITY'].max(),
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Set layout
    market_type_full = "Negative" if market_type == 'NEG' else "Positive"
    fig.update_layout(
        title=f"Average Merit Order List - {market_type_full} aFRR {data_market} Market",
        xaxis_title=f"Cumulative Capacity ({capacity_unit})",
        yaxis_title=f"Price ({price_unit})",
        height=600,
        width=1000,
        hovermode="closest",
        legend=dict(x=0.01, y=0.99),
        template="plotly_white"
    )
    
    return fig

# ----------------------
# Plot All Daily MOLs for a Product (Long-term)
# ----------------------
def plot_daily_mols_for_product(processed_df, product, data_market='ENERGY'):
    """
    Plot all daily MOLs for a selected product over a date range.
    Each curve is for one day.
    """
    if processed_df is None or len(processed_df) == 0:
        return None
    
    # Filter for the selected product
    filtered_df = processed_df[processed_df['PRODUCT'] == product].copy()
    if len(filtered_df) == 0:
        return None
    
    # Ensure date column
    if 'DATE' not in filtered_df.columns:
        filtered_df['DATE'] = pd.to_datetime(filtered_df['DOWNLOAD_DATE'])
    
    # Get units
    price_unit = filtered_df['PRICE_UNIT'].iloc[0]
    capacity_unit = filtered_df['CAPACITY_UNIT'].iloc[0]
    
    # Get all unique days
    days = sorted(filtered_df['DATE'].dt.date.unique())
    if len(days) == 0:
        return None
    
    fig = go.Figure()
    color_scale = px.colors.sequential.Viridis
    n_colors = len(color_scale)
    
    for i, day in enumerate(days):
        day_df = filtered_df[filtered_df['DATE'].dt.date == day]
        if len(day_df) == 0:
            continue
        # Sort by price (direction depends on product type)
        is_negative = 'NEG' in product
        ascending = (data_market == 'CAPACITY' and is_negative) or (not is_negative)
        sorted_df = day_df.sort_values('SIGNED_PRICE', ascending=ascending)
        sorted_df['CUMULATIVE_CAPACITY'] = sorted_df['CAPACITY_COL'].cumsum()
        x_values = [0] + list(sorted_df['CUMULATIVE_CAPACITY'])
        y_values = [sorted_df['SIGNED_PRICE'].iloc[0]] + list(sorted_df['SIGNED_PRICE'])
        color = color_scale[i % n_colors]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color=color, width=2, shape='hv'),
            name=str(day),
            hoverinfo='text',
            hovertext=[f"Date: {day}<br>Price: {y:.2f} {price_unit}<br>Capacity: {x:.2f} {capacity_unit}" for x, y in zip(x_values, y_values)]
        ))
    
    fig.update_layout(
        title=f"Daily Merit Order Curves for {product}",
        xaxis_title=f"Cumulative Capacity ({capacity_unit})",
        yaxis_title=f"Price ({price_unit})",
        height=600,
        width=1000,
        hovermode="closest",
        legend_title="Date",
        template="plotly_white"
    )
    return fig

# ----------------------
# VWAP Price Trend for a Product
# ----------------------
def plot_vwap_price_trend(processed_df, product, data_market='ENERGY'):
    """
    Plot VWAP price trend for a product with confidence band (std dev).
    """
    if processed_df is None or len(processed_df) == 0:
        return None
    
    # Filter for the selected product
    filtered_df = processed_df[processed_df['PRODUCT'] == product].copy()
    if len(filtered_df) == 0:
        return None
    
    # Ensure date column
    if 'DATE' not in filtered_df.columns:
        filtered_df['DATE'] = pd.to_datetime(filtered_df['DOWNLOAD_DATE'])
    
    # Group by day and calculate VWAP and std
    def vwap(group):
        v = group['CAPACITY_COL']
        p = group['SIGNED_PRICE']
        return (p * v).sum() / v.sum() if v.sum() != 0 else np.nan
    
    daily = filtered_df.groupby(filtered_df['DATE'].dt.date).apply(
        lambda g: pd.Series({
            'VWAP': vwap(g),
            'STD': g['SIGNED_PRICE'].std(),
            'COUNT': len(g)
        })
    ).reset_index().rename(columns={'DATE': 'Day'})
    
    price_unit = filtered_df['PRICE_UNIT'].iloc[0]
    
    fig = go.Figure()
    # VWAP line
    fig.add_trace(go.Scatter(
        x=daily['Day'],
        y=daily['VWAP'],
        mode='lines+markers',
        name='VWAP',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    # Confidence band (VWAP ± STD)
    fig.add_trace(go.Scatter(
        x=list(daily['Day']) + list(daily['Day'])[::-1],
        y=list(daily['VWAP'] + daily['STD']) + list((daily['VWAP'] - daily['STD'])[::-1]),
        fill='toself',
        fillcolor='rgba(0,100,200,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 STD',
        showlegend=True
    ))
    fig.update_layout(
        title=f"VWAP Price Trend for {product}",
        xaxis_title="Date",
        yaxis_title=f"VWAP Price ({price_unit})",
        height=500,
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

# ----------------------
# Test Functions for Debugging
# ----------------------
def test_api_connection():
    """
    Test the API connection with a simple request.
    """
    try:
        # Test with today's date
        test_date = datetime.now().strftime('%Y-%m-%d')
        test_url = f"https://www.regelleistung.net/apps/cpp-publisher/api/v1/download/tenders/anonymousresults?&productTypes=aFRR&market=ENERGY&exportFormat=xlsx&date={test_date}&countryCodeA2=DE"
        
        st.write(f"🔍 Testing API connection...")
        st.write(f"📡 URL: {test_url}")
        
        response = requests.get(test_url, timeout=30)
        st.write(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            st.success("✅ API connection successful!")
            # Try to read a small sample
            try:
                df = pd.read_excel(io.BytesIO(response.content))
                st.success(f"✅ Data parsing successful! Sample: {len(df)} records")
                return True
            except Exception as e:
                st.error(f"❌ Data parsing failed: {e}")
                return False
        else:
            st.error(f"❌ API request failed with status {response.status_code}")
            st.write(f"Response content: {response.text[:500]}...")
            return False
            
    except Exception as e:
        st.error(f"❌ API connection test failed: {e}")
        return False

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="aFRR Energy & Capacity Bids MOL", layout="wide")
st.title("aFRR Energy & Capacity Bids Merit Order List Visualization")
st.markdown("""
This app downloads anonymous aFRR energy and capacity bids from the regelleistung.net API and visualizes the Merit Order List (MOL) for selected delivery periods.
""")

# Sidebar for data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Data Source", ["Download from API", "Upload Excel Files"])

# Date selection
def_date = datetime.now().date()
selected_date = st.sidebar.date_input("Select Date", def_date)

# Market selection
st.sidebar.header("Markets to Download")
download_energy = st.sidebar.checkbox("Download aFRR Energy", value=True)
download_capacity = st.sidebar.checkbox("Download aFRR Capacity", value=True)

# Data loading
if data_source == "Download from API":
    if st.sidebar.button("Download Data"):
        with st.spinner("Downloading data..."):
            # Download energy data
            if download_energy:
                energy_df = download_afrr_data(selected_date.strftime('%Y-%m-%d'), 'ENERGY')
                if energy_df is not None:
                    st.session_state['energy_df'] = energy_df
                    st.success(f"Energy data downloaded: {len(energy_df)} records")
            
            # Download capacity data
            if download_capacity:
                capacity_df = download_afrr_data(selected_date.strftime('%Y-%m-%d'), 'CAPACITY')
                if capacity_df is not None:
                    st.session_state['capacity_df'] = capacity_df
                    st.success(f"Capacity data downloaded: {len(capacity_df)} records")

elif data_source == "Upload Excel Files":
    st.sidebar.header("Upload Files")
    energy_file = st.sidebar.file_uploader("Upload Energy Excel File", type=["xlsx"], key="energy")
    capacity_file = st.sidebar.file_uploader("Upload Capacity Excel File", type=["xlsx"], key="capacity")
    
    if energy_file is not None:
        energy_df = load_local_excel(energy_file, 'ENERGY')
        if energy_df is not None:
            st.session_state['energy_df'] = energy_df
            st.success(f"Energy data loaded: {len(energy_df)} records")
    
    if capacity_file is not None:
        capacity_df = load_local_excel(capacity_file, 'CAPACITY')
        if capacity_df is not None:
            st.session_state['capacity_df'] = capacity_df
            st.success(f"Capacity data loaded: {len(capacity_df)} records")

# Use session state to persist data
energy_df = st.session_state.get('energy_df', None)
capacity_df = st.session_state.get('capacity_df', None)

# Main content area
if energy_df is not None or capacity_df is not None:
    # Create tabs for different markets
    tab_names = []
    if energy_df is not None:
        tab_names.append("aFRR Energy")
    if capacity_df is not None:
        tab_names.append("aFRR Capacity")
    
    if len(tab_names) > 0:
        tabs = st.tabs(tab_names)
        
        # Energy tab
        if energy_df is not None:
            with tabs[tab_names.index("aFRR Energy")]:
                st.header("aFRR Energy Bids Merit Order List Visualization")
                st.success(f"Energy data loaded. Number of records: {len(energy_df)}")
                
                processed_energy_df, energy_products = process_data(energy_df)
                if processed_energy_df is not None:
                    st.write("### Energy Data Preview", processed_energy_df.head())
                    
                    # Sub-tabs for energy visualization
                    energy_tab1, energy_tab2, energy_tab3 = st.tabs(["All MOLs (NEG)", "All MOLs (POS)", "Single Product MOL"])
                    
                    with energy_tab1:
                        st.write("#### All Negative aFRR Energy Merit Order Curves")
                        fig_neg = plot_daily_merit_order_curves(processed_energy_df, market_type='NEG', data_market='ENERGY')
                        if fig_neg:
                            st.plotly_chart(fig_neg, use_container_width=True)
                            if st.button("Download NEG Energy MOL Plot as HTML", key="neg_energy_html"):
                                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                                fig_neg.write_html(tmpfile.name)
                                with open(tmpfile.name, "rb") as f:
                                    st.download_button("Download HTML", f, file_name="neg_energy_mol.html")
                    
                    with energy_tab2:
                        st.write("#### All Positive aFRR Energy Merit Order Curves")
                        fig_pos = plot_daily_merit_order_curves(processed_energy_df, market_type='POS', data_market='ENERGY')
                        if fig_pos:
                            st.plotly_chart(fig_pos, use_container_width=True)
                            if st.button("Download POS Energy MOL Plot as HTML", key="pos_energy_html"):
                                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                                fig_pos.write_html(tmpfile.name)
                                with open(tmpfile.name, "rb") as f:
                                    st.download_button("Download HTML", f, file_name="pos_energy_mol.html")
                    
                    with energy_tab3:
                        st.write("#### Select an Energy Product to Visualize its MOL")
                        if energy_products is not None and len(energy_products) > 0:
                            product = st.selectbox("Select Energy Product", sorted(energy_products), key="energy_product")
                            fig = plot_merit_order_list(processed_energy_df, product)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                if st.button("Download Energy Product MOL Plot as HTML", key="prod_energy_html"):
                                    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                                    fig.write_html(tmpfile.name)
                                    with open(tmpfile.name, "rb") as f:
                                        st.download_button("Download HTML", f, file_name=f"energy_mol_{product}.html")
        
        # Capacity tab
        if capacity_df is not None:
            with tabs[tab_names.index("aFRR Capacity")]:
                st.header("aFRR Capacity Bids Merit Order List Visualization")
                st.success(f"Capacity data loaded. Number of records: {len(capacity_df)}")
                
                processed_capacity_df, capacity_products = process_data(capacity_df)
                if processed_capacity_df is not None:
                    st.write("### Capacity Data Preview", processed_capacity_df.head())
                    
                    # Sub-tabs for capacity visualization
                    capacity_tab1, capacity_tab2, capacity_tab3 = st.tabs(["All MOLs (NEG)", "All MOLs (POS)", "Single Product MOL"])
                    
                    with capacity_tab1:
                        st.write("#### All Negative aFRR Capacity Merit Order Curves")
                        fig_neg = plot_daily_merit_order_curves(processed_capacity_df, market_type='NEG', data_market='CAPACITY')
                        if fig_neg:
                            st.plotly_chart(fig_neg, use_container_width=True)
                            if st.button("Download NEG Capacity MOL Plot as HTML", key="neg_capacity_html"):
                                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                                fig_neg.write_html(tmpfile.name)
                                with open(tmpfile.name, "rb") as f:
                                    st.download_button("Download HTML", f, file_name="neg_capacity_mol.html")
                    
                    with capacity_tab2:
                        st.write("#### All Positive aFRR Capacity Merit Order Curves")
                        fig_pos = plot_daily_merit_order_curves(processed_capacity_df, market_type='POS', data_market='CAPACITY')
                        if fig_pos:
                            st.plotly_chart(fig_pos, use_container_width=True)
                            if st.button("Download POS Capacity MOL Plot as HTML", key="pos_capacity_html"):
                                tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                                fig_pos.write_html(tmpfile.name)
                                with open(tmpfile.name, "rb") as f:
                                    st.download_button("Download HTML", f, file_name="pos_capacity_mol.html")
                    
                    with capacity_tab3:
                        st.write("#### Select a Capacity Product to Visualize its MOL")
                        if capacity_products is not None and len(capacity_products) > 0:
                            product = st.selectbox("Select Capacity Product", sorted(capacity_products), key="capacity_product")
                            fig = plot_merit_order_list(processed_capacity_df, product)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                if st.button("Download Capacity Product MOL Plot as HTML", key="prod_capacity_html"):
                                    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                                    fig.write_html(tmpfile.name)
                                    with open(tmpfile.name, "rb") as f:
                                        st.download_button("Download HTML", f, file_name=f"capacity_mol_{product}.html")
else:
    st.info("Please download or upload data to begin.")

# ----------------------
# Long-term Analysis Hub
# ----------------------
st.markdown("---")
st.header("🚀 Long-term Analysis Hub")
st.markdown("""
Analyze trends and patterns over extended periods. Download data for multiple days and explore market dynamics over time.
""")

# Long-term analysis sidebar
st.sidebar.markdown("---")
st.sidebar.header("📊 Long-term Analysis")

# Date range selection
st.sidebar.subheader("Date Range")
analysis_type = st.sidebar.selectbox(
    "Analysis Period",
    ["Custom Range", "Last 7 days", "Last 14 days", "Last 30 days", "Last 90 days"],
    help="Choose a predefined period or set a custom date range"
)

if analysis_type == "Custom Range":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
else:
    # Predefined ranges
    days_map = {
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Last 90 days": 90
    }
    start_date, end_date = get_date_range_suggestion(days_map[analysis_type])
    st.sidebar.write(f"**Date Range:** {start_date} to {end_date}")

# Market selection for long-term analysis
st.sidebar.subheader("Markets to Analyze")
lt_download_energy = st.sidebar.checkbox("Analyze aFRR Energy", value=True, key="lt_energy")
lt_download_capacity = st.sidebar.checkbox("Analyze aFRR Capacity", value=True, key="lt_capacity")

# Market type selection
st.sidebar.subheader("Market Type")
lt_market_type = st.sidebar.radio("Focus on", ["NEG", "POS"], key="lt_market_type")

# Download button for long-term analysis
if st.sidebar.button("🚀 Start Long-term Analysis", type="primary"):
    if start_date > end_date:
        st.error("Start date must be before end date!")
    else:
        total_days = (end_date - start_date).days + 1
        estimated_records = total_days * 110000  # Rough estimate based on your data
        
        st.warning(f"⚠️ **Performance Notice:** This will download ~{estimated_records:,} records over {total_days} days. This may take several minutes.")
        
        # Store the download request in session state
        st.session_state['lt_download_requested'] = True
        st.session_state['lt_start_date'] = start_date
        st.session_state['lt_end_date'] = end_date
        st.session_state['lt_download_energy'] = lt_download_energy
        st.session_state['lt_download_capacity'] = lt_download_capacity
        st.rerun()

# Handle the actual download (separate from the button)
if st.session_state.get('lt_download_requested', False):
    # Clear the request flag
    st.session_state['lt_download_requested'] = False
    
    # Get the stored parameters
    start_date = st.session_state.get('lt_start_date')
    end_date = st.session_state.get('lt_end_date')
    lt_download_energy = st.session_state.get('lt_download_energy', True)
    lt_download_capacity = st.session_state.get('lt_download_capacity', True)
    
    if start_date and end_date:
        total_days = (end_date - start_date).days + 1
        estimated_records = total_days * 110000
        
        st.info(f"🔄 Starting download for {total_days} days ({start_date} to {end_date})")
        st.info(f"📊 Estimated records: ~{estimated_records:,}")
        
        with st.spinner("Downloading long-term data..."):
            progress_bar = st.progress(0)
            
            # Download energy data
            if lt_download_energy:
                st.write("📊 Downloading Energy data...")
                try:
                    lt_energy_df = download_afrr_data_batch(start_date, end_date, 'ENERGY', progress_bar)
                    if lt_energy_df is not None:
                        st.session_state['lt_energy_df'] = lt_energy_df
                        st.success(f"✅ Energy data downloaded: {len(lt_energy_df):,} records")
                    else:
                        st.error("❌ Failed to download Energy data")
                except Exception as e:
                    st.error(f"❌ Error downloading Energy data: {e}")
            
            # Download capacity data
            if lt_download_capacity:
                st.write("📊 Downloading Capacity data...")
                try:
                    lt_capacity_df = download_afrr_data_batch(start_date, end_date, 'CAPACITY', progress_bar)
                    if lt_capacity_df is not None:
                        st.session_state['lt_capacity_df'] = lt_capacity_df
                        st.success(f"✅ Capacity data downloaded: {len(lt_capacity_df):,} records")
                    else:
                        st.error("❌ Failed to download Capacity data")
                except Exception as e:
                    st.error(f"❌ Error downloading Capacity data: {e}")
            
            progress_bar.empty()
            st.success("🎉 Long-term data download completed!")
            st.rerun()

# Display long-term analysis results
lt_energy_df = st.session_state.get('lt_energy_df', None)
lt_capacity_df = st.session_state.get('lt_capacity_df', None)

if lt_energy_df is not None or lt_capacity_df is not None:
    st.success("📈 Long-term data available for analysis!")
    
    # Create tabs for long-term analysis
    lt_tab_names = []
    if lt_energy_df is not None:
        lt_tab_names.append("📊 Energy Trends")
    if lt_capacity_df is not None:
        lt_tab_names.append("📊 Capacity Trends")
    
    if len(lt_tab_names) > 0:
        lt_tabs = st.tabs(lt_tab_names)
        
        # Energy trends tab
        if lt_energy_df is not None:
            with lt_tabs[lt_tab_names.index("📊 Energy Trends")]:
                st.header(f"📊 aFRR Energy Trends Analysis ({lt_market_type})")
                
                # Get unique products for dropdown
                all_products = lt_energy_df[lt_energy_df['PRODUCT'].str.contains(lt_market_type, na=False)]['PRODUCT'].unique()
                if len(all_products) == 0:
                    st.warning(f"No {lt_market_type} Energy products found for the selected period.")
                else:
                    selected_product = st.selectbox("Select Energy Product for Analysis", sorted(all_products), key="lt_energy_product")
                    
                    # Filter data for selected product
                    product_energy_df = lt_energy_df[lt_energy_df['PRODUCT'] == selected_product]
                    
                    # Analyze trends for selected product
                    energy_analysis = analyze_long_term_trends(product_energy_df, lt_market_type, 'ENERGY')
                    
                    if energy_analysis:
                        # Display overall statistics
                        stats = energy_analysis['overall_stats']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Records", f"{stats['total_records']:,}")
                        with col2:
                            st.metric("Date Range", stats['date_range'])
                        with col3:
                            st.metric("Avg Price", f"{stats['avg_price']:.2f} €/MWh")
                        with col4:
                            st.metric("Total Capacity", f"{stats['total_capacity']:,.0f} MW")
                        
                        # 1. Daily MOLs for selected product
                        st.write(f"#### Daily Merit Order Curves for {selected_product}")
                        daily_mol_fig = plot_daily_mols_for_product(energy_analysis['processed_df'], selected_product, 'ENERGY')
                        if daily_mol_fig:
                            st.plotly_chart(daily_mol_fig, use_container_width=True)
                        
                        # 2. VWAP price trend
                        st.write(f"#### VWAP Price Trend for {selected_product}")
                        vwap_fig = plot_vwap_price_trend(energy_analysis['processed_df'], selected_product, 'ENERGY')
                        if vwap_fig:
                            st.plotly_chart(vwap_fig, use_container_width=True)
                        
                        # 3. Capacity trend
                        st.write(f"#### Capacity Trends Over Time for {selected_product}")
                        capacity_fig = plot_capacity_trends(energy_analysis['daily_stats'], lt_market_type, 'ENERGY')
                        if capacity_fig:
                            st.plotly_chart(capacity_fig, use_container_width=True)
                        
                        # 4. Daily statistics table
                        st.write(f"#### Daily Statistics for {selected_product}")
                        st.dataframe(energy_analysis['daily_stats'], use_container_width=True)
                    else:
                        st.warning(f"No {lt_market_type} Energy data found for {selected_product} in the selected period.")
        
        # Capacity trends tab
        if lt_capacity_df is not None:
            with lt_tabs[lt_tab_names.index("📊 Capacity Trends")]:
                st.header(f"📊 aFRR Capacity Trends Analysis ({lt_market_type})")
                
                # Get unique products for dropdown
                all_products = lt_capacity_df[lt_capacity_df['PRODUCT'].str.contains(lt_market_type, na=False)]['PRODUCT'].unique()
                if len(all_products) == 0:
                    st.warning(f"No {lt_market_type} Capacity products found for the selected period.")
                else:
                    selected_product = st.selectbox("Select Capacity Product for Analysis", sorted(all_products), key="lt_capacity_product")
                    
                    # Filter data for selected product
                    product_capacity_df = lt_capacity_df[lt_capacity_df['PRODUCT'] == selected_product]
                    
                    # Analyze trends for selected product
                    capacity_analysis = analyze_long_term_trends(product_capacity_df, lt_market_type, 'CAPACITY')
                    
                    if capacity_analysis:
                        # Display overall statistics
                        stats = capacity_analysis['overall_stats']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Records", f"{stats['total_records']:,}")
                        with col2:
                            st.metric("Date Range", stats['date_range'])
                        with col3:
                            st.metric("Avg Price", f"{stats['avg_price']:.2f} €/MW/h")
                        with col4:
                            st.metric("Total Capacity", f"{stats['total_capacity']:,.0f} MW")
                        
                        # 1. Daily MOLs for selected product
                        st.write(f"#### Daily Merit Order Curves for {selected_product}")
                        daily_mol_fig = plot_daily_mols_for_product(capacity_analysis['processed_df'], selected_product, 'CAPACITY')
                        if daily_mol_fig:
                            st.plotly_chart(daily_mol_fig, use_container_width=True)
                        
                        # 2. VWAP price trend
                        st.write(f"#### VWAP Price Trend for {selected_product}")
                        vwap_fig = plot_vwap_price_trend(capacity_analysis['processed_df'], selected_product, 'CAPACITY')
                        if vwap_fig:
                            st.plotly_chart(vwap_fig, use_container_width=True)
                        
                        # 3. Capacity trend
                        st.write(f"#### Capacity Trends Over Time for {selected_product}")
                        capacity_fig = plot_capacity_trends(capacity_analysis['daily_stats'], lt_market_type, 'CAPACITY')
                        if capacity_fig:
                            st.plotly_chart(capacity_fig, use_container_width=True)
                        
                        # 4. Daily statistics table
                        st.write(f"#### Daily Statistics for {selected_product}")
                        st.dataframe(capacity_analysis['daily_stats'], use_container_width=True)
                    else:
                        st.warning(f"No {lt_market_type} Capacity data found for {selected_product} in the selected period.")
else:
    st.info("💡 Use the sidebar to configure and start a long-term analysis.")

# ----------------------
# Requirements file for easy sharing
# ----------------------
# Create requirements.txt
with open("requirements.txt", "w") as f:
    f.write("streamlit\nplotly\npandas\nnumpy\nopenpyxl\nrequests\n")

# Show usage instructions
st.sidebar.markdown("""
**Usage:**
1. Install requirements: `pip install -r requirements.txt`
2. Run: `streamlit run afrrenergybids.py`
3. Use the web UI in your browser.
""")

# Debug section
st.sidebar.markdown("---")
st.sidebar.header("🔧 Debug Tools")
if st.sidebar.button("🧪 Test API Connection"):
    test_api_connection()


