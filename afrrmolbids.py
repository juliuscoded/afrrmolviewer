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
import csv
import base64

# --- Activation Data Functions (for 4s overlay) ---
def download_and_parse_activation_csv(date_str):
    """
    Download and parse the 4s activation CSV for a given date.
    Returns a DataFrame with columns: Zeit (ISO 8601), 50HZT_POS, 50HZT_NEG
    """
    url = f"https://api.transnetbw.de/picasso-cbmp/csv?date={date_str}&lang=de"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # The CSV uses semicolon as separator
        content = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(content), sep=';')
        
        # Extract only the 50HZ columns we need
        time_col = df.columns[0].replace('﻿"', '').replace('"', '')  # Remove BOM and quotes
        pos_col = '50HZT_POS'
        neg_col = '50HZT_NEG'
        
        # Check if our expected columns exist
        if pos_col not in df.columns or neg_col not in df.columns:
            st.error(f"Expected columns {pos_col} and {neg_col} not found")
            return None
        
        # Create a clean DataFrame with just the 50HZ columns we need
        clean_df = df[[time_col, pos_col, neg_col]].copy()
        clean_df.columns = ['Zeit', '50HZT_POS', '50HZT_NEG']
        
        # Convert Zeit to datetime
        clean_df['Zeit'] = pd.to_datetime(clean_df['Zeit'])
        
        # Handle timezone - TransnetBW data is in UTC
        if clean_df['Zeit'].dt.tz is None:
            clean_df['Zeit'] = clean_df['Zeit'].dt.tz_localize('UTC')
        
        # Add a CET/CEST column for user display
        clean_df['Zeit_CET'] = clean_df['Zeit'].dt.tz_convert('Europe/Berlin')
        
        # Convert prices to numeric, coerce errors
        clean_df['50HZT_POS'] = pd.to_numeric(clean_df['50HZT_POS'], errors='coerce')
        clean_df['50HZT_NEG'] = pd.to_numeric(clean_df['50HZT_NEG'], errors='coerce')
        
        # Remove rows where both prices are NaN (no activation data)
        clean_df = clean_df.dropna(subset=['50HZT_POS', '50HZT_NEG'], how='all')
        
        if len(clean_df) == 0:
            st.warning(f"No valid 50HZ activation data found for {date_str}")
            return None
        
        return clean_df
    except Exception as e:
        st.error(f"Error downloading or parsing 50HZ activation CSV: {e}")
        return None

def get_activation_points_for_product(activation_df, product_type='NEG'):
    """
    For a given activation DataFrame and product type (NEG or POS),
    return a DataFrame of activation points where activation occurred (POS != NEG).
    """
    if activation_df is None or len(activation_df) == 0:
        return None
    
    # Find rows where activation occurred (POS != NEG)
    activation_mask = activation_df['50HZT_NEG'] != activation_df['50HZT_POS']
    
    # Include both Zeit and Zeit_CET columns for proper display
    columns_to_include = ['Zeit', '50HZT_NEG', '50HZT_POS']
    if 'Zeit_CET' in activation_df.columns:
        columns_to_include.append('Zeit_CET')
    
    if product_type == 'NEG':
        # For NEG products, use NEG price when activation occurred
        result = activation_df.loc[activation_mask, columns_to_include].rename(columns={'50HZT_NEG': 'Activation_Price'})
    else:
        # For POS products, use POS price when activation occurred
        result = activation_df.loc[activation_mask, columns_to_include].rename(columns={'50HZT_POS': 'Activation_Price'})
    
    # Clean the result - remove rows with NaN or NaT values
    result = result.dropna(subset=['Activation_Price', 'Zeit'])
    
    return result.reset_index(drop=True)

def extract_period_from_product(product_name, selected_date):
    """
    Extract the 15-min period from a product name.
    Example: 'POS_001' -> ('2025-06-30', '00:00', '00:15') for the first period of the day
    """
    try:
        # Extract the product number (e.g., '001' from 'POS_001')
        if '_' in product_name:
            product_number_str = product_name.split('_')[-1]
            if product_number_str.isdigit():
                product_number = int(product_number_str)
                
                # Calculate the time period (15-minute intervals)
                # Period 1 = 00:00-00:15, Period 2 = 00:15-00:30, etc.
                start_minutes = (product_number - 1) * 15
                end_minutes = product_number * 15
                
                # Convert to HH:MM format
                start_hour = start_minutes // 60
                start_minute = start_minutes % 60
                end_hour = end_minutes // 60
                end_minute = end_minutes % 60
                
                start_time = f"{start_hour:02d}:{start_minute:02d}"
                end_time = f"{end_hour:02d}:{end_minute:02d}"
                
                # Use the selected date
                date_str = selected_date.strftime('%Y-%m-%d')
                
                return date_str, start_time, end_time
        
        return None, None, None
    except:
        return None, None, None

def filter_activation_by_period(activation_df, product_name, selected_date):
    """
    Filter activation data to only include timestamps within the specific 15-min period of the product.
    The product periods are in CEST (local time), but we need to convert to UTC for filtering the data.
    """
    if activation_df is None or len(activation_df) == 0:
        return None
    
    date_str, start_time, end_time = extract_period_from_product(product_name, selected_date)
    if not all([date_str, start_time, end_time]):
        return None
    
    # Create datetime objects for the period boundaries in CEST (local time)
    # Then convert to UTC for filtering the activation data
    start_datetime_cest = pd.to_datetime(f"{date_str} {start_time}").tz_localize('Europe/Berlin')
    end_datetime_cest = pd.to_datetime(f"{date_str} {end_time}").tz_localize('Europe/Berlin')
    
    # Convert to UTC for filtering (activation data is in UTC)
    start_datetime_utc = start_datetime_cest.tz_convert('UTC')
    end_datetime_utc = end_datetime_cest.tz_convert('UTC')
    
    # Filter activation data to only include timestamps within this period
    period_mask = (activation_df['Zeit'] >= start_datetime_utc) & (activation_df['Zeit'] < end_datetime_utc)
    filtered_df = activation_df[period_mask].copy()
    
    if len(filtered_df) == 0:
        return None
    
    return filtered_df

# ----------------------
# Activation Visualization Functions
# ----------------------
def create_mol_activation_animation(mol_df, activation_points, product_type='NEG', y_axis_scale='auto', max_y_scale=None):
    """
    Create a beautiful animated visualization showing how the MOL gets activated over time.
    Shows the dynamic 4-second activation with a moving colored area from left to right.
    """
    if mol_df is None or len(mol_df) == 0 or activation_points is None or len(activation_points) == 0:
        return None
    
    # Clean activation data - remove rows with NaN or NaT values
    activation_points = activation_points.dropna(subset=['Activation_Price', 'Zeit'])
    if len(activation_points) == 0:
        st.warning("No valid activation data found after cleaning")
        return None
    
    # Sort activation points by time
    activation_points = activation_points.sort_values('Zeit')
    
    # Dynamic y-axis scaling for better visibility
    price_range = mol_df['SIGNED_PRICE'].max() - mol_df['SIGNED_PRICE'].min()
    if y_axis_scale == 'auto':
        # Use data-driven scaling with some padding
        y_min = mol_df['SIGNED_PRICE'].min() - price_range * 0.1
        y_max = mol_df['SIGNED_PRICE'].max() + price_range * 0.1
    else:
        # Use fixed scale (15k for POS, -15k for NEG)
        is_negative = 'NEG' in product_type
        if is_negative:
            y_min = -15000
            y_max = 0
        else:
            y_min = 0
            y_max = 15000
    
    # Override with custom max scale if provided
    if max_y_scale is not None:
        is_negative = 'NEG' in product_type
        if is_negative:
            y_min = -max_y_scale
            y_max = 0
        else:
            y_min = 0
            y_max = max_y_scale
    
    # Create frames for animation (one frame per activation event)
    frames = []
    for i, (_, activation) in enumerate(activation_points.iterrows()):
        # Skip if activation price is NaN or NaT
        if pd.isna(activation['Activation_Price']):
            continue
            
        # Calculate which bids would be activated at this price
        if product_type == 'NEG':
            activated_mask = mol_df['SIGNED_PRICE'] <= activation['Activation_Price']
        else:
            activated_mask = mol_df['SIGNED_PRICE'] >= activation['Activation_Price']
        
        # Create frame data
        frame_data = []
        
        # 1. Add the complete MOL curve (black line, no fill)
        x_values = [0] + list(mol_df['CUMULATIVE_CAPACITY'])
        y_values = [mol_df['SIGNED_PRICE'].iloc[0]] + list(mol_df['SIGNED_PRICE'])
        
        frame_data.append(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color='#2E2E2E', width=4, shape='hv'),
            name='MOL Curve',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 2. Add the dynamic activated area (beautiful gradient fill)
        if activated_mask.any():
            # Find the activation capacity
            activation_capacity = mol_df.loc[activated_mask.idxmax(), 'CUMULATIVE_CAPACITY']
            
            # Create the fill area from 0 to activation capacity
            fill_mask = mol_df['CUMULATIVE_CAPACITY'] <= activation_capacity
            fill_df = mol_df[fill_mask]
            
            if len(fill_df) > 0:
                fill_x = [0] + list(fill_df['CUMULATIVE_CAPACITY'])
                fill_y = [fill_df['SIGNED_PRICE'].iloc[0]] + list(fill_df['SIGNED_PRICE'])
                
                # Beautiful gradient fill for activated area
                frame_data.append(go.Scatter(
                    x=fill_x,
                    y=fill_y,
                    mode='lines',
                    line=dict(color='rgba(255,215,0,0.8)', width=0),  # Golden color
                    fill='tozeroy',
                    fillcolor='rgba(255,215,0,0.4)',  # Semi-transparent golden fill
                    name='Activated Area',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # 3. Add activation boundary line (vertical line at current capacity)
            frame_data.append(go.Scatter(
                x=[activation_capacity, activation_capacity],
                y=[y_min, activation['Activation_Price']],
                mode='lines',
                line=dict(color='#FF6B35', width=4, dash='solid'),  # Orange solid line
                name='Activation Boundary',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 4. Add activation price line (horizontal line at current price)
        max_capacity = mol_df['CUMULATIVE_CAPACITY'].max()
        frame_data.append(go.Scatter(
            x=[0, max_capacity],
            y=[activation['Activation_Price'], activation['Activation_Price']],
            mode='lines',
            line=dict(color='#FF6B35', width=3, dash='dash'),  # Orange dashed line
            name='Activation Price',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 5. Add activation capacity annotation
        if activated_mask.any():
            activation_capacity = mol_df.loc[activated_mask.idxmax(), 'CUMULATIVE_CAPACITY']
            frame_data.append(go.Scatter(
                x=[activation_capacity],
                y=[activation['Activation_Price']],
                mode='markers',
                marker=dict(
                    color='#FF6B35',
                    size=12,
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                name='Current Activation Point',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Get CET time for display
        if 'Zeit_CET' in activation:
            display_time = activation['Zeit_CET'].strftime('%H:%M:%S')
        else:
            display_time = activation['Zeit'].strftime('%H:%M:%S')
        
        # Create frame with beautiful layout
        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            layout=go.Layout(
                title=dict(
                    text=f"<b>MOL Activation at {display_time}</b><br><span style='font-size: 14px; color: #666;'>Price: {activation['Activation_Price']:.2f} €/MWh | Capacity: {activation_capacity:.0f} MW</span>",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18, color='#2E2E2E')
                ),
                yaxis=dict(
                    range=[y_min, y_max],
                    gridcolor='rgba(128,128,128,0.2)',
                    zerolinecolor='rgba(128,128,128,0.5)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12)
                ),
                xaxis=dict(
                    gridcolor='rgba(128,128,128,0.2)',
                    zerolinecolor='rgba(128,128,128,0.5)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12)
                ),
                plot_bgcolor='rgba(255,255,255,0.9)',
                paper_bgcolor='white',
                margin=dict(t=100, b=80, l=80, r=80)
            )
        ))
    
    # Create the main figure with enhanced controls
    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
        layout=go.Layout(
            title=dict(
                text="<b>MOL Activation Animation</b><br><span style='font-size: 14px; color: #666;'>Watch the dynamic 4-second activation over time</span>",
                x=0.5,
                xanchor='center',
                font=dict(size=20, color='#2E2E2E')
            ),
            xaxis_title="Cumulative Capacity (MW)",
            yaxis_title="Price (€/MWh)",
            yaxis=dict(
                range=[y_min, y_max],
                gridcolor='rgba(128,128,128,0.2)',
                zerolinecolor='rgba(128,128,128,0.5)',
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            xaxis=dict(
                gridcolor='rgba(128,128,128,0.2)',
                zerolinecolor='rgba(128,128,128,0.5)',
                title_font=dict(size=14),
                tickfont=dict(size=12)
            ),
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='white',
            margin=dict(t=100, b=80, l=80, r=80),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'x': 0.1,
                'y': 1.1,
                'xanchor': 'left',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 800, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 300}}]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                    }
                ],
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': '#2E2E2E',
                'borderwidth': 1
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[frame.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': f"{i+1}",
                        'method': 'animate'
                    } for i, frame in enumerate(frames)
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Frame: ', 'font': {'size': 14}},
                'len': 0.8,
                'x': 0.1,
                'xanchor': 'left',
                'y': -0.25,
                'yanchor': 'bottom',
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': '#2E2E2E',
                'borderwidth': 1,
                'tickcolor': '#2E2E2E'
            }]
        )
    )
    
    return fig

def download_animation_as_gif(fig, filename="mol_animation.gif"):
    """
    Download the Plotly animation as a GIF file.
    Note: This creates a static image sequence that can be converted to GIF.
    """
    try:
        # Create a temporary HTML file with the animation
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
            fig.write_html(tmp_file.name)
            
            # Read the HTML content
            with open(tmp_file.name, 'r') as f:
                html_content = f.read()
            
            # Clean up temporary file
            import os
            os.unlink(tmp_file.name)
        
        # Create download button for HTML file
        st.download_button(
            label="📥 Download Animation as HTML",
            data=html_content,
            file_name=filename.replace('.gif', '.html'),
            mime="text/html"
        )
        
        st.info("💡 **Note:** For GIF export, download the HTML file and use a screen recording tool or browser extension to capture the animation.")
        
        return True
    except Exception as e:
        st.error(f"Error creating download: {e}")
        return False

def create_price_frequency_analysis(activation_points, product_type='NEG'):
    """
    Create a beautiful histogram showing the frequency of activation prices.
    """
    if activation_points is None or len(activation_points) == 0:
        return None
    
    # Create price bins (adjust bin size based on price range)
    prices = activation_points['Activation_Price']
    price_range = prices.max() - prices.min()
    
    # Use more bins for better resolution
    if price_range < 10:
        nbins = 20  # More bins for small ranges
    elif price_range < 100:
        nbins = 15
    else:
        nbins = 10
    
    fig = go.Figure()
    
    # Beautiful histogram with gradient colors
    fig.add_trace(go.Histogram(
        x=prices,
        nbinsx=nbins,
        name='Activation Frequency',
        marker=dict(
            color=prices,
            colorscale='Viridis',
            showscale=False
        ),
        opacity=0.8,
        hovertemplate="<b>Price Range:</b> %{x}<br><b>Frequency:</b> %{y}<extra></extra>"
    ))
    
    # Enhanced layout
    fig.update_layout(
        title=dict(
            text=f"<b>Activation Price Frequency Distribution ({product_type})</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='#2E2E2E')
        ),
        xaxis_title="Activation Price (€/MWh)",
        yaxis_title="Frequency",
        height=500,
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='white',
        margin=dict(t=80, b=80, l=80, r=80),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.5)',
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            zerolinecolor='rgba(128,128,128,0.5)',
            title_font=dict(size=14),
            tickfont=dict(size=12)
        )
    )
    
    return fig

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
def plot_merit_order_list(df, product, y_axis_scale='auto', max_y_scale=None, show_fill=True):
    """
    Plot the Merit Order List for a specific product.
    Works for both ENERGY and CAPACITY markets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Processed DataFrame containing the aFRR data
    product : str
        The product to plot
    y_axis_scale : str
        'auto' for automatic scaling, 'fixed' for fixed scale
    max_y_scale : float, optional
        Maximum y-axis value when using fixed scale
    show_fill : bool
        Whether to show filled area under the MOL curve
        
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
        fill='tozeroy' if show_fill else None,
        fillcolor=color if show_fill else None,
        opacity=0.7 if show_fill else 1.0,
        name='Merit Order',
        hoverinfo='text',
        hovertext=[f"Price: {y:.2f} {price_unit}<br>Capacity: {x:.2f} {capacity_unit}" for x, y in zip(x_values, y_values)]
    ))
    
    # Add points for individual bids (MUCH SMALLER)
    fig.add_trace(go.Scatter(
        x=sorted_df['CUMULATIVE_CAPACITY'],
        y=sorted_df['SIGNED_PRICE'],
        mode='markers',
        marker=dict(color='red', size=3),  # Reduced from 8 to 3
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
    
    # Dynamic y-axis scaling for better visibility
    price_range = sorted_df['SIGNED_PRICE'].max() - sorted_df['SIGNED_PRICE'].min()
    if y_axis_scale == 'auto':
        # Use data-driven scaling with some padding
        y_min = sorted_df['SIGNED_PRICE'].min() - price_range * 0.1
        y_max = sorted_df['SIGNED_PRICE'].max() + price_range * 0.1
    else:
        # Use fixed scale (15k for POS, -15k for NEG)
        if is_negative:
            y_min = -15000
            y_max = 0
        else:
            y_min = 0
            y_max = 15000
    
    # Override with custom max scale if provided
    if max_y_scale is not None:
        if is_negative:
            y_min = -max_y_scale
            y_max = 0
        else:
            y_min = 0
            y_max = max_y_scale
    
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
        template="plotly_white",
        yaxis=dict(range=[y_min, y_max])
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
    
    # Create individual MOL traces with proper names for legend
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
        
        # Extract time from product name for better legend labels
        if '_' in product:
            period_part = product.split('_')[1]
            try:
                period_num = int(period_part)
                # Convert period number to time (assuming 15-minute intervals starting from 00:00)
                hours = (period_num - 1) // 4
                minutes = ((period_num - 1) % 4) * 15
                time_label = f"{hours:02d}:{minutes:02d}"
            except:
                time_label = f"Period {period_num}"
        else:
            time_label = f"Period {i+1}"
        
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            line=dict(color=color, width=1.5, shape='hv'),
            name=f"{product} ({time_label})",
            hovertext=[f"Product: {product}<br>Time: {time_label}<br>Price: {y:.2f} {price_unit}<br>Capacity: {x:.2f} {capacity_unit}" for x, y in zip(x_values, y_values)],
            hoverinfo='text',
            visible=True,  # All traces visible by default
            legendgroup=f"group_{i}",
            legendgrouptitle_text=f"Period {i+1}"
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
                x=1.08,
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
        template="plotly_white",
        showlegend=True,  # Enable legend
        legend=dict(
            x=1.15,
            y=0.5,
            yanchor='middle',
            xanchor='left',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            font=dict(size=10)
        )
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
def plot_boxplot_price_trend(processed_df, product, data_market='ENERGY'):
    """
    Plot a box plot of bid prices for each day for the selected product.
    Optionally overlay the VWAP as a line.
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
    
    price_unit = filtered_df['PRICE_UNIT'].iloc[0]
    
    # Prepare data for box plot
    filtered_df['Day'] = filtered_df['DATE'].dt.date
    
    fig = go.Figure()
    # Box plot for each day
    for day in sorted(filtered_df['Day'].unique()):
        day_df = filtered_df[filtered_df['Day'] == day]
        fig.add_trace(go.Box(
            y=day_df['SIGNED_PRICE'],
            x=[day] * len(day_df),
            name=str(day),
            boxpoints='outliers',
            marker_color='blue',
            line=dict(width=1),
            showlegend=False
        ))
    # Optionally overlay VWAP as a line
    def vwap(group):
        v = group['CAPACITY_COL']
        p = group['SIGNED_PRICE']
        return (p * v).sum() / v.sum() if v.sum() != 0 else np.nan
    vwap_by_day = filtered_df.groupby('Day').apply(vwap)
    fig.add_trace(go.Scatter(
        x=vwap_by_day.index,
        y=vwap_by_day.values,
        mode='lines+markers',
        name='VWAP',
        line=dict(color='orange', width=2),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title=f"Bid Price Distribution (Box Plot) for {product}",
        xaxis_title="Date",
        yaxis_title=f"Bid Price ({price_unit})",
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

def test_activation_data_download():
    """
    Test the 50HZ activation data download specifically.
    """
    st.write("🧪 Testing 50HZ Activation Data Download")
    
    # Test with today's date
    test_date = datetime.now().strftime('%Y-%m-%d')
    st.write(f"📅 Testing with date: {test_date}")
    
    # Test the download function
    activation_df = download_and_parse_activation_csv(test_date)
    
    if activation_df is not None:
        st.success(f"✅ 50HZ activation data download successful!")
        st.write(f"📊 Downloaded {len(activation_df)} rows")
        
        # Test period extraction for a sample product
        sample_product = "NEG_001"  # First period of the day
        st.write(f"🔍 Testing period extraction for: {sample_product}")
        
        # Extract period
        date_str, start_time, end_time = extract_period_from_product(sample_product, datetime.now().date())
        st.write(f"⏰ Extracted period: {date_str} {start_time} to {end_time}")
        
        # Test filtering
        filtered_df = filter_activation_by_period(activation_df, sample_product, datetime.now().date())
        if filtered_df is not None:
            st.success(f"✅ Period filtering successful! Found {len(filtered_df)} 50HZ events")
        else:
            st.warning("⚠️ Period filtering returned no 50HZ data")
    else:
        st.error("❌ 50HZ activation data download failed")
    
    return activation_df is not None

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="aFRR Energy & Capacity Bids MOL", layout="wide")
st.title("aFRR Energy & Capacity Bids Merit Order List Visualization")

# Add explanation section
with st.expander("🎯 Visualization Improvements & FAQ", expanded=False):
    st.markdown("""
    ### Recent Improvements:
    
    **🔧 Fixed Issues:**
    - **Red dots are now much smaller** (size 3 instead of 8) for better visibility
    - **Dynamic Y-axis scaling** - choose 'auto' for data-driven scaling or 'fixed' for custom range
    - **Improved activation animation** - shows moving filled area instead of yellow X markers
    - **Better visibility** for typical activation prices (usually 50-500 €/MWh vs 15k range)
    
    **❓ FAQ - Why does the MOL change in animation?**
    The MOL curve itself doesn't change! What you see is:
    - **Black line**: The complete MOL curve (always the same, no fill)
    - **Yellow filled area**: Only the area under the curve up to the current 4-second activation price
    - **Orange dashed lines**: Show the current activation price and capacity
    
    **🎨 New Animation Features:**
    - **Static MOL line**: Black line shows the complete MOL curve (never changes)
    - **Growing yellow area**: Yellow area fills under the curve based on 4-second activation price
    - **Horizontal price line**: Shows current activation price across the chart
    - **Vertical capacity line**: Shows how much capacity is activated
    - **Dynamic scaling**: Y-axis automatically adjusts to show relevant price ranges
    
    **📊 Visibility Options:**
    - **Auto scaling**: Y-axis adjusts to your data (recommended for most cases)
    - **Fixed scaling**: Set custom Y-axis range (useful for comparing different periods)
    - **Smaller bid markers**: Red dots are now much less intrusive
    """)

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
                            
                            # Visualization controls
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                y_axis_scale = st.selectbox("Y-Axis Scale", ["auto", "fixed"], key="energy_y_scale")
                            with col2:
                                max_y_scale = st.number_input("Max Y-Scale (€/MWh)", min_value=100, max_value=20000, value=1000, step=100, key="energy_max_y")
                            with col3:
                                overlay_activation = st.checkbox("Overlay 4s Activation Data", value=False, key="overlay_activation")
                            
                            fig = plot_merit_order_list(processed_energy_df, product, y_axis_scale=y_axis_scale, max_y_scale=max_y_scale if y_axis_scale == "fixed" else None)
                            if fig:
                                if overlay_activation:
                                    date_str = selected_date.strftime('%Y-%m-%d')
                                    activation_df = download_and_parse_activation_csv(date_str)
                                    if activation_df is not None:
                                        # Filter activation data to the specific period of the selected product
                                        period_activation_df = filter_activation_by_period(activation_df, product, selected_date)
                                        if period_activation_df is not None:
                                            product_type = 'NEG' if 'NEG' in product else 'POS'
                                            activation_points = get_activation_points_for_product(period_activation_df, product_type)
                                            if activation_points is not None and not activation_points.empty:
                                                # Show timezone info for clarity
                                                st.success(f"Found {len(activation_points)} activation events for {product}")
                                                
                                                # Get the actual period for this product
                                                date_str, start_time, end_time = extract_period_from_product(product, selected_date)
                                                if all([date_str, start_time, end_time]):
                                                    period_info = f"{start_time}-{end_time} CEST"
                                                else:
                                                    period_info = "unknown period"
                                                
                                                st.info(f"📅 Product period: {product} ({period_info}) | 📊 Data: {len(period_activation_df)} timestamps from 4-second intervals")
                                                
                                                # Create sub-tabs for different activation visualizations
                                                activation_tab1, activation_tab2, activation_tab3 = st.tabs([
                                                    "MOL with Overlay", 
                                                    "Activation Animation", 
                                                    "Price Frequency"
                                                ])
                                                
                                                with activation_tab1:
                                                    # Original MOL with activation overlay
                                                    fig_with_overlay = plot_merit_order_list(processed_energy_df, product, y_axis_scale=y_axis_scale, max_y_scale=max_y_scale if y_axis_scale == "fixed" else None)
                                                    if fig_with_overlay:
                                                        # Add activation overlay to the figure
                                                        mol_df = processed_energy_df[processed_energy_df['PRODUCT'] == product].copy()
                                                        is_negative = 'NEG' in product
                                                        ascending = not is_negative
                                                        mol_df = mol_df.sort_values('SIGNED_PRICE', ascending=ascending)
                                                        mol_df['CUMULATIVE_CAPACITY'] = mol_df['CAPACITY_COL'].cumsum()
                                                        
                                                        # For each activation price, find the corresponding x (cumulative capacity)
                                                        activation_x = []
                                                        activation_times = []
                                                        activation_prices = []
                                                        for _, row in activation_points.iterrows():
                                                            price = row['Activation_Price']
                                                            if product_type == 'NEG':
                                                                mask = mol_df['SIGNED_PRICE'] <= price
                                                            else:
                                                                mask = mol_df['SIGNED_PRICE'] >= price
                                                            idx = mask.idxmax() if mask.any() else mol_df.index[-1]
                                                            x_val = mol_df.loc[idx, 'CUMULATIVE_CAPACITY']
                                                            activation_x.append(x_val)
                                                            activation_prices.append(price)
                                                            
                                                            # Use CET time for display if available
                                                            if 'Zeit_CET' in row:
                                                                time_str = row['Zeit_CET'].strftime('%H:%M:%S')
                                                            else:
                                                                time_str = row['Zeit'].strftime('%H:%M:%S')
                                                            activation_times.append(time_str)
                                                        
                                                        # Simple activation points overlay - just mark the activated prices on the curve
                                                        fig_with_overlay.add_trace(go.Scatter(
                                                            x=activation_x,
                                                            y=activation_prices,
                                                            mode='markers',
                                                            marker=dict(
                                                                color='#FF6B35',
                                                                size=8,
                                                                symbol='diamond',
                                                                line=dict(color='white', width=1)
                                                            ),
                                                            name='4s Activation Points',
                                                            hovertext=[f"Time: {t}<br>Price: {p:.2f} €/MWh<br>Capacity: {x:.0f} MW" 
                                                                     for t, p, x in zip(activation_times, activation_prices, activation_x)],
                                                            hoverinfo='text'
                                                        ))
                                                        st.plotly_chart(fig_with_overlay, use_container_width=True)
                                                
                                                with activation_tab2:
                                                    # MOL activation animation
                                                    st.write("#### MOL Activation Animation")
                                                    st.write("Watch how the MOL gets activated over time during this 15-min period")
                                                    animation_fig = create_mol_activation_animation(mol_df, activation_points, product_type, y_axis_scale=y_axis_scale, max_y_scale=max_y_scale if y_axis_scale == "fixed" else None)
                                                    if animation_fig:
                                                        st.plotly_chart(animation_fig, use_container_width=True)
                                                        
                                                        # Add download section
                                                        st.markdown("---")
                                                        st.write("#### 📥 Download Animation")
                                                        download_animation_as_gif(animation_fig, f"mol_animation_{product}_{selected_date.strftime('%Y%m%d')}.html")
                                                    else:
                                                        st.warning("Could not create animation")
                                                
                                                with activation_tab3:
                                                    # Price frequency analysis
                                                    st.write("#### Activation Price Frequency Analysis")
                                                    st.write("Distribution of activation prices during this 15-min period")
                                                    freq_fig = create_price_frequency_analysis(activation_points, product_type)
                                                    if freq_fig:
                                                        st.plotly_chart(freq_fig, use_container_width=True)
                                                        
                                                        # Add summary statistics
                                                        col1, col2, col3, col4 = st.columns(4)
                                                        with col1:
                                                            st.metric("Total Activations", len(activation_points))
                                                        with col2:
                                                            st.metric("Avg Price", f"{activation_points['Activation_Price'].mean():.2f} €/MWh")
                                                        with col3:
                                                            st.metric("Min Price", f"{activation_points['Activation_Price'].min():.2f} €/MWh")
                                                        with col4:
                                                            st.metric("Max Price", f"{activation_points['Activation_Price'].max():.2f} €/MWh")
                                                    else:
                                                        st.warning("Could not create frequency analysis")
                                            else:
                                                st.info(f"No activation events found for {product} in the selected period")
                                        else:
                                            st.warning(f"Could not filter activation data for period of {product}")
                                    else:
                                        st.error("Failed to download activation data")
                                else:
                                    # Show regular MOL without overlay
                                    st.plotly_chart(fig, use_container_width=True)
        
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
                            
                            # Visualization controls
                            col1, col2 = st.columns(2)
                            with col1:
                                y_axis_scale_cap = st.selectbox("Y-Axis Scale", ["auto", "fixed"], key="capacity_y_scale")
                            with col2:
                                max_y_scale_cap = st.number_input("Max Y-Scale (€/MW/h)", min_value=10, max_value=2000, value=100, step=10, key="capacity_max_y")
                            
                            fig = plot_merit_order_list(processed_capacity_df, product, y_axis_scale=y_axis_scale_cap, max_y_scale=max_y_scale_cap if y_axis_scale_cap == "fixed" else None)
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
                        
                        # 2. Box plot of bid prices
                        st.write(f"#### Bid Price Distribution (Box Plot) for {selected_product}")
                        boxplot_fig = plot_boxplot_price_trend(energy_analysis['processed_df'], selected_product, 'ENERGY')
                        if boxplot_fig:
                            st.plotly_chart(boxplot_fig, use_container_width=True)
                        
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
                        
                        # 2. Box plot of bid prices
                        st.write(f"#### Bid Price Distribution (Box Plot) for {selected_product}")
                        boxplot_fig = plot_boxplot_price_trend(capacity_analysis['processed_df'], selected_product, 'CAPACITY')
                        if boxplot_fig:
                            st.plotly_chart(boxplot_fig, use_container_width=True)
                        
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

if st.sidebar.button("🧪 Test Activation Data"):
    test_activation_data_download()


