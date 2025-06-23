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
        
        return df
    except Exception as e:
        st.error(f"Error downloading {market} data: {e}")
        return None

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


