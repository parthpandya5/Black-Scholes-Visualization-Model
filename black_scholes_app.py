import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📈",
    layout="wide"
)

# Set matplotlib to use transparent background and light text for dark mode
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: transparent !important;
        padding: 10px;
        border-radius: 5px;
    }
    [data-testid="stMetricValue"] {
        background-color: transparent !important;
    }
    [data-testid="stMetricLabel"] {
        background-color: transparent !important;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(38, 39, 48, 0.4) !important;
        border: 1px solid rgba(250, 250, 250, 0.2);
        padding: 15px;
        border-radius: 8px;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Black-Scholes calculation functions
def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega (same for call and put)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Theta
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return delta, gamma, vega, theta, rho

# Title and description
st.title("📈 Black-Scholes Option Pricing Model")
st.markdown("### Interactive visualization and sensitivity analysis for European options")

# Sidebar for parameter inputs
st.sidebar.header("Option Parameters")

option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

col1, col2 = st.sidebar.columns(2)
with col1:
    S = st.number_input("Spot Price ($)", min_value=1.0, value=100.0, step=1.0)
    T = st.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, step=0.1)
    sigma = st.number_input("Volatility (σ)", min_value=0.01, max_value=2.0, value=0.2, step=0.01)

with col2:
    K = st.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=1.0)
    r = st.number_input("Risk-free Rate", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### Sensitivity Analysis Ranges")
vol_min = st.sidebar.slider("Min Volatility", 0.05, 0.5, 0.1, 0.05)
vol_max = st.sidebar.slider("Max Volatility", 0.5, 1.5, 0.5, 0.05)
spot_range = st.sidebar.slider("Spot Price Range (%)", 10, 100, 50, 5)

# Calculate option price and Greeks
if option_type == "Call":
    option_price = black_scholes_call(S, K, T, r, sigma)
else:
    option_price = black_scholes_put(S, K, T, r, sigma)

delta, gamma, vega, theta, rho = calculate_greeks(S, K, T, r, sigma, option_type.lower())

# Display results in metrics
st.markdown("## Option Price and Greeks")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Option Price", f"${option_price:.2f}")
with col2:
    st.metric("Delta (Δ)", f"{delta:.4f}")
with col3:
    st.metric("Gamma (Γ)", f"{gamma:.4f}")
with col4:
    st.metric("Vega (ν)", f"{vega:.4f}")
with col5:
    st.metric("Theta (Θ)", f"{theta:.4f}")
with col6:
    st.metric("Rho (ρ)", f"{rho:.4f}")

# Delta Volume/Hedging Section
st.markdown("## 📊 Delta Hedging Analysis")
num_contracts = st.number_input("Number of Option Contracts", min_value=1, value=10, step=1, 
                                help="1 contract = 100 options")
total_options = num_contracts * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Options", f"{total_options:,}")
    st.metric("Total Delta Exposure", f"{delta * total_options:,.2f}", 
             help="Equivalent shares exposure")
with col2:
    shares_to_hedge = abs(delta * total_options)
    st.metric("Shares to Hedge", f"{shares_to_hedge:,.2f}",
             help="Number of underlying shares needed for delta-neutral hedge")
    hedge_cost = shares_to_hedge * S
    st.metric("Hedge Cost", f"${hedge_cost:,.2f}",
             help="Cost to purchase hedging shares")
with col3:
    total_premium = option_price * total_options
    st.metric("Total Premium", f"${total_premium:,.2f}",
             help="Total cost/value of option position")
    hedge_direction = "SHORT" if option_type == "Call" else "LONG"
    st.metric("Hedge Direction", hedge_direction,
             help=f"To delta hedge, {hedge_direction} the underlying")

st.info(f"💡 **Delta Hedging Strategy:** To maintain a delta-neutral position, you need to {hedge_direction} " +
        f"{shares_to_hedge:,.0f} shares of the underlying asset at ${S:.2f} per share.")

# Create tabs for different visualizations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Sensitivity", "🔥 Volatility-Spot Heatmap", "📈 Greeks", "📉 Profit/Loss", "🔄 Delta Volume"])

with tab1:
    st.markdown("### Option Price Sensitivity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price vs Spot Price
        spot_prices = np.linspace(S * (1 - spot_range/100), S * (1 + spot_range/100), 100)
        if option_type == "Call":
            prices = [black_scholes_call(s, K, T, r, sigma) for s in spot_prices]
        else:
            prices = [black_scholes_put(s, K, T, r, sigma) for s in spot_prices]
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig1.patch.set_facecolor('none')
        fig1.patch.set_alpha(0)
        ax1.patch.set_facecolor('none')
        ax1.patch.set_alpha(0)
        ax1.plot(spot_prices, prices, linewidth=2, color='#1f77b4')
        ax1.axvline(S, color='red', linestyle='--', alpha=0.7, label='Current Spot')
        ax1.axhline(option_price, color='green', linestyle='--', alpha=0.7, label='Current Price')
        ax1.set_xlabel('Spot Price ($)', fontsize=12, color='white')
        ax1.set_ylabel('Option Price ($)', fontsize=12, color='white')
        ax1.set_title(f'{option_type} Option Price vs Spot Price', fontsize=14, fontweight='bold', color='white')
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.legend(facecolor='#262730', edgecolor='white')
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig1, transparent=True)
        plt.close(fig1)
    
    with col2:
        # Price vs Volatility
        volatilities = np.linspace(vol_min, vol_max, 100)
        if option_type == "Call":
            prices_vol = [black_scholes_call(S, K, T, r, v) for v in volatilities]
        else:
            prices_vol = [black_scholes_put(S, K, T, r, v) for v in volatilities]
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.patch.set_facecolor('none')
        fig2.patch.set_alpha(0)
        ax2.patch.set_facecolor('none')
        ax2.patch.set_alpha(0)
        ax2.plot(volatilities, prices_vol, linewidth=2, color='#ff7f0e')
        ax2.axvline(sigma, color='red', linestyle='--', alpha=0.7, label='Current Volatility')
        ax2.axhline(option_price, color='green', linestyle='--', alpha=0.7, label='Current Price')
        ax2.set_xlabel('Volatility (σ)', fontsize=12, color='white')
        ax2.set_ylabel('Option Price ($)', fontsize=12, color='white')
        ax2.set_title(f'{option_type} Option Price vs Volatility', fontsize=14, fontweight='bold', color='white')
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.legend(facecolor='#262730', edgecolor='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig2, transparent=True)
        plt.close(fig2)

with tab2:
    st.markdown("### Sensitivity Heatmap: Volatility vs Spot Price")
    
    # Create meshgrid for heatmap
    spot_range_heat = np.linspace(S * (1 - spot_range/100), S * (1 + spot_range/100), 30)
    vol_range_heat = np.linspace(vol_min, vol_max, 30)
    
    option_prices_grid = np.zeros((len(vol_range_heat), len(spot_range_heat)))
    
    for i, v in enumerate(vol_range_heat):
        for j, s in enumerate(spot_range_heat):
            if option_type == "Call":
                option_prices_grid[i, j] = black_scholes_call(s, K, T, r, v)
            else:
                option_prices_grid[i, j] = black_scholes_put(s, K, T, r, v)
    
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    fig3.patch.set_facecolor('none')
    fig3.patch.set_alpha(0)
    ax3.patch.set_facecolor('none')
    ax3.patch.set_alpha(0)
    sns.heatmap(option_prices_grid, 
                xticklabels=np.round(spot_range_heat, 1)[::3],
                yticklabels=np.round(vol_range_heat, 2)[::3],
                cmap='YlOrRd', 
                annot=False, 
                fmt='.2f',
                cbar_kws={'label': 'Option Price ($)'},
                ax=ax3)
    ax3.set_xlabel('Spot Price ($)', fontsize=12, color='white')
    ax3.set_ylabel('Volatility (σ)', fontsize=12, color='white')
    ax3.set_title(f'{option_type} Option Price Heatmap', fontsize=14, fontweight='bold', color='white')
    ax3.tick_params(colors='white')
    cbar = ax3.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    st.pyplot(fig3, transparent=True)
    plt.close(fig3)
    
    st.info(f"💡 Current position: Spot=${S:.2f}, Volatility={sigma:.2f}, Option Price=${option_price:.2f}")

with tab3:
    st.markdown("### Greeks Sensitivity Analysis")
    
    spot_prices_greeks = np.linspace(S * 0.7, S * 1.3, 100)
    
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    
    for s in spot_prices_greeks:
        d, g, v, t, _ = calculate_greeks(s, K, T, r, sigma, option_type.lower())
        deltas.append(d)
        gammas.append(g)
        vegas.append(v)
        thetas.append(t)
    
    fig4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig4.patch.set_facecolor('none')
    fig4.patch.set_alpha(0)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.patch.set_facecolor('none')
        ax.patch.set_alpha(0)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
    
    # Delta
    ax1.plot(spot_prices_greeks, deltas, linewidth=2, color='#1f77b4')
    ax1.axvline(S, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Spot Price ($)', color='white')
    ax1.set_ylabel('Delta', color='white')
    ax1.set_title('Delta vs Spot Price', fontweight='bold', color='white')
    ax1.grid(True, alpha=0.3, color='gray')
    
    # Gamma
    ax2.plot(spot_prices_greeks, gammas, linewidth=2, color='#ff7f0e')
    ax2.axvline(S, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Spot Price ($)', color='white')
    ax2.set_ylabel('Gamma', color='white')
    ax2.set_title('Gamma vs Spot Price', fontweight='bold', color='white')
    ax2.grid(True, alpha=0.3, color='gray')
    
    # Vega
    ax3.plot(spot_prices_greeks, vegas, linewidth=2, color='#2ca02c')
    ax3.axvline(S, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Spot Price ($)', color='white')
    ax3.set_ylabel('Vega', color='white')
    ax3.set_title('Vega vs Spot Price', fontweight='bold', color='white')
    ax3.grid(True, alpha=0.3, color='gray')
    
    # Theta
    ax4.plot(spot_prices_greeks, thetas, linewidth=2, color='#d62728')
    ax4.axvline(S, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Spot Price ($)', color='white')
    ax4.set_ylabel('Theta', color='white')
    ax4.set_title('Theta vs Spot Price', fontweight='bold', color='white')
    ax4.grid(True, alpha=0.3, color='gray')
    
    plt.tight_layout()
    st.pyplot(fig4, transparent=True)
    plt.close(fig4)

with tab4:
    st.markdown("### Profit/Loss Analysis at Expiration")
    
    spot_at_exp = np.linspace(S * 0.5, S * 1.5, 100)
    
    if option_type == "Call":
        payoff = np.maximum(spot_at_exp - K, 0)
    else:
        payoff = np.maximum(K - spot_at_exp, 0)
    
    profit = payoff - option_price
    
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    fig5.patch.set_facecolor('none')
    fig5.patch.set_alpha(0)
    ax5.patch.set_facecolor('none')
    ax5.patch.set_alpha(0)
    ax5.plot(spot_at_exp, profit, linewidth=2.5, color='#1f77b4', label='Profit/Loss')
    ax5.axhline(0, color='white', linestyle='-', linewidth=0.8)
    ax5.axvline(K, color='red', linestyle='--', alpha=0.7, label='Strike Price')
    ax5.fill_between(spot_at_exp, profit, 0, where=(profit > 0), alpha=0.3, color='green', label='Profit Zone')
    ax5.fill_between(spot_at_exp, profit, 0, where=(profit < 0), alpha=0.3, color='red', label='Loss Zone')
    ax5.set_xlabel('Spot Price at Expiration ($)', fontsize=12, color='white')
    ax5.set_ylabel('Profit/Loss ($)', fontsize=12, color='white')
    ax5.set_title(f'{option_type} Option Profit/Loss Diagram', fontsize=14, fontweight='bold', color='white')
    ax5.grid(True, alpha=0.3, color='gray')
    ax5.legend(facecolor='#262730', edgecolor='white')
    ax5.tick_params(colors='white')
    for spine in ax5.spines.values():
        spine.set_edgecolor('white')
    st.pyplot(fig5, transparent=True)
    plt.close(fig5)
    
    # Calculate breakeven
    if option_type == "Call":
        breakeven = K + option_price
        st.success(f"📊 Breakeven Price: ${breakeven:.2f} (Strike + Premium)")
        st.info(f"💰 Maximum Loss: ${option_price:.2f} (Premium paid)")
        st.info(f"📈 Maximum Profit: Unlimited")
    else:
        breakeven = K - option_price
        st.success(f"📊 Breakeven Price: ${breakeven:.2f} (Strike - Premium)")
        st.info(f"💰 Maximum Loss: ${option_price:.2f} (Premium paid)")
        st.info(f"📈 Maximum Profit: ${breakeven:.2f} (if spot goes to $0)")

with tab5:
    st.markdown("### Delta Volume and Hedging Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Delta Volume vs Spot Price
        spot_prices_delta = np.linspace(S * 0.7, S * 1.3, 100)
        deltas_vol = []
        shares_needed = []
        
        for s in spot_prices_delta:
            d, _, _, _, _ = calculate_greeks(s, K, T, r, sigma, option_type.lower())
            deltas_vol.append(d)
            shares_needed.append(abs(d * total_options))
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        fig6.patch.set_facecolor('none')
        fig6.patch.set_alpha(0)
        ax6.patch.set_facecolor('none')
        ax6.patch.set_alpha(0)
        ax6.plot(spot_prices_delta, shares_needed, linewidth=2.5, color='#1f77b4')
        ax6.axvline(S, color='red', linestyle='--', alpha=0.7, label='Current Spot')
        ax6.axhline(shares_to_hedge, color='green', linestyle='--', alpha=0.7, label='Current Hedge')
        ax6.set_xlabel('Spot Price ($)', fontsize=12, color='white')
        ax6.set_ylabel('Shares Needed to Hedge', fontsize=12, color='white')
        ax6.set_title(f'Delta Hedging Volume vs Spot Price ({num_contracts} contracts)', fontsize=14, fontweight='bold', color='white')
        ax6.grid(True, alpha=0.3, color='gray')
        ax6.legend(facecolor='#262730', edgecolor='white')
        ax6.tick_params(colors='white')
        for spine in ax6.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig6, transparent=True)
        plt.close(fig6)
    
    with col2:
        # Delta Volume vs Time
        times = np.linspace(T, 0.01, 100)
        deltas_time = []
        shares_time = []
        
        for t in times:
            d, _, _, _, _ = calculate_greeks(S, K, t, r, sigma, option_type.lower())
            deltas_time.append(d)
            shares_time.append(abs(d * total_options))
        
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        fig7.patch.set_facecolor('none')
        fig7.patch.set_alpha(0)
        ax7.patch.set_facecolor('none')
        ax7.patch.set_alpha(0)
        ax7.plot(times, shares_time, linewidth=2.5, color='#ff7f0e')
        ax7.axvline(T, color='red', linestyle='--', alpha=0.7, label='Current Time')
        ax7.axhline(shares_to_hedge, color='green', linestyle='--', alpha=0.7, label='Current Hedge')
        ax7.set_xlabel('Time to Maturity (Years)', fontsize=12, color='white')
        ax7.set_ylabel('Shares Needed to Hedge', fontsize=12, color='white')
        ax7.set_title(f'Delta Hedging Volume vs Time Decay ({num_contracts} contracts)', fontsize=14, fontweight='bold', color='white')
        ax7.grid(True, alpha=0.3, color='gray')
        ax7.legend(facecolor='#262730', edgecolor='white')
        ax7.tick_params(colors='white')
        for spine in ax7.spines.values():
            spine.set_edgecolor('white')
        ax7.invert_xaxis()  # So time flows left to right towards expiration
        st.pyplot(fig7, transparent=True)
        plt.close(fig7)
    
    # Delta Volume Heatmap
    st.markdown("### Delta Volume Heatmap: Spot Price vs Volatility")
    
    spot_range_delta = np.linspace(S * 0.7, S * 1.3, 25)
    vol_range_delta = np.linspace(vol_min, vol_max, 25)
    
    delta_grid = np.zeros((len(vol_range_delta), len(spot_range_delta)))
    shares_grid = np.zeros((len(vol_range_delta), len(spot_range_delta)))
    
    for i, v in enumerate(vol_range_delta):
        for j, s in enumerate(spot_range_delta):
            d, _, _, _, _ = calculate_greeks(s, K, T, r, v, option_type.lower())
            delta_grid[i, j] = d
            shares_grid[i, j] = abs(d * total_options)
    
    fig8, (ax8, ax9) = plt.subplots(1, 2, figsize=(16, 6))
    fig8.patch.set_facecolor('none')
    fig8.patch.set_alpha(0)
    
    for ax in [ax8, ax9]:
        ax.patch.set_facecolor('none')
        ax.patch.set_alpha(0)
        ax.tick_params(colors='white')
    
    # Delta heatmap
    sns.heatmap(delta_grid, 
                xticklabels=np.round(spot_range_delta, 1)[::3],
                yticklabels=np.round(vol_range_delta, 2)[::3],
                cmap='RdYlGn', 
                center=0.5,
                annot=False,
                cbar_kws={'label': 'Delta'},
                ax=ax8)
    ax8.set_xlabel('Spot Price ($)', fontsize=11, color='white')
    ax8.set_ylabel('Volatility (σ)', fontsize=11, color='white')
    ax8.set_title('Delta Values', fontsize=13, fontweight='bold', color='white')
    cbar8 = ax8.collections[0].colorbar
    cbar8.ax.yaxis.set_tick_params(color='white')
    cbar8.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar8.ax.axes, 'yticklabels'), color='white')
    
    # Shares needed heatmap
    sns.heatmap(shares_grid, 
                xticklabels=np.round(spot_range_delta, 1)[::3],
                yticklabels=np.round(vol_range_delta, 2)[::3],
                cmap='viridis', 
                annot=False,
                cbar_kws={'label': 'Shares'},
                ax=ax9)
    ax9.set_xlabel('Spot Price ($)', fontsize=11, color='white')
    ax9.set_ylabel('Volatility (σ)', fontsize=11, color='white')
    ax9.set_title(f'Shares to Hedge ({num_contracts} contracts)', fontsize=13, fontweight='bold', color='white')
    cbar9 = ax9.collections[0].colorbar
    cbar9.ax.yaxis.set_tick_params(color='white')
    cbar9.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar9.ax.axes, 'yticklabels'), color='white')
    
    plt.tight_layout()
    st.pyplot(fig8, transparent=True)
    plt.close(fig8)
    
    # Summary table
    st.markdown("### Delta Hedging Summary Table")
    
    summary_spots = np.linspace(S * 0.8, S * 1.2, 9)
    summary_data = []
    
    for s in summary_spots:
        d, g, _, _, _ = calculate_greeks(s, K, T, r, sigma, option_type.lower())
        shares = abs(d * total_options)
        cost = shares * s
        summary_data.append({
            'Spot Price': f'${s:.2f}',
            'Delta': f'{d:.4f}',
            'Gamma': f'{g:.4f}',
            'Shares to Hedge': f'{shares:,.0f}',
            'Hedge Cost': f'${cost:,.2f}',
            'Total Exposure': f'${abs(d * total_options * s):,.2f}'
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.info(f"💡 **Current Position:** With {num_contracts} contracts ({total_options:,} options), " +
            f"you need to {hedge_direction} {shares_to_hedge:,.0f} shares at ${S:.2f} " +
            f"for a total hedge cost of ${hedge_cost:,.2f}")


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Black-Scholes Option Pricing Model</strong></p>
    <p>Built with Python, Streamlit, NumPy, SciPy, and Seaborn</p>
    <p style='font-size: 12px;'>For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)
