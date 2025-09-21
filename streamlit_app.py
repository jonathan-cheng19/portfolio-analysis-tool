import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import itertools
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# =========================
# Utility computations
# =========================
def compute_rsi(close_series: pd.Series, period: int = 14) -> float:
    if len(close_series) < period + 1:
        return np.nan
    delta = close_series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def annualized_vol_from_returns(ret: pd.Series) -> float:
    if ret.std() == 0 or ret.empty:
        return 0.0
    return float(ret.std() * np.sqrt(252))

def beta_vs_market(asset_ret: pd.Series, mkt_ret: pd.Series) -> float:
    df = pd.concat([asset_ret, mkt_ret], axis=1).dropna()
    if df.shape[0] < 20:
        return np.nan
    x = df.iloc[:,1].values
    y = df.iloc[:,0].values
    denom = np.var(x, ddof=1)
    if denom == 0:
        return np.nan
    beta = np.cov(y, x, ddof=1)[0,1] / denom
    return float(beta)

def pct_from_52w_low(close_series: pd.Series) -> float:
    if close_series.empty:
        return np.nan
    low_52 = close_series[-252:].min() if len(close_series) >= 252 else close_series.min()
    last = close_series.iloc[-1]
    return float((last / low_52 - 1) * 100) if low_52 > 0 else np.nan

def last_day_change(close_series: pd.Series) -> float:
    if len(close_series) < 2:
        return np.nan
    return float((close_series.iloc[-1] / close_series.iloc[-2] - 1) * 100)

def sharpe_from_returns(ret: pd.Series, daily_rf: float) -> float:
    if ret.std() == 0 or ret.empty:
        return 0.0
    excess = ret - daily_rf
    return float((excess.mean() * 252) / (ret.std() * np.sqrt(252)))

def normalize_weights(weights_dict):
    total = sum(weights_dict.values())
    if total == 0:
        n = len(weights_dict)
        return {k: 1.0/n for k in weights_dict}
    return {k: v/total for k, v in weights_dict.items()}

def rolling_sharpe(returns: pd.Series, window: int = 60, daily_rf: float = 0.0) -> pd.Series:
    if returns.empty or window < 5:
        return pd.Series(dtype=float)
    er = returns - daily_rf
    roll_mean = er.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    sharpe = (roll_mean * 252) / (roll_std * np.sqrt(252))
    return sharpe.dropna()

def drawdown(cumulative: pd.Series):
    if cumulative.empty:
        return pd.Series(dtype=float), np.nan
    peak = cumulative.cummax()
    dd = cumulative / peak - 1.0
    mdd = float(dd.min()) if not dd.empty else np.nan
    return dd, mdd

# =========================
# Max Sharpe optimization
# =========================
def max_sharpe_weights(returns_df: pd.DataFrame, daily_rf: float) -> dict:
    """
    Long-only, sum-to-1 max Sharpe weights.
    First try unconstrained tangency closed-form; if any negative -> Dirichlet random search fallback.
    """
    if returns_df.empty or returns_df.shape[1] == 0:
        return {}

    mu = returns_df.mean()
    Sigma = returns_df.cov()
    tickers = list(returns_df.columns)

    try:
        Sigma_inv = np.linalg.pinv(Sigma.values)
    except Exception:
        Sigma_inv = np.linalg.pinv(Sigma.values)

    ones = np.ones(len(tickers))
    k = Sigma_inv @ (mu.values - daily_rf * ones)
    if np.allclose(k.sum(), 0):
        k = np.maximum(k, 0)

    w_unconstrained = k / np.sum(k) if np.sum(k) != 0 else np.ones(len(tickers)) / len(tickers)

    if np.all(w_unconstrained >= -1e-9):
        w_unconstrained = np.clip(w_unconstrained, 0, None)
        w_unconstrained = w_unconstrained / w_unconstrained.sum() if w_unconstrained.sum() > 0 else np.ones(len(tickers))/len(tickers)
        return dict(zip(tickers, map(float, w_unconstrained)))

    # Fallback: random Dirichlet search
    def sharpe_for_w(w):
        pr = (returns_df @ w)
        ex = pr - daily_rf
        sd = pr.std()
        return -((ex.mean() * 252) / (sd * np.sqrt(252)) if sd != 0 else -1e9)

    best_w = None
    best_s = 1e9
    rng = np.random.default_rng(42)
    trials = 20000
    for _ in range(trials):
        w = rng.dirichlet(np.ones(len(tickers)))
        s = sharpe_for_w(w)
        if s < best_s:
            best_s, best_w = s, w

    return dict(zip(tickers, map(float, best_w)))

# =========================
# Suggestion universe
# =========================
SUGGESTION_UNIVERSE = [
    "PHAR","PRDO","TNL","AXL","EFSC","TRS","LEVI","GLDD","TNGX","SKIN","GTES","MCY","MCB","COUR",
    "LAW","RCKY","CZWI","GILT","SMP","TILE","INDB","JOUT","ATGE","PARR","CTRE","CALX","HBT","BWFG",
    "KAR","DAKT","EYE","CARE","VSTA","URBN","PHIN","JHG","FVCB","EGO","CBAN","MLCO","NWPX","DXPE",
    "HG","OSK","ALLT","PRA","EVI","BWB","DINO","BBW","TCBI","KNSA","ALRS","SUPN","PCB","AM","FINW",
    "THFF","SHBI","PAHC","CADE","TIGO","MED","VBTX","UFCS","HWC","HBM","ASB","VTOL","RERE","BWA",
    "ENIC","ASAIY","SFST","CSTM","HCI","TBPH","HALO","FBIO","HMN","BWAY","ADT","RLX","BPOP","CENX",
    "SHCO","CECO","ATAI","TCBX","HITI","PSTL","INFU","MRBK","ANIP","PRE","AMRX","GTX","MUX","LAUR",
    "GAU","HRTG","ASYS","THG","JMPLY","CIA","FSM","ARREF","WDH","BBU","IFS","GCT","OVID","OZK","XHR",
    "VALN","TTMI","TATT","EGAN","UPWK","BVN","IBEX","NKTR","BTG","EZPW","BTSG","HUT","LYFT","KE",
    "BLX","FET","RRGB","PBI","RMNI","OPRX","FLGT","DOMO","OMF","SNDL","OUST","RIOT","IAG","PDLB",
    "LASR","NGL","BKTI","WLDN","MPAA","LPL","CCSI","TPC","DRD","FTEK","ESEA","AVAH","AUGO","PRCH",
    "GNW","XERS","TLS","CDTX","FSI","OBE","NGD","ITRG","HHH","VSAT","CYD","GSL","PGY","COMM","PSIX",
    "GECC","SOHU","KPLT","SSRM","SCZMF","IMPP","ZEPP", "STCE", "GDMN", "WGMI", "DAPP", "BITQ", "BKCH",
    "FDIG", "SPRX", "RING", "URA", "GDXJ", "SLVP", "SIL", "URNJ", "GDX", "SDGJ", "BITW", "ARKW", "SGDM",
    "GOAU", "CHAT", "NUKZ", "ETHE", "SILJ", "BLOK", "CRPT", "ETHA", "TETH", "ETHW", "FETH", "ETHV", "ETH",
    "EZET", "SHLD"
]

# =========================
# Streamlit Configuration
# =========================
st.set_page_config(
    page_title="Portfolio Analysis Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .sharpe-interpretation {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .excellent { background-color: #d4edda; color: #155724; }
    .good { background-color: #cce5ff; color: #004085; }
    .acceptable { background-color: #fff3cd; color: #856404; }
    .poor { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# =========================
# Main App Functions
# =========================
@st.cache_data
def fetch_stock_data(tickers, period):
    """Fetch stock data with caching"""
    portfolio_data = {}
    successful_tickers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Fetching data for {ticker}...")
        try:
            data = yf.Ticker(ticker).history(period=period)
            if not data.empty:
                portfolio_data[ticker] = data
                successful_tickers.append(ticker)
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {e}")
        
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    return portfolio_data, successful_tickers

@st.cache_data
def get_risk_free_rate():
    """Get risk-free rate with caching"""
    try:
        treasury_data = yf.Ticker("^IRX").history(period="5d")
        annual_rf_rate = treasury_data['Close'].iloc[-1] / 100 if not treasury_data.empty else 0.045
    except:
        annual_rf_rate = 0.045
    return annual_rf_rate

def analyze_portfolio(tickers, period):
    """Main portfolio analysis function"""
    if len(tickers) < 2:
        st.error("Please enter at least 2 ticker symbols.")
        return None
    
    # Fetch data
    portfolio_data, successful_tickers = fetch_stock_data(tickers, period)
    
    if len(successful_tickers) < 2:
        st.error("Could not fetch data for enough stocks. Please check ticker symbols.")
        return None
    
    # Calculate returns
    price_data = {t: d['Close'] for t, d in portfolio_data.items()}
    price_df = pd.DataFrame(price_data).dropna()
    portfolio_returns = price_df.pct_change().dropna()
    
    # Get risk-free rate
    annual_rf_rate = get_risk_free_rate()
    daily_rf_rate = annual_rf_rate / 252.0
    
    # Optimize weights for max Sharpe
    indiv_ret = portfolio_returns[successful_tickers]
    opt_weights = max_sharpe_weights(indiv_ret, daily_rf_rate)
    
    # Calculate portfolio returns using optimal weights
    w_series = pd.Series({t: opt_weights.get(t, 0.0) for t in indiv_ret.columns}, dtype=float)
    w_series = w_series / (w_series.sum() if w_series.sum() != 0 else 1.0)
    
    portfolio_ret = (indiv_ret * w_series).sum(axis=1)
    
    # Calculate metrics
    excess_returns = portfolio_ret - daily_rf_rate
    mean_return = portfolio_ret.mean()
    std_return = portfolio_ret.std()
    mean_excess_return = excess_returns.mean()
    
    sharpe_ratio = mean_excess_return / std_return if std_return != 0 else 0
    annualized_return = mean_return * 252
    annualized_volatility = std_return * np.sqrt(252)
    annualized_sharpe = (mean_excess_return * 252) / (std_return * np.sqrt(252)) if std_return != 0 else 0
    
    # Calculate covariances
    covariances = {}
    for t1, t2 in itertools.combinations(successful_tickers, 2):
        covariances[f"[{t1} -> {t2}]"] = np.cov(indiv_ret[t1], indiv_ret[t2])[0,1]
    
    return {
        'tickers': successful_tickers,
        'weights': {k: float(v) for k, v in w_series.items()},
        'portfolio_returns': portfolio_ret,
        'individual_returns': indiv_ret,
        'sharpe_metrics': {
            'daily_return': mean_return,
            'daily_volatility': std_return,
            'daily_sharpe': sharpe_ratio,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'annualized_sharpe': annualized_sharpe,
            'risk_free_rate': annual_rf_rate,
            'total_days': len(portfolio_ret)
        },
        'covariances': covariances,
        'price_data': price_df,
        'portfolio_data': portfolio_data
    }

def display_portfolio_summary(analysis_results):
    """Display portfolio summary with metrics"""
    st.header("📋 Portfolio Summary")
    
    res = analysis_results
    weights = res['weights']
    sm = res['sharpe_metrics']
    
    # Portfolio composition
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Portfolio Composition")
        st.write(f"**Stocks:** {', '.join(res['tickers'])}")
        st.write(f"**Number of stocks:** {len(res['tickers'])}")
        st.write("**Weighting:** Max-Sharpe (long-only, sum = 1.0)")
        
        # Weights table
        weight_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weight_df['Weight %'] = weight_df['Weight'] * 100
        weight_df = weight_df.sort_values('Weight', ascending=False)
        st.dataframe(weight_df.style.format({'Weight': '{:.4f}', 'Weight %': '{:.2f}%'}))
    
    with col2:
        # Weights chart
        fig = px.bar(
            x=list(weights.values()), 
            y=list(weights.keys()),
            orientation='h',
            title="Portfolio Weights",
            labels={'x': 'Weight', 'y': 'Ticker'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics
    st.subheader("📈 Portfolio Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Annualized Return",
            f"{sm['annualized_return']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Annualized Volatility",
            f"{sm['annualized_volatility']*100:.2f}%"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{sm['annualized_sharpe']:.4f}"
        )
    
    with col4:
        st.metric(
            "Risk-Free Rate",
            f"{sm['risk_free_rate']*100:.2f}%"
        )
    
    # Sharpe interpretation
    s = sm['annualized_sharpe']
    if s > 2.0:
        interp = "Excellent (> 2.0)"
        css_class = "excellent"
    elif s > 1.0:
        interp = "Good (1.0 – 2.0)"
        css_class = "good"
    elif s > 0.5:
        interp = "Acceptable (0.5 – 1.0)"
        css_class = "acceptable"
    else:
        interp = "Poor (≤ 0.5)"
        css_class = "poor"
    
    st.markdown(f"""
    <div class="sharpe-interpretation {css_class}">
        <strong>Sharpe Ratio Interpretation:</strong> {interp}
    </div>
    """, unsafe_allow_html=True)

def display_individual_performance(analysis_results):
    """Display individual stock performance table"""
    st.header("📊 Individual Stock Performance")
    
    res = analysis_results
    indiv = res['individual_returns']
    weights = res['weights']
    
    # Calculate individual stock metrics
    stats_data = []
    for ticker in res['tickers']:
        r = indiv[ticker].dropna()
        ann_ret = r.mean() * 252 if not r.empty else float('nan')
        ann_vol = r.std() * np.sqrt(252) if not r.empty else float('nan')
        weight = weights.get(ticker, 0.0)
        
        # Get current price data for additional metrics
        try:
            hist = res['portfolio_data'][ticker]
            close = hist['Close']
            day_change = last_day_change(close)
            from_52w_low = pct_from_52w_low(close)
            rsi_val = compute_rsi(close, period=14)
        except:
            day_change = np.nan
            from_52w_low = np.nan
            rsi_val = np.nan
        
        stats_data.append({
            'Ticker': ticker,
            'Weight': weight,
            'Ann. Return': ann_ret,
            'Ann. Volatility': ann_vol,
            'Day Change': day_change,
            '% from 52W Low': from_52w_low,
            'RSI(14)': rsi_val
        })
    
    df = pd.DataFrame(stats_data)
    
    # Format and display
    formatted_df = df.style.format({
        'Weight': '{:.1%}',
        'Ann. Return': '{:.1%}',
        'Ann. Volatility': '{:.1%}',
        'Day Change': '{:.2f}%',
        '% from 52W Low': '{:.2f}%',
        'RSI(14)': '{:.1f}'
    })
    
    st.dataframe(formatted_df, use_container_width=True)

def display_portfolio_charts(analysis_results, timeframe="1y"):
    """Display portfolio performance charts"""
    st.header("📈 Portfolio Charts")
    
    # Timeframe selector
    col1, col2 = st.columns([1, 4])
    with col1:
        timeframe = st.selectbox(
            "Select Timeframe:",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
    
    try:
        # Fetch data for selected timeframe
        chart_data = {}
        for ticker in analysis_results['tickers']:
            data = yf.Ticker(ticker).history(period=timeframe)
            if not data.empty:
                chart_data[ticker] = data['Close']
        
        if not chart_data:
            st.warning("No data available for selected timeframe")
            return
        
        df_prices = pd.DataFrame(chart_data).dropna()
        
        # 1. Normalized performance chart
        st.subheader("Portfolio Performance (Normalized)")
        fig = go.Figure()
        
        for ticker, prices in chart_data.items():
            normalized = (prices / prices.iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=normalized.index,
                y=normalized.values,
                name=ticker,
                mode='lines'
            ))
        
        fig.update_layout(
            title="Normalized Price Performance (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Returns distribution
        st.subheader("Returns Distribution")
        returns_data = df_prices.pct_change().dropna()
        
        fig = go.Figure()
        for ticker in returns_data.columns:
            fig.add_trace(go.Histogram(
                x=returns_data[ticker],
                name=ticker,
                opacity=0.7,
                nbinsx=30
            ))
        
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Returns",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. Risk-Return scatter
        st.subheader("Risk-Return Profile")
        risk_return_data = []
        for ticker in returns_data.columns:
            r = returns_data[ticker]
            annual_return = r.mean() * 252
            annual_risk = r.std() * np.sqrt(252)
            risk_return_data.append({
                'Ticker': ticker,
                'Annual Return': annual_return,
                'Annual Risk': annual_risk
            })
        
        rr_df = pd.DataFrame(risk_return_data)
        
        fig = px.scatter(
            rr_df,
            x='Annual Risk',
            y='Annual Return',
            text='Ticker',
            title="Risk vs Return Profile"
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Correlation heatmap
        st.subheader("Correlation Matrix")
        corr_matrix = returns_data.corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Asset Correlation Matrix"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating charts: {str(e)}")

def display_buy_hold_analysis(analysis_results, timeframe="1y"):
    """Display buy & hold analysis with interactive charts"""
    st.header("💰 Buy & Hold Analysis")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        timeframe = st.selectbox(
            "Select Timeframe:",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            key="bh_timeframe"
        )
    
    with col2:
        window = st.selectbox(
            "Rolling Sharpe Window:",
            [20, 30, 60, 90, 120, 252],
            index=2,
            key="rs_window"
        )
    
    try:
        # Fetch prices
        chart_data = {}
        for ticker in analysis_results['tickers']:
            data = yf.Ticker(ticker).history(period=timeframe)
            if not data.empty:
                chart_data[ticker] = data['Close']
        
        if not chart_data:
            st.warning("No data available for selected timeframe")
            return
        
        df_prices = pd.DataFrame(chart_data).dropna()
        returns_df = df_prices.pct_change().dropna()
        
        # Get weights and calculate portfolio returns
        weights = analysis_results.get('weights', {})
        w = {t: weights.get(t, 0.0) for t in df_prices.columns}
        total_weight = sum(w.values())
        if total_weight == 0:
            w = {t: 1.0/len(df_prices.columns) for t in df_prices.columns}
        else:
            w = {t: v/total_weight for t, v in w.items()}
        
        portfolio_returns = (returns_df * pd.Series(w)).sum(axis=1)
        cumulative_value = (1 + portfolio_returns).cumprod() * 100
        
        # 1. Buy & Hold Performance
        st.subheader("Portfolio Value Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_value.index,
            y=cumulative_value.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=f"Buy & Hold Portfolio Value ({timeframe})",
            xaxis_title="Date",
            yaxis_title="Value (Base=100)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Rolling Sharpe Ratio
        st.subheader("Rolling Sharpe Ratio")
        daily_rf = analysis_results['sharpe_metrics']['risk_free_rate'] / 252.0
        rolling_sharpe_series = rolling_sharpe(portfolio_returns, window=window, daily_rf=daily_rf)
        
        if not rolling_sharpe_series.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_sharpe_series.index,
                y=rolling_sharpe_series.values,
                mode='lines',
                name=f'Rolling Sharpe ({window}D)',
                line=dict(width=2)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f"Rolling Sharpe Ratio ({window}-day window)",
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 3. Drawdown Analysis
        st.subheader("Drawdown Analysis")
        dd_series, max_dd = drawdown(cumulative_value)
        
        if not dd_series.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dd_series.index,
                y=dd_series.values * 100,
                fill='tonexty',
                mode='lines',
                name='Drawdown',
                line=dict(width=1),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            fig.update_layout(
                title=f"Portfolio Drawdown (Max DD: {max_dd*100:.2f}%)",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_return = ((cumulative_value.iloc[-1] / 100) - 1) * 100
            st.metric("Total Return", f"{total_return:.2f}%")
        
        with col2:
            annual_vol = portfolio_returns.std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{annual_vol:.2f}%")
        
        with col3:
            if not rolling_sharpe_series.empty:
                avg_sharpe = rolling_sharpe_series.mean()
                st.metric("Average Rolling Sharpe", f"{avg_sharpe:.3f}")
    
    except Exception as e:
        st.error(f"Error in buy & hold analysis: {str(e)}")

def display_risk_analysis(analysis_results):
    """Display risk analysis and beta visualizations"""
    st.header("⚠️ Risk & Beta Analysis")
    
    try:
        # Fetch market data (SPY)
        market_data = yf.Ticker("SPY").history(
            start=analysis_results['portfolio_returns'].index.min() - pd.Timedelta(days=10),
            end=analysis_results['portfolio_returns'].index.max() + pd.Timedelta(days=10)
        )['Close'].pct_change().dropna()
        
        # Calculate betas
        betas = {}
        for ticker in analysis_results['tickers']:
            asset_ret = analysis_results['individual_returns'][ticker].dropna()
            betas[ticker] = beta_vs_market(asset_ret, market_data)
        
        # 1. Beta Bar Chart
        st.subheader("Beta vs SPY")
        beta_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
        beta_df = beta_df.sort_values('Beta', ascending=True)
        
        fig = px.bar(
            beta_df,
            x='Beta',
            y=beta_df.index,
            orientation='h',
            title="Beta vs SPY (β > 1 = More volatile than market)"
        )
        fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta = 1.0")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Rolling Beta for selected stock
        st.subheader("Rolling Beta Analysis")
        selected_ticker = st.selectbox(
            "Select ticker for rolling beta:",
            analysis_results['tickers']
        )
        
        if selected_ticker:
            # Calculate rolling beta
            asset_ret = analysis_results['individual_returns'][selected_ticker].dropna()
            combined_data = pd.concat([asset_ret, market_data], axis=1).dropna()
            combined_data.columns = ['asset', 'market']
            
            window = 60
            rolling_betas = []
            
            if len(combined_data) >= window:
                for i in range(window, len(combined_data)):
                    window_data = combined_data.iloc[i-window:i]
                    beta = beta_vs_market(window_data['asset'], window_data['market'])
                    rolling_betas.append({
                        'date': combined_data.index[i],
                        'beta': beta
                    })
                
                rb_df = pd.DataFrame(rolling_betas).set_index('date')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=rb_df.index,
                    y=rb_df['beta'],
                    mode='lines',
                    name=f'{selected_ticker} Rolling Beta',
                    line=dict(width=2)
                ))
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Market Beta = 1.0")
                
                fig.update_layout(
                    title=f"Rolling 60-Day Beta: {selected_ticker}",
                    xaxis_title="Date",
                    yaxis_title="Beta",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 3. Correlation Analysis
        st.subheader("Portfolio Correlation Analysis")
        corr_matrix = analysis_results['individual_returns'].corr()
        
        # Create correlation network visualization
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Asset Correlation Heatmap",
            zmin=-1, zmax=1
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # PCA Analysis
        st.subheader("Principal Component Analysis")
        cov_matrix = analysis_results['individual_returns'].cov()
        eigenvalues, _ = np.linalg.eigh(cov_matrix.values)
        eigenvalues = eigenvalues[::-1]  # Sort descending
        explained_var = eigenvalues / eigenvalues.sum()
        cumulative_var = np.cumsum(explained_var)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(eigenvalues) + 1)),
            y=cumulative_var,
            mode='lines+markers',
            name='Cumulative Variance Explained',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title="PCA Scree Plot - Cumulative Variance Explained",
            xaxis_title="Principal Component",
            yaxis_title="Cumulative Variance Explained",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Beta summary table
        st.subheader("Beta Summary")
        beta_summary_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
        beta_summary_df['Risk Level'] = beta_summary_df['Beta'].apply(
            lambda x: 'High (β > 1.5)' if x > 1.5 else 
                     'Moderate-High (1.0 < β ≤ 1.5)' if x > 1.0 else
                     'Moderate-Low (0.5 < β ≤ 1.0)' if x > 0.5 else
                     'Low (β ≤ 0.5)' if pd.notnull(x) else 'N/A'
        )
        beta_summary_df = beta_summary_df.sort_values('Beta', ascending=False)
        st.dataframe(beta_summary_df.style.format({'Beta': '{:.3f}'}), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in risk analysis: {str(e)}")

def display_suggestions(analysis_results):
    """Display portfolio optimization suggestions"""
    st.header("💡 Portfolio Optimization Suggestions")
    
    try:
        base_returns = analysis_results['portfolio_returns']
        base_sharpe = sharpe_from_returns(base_returns, analysis_results['sharpe_metrics']['risk_free_rate']/252)
        current_tickers = analysis_results['tickers']
        current_weights = analysis_results.get('weights', {t: 1.0/len(current_tickers) for t in current_tickers})
        current_weights = normalize_weights(current_weights)
        
        start_date = base_returns.index.min() - pd.Timedelta(days=5)
        end_date = base_returns.index.max() + pd.Timedelta(days=5)
        
        suggestions = []
        
        # Progress bar for suggestions calculation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Test adding new stocks from universe
        universe_sample = SUGGESTION_UNIVERSE[:50]  # Limit for performance
        
        for i, candidate in enumerate(universe_sample):
            if candidate in current_tickers:
                continue
                
            status_text.text(f"Evaluating {candidate}...")
            
            try:
                candidate_data = yf.Ticker(candidate).history(start=start_date, end=end_date)
                if candidate_data.empty:
                    continue
                    
                candidate_returns = candidate_data['Close'].pct_change().dropna()
                combined_data = pd.concat([base_returns, candidate_returns], axis=1).dropna()
                
                if combined_data.shape[0] < len(base_returns) * 0.5:
                    continue
                
                # Test adding 10% weight
                new_weights = {**current_weights}
                for k in new_weights:
                    new_weights[k] *= 0.90
                new_weights[candidate] = 0.10
                
                # Calculate new portfolio returns
                individual_returns = analysis_results['individual_returns']
                aligned_returns = pd.concat([individual_returns, candidate_returns.rename(candidate)], axis=1).dropna()
                
                if aligned_returns.empty:
                    continue
                    
                new_portfolio_returns = (aligned_returns * pd.Series(new_weights)).sum(axis=1)
                new_sharpe = sharpe_from_returns(new_portfolio_returns, analysis_results['sharpe_metrics']['risk_free_rate']/252)
                
                delta_sharpe = new_sharpe - base_sharpe
                
                if delta_sharpe > 0:
                    suggestions.append({
                        'Type': 'Add',
                        'Action': f'Add {candidate}',
                        'Delta Sharpe': delta_sharpe,
                        'Description': f'Add {candidate} with 10% weight'
                    })
            
            except Exception:
                continue
            
            progress_bar.progress((i + 1) / len(universe_sample))
        
        progress_bar.empty()
        status_text.empty()
        
        # Test removing existing stocks
        for ticker in current_tickers:
            try:
                remaining_weights = {k: v for k, v in current_weights.items() if k != ticker}
                if not remaining_weights:
                    continue
                    
                remaining_weights = normalize_weights(remaining_weights)
                individual_returns = analysis_results['individual_returns']
                
                remaining_columns = [c for c in individual_returns.columns if c in remaining_weights]
                if not remaining_columns:
                    continue
                    
                aligned_remaining = individual_returns[remaining_columns].dropna()
                if aligned_remaining.empty:
                    continue
                    
                remaining_portfolio_returns = (aligned_remaining * pd.Series(remaining_weights)).sum(axis=1)
                remaining_sharpe = sharpe_from_returns(remaining_portfolio_returns, analysis_results['sharpe_metrics']['risk_free_rate']/252)
                
                delta_sharpe = remaining_sharpe - base_sharpe
                
                if delta_sharpe > 0:
                    suggestions.append({
                        'Type': 'Remove',
                        'Action': f'Remove {ticker}',
                        'Delta Sharpe': delta_sharpe,
                        'Description': f'Remove {ticker} and reweight remaining stocks'
                    })
            
            except Exception:
                continue
        
        # Display suggestions
        if suggestions:
            suggestions_df = pd.DataFrame(suggestions)
            suggestions_df = suggestions_df.sort_values('Delta Sharpe', ascending=False).head(10)
            
            st.subheader("Top 10 Suggestions")
            
            # Color code suggestions
            def color_delta(val):
                if val > 0.1:
                    return 'background-color: #d4edda'  # Green
                elif val > 0.05:
                    return 'background-color: #cce5ff'  # Blue
                else:
                    return 'background-color: #fff3cd'  # Yellow
            
            styled_df = suggestions_df.style.format({'Delta Sharpe': '{:+.4f}'})
            styled_df = styled_df.applymap(color_delta, subset=['Delta Sharpe'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Allow user to apply suggestions
            st.subheader("Apply Suggestion")
            selected_suggestion = st.selectbox(
                "Select a suggestion to apply:",
                options=range(len(suggestions_df)),
                format_func=lambda x: f"{suggestions_df.iloc[x]['Action']} (Δ Sharpe: {suggestions_df.iloc[x]['Delta Sharpe']:+.4f})"
            )
            
            if st.button("Apply Selected Suggestion"):
                suggestion = suggestions_df.iloc[selected_suggestion]
                if suggestion['Type'] == 'Add':
                    new_ticker = suggestion['Action'].replace('Add ', '')
                    st.session_state.suggested_tickers = current_tickers + [new_ticker]
                elif suggestion['Type'] == 'Remove':
                    remove_ticker = suggestion['Action'].replace('Remove ', '')
                    st.session_state.suggested_tickers = [t for t in current_tickers if t != remove_ticker]
                
                st.success(f"Suggestion applied! New tickers: {', '.join(st.session_state.suggested_tickers)}")
                st.info("Please update the ticker input above and re-run analysis to see the results.")
        
        else:
            st.info("No beneficial suggestions found. Your current portfolio appears well-optimized!")
    
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")

def display_raw_data(analysis_results):
    """Display raw portfolio data"""
    st.header("📋 Raw Data")
    
    # Portfolio returns data
    portfolio_returns = analysis_results['portfolio_returns']
    cumulative_value = (1 + portfolio_returns).cumprod() * 100
    
    # Create data table
    data_table = pd.DataFrame({
        'Date': portfolio_returns.index,
        'Daily Return (%)': portfolio_returns.values * 100,
        'Portfolio Value': cumulative_value.values
    })
    
    # Display with formatting
    st.subheader("Portfolio Performance Data")
    st.dataframe(
        data_table.style.format({
            'Daily Return (%)': '{:.4f}%',
            'Portfolio Value': '{:.2f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = data_table.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"portfolio_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )
    
    # Individual stock returns
    st.subheader("Individual Stock Returns")
    individual_returns = analysis_results['individual_returns'] * 100  # Convert to percentage
    
    st.dataframe(
        individual_returns.style.format('{:.4f}%'),
        use_container_width=True,
        height=400
    )
    
    # Download individual returns
    individual_csv = individual_returns.to_csv()
    st.download_button(
        label="Download Individual Returns as CSV",
        data=individual_csv,
        file_name=f"individual_returns_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )

# =========================
# Main Streamlit App
# =========================
def main():
    st.markdown('<h1 class="main-header">📊 Portfolio Analysis Calculator</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'suggested_tickers' not in st.session_state:
        st.session_state.suggested_tickers = []
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Portfolio Configuration")
        
        # Ticker input
        default_tickers = ", ".join(st.session_state.suggested_tickers) if st.session_state.suggested_tickers else "AAPL, MSFT, GOOGL, TSLA, AMZN"
        ticker_input = st.text_area(
            "Enter Ticker Symbols:",
            value=default_tickers,
            help="Enter stock ticker symbols separated by commas or spaces"
        )
        
        # Analysis period
        period = st.selectbox(
            "Analysis Period:",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        # Analyze button
        if st.button("🔍 Analyze Portfolio", type="primary"):
            if ticker_input.strip():
                # Parse tickers
                if ',' in ticker_input:
                    tickers = [t.strip().upper() for t in ticker_input.split(',')]
                else:
                    tickers = ticker_input.upper().split()
                
                tickers = list(dict.fromkeys([t for t in tickers if t]))  # Remove duplicates
                
                if len(tickers) >= 2:
                    with st.spinner("Analyzing portfolio... This may take a moment."):
                        st.session_state.analysis_results = analyze_portfolio(tickers, period)
                    
                    if st.session_state.analysis_results:
                        st.success("Analysis complete!")
                        # Clear suggested tickers after successful analysis
                        st.session_state.suggested_tickers = []
                else:
                    st.error("Please enter at least 2 ticker symbols.")
            else:
                st.error("Please enter ticker symbols.")
        
        # Clear button
        if st.button("🗑️ Clear Results"):
            st.session_state.analysis_results = None
            st.session_state.suggested_tickers = []
            st.rerun()
        
        # Info section
        st.markdown("---")
        st.markdown("**Features:**")
        st.markdown("• Max-Sharpe weight optimization")
        st.markdown("• Risk and return analysis")
        st.markdown("• Interactive visualizations")
        st.markdown("• Portfolio suggestions")
        st.markdown("• Buy & hold analysis")
    
    # Main content area
    if st.session_state.analysis_results:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Summary", 
            "📈 Charts", 
            "💰 Buy & Hold", 
            "⚠️ Risk Analysis", 
            "💡 Suggestions", 
            "📊 Performance", 
            "📋 Raw Data"
        ])
        
        with tab1:
            display_portfolio_summary(st.session_state.analysis_results)
        
        with tab2:
            display_portfolio_charts(st.session_state.analysis_results)
        
        with tab3:
            display_buy_hold_analysis(st.session_state.analysis_results)
        
        with tab4:
            display_risk_analysis(st.session_state.analysis_results)
        
        with tab5:
            display_suggestions(st.session_state.analysis_results)
        
        with tab6:
            display_individual_performance(st.session_state.analysis_results)
        
        with tab7:
            display_raw_data(st.session_state.analysis_results)
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to the Portfolio Analysis Calculator! 👋
        
        This application helps you analyze and optimize your investment portfolio using modern portfolio theory and quantitative finance techniques.
        
        ### How to get started:
        1. **Enter ticker symbols** in the sidebar (e.g., AAPL, MSFT, GOOGL)
        2. **Select analysis period** (recommended: 1 year)
        3. **Click "Analyze Portfolio"** to generate insights
        
        ### What you'll get:
        - **Optimized weights** using Maximum Sharpe Ratio methodology
        - **Risk metrics** including volatility, beta, and correlation analysis
        - **Interactive charts** showing performance, returns distribution, and risk profiles
        - **Buy & hold analysis** with drawdown and rolling Sharpe calculations
        - **Portfolio suggestions** to potentially improve your Sharpe ratio
        - **Detailed performance metrics** for individual stocks
        
        ### Features:
        - 📊 **Modern Portfolio Theory** optimization
        - 📈 **Interactive visualizations** powered by Plotly
        - 💡 **AI-driven suggestions** for portfolio improvement
        - 📱 **Responsive design** that works on all devices
        - 💾 **Export capabilities** for further analysis
        
        **Ready to optimize your portfolio? Enter your tickers in the sidebar to begin!**
        """)
        
        # Example portfolios
        st.markdown("### 💡 Try these example portfolios:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏦 Conservative Portfolio"):
                st.session_state.suggested_tickers = ["VTI", "BND", "VEA", "VWO"]
                st.rerun()
        
        with col2:
            if st.button("⚖️ Balanced Portfolio"):
                st.session_state.suggested_tickers = ["SPY", "QQQ", "VEA", "BND", "VNQ"]
                st.rerun()
        
        with col3:
            if st.button("🚀 Growth Portfolio"):
                st.session_state.suggested_tickers = ["TSLA", "NVDA", "AMZN", "GOOGL", "MSFT"]
                st.rerun()

if __name__ == "__main__":
    main()