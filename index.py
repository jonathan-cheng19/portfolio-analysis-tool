import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from datetime import datetime, timedelta
import itertools
import threading
import warnings
import mplcursors
import matplotlib.dates as mdates

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
# Suggestion universe (updated)
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

class PortfolioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Analysis Calculator")
        self.root.geometry("1500x1000")
        self.root.configure(bg='#f0f0f0')
        
        self.portfolio_data = {}
        self.portfolio_returns = pd.DataFrame()
        self.analysis_results = None
        self.current_timeframe = "1Y"
        
        # Suggestions / beta caching
        self.suggestions_df = pd.DataFrame()
        self.suggest_period = "1y"
        self.market_returns_cache = None
        self.applied_weights = None
        
        self.create_widgets()
        plt.style.use('default')
        sns.set_palette("husl")

    def _max_sharpe_weights(self, returns_df: pd.DataFrame, daily_rf: float) -> dict:
        """
        Long-only, sum-to-1 max Sharpe weights.
        First try unconstrained tangency closed-form; if any negative -> Dirichlet random search fallback.
        """
        if returns_df.empty or returns_df.shape[1] == 0:
            return {}

        mu = returns_df.mean()            # daily means
        Sigma = returns_df.cov()
        tickers = list(returns_df.columns)

        # Guard against singular covariance
        try:
            Sigma_inv = np.linalg.pinv(Sigma.values)
        except Exception:
            Sigma_inv = np.linalg.pinv(Sigma.values)

        ones = np.ones(len(tickers))
        # Unconstrained tangency (with risk-free)
        k = Sigma_inv @ (mu.values - daily_rf * ones)
        if np.allclose(k.sum(), 0):
            k = np.maximum(k, 0)

        w_unconstrained = k / np.sum(k) if np.sum(k) != 0 else np.ones(len(tickers)) / len(tickers)

        # If all weights >= 0, take it
        if np.all(w_unconstrained >= -1e-9):
            w_unconstrained = np.clip(w_unconstrained, 0, None)
            w_unconstrained = w_unconstrained / w_unconstrained.sum() if w_unconstrained.sum() > 0 else np.ones(len(tickers))/len(tickers)
            return dict(zip(tickers, map(float, w_unconstrained)))

        # Fallback: random Dirichlet search (long-only, sum=1)
        def sharpe_for_w(w):
            pr = (returns_df @ w)
            ex = pr - daily_rf
            sd = pr.std()
            return -((ex.mean() * 252) / (sd * np.sqrt(252)) if sd != 0 else -1e9)  # negative for minimization style

        best_w = None
        best_s = 1e9
        rng = np.random.default_rng(42)
        trials = 20000  # fast but decent; bump if you want even finer
        for _ in range(trials):
            w = rng.dirichlet(np.ones(len(tickers)))
            s = sharpe_for_w(w)
            if s < best_s:
                best_s, best_w = s, w

        return dict(zip(tickers, map(float, best_w)))

    # ---------- helpers for scrollable areas ----------
    def _make_scroll_area(self, parent):
        """Return (canvas, inner_frame, scrollbar, window_id) for a vertical scroll area."""
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        # keep scrollregion & width synced
        def on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", on_inner_configure)

        def on_canvas_configure(event):
            canvas.itemconfig(win_id, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)

        # mouse wheel
        def _on_mousewheel(event):
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_linux_scroll_up(_): canvas.yview_scroll(-3, "units")
        def _on_linux_scroll_down(_): canvas.yview_scroll(3, "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)        # Windows / macOS
        canvas.bind_all("<Button-4>", _on_linux_scroll_up)     # Linux
        canvas.bind_all("<Button-5>", _on_linux_scroll_down)   # Linux

        return canvas, inner, vbar, win_id

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        title_label = ttk.Label(main_frame, text="Portfolio Analysis Calculator", font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        self.create_input_panel(main_frame)
        self.create_results_panel(main_frame)
    
    def create_input_panel(self, parent):
        input_frame = ttk.LabelFrame(parent, text="Portfolio Input", padding="15")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        input_frame.columnconfigure(0, weight=1)

        ttk.Label(input_frame, text="Enter Ticker Symbols:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Label(input_frame, text="(Separated by commas or spaces)", font=('Arial', 8)).grid(row=1, column=0, sticky=tk.W, pady=(0, 10))

        self.ticker_entry = tk.Text(input_frame, height=3, width=30, font=('Arial', 10))
        self.ticker_entry.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.ticker_entry.insert("1.0", "AAPL, MSFT, GOOGL, TSLA, AMZN")

        # (Removed Equal/Custom weighting UI)

        ttk.Label(input_frame, text="Analysis Period:", font=('Arial', 10, 'bold')).grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(input_frame, textvariable=self.period_var, values=["1mo", "3mo", "6mo", "1y", "2y", "5y"], state="readonly", width=10)
        period_combo.grid(row=4, column=0, sticky=tk.W, pady=(0, 20))

        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        self.analyze_button = ttk.Button(button_frame, text="Analyze Portfolio", command=self.start_analysis)
        self.analyze_button.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(button_frame, text="Save Results", command=self.save_results).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))

        # Suggestions controls
        sugg_frame = ttk.LabelFrame(input_frame, text="Sharpe Boost Suggestions", padding="10")
        sugg_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        ttk.Label(sugg_frame, text="Candidate Universe Scanned Automatically").grid(row=0, column=0, sticky=tk.W)
        self.refresh_sugg_btn = ttk.Button(sugg_frame, text="Refresh Suggestions", command=self.refresh_suggestions)
        self.refresh_sugg_btn.grid(row=0, column=1, sticky=tk.E)
        self.suggestion_tree = ttk.Treeview(sugg_frame, columns=("Type","Ticker","Est ΔSharpe","Note"), show="headings", height=8)
        for col, w in zip(["Type","Ticker","Est ΔSharpe","Note"], [80,120,100,220]):
            self.suggestion_tree.heading(col, text=col)
            self.suggestion_tree.column(col, width=w, anchor=tk.W)
        self.suggestion_tree.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(8,4))
        self.apply_sugg_btn = ttk.Button(sugg_frame, text="Apply Selected Suggestion", command=self.apply_selected_suggestion)
        self.apply_sugg_btn.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.progress = ttk.Progressbar(input_frame, mode='indeterminate')
        self.progress.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self.status_label = ttk.Label(input_frame, text="Ready to analyze portfolio", foreground="green")
        self.status_label.grid(row=8, column=0, sticky=tk.W, pady=(5, 0))
    
    def create_results_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.create_summary_tab()
        self.create_charts_tab()
        self.create_buy_hold_tab()
        self.create_covariance_tab()
        self.create_fundamentals_tab()
        self.create_data_tab()
        self.create_risk_visuals_tab()
    
    def create_summary_tab(self):
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="Portfolio Summary")

        # Monospaced text (already used in your code)
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=20, width=60, font=('Courier', 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))
        self.summary_text.insert(tk.END, "Portfolio analysis results will appear here after running analysis.\n\n")
        self.summary_text.insert(tk.END, "Features:\n• Max-Sharpe weights\n• Individual stock performance\n• Pairwise covariance analysis\n• Risk and return metrics\n• Diversification assessment\n\nEnter your portfolio tickers and click 'Analyze Portfolio' to begin.")

        # A dedicated area for the weights bar chart (drawn by update_summary_tab)
        self.summary_chart_frame = ttk.Frame(summary_frame)
        self.summary_chart_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

    def create_charts_tab(self):
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="Portfolio Charts")

        timeframe_frame = ttk.Frame(charts_frame)
        timeframe_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        ttk.Label(timeframe_frame, text="Time Frame:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        timeframes = [("1D", "1d"), ("1W", "5d"), ("1M", "1mo"), ("3M", "3mo"), ("6M", "6mo"), ("YTD", "ytd"), ("1Y", "1y")]
        self.timeframe_buttons = {}
        for label, value in timeframes:
            btn = ttk.Button(timeframe_frame, text=label, width=6, command=lambda v=value, l=label: self.update_chart_timeframe(v, l))
            btn.pack(side=tk.LEFT, padx=2)
            self.timeframe_buttons[label] = btn

        # Scrollable stack of charts
        self.chart_canvas, self.chart_inner, _, _ = self._make_scroll_area(charts_frame)
        self.chart_placeholder = ttk.Label(self.chart_inner, text="Portfolio performance charts will appear here after analysis.", font=('Arial', 11), justify=tk.CENTER)
        self.chart_placeholder.pack(expand=True, pady=10)

    # ================
    # Buy & Hold (scrollable stacked)
    # ================
    def create_buy_hold_tab(self):
        bh_frame = ttk.Frame(self.notebook)
        self.notebook.add(bh_frame, text="Buy & Hold")

        top = ttk.Frame(bh_frame); top.pack(fill=tk.X, padx=10, pady=(10,0))
        ttk.Label(top, text="Buy & Hold Time Frame:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.bh_tf_buttons = {}
        for label, value in [("1M","1mo"),("3M","3mo"),("6M","6mo"),("YTD","ytd"),("1Y","1y"),("2Y","2y"),("5Y","5y")]:
            b = ttk.Button(top, text=label, width=6, command=lambda v=value,l=label: self.update_buy_hold(v,l))
            b.pack(side=tk.LEFT, padx=2); self.bh_tf_buttons[label] = b

        ttk.Label(top, text="   |   ").pack(side=tk.LEFT)
        ttk.Label(top, text="Rolling Sharpe window:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.rs_window_var = tk.StringVar(value="60")
        self.rs_window = ttk.Combobox(top, textvariable=self.rs_window_var, values=["20","30","60","90","120","252"], state="readonly", width=6)
        self.rs_window.pack(side=tk.LEFT, padx=(5,2))
        ttk.Button(top, text="Apply", command=lambda: self.update_buy_hold(getattr(self, "_last_bh_period", "1y"), getattr(self, "_last_bh_label", "1Y"))).pack(side=tk.LEFT)

        # Scrollable stack for the three plots
        self.bh_canvas, self.bh_inner, _, _ = self._make_scroll_area(bh_frame)
        ttk.Label(self.bh_inner, text="Buy & Hold portfolio chart will appear here.", font=('Arial', 11)).pack(expand=True, pady=10)

        self._last_bh_period = "1y"
        self._last_bh_label = "1Y"

    def update_buy_hold(self, period_value, label):
        if not self.analysis_results: 
            return

        self._last_bh_period = period_value
        self._last_bh_label = label

        for w in self.bh_inner.winfo_children(): w.destroy()
        try:
            # fetch prices
            chart_data = {}
            for t in self.analysis_results['tickers']:
                d = yf.Ticker(t).history(period=period_value)
                if not d.empty:
                    chart_data[t] = d['Close']
            if not chart_data:
                ttk.Label(self.bh_inner, text="No data available for selected timeframe").pack(expand=True, pady=10); 
                return

            df_px = pd.DataFrame(chart_data).dropna()
            ret_df = df_px.pct_change().dropna()

            # weights: use optimized weights and align to available tickers, then normalize
            base_w = self.analysis_results.get('weights', {})
            w = {t: base_w.get(t, 0.0) for t in df_px.columns}
            s = sum(w.values())
            if s == 0:
                w = {t: 1.0/len(df_px.columns) for t in df_px.columns}
            else:
                w = {t: v/s for t, v in w.items()}

            # portfolio returns & cumulative
            port_ret = (ret_df * pd.Series(w)).sum(axis=1)
            cum = (1 + port_ret).cumprod() * 100

            # rolling sharpe & drawdown
            daily_rf = self.analysis_results['sharpe_metrics']['risk_free_rate'] / 252.0
            window = int(self.rs_window_var.get())
            roll_sh = rolling_sharpe(port_ret, window=window, daily_rf=daily_rf)
            dd_series, mdd = drawdown(cum)

            # (A) Buy & Hold curve (keeps existing cursor you already have)
            fig1 = Figure(figsize=(12,4), dpi=100, constrained_layout=True)
            ax1 = fig1.add_subplot(1,1,1)
            (line,) = ax1.plot(cum.index, cum.values, linewidth=2)
            ax1.set_title(f"Buy & Hold Portfolio Value ({label})", fontweight='bold')
            ax1.set_ylabel("Value (Base=100)")
            ax1.grid(True, alpha=0.3)
            xdates = mdates.date2num(cum.index.to_pydatetime())
            cursor = mplcursors.cursor(line, hover=True)
            @cursor.connect("add")
            def _on_add(sel):
                x, _ = sel.target
                idx = int(np.argmin(np.abs(xdates - x)))
                dt = cum.index[idx]
                val = cum.iloc[idx]
                sel.annotation.set(text=f"{dt:%Y-%m-%d}\n{val:.2f}",
                                fontsize=9,
                                bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.9))
            canvas1 = FigureCanvasTkAgg(fig1, self.bh_inner); canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,14))

            # (B) Rolling Sharpe with tooltip (unchanged)
            fig2 = Figure(figsize=(12,3.6), dpi=100, constrained_layout=True)
            ax2 = fig2.add_subplot(1,1,1)
            if not roll_sh.empty:
                (line2,) = ax2.plot(roll_sh.index, roll_sh.values, linewidth=1.8)
                x2 = mdates.date2num(roll_sh.index.to_pydatetime())
                cur2 = mplcursors.cursor(line2, hover=True)
                @cur2.connect("add")
                def _on_add_rs(sel):
                    xx, _ = sel.target
                    idx = int(np.argmin(np.abs(x2 - xx)))
                    sel.annotation.set(text=f"{roll_sh.index[idx]:%Y-%m-%d}\nSharpe={roll_sh.iloc[idx]:.3f}",
                                    fontsize=9,
                                    bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95))
            ax2.axhline(0, linestyle='--', linewidth=1)
            ax2.set_title(f"Rolling Sharpe (window={window})", fontweight='bold')
            ax2.set_ylabel("Sharpe"); ax2.grid(True, alpha=0.3)
            canvas2 = FigureCanvasTkAgg(fig2, self.bh_inner); canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,14))

            # (C) Drawdown with tooltip (unchanged)
            fig3 = Figure(figsize=(12,3.6), dpi=100, constrained_layout=True)
            ax3 = fig3.add_subplot(1,1,1)
            if not dd_series.empty:
                ax3.fill_between(dd_series.index, dd_series.values, 0, step=None, alpha=0.4)
                (line3,) = ax3.plot(dd_series.index, dd_series.values, linewidth=1.5)
                x3 = mdates.date2num(dd_series.index.to_pydatetime())
                cur3 = mplcursors.cursor(line3, hover=True)
                @cur3.connect("add")
                def _on_add_dd(sel):
                    xx, _ = sel.target
                    idx = int(np.argmin(np.abs(x3 - xx)))
                    sel.annotation.set(text=f"{dd_series.index[idx]:%Y-%m-%d}\nDD={dd_series.iloc[idx]*100:.2f}%",
                                    fontsize=9,
                                    bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95))
            ax3.set_title(f"Drawdown (Max DD: {mdd*100:.2f}%)", fontweight='bold')
            ax3.set_ylabel("Drawdown"); ax3.grid(True, alpha=0.3)
            ax3.set_ylim(min(-1, dd_series.min() if not dd_series.empty else -0.1), 0.02)
            canvas3 = FigureCanvasTkAgg(fig3, self.bh_inner); canvas3.draw()
            canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,10))

        except Exception as e:
            ttk.Label(self.bh_inner, text=f"Error: {e}", foreground="red").pack(expand=True, pady=10)

    # ==========================
    # Financial Ratios
    # ==========================
    def create_fundamentals_tab(self):
        fr = ttk.Frame(self.notebook)
        self.notebook.add(fr, text="Financial Ratios")

        ctrl = ttk.Frame(fr); ctrl.pack(fill=tk.X, padx=10, pady=(10,0))
        ttk.Label(ctrl, text="Sort by:", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.fund_sort_key = tk.StringVar(value="Ticker")
        self.fund_sort_order_desc = tk.BooleanVar(value=False)
        self.fund_sort_combo = ttk.Combobox(ctrl, textvariable=self.fund_sort_key,
            values=["Ticker","Day %","% from 52w Low","Ann Vol","Sector","RSI(14)","Insider %","Sharpe (ann)"],
            state="readonly", width=22)
        self.fund_sort_combo.pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(ctrl, text="Descending", variable=self.fund_sort_order_desc).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Apply Sort", command=self.apply_fund_sort).pack(side=tk.LEFT, padx=6)

        cols = ["Ticker","Day %","% from 52w Low","Ann Vol","Sector","RSI(14)","Insider %","Sharpe (ann)"]
        self.fund_tree = ttk.Treeview(fr, columns=cols, show="headings", height=18)
        for c, w in zip(cols, [80,90,120,90,180,80,90,100]):
            self.fund_tree.heading(c, text=c, command=lambda col=c: self.sort_fund_table(col))
            self.fund_tree.column(c, width=w, anchor=tk.W)
        self.fund_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,0), pady=10)
        sb = ttk.Scrollbar(fr, orient=tk.VERTICAL, command=self.fund_tree.yview)
        self.fund_tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,10), pady=10)
        self._fund_rows_cache = []

    # ==========================
    # Covariance Tab
    # ==========================
    def create_covariance_tab(self):
        cov_frame = ttk.Frame(self.notebook)
        self.notebook.add(cov_frame, text="Covariance Matrix")
        self.covariance_text = scrolledtext.ScrolledText(cov_frame, height=15, width=60, font=('Courier', 9))
        self.covariance_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.covariance_text.insert(tk.END, "Detailed covariance analysis will appear here after running analysis.\n\n")
        self.covariance_text.insert(tk.END, "This will include:\n• Pairwise covariances in [STOCK1 -> STOCK2]: value format\n• Correlation coefficients\n• Diversification insights\n• Risk concentration analysis\n")

    # ======================================
    # Risk & Beta Visualizations
    # ======================================
    def create_risk_visuals_tab(self):
        rv = ttk.Frame(self.notebook)
        self.notebook.add(rv, text="Risk & Beta Visuals")

        # Make a single stacked + scrollable column for all risk/beta visuals
        self.rv_canvas, self.rv_inner, _, _ = self._make_scroll_area(rv)

        # Controls row (stays pinned at the top of the inner frame)
        ctrl = ttk.Frame(self.rv_inner)
        ctrl.pack(fill=tk.X, padx=5, pady=(0,6))

        ttk.Label(ctrl, text="Rolling Beta (select ticker):", font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        self.beta_select = ttk.Combobox(ctrl, values=[], state="readonly", width=12)
        self.beta_select.pack(side=tk.LEFT, padx=(6,0))
        self.beta_select.bind("<<ComboboxSelected>>", lambda e: self.draw_rolling_beta())

        # Containers (stacked)
        self.beta_frame = ttk.LabelFrame(self.rv_inner, text="Beta vs SPY", padding=8)
        self.beta_frame.pack(fill=tk.BOTH, expand=True, pady=(0,8))

        self.rolling_beta_frame = ttk.LabelFrame(self.rv_inner, text="Rolling 60D Beta", padding=8)
        self.rolling_beta_frame.pack(fill=tk.BOTH, expand=True, pady=(0,8))

        self.arc_frame = ttk.LabelFrame(self.rv_inner, text="Covariance Arc Diagram", padding=8)
        self.arc_frame.pack(fill=tk.BOTH, expand=True, pady=(0,8))

        self.cluster_pca_frame = ttk.LabelFrame(self.rv_inner, text="Correlation Clustermap & PCA Scree", padding=8)
        self.cluster_pca_frame.pack(fill=tk.BOTH, expand=True, pady=(0,8))

    def draw_beta_bar(self):
        for w in self.beta_frame.winfo_children(): w.destroy()
        if not self.analysis_results: return
        try:
            # ensure market cache covers portfolio span
            if (self.market_returns_cache is None or
                self.market_returns_cache.index.min() > self.portfolio_returns.index.min() or
                self.market_returns_cache.index.max() < self.portfolio_returns.index.max()):
                mkt = yf.Ticker("SPY").history(
                    start=self.portfolio_returns.index.min()-pd.Timedelta(days=5),
                    end=self.portfolio_returns.index.max()+pd.Timedelta(days=5))
                self.market_returns_cache = mkt['Close'].pct_change().dropna()

            betas = {}
            for t in self.analysis_results['tickers']:
                asset_ret = self.portfolio_returns[t].dropna()
                betas[t] = beta_vs_market(asset_ret, self.market_returns_cache)

            fig = Figure(figsize=(12,3.6), dpi=100, constrained_layout=True)
            ax = fig.add_subplot(1,1,1)
            labels = list(betas.keys()); vals = [betas[k] for k in labels]
            bars = ax.bar(labels, vals)
            ax.axhline(1.0, linestyle='--', linewidth=1)
            ax.set_title("Beta vs SPY", fontweight='bold')
            ax.set_ylabel("Beta"); ax.grid(True, axis='y', alpha=0.3)
            ax.set_xticklabels(labels, rotation=45, ha='right')

            # mplcursors on bars
            cur = mplcursors.cursor(bars, hover=True)
            @cur.connect("add")
            def _on_add(sel):
                i = sel.index
                sel.annotation.set(text=f"{labels[i]}: {vals[i]:.3f}",
                                fontsize=9,
                                bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95))

            canvas = FigureCanvasTkAgg(fig, self.beta_frame); canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # populate dropdown if empty
            self.beta_select['values'] = labels
            if not self.beta_select.get() and labels: self.beta_select.set(labels[0])

        except Exception as e:
            ttk.Label(self.beta_frame, text=f"Beta error: {e}", foreground="red").pack()


    def draw_rolling_beta(self):
        for w in self.rolling_beta_frame.winfo_children(): w.destroy()
        if not self.analysis_results: return
        t = self.beta_select.get()
        if not t: return
        try:
            mkt = yf.Ticker("SPY").history(
                start=self.portfolio_returns.index.min()-pd.Timedelta(days=10),
                end=self.portfolio_returns.index.max()+pd.Timedelta(days=10)
            )['Close'].pct_change().dropna()
            a = self.portfolio_returns[t].dropna()
            df = pd.concat([a, mkt], axis=1).dropna(); df.columns = ['asset','mkt']
            window = 60
            if df.shape[0] < window:
                ttk.Label(self.rolling_beta_frame, text="Not enough data for rolling beta.").pack(); return

            roll = []
            for i in range(window, len(df)):
                roll.append((df.index[i], beta_vs_market(df['asset'].iloc[i-window:i], df['mkt'].iloc[i-window:i])))
            rb = pd.DataFrame(roll, columns=['date','beta']).set_index('date')

            fig = Figure(figsize=(12,3.6), dpi=100, constrained_layout=True)
            ax = fig.add_subplot(1,1,1)
            (line,) = ax.plot(rb.index, rb['beta'], linewidth=1.8)
            ax.axhline(1.0, linestyle='--', linewidth=1)
            ax.set_title(f"Rolling 60D Beta: {t}", fontweight='bold'); ax.grid(True, alpha=0.3)

            # mplcursors on line (date/value)
            xdates = mdates.date2num(rb.index.to_pydatetime())
            cur = mplcursors.cursor(line, hover=True)
            @cur.connect("add")
            def _on_add(sel):
                x, y = sel.target
                idx = int(np.argmin(np.abs(xdates - x)))
                dt = rb.index[idx]; val = rb['beta'].iloc[idx]
                sel.annotation.set(text=f"{dt:%Y-%m-%d}\nβ={val:.3f}",
                                fontsize=9,
                                bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95))

            canvas = FigureCanvasTkAgg(fig, self.rolling_beta_frame); canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            ttk.Label(self.rolling_beta_frame, text=f"Rolling beta error: {e}", foreground="red").pack()


    def draw_covariance_arc(self):
        for w in self.arc_frame.winfo_children(): w.destroy()
        if not self.analysis_results: return
        try:
            ret = self.analysis_results['individual_returns']
            cov = ret.cov(); labels = list(cov.columns)

            fig = Figure(figsize=(12,3.6), dpi=100, constrained_layout=True)
            ax = fig.add_subplot(1,1,1)
            n = len(labels); xs = np.linspace(0, 1, n)
            pts = ax.scatter(xs, np.zeros(n), s=60)
            for i, label in enumerate(labels):
                ax.text(xs[i], -0.06, label, ha='center', va='top', rotation=45)

            max_abs = np.abs(cov.values[np.triu_indices(n,1)]).max() if n>1 else 1
            for i in range(n):
                for j in range(i+1, n):
                    strength = abs(cov.iloc[i,j]) / (max_abs if max_abs!=0 else 1)
                    if strength > 0:
                        xm = (xs[i] + xs[j]) / 2
                        r = (xs[j] - xs[i]) / 2
                        theta = np.linspace(0, np.pi, 60)
                        x_arc = xm + r * np.cos(theta); y_arc = r * np.sin(theta)
                        ax.plot(x_arc, y_arc, alpha=0.2 + 0.6*strength, linewidth=1 + 3*strength)

            ax.axis('off'); ax.set_title("Covariance Arc Diagram", fontweight='bold')

            # mplcursors on the node scatter -> show ticker
            cur = mplcursors.cursor(pts, hover=True)
            @cur.connect("add")
            def _on_add(sel):
                i = sel.index
                sel.annotation.set(text=f"{labels[i]}", fontsize=9,
                                bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95))

            canvas = FigureCanvasTkAgg(fig, self.arc_frame); canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            ttk.Label(self.arc_frame, text=f"Arc diagram error: {e}", foreground="red").pack()

    def draw_cluster_and_pca(self):
        for w in self.cluster_pca_frame.winfo_children(): w.destroy()
        if not self.analysis_results: return
        try:
            ret = self.analysis_results['individual_returns']
            corr = ret.corr()

            fig = Figure(figsize=(12,4.6), dpi=100, constrained_layout=True)
            ax = fig.add_subplot(1,2,1)
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
            tickers = list(corr.index)
            ax.set_xticks(range(len(tickers))); ax.set_yticks(range(len(tickers)))
            ax.set_xticklabels(tickers, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tickers, fontsize=8)
            ax.set_title("Correlation Heatmap", fontweight='bold')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Corr')

            # PCA Scree (cumulative)
            ax2 = fig.add_subplot(1,2,2)
            cov = ret.cov(); vals, _ = np.linalg.eigh(cov.values); vals = vals[::-1]
            exp_var = vals / vals.sum() if vals.sum()!=0 else vals
            (line2,) = ax2.plot(range(1, len(vals)+1), exp_var.cumsum(), marker='o')
            ax2.set_title("PCA Scree (Cumulative Variance)", fontweight='bold')
            ax2.set_xlabel("Components"); ax2.set_ylabel("Cumulative Variance Explained")
            ax2.grid(True, alpha=0.3)

            # mplcursors: heatmap cell (i,j) -> show tickers + value
            # We'll create a cursor tied to the image artist and compute indices from mouse event.
            cur1 = mplcursors.cursor(im, hover=True)
            @cur1.connect("add")
            def _on_add_heat(sel):
                # transform mouse to image indices
                x, y = sel.target  # float image coords
                j = int(round(x)); i = int(round(y))
                if 0 <= i < len(tickers) and 0 <= j < len(tickers):
                    val = corr.iloc[i, j]
                    sel.annotation.set(text=f"{tickers[i]} ↔ {tickers[j]}\nρ={val:.3f}",
                                    fontsize=9,
                                    bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95))
                else:
                    sel.annotation.set_visible(False)

            # mplcursors on scree line points
            cur2 = mplcursors.cursor(line2, hover=True)
            @cur2.connect("add")
            def _on_add_scree(sel):
                x, y = sel.target
                comp = int(round(x))
                if 1 <= comp <= len(vals):
                    sel.annotation.set(text=f"Comp {comp}\nCumVar={y:.3f}",
                                    fontsize=9,
                                    bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95))

            canvas = FigureCanvasTkAgg(fig, self.cluster_pca_frame); canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            ttk.Label(self.cluster_pca_frame, text=f"Cluster/PCA error: {e}", foreground="red").pack()

    # ======================
    # Raw Data Tab
    # ======================
    def create_data_tab(self):
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Raw Data")
        columns = ('Date', 'Portfolio Value', 'Daily Return')
        self.data_tree = ttk.Treeview(data_frame, columns=columns, show='tree headings', height=20)
        self.data_tree.column('#0', width=0, stretch=False)
        self.data_tree.column('Date', width=100, anchor=tk.W)
        self.data_tree.column('Portfolio Value', width=120, anchor=tk.E)
        self.data_tree.column('Daily Return', width=120, anchor=tk.E)
        self.data_tree.heading('Date', text='Date', anchor=tk.W)
        self.data_tree.heading('Portfolio Value', text='Portfolio Value', anchor=tk.E)
        self.data_tree.heading('Daily Return', text='Daily Return (%)', anchor=tk.E)
        data_scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=data_scrollbar.set)
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
    
    def toggle_weight_input(self):
        for widget in self.weights_frame.winfo_children():
            widget.destroy()
        if self.weight_var.get() == "custom":
            ttk.Label(self.weights_frame, text="Enter weights (sum to 1.0):").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
            self.weights_note = ttk.Label(self.weights_frame, text="Weight inputs will appear after entering tickers", foreground="gray")
            self.weights_note.grid(row=1, column=0, columnspan=2, sticky=tk.W)
    
    def create_weight_inputs(self, tickers):
        if self.weight_var.get() != "custom":
            return
        for widget in self.weights_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.weights_frame, text="Enter weights (sum to 1.0):", font=('Arial', 9, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        self.weight_entries = {}
        equal_weight = 1.0 / len(tickers)
        for i, ticker in enumerate(tickers):
            ttk.Label(self.weights_frame, text=f"{ticker}:").grid(row=i+1, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(self.weights_frame, width=10)
            entry.grid(row=i+1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
            entry.insert(0, f"{equal_weight:.3f}")
            self.weight_entries[ticker] = entry
    
    def validate_tickers(self, ticker_text):
        if not ticker_text.strip():
            return []
        if ',' in ticker_text:
            tickers = [t.strip().upper() for t in ticker_text.split(',')]
        else:
            tickers = ticker_text.upper().split()
        tickers = list(dict.fromkeys([t for t in tickers if t]))
        return tickers
    
    def validate_weights(self, tickers):
        if self.weight_var.get() == "equal":
            return None
        weights = {}; total_weight = 0
        try:
            for ticker in tickers:
                weight_str = self.weight_entries[ticker].get().strip()
                weight = float(weight_str[:-1])/100 if weight_str.endswith('%') else float(weight_str)
                if weight < 0: raise ValueError(f"Weight for {ticker} cannot be negative")
                if weight > 1: raise ValueError(f"Weight for {ticker} cannot be greater than 1.0")
                weights[ticker] = weight; total_weight += weight
            if abs(total_weight - 1.0) > 0.01:
                response = messagebox.askyesno("Weight Validation", f"Weights sum to {total_weight:.3f}, not 1.0.\n\nNormalize weights to sum to 1.0?")
                if response:
                    for ticker in weights: weights[ticker] = weights[ticker] / total_weight
                else:
                    return None
            return weights
        except ValueError as e:
            messagebox.showerror("Weight Error", str(e)); return None
    
    def start_analysis(self):
        ticker_text = self.ticker_entry.get("1.0", tk.END).strip()
        tickers = self.validate_tickers(ticker_text)
        if len(tickers) < 2:
            messagebox.showerror("Input Error", "Please enter at least 2 ticker symbols."); return

        self.analyze_button.configure(state='disabled'); self.progress.start()
        self.status_label.configure(text="Analyzing portfolio...", foreground="orange")
        analysis_thread = threading.Thread(target=self.run_analysis, args=(tickers,))
        analysis_thread.daemon = True; analysis_thread.start()

    def run_analysis(self, tickers):
        try:
            self.root.after(0, lambda: self.status_label.configure(text="Fetching stock data...", foreground="orange"))
            portfolio_data = {}; successful_tickers = []; period = self.period_var.get()
            for ticker in tickers:
                try:
                    data = yf.Ticker(ticker).history(period=period)
                    if not data.empty:
                        portfolio_data[ticker] = data; successful_tickers.append(ticker)
                except:
                    continue
            if len(successful_tickers) < 2:
                raise ValueError("Could not fetch data for enough stocks. Please check ticker symbols.")
            self.portfolio_data = portfolio_data

            self.root.after(0, lambda: self.status_label.configure(text="Calculating returns...", foreground="orange"))
            price_data = {t: d['Close'] for t, d in portfolio_data.items()}
            price_df = pd.DataFrame(price_data).dropna()
            self.portfolio_returns = price_df.pct_change().dropna()

            self.root.after(0, lambda: self.status_label.configure(text="Estimating risk-free rate...", foreground="orange"))
            try:
                treasury_data = yf.Ticker("^IRX").history(period="5d")
                annual_rf_rate = treasury_data['Close'].iloc[-1] / 100 if not treasury_data.empty else 0.045
            except:
                annual_rf_rate = 0.045
            daily_rf_rate = annual_rf_rate / 252.0

            # ==== NEW: Max-Sharpe weights (long-only, sum=1) ====
            self.root.after(0, lambda: self.status_label.configure(text="Optimizing weights for max Sharpe...", foreground="orange"))
            indiv_ret = self.portfolio_returns[successful_tickers]
            opt_w = self._max_sharpe_weights(indiv_ret, daily_rf_rate)  # dict ticker->weight

            # align, normalize (safety)
            w_series = pd.Series({t: opt_w.get(t, 0.0) for t in indiv_ret.columns}, dtype=float)
            w_series = w_series / (w_series.sum() if w_series.sum()!=0 else 1.0)

            # portfolio return series using optimal weights
            portfolio_returns = (indiv_ret * w_series).sum(axis=1)

            self.root.after(0, lambda: self.status_label.configure(text="Calculating Sharpe & covariances...", foreground="orange"))
            excess_returns = portfolio_returns - daily_rf_rate
            mean_return = portfolio_returns.mean(); std_return = portfolio_returns.std(); mean_excess_return = excess_returns.mean()
            sharpe_ratio = mean_excess_return / std_return if std_return != 0 else 0
            annualized_return = mean_return * 252
            annualized_volatility = std_return * np.sqrt(252)
            annualized_sharpe = (mean_excess_return * 252) / (std_return * np.sqrt(252)) if std_return != 0 else 0

            covariances = {}
            for t1, t2 in itertools.combinations(successful_tickers, 2):
                covariances[f"[{t1} -> {t2}]"] = np.cov(indiv_ret[t1], indiv_ret[t2])[0,1]
                covariances[f"[{t2} -> {t1}]"] = covariances[f"[{t1} -> {t2}]"]

            self.analysis_results = {
                'tickers': successful_tickers,
                'weights': {k: float(v) for k, v in w_series.items()},   # store optimal weights
                'portfolio_returns': portfolio_returns,
                'individual_returns': indiv_ret,
                'sharpe_metrics': {
                    'daily_return': mean_return,
                    'daily_volatility': std_return,
                    'daily_sharpe': sharpe_ratio,
                    'annualized_return': annualized_return,
                    'annualized_volatility': annualized_volatility,
                    'annualized_sharpe': annualized_sharpe,
                    'risk_free_rate': annual_rf_rate,
                    'total_days': len(portfolio_returns)
                },
                'covariances': covariances,
                'price_data': price_df
            }

            try:
                mkt = yf.Ticker("SPY").history(
                    start=price_df.index.min()-pd.Timedelta(days=10),
                    end=price_df.index.max()+pd.Timedelta(days=10)
                )['Close'].pct_change().dropna()
                self.market_returns_cache = mkt
            except:
                self.market_returns_cache = None

            self.root.after(0, self.analysis_complete)
        except Exception as e:
            self.root.after(0, lambda: self.analysis_error(f"Analysis failed: {str(e)}"))
    
    def analysis_complete(self):
        self.progress.stop(); self.analyze_button.configure(state='normal')
        self.status_label.configure(text="Analysis complete!", foreground="green")
        self.update_summary_tab(); self.update_covariance_tab(); self.update_data_tab()
        self.update_charts("1y", "1Y"); self.update_buy_hold("1y", "1Y")
        self.populate_financial_ratios()
        self.draw_beta_bar(); self.draw_rolling_beta(); self.draw_covariance_arc(); self.draw_cluster_and_pca()
        self.refresh_suggestions()
        self.notebook.select(0)

    def analysis_error(self, error_msg):
        self.progress.stop(); self.analyze_button.configure(state='normal')
        self.status_label.configure(text="Analysis failed", foreground="red")
        messagebox.showerror("Analysis Error", error_msg)
    
    def update_summary_tab(self):
        if not self.analysis_results: 
            return

        res = self.analysis_results
        tickers = res['tickers']
        weights = res.get('weights', {})

        # ---------- helpers ----------
        def pct(x, digits=2):
            try:
                return f"{x*100:.{digits}f}%"
            except Exception:
                return "N/A"

        def line(ch='-', n=70):
            return ch * n

        # Sort weights desc and prepare rows
        weight_items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        total_w = sum(w for _, w in weight_items) if weight_items else 0.0

        # Individual stats table (annualized)
        indiv = res['individual_returns']
        stats_rows = []
        for t in tickers:
            r = indiv[t].dropna()
            ann_ret = r.mean() * 252 if not r.empty else float('nan')
            ann_vol = r.std() * (252**0.5) if not r.empty else float('nan')
            w = weights.get(t, 0.0)
            stats_rows.append((t, w, ann_ret, ann_vol))

        # column widths
        col_t, col_w, col_r, col_v = 10, 10, 12, 12
        header_weights = f"{'Ticker'.ljust(col_t)}{'Weight'.rjust(col_w)}"
        header_stats   = f"{'Ticker'.ljust(col_t)}{'Weight'.rjust(col_w)}{'AnnRet'.rjust(col_r)}{'AnnVol'.rjust(col_v)}"

        # Build text
        out = []
        out.append(line('='))
        out.append("PORTFOLIO ANALYSIS RESULTS")
        out.append(line('='))

        out.append("\n📋 PORTFOLIO COMPOSITION:")
        out.append(f"  Stocks: {', '.join(tickers)}")
        out.append(f"  Number of stocks: {len(tickers)}")
        out.append("  Weighting: Max-Sharpe (long-only, sum = 1.0)")
        out.append("")
        out.append("  Weights")
        out.append("  " + header_weights)
        out.append("  " + line('-', len(header_weights)))
        if weight_items:
            for t, w in weight_items:
                out.append(f"  {t.ljust(col_t)}{pct(w,2).rjust(col_w)}")
        else:
            out.append("  (No weights available)")
        out.append("  " + line('-', len(header_weights)))
        out.append(f"  {'TOTAL'.ljust(col_t)}{pct(total_w,2).rjust(col_w)}")

        # Sharpe metrics
        sm = res['sharpe_metrics']
        out.append("\n📈 PORTFOLIO SHARPE RATIO (" + self.period_var.get().upper() + "):")
        out.append(f"  Risk-free (Treasury): {pct(sm['risk_free_rate'],2)}")
        out.append(f"  Annualized Return:    {pct(sm['annualized_return'],2)}")
        out.append(f"  Annualized Volatility:{pct(sm['annualized_volatility'],2)}")
        out.append(f"  Sharpe Ratio:         {sm['annualized_sharpe']:.4f}")

        s = sm['annualized_sharpe']
        interp = ("Excellent (> 2.0)" if s > 2.0 else
                "Good (1.0 – 2.0)" if s > 1.0 else
                "Acceptable (0.5 – 1.0)" if s > 0.5 else
                "Poor (0 – 0.5)" if s > 0   else
                "Very Poor (< 0)")
        out.append(f"  Interpretation:       {interp}")

        # Per-ticker stats
        out.append("\n📊 INDIVIDUAL STOCK PERFORMANCE (annualized):")
        out.append("  " + header_stats)
        out.append("  " + line('-', len(header_stats)))
        for t, w, ar, av in sorted(stats_rows, key=lambda r: r[2], reverse=True):
            ar_s = pct(ar, 2) if ar == ar else "N/A"
            av_s = pct(av, 2) if av == av else "N/A"
            out.append(f"  {t.ljust(col_t)}{pct(w,2).rjust(col_w)}{ar_s.rjust(col_r)}{av_s.rjust(col_v)}")

        # Covariance / diversification snippet
        covs = res['covariances']
        unique_pairs = {}
        for k, v in covs.items():
            if '->' in k:
                a, b = [s.strip() for s in k.strip('[]').split('->')]
                unique_pairs[tuple(sorted((a, b)))] = v
        if unique_pairs:
            sp = sorted(unique_pairs.items(), key=lambda x: abs(x[1]), reverse=True)
            (hi1, hi2), hi_v = sp[0][0], sp[0][1]
            (lo1, lo2), lo_v = sp[-1][0], sp[-1][1]
            avg_cov = float(np.mean(list(unique_pairs.values())))
            out.append("\n💡 PORTFOLIO INSIGHTS:")
            out.append(f"  Highest covariance: {hi1}-{hi2}  (cov: {hi_v:.6f})")
            out.append(f"  Lowest  covariance: {lo1}-{lo2}  (cov: {lo_v:.6f})")
            divers = ("Low – stocks move together" if avg_cov > 0.0001 else
                    "Moderate – some diversification" if avg_cov > 0.00005 else
                    "Good – well diversified")
            out.append(f"  Avg pairwise covariance: {avg_cov:.6f}")
            out.append(f"  Diversification level:   {divers}")

        out.append(f"\n📅 Analysis Period: {sm['total_days']} trading days")

        # Write text
        self.summary_text.configure(state='normal')
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "\n".join(out))
        self.summary_text.configure(state='normal')

        # ===== Draw bar chart of weights (sorted high -> low) =====
        for w in self.summary_chart_frame.winfo_children():
            w.destroy()

        if weight_items:
            from matplotlib.ticker import PercentFormatter
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import mplcursors
            import matplotlib.dates as mdates  # safe; no-op here but avoids circular imports if used elsewhere

            labels = [k for k, _ in weight_items]
            vals   = [v for _, v in weight_items]

            fig = Figure(figsize=(12, 3.2), dpi=100, constrained_layout=True)
            ax = fig.add_subplot(1,1,1)
            bars = ax.bar(range(len(vals)), vals)
            ax.set_title("Portfolio Weights (Max-Sharpe)", fontweight='bold')
            ax.set_ylabel("Weight")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, axis='y', alpha=0.3)
            # Add values on bars
            try:
                ax.bar_label(bars, labels=[f"{v*100:.1f}%" for v in vals], padding=3, fontsize=9)
            except Exception:
                pass

            # Hover tooltip for quick read
            cur = mplcursors.cursor(bars, hover=True)
            @cur.connect("add")
            def _on_add(sel):
                i = sel.index
                sel.annotation.set(
                    text=f"{labels[i]}: {vals[i]*100:.2f}%",
                    fontsize=9,
                    bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.95)
                )

            canvas = FigureCanvasTkAgg(fig, self.summary_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_covariance_tab(self):
        if not self.analysis_results: return
        results = self.analysis_results
        self.covariance_text.delete(1.0, tk.END)
        cov_text = f"""{'='*60}
DETAILED COVARIANCE ANALYSIS
{'='*60}

🔄 PAIRWISE COVARIANCES (Daily):
"""
        covariances = results['covariances']; unique_pairs = {}
        for cov_key, cov_value in covariances.items():
            if '->' in cov_key:
                t1, t2 = [t.strip() for t in cov_key.strip('[]').split('->')]
                unique_pairs[tuple(sorted([t1, t2]))] = cov_value
        sorted_pairs = sorted(unique_pairs.items(), key=lambda x: abs(x[1]), reverse=True)
        for (t1, t2), cov_value in sorted_pairs:
            cov_text += f"  [{t1} -> {t2}]: {cov_value:.6f}\n"
        cov_text += f"\n📊 CORRELATION COEFFICIENTS:\n"
        for (t1, t2), _ in sorted_pairs:
            r1 = results['individual_returns'][t1]; r2 = results['individual_returns'][t2]
            cov_text += f"  [{t1} -> {t2}]: {np.corrcoef(r1, r2)[0,1]:.4f}\n"
        cov_text += f"\n⚠️ RISK ANALYSIS:\n"
        volatilities = {t: results['individual_returns'][t].std()*np.sqrt(252) for t in results['individual_returns'].columns}
        max_vol_ticker = max(volatilities, key=volatilities.get); min_vol_ticker = min(volatilities, key=volatilities.get)
        cov_text += f"  Highest risk stock: {max_vol_ticker} ({volatilities[max_vol_ticker]*100:.2f}% volatility)\n"
        cov_text += f"  Lowest risk stock: {min_vol_ticker} ({volatilities[min_vol_ticker]*100:.2f}% volatility)\n"
        avg_correlation = np.mean([np.corrcoef(results['individual_returns'][t1], results['individual_returns'][t2])[0,1] 
                                  for t1, t2 in itertools.combinations(results['tickers'], 2)])
        cov_text += f"\n🎯 DIVERSIFICATION METRICS:\n  Average correlation: {avg_correlation:.4f}\n"
        div_assessment = ("Low diversification - consider adding uncorrelated assets" if avg_correlation > 0.7 else
                          "Moderate diversification - room for improvement" if avg_correlation > 0.4 else
                          "Good diversification - portfolio is well spread")
        cov_text += f"  Assessment: {div_assessment}\n"
        self.covariance_text.insert(tk.END, cov_text)
    
    def update_data_tab(self):
        if not self.analysis_results: return
        for item in self.data_tree.get_children(): self.data_tree.delete(item)
        results = self.analysis_results; portfolio_returns = results['portfolio_returns']
        cumulative_value = (1 + portfolio_returns).cumprod() * 100
        for date, ret_val in portfolio_returns.items():
            self.data_tree.insert('', tk.END, values=(date.strftime('%Y-%m-%d'), f"${cumulative_value[date]:.2f}", f"{ret_val*100:.2f}%"))
    
    def update_chart_timeframe(self, timeframe_value, timeframe_label):
        self.current_timeframe = timeframe_label
        for label, btn in self.timeframe_buttons.items():
            btn.configure(style=('Accent.TButton' if label == timeframe_label else 'TButton'))
        self.update_charts(timeframe_value, timeframe_label)
    
    # -------------- Stacked charts (scrollable) --------------
    def update_charts(self, timeframe_value, timeframe_label):
        if not self.analysis_results: return
        for widget in self.chart_inner.winfo_children(): widget.destroy()
        try:
            chart_data = {}
            for ticker in self.analysis_results['tickers']:
                data = yf.Ticker(ticker).history(period=timeframe_value)
                if not data.empty: chart_data[ticker] = data['Close']
            if not chart_data:
                ttk.Label(self.chart_inner, text="No data available for selected timeframe").pack(expand=True, pady=10)
                return

            # (1) Portfolio performance (normalized) — no cursors
            fig1 = Figure(figsize=(12,4), dpi=100, constrained_layout=True)
            ax1 = fig1.add_subplot(1,1,1)
            for ticker, prices in chart_data.items():
                norm = (prices / prices.iloc[0]) * 100
                ax1.plot(norm.index, norm, label=ticker, linewidth=2)
            ax1.set_title(f'Portfolio Performance ({timeframe_label})', fontweight='bold')
            ax1.set_ylabel('Normalized Price (Base = 100)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(ncol=4, fontsize=9, frameon=False)
            c1 = FigureCanvasTkAgg(fig1, self.chart_inner); c1.draw()
            c1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,14))

            # compute returns once
            returns_data = {t: p.pct_change().dropna() for t, p in chart_data.items()}

            # (2) Returns distribution — no cursors
            fig2 = Figure(figsize=(12,3.6), dpi=100, constrained_layout=True)
            ax2 = fig2.add_subplot(1,1,1)
            for t, r in returns_data.items():
                ax2.hist(r, bins=30, alpha=0.45, density=True, label=t)
            ax2.set_title('Returns Distribution', fontweight='bold')
            ax2.set_xlabel('Daily Returns'); ax2.set_ylabel('Density'); ax2.grid(True, alpha=0.3)
            ax2.legend(ncol=4, fontsize=8, frameon=False)
            c2 = FigureCanvasTkAgg(fig2, self.chart_inner); c2.draw()
            c2.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,14))

            # (3) Risk-Return — no cursors
            fig3 = Figure(figsize=(12,3.6), dpi=100, constrained_layout=True)
            ax3 = fig3.add_subplot(1,1,1)
            for t, r in returns_data.items():
                annual_return = r.mean() * 252
                annual_risk = r.std() * np.sqrt(252)
                ax3.scatter(annual_risk, annual_return, s=90)
                ax3.annotate(t, (annual_risk, annual_return), xytext=(5, 5),
                            textcoords='offset points', fontsize=9)
            ax3.set_title('Risk-Return Profile', fontweight='bold')
            ax3.set_xlabel('Annual Risk (Volatility)'); ax3.set_ylabel('Annual Return'); ax3.grid(True, alpha=0.3)
            c3 = FigureCanvasTkAgg(fig3, self.chart_inner); c3.draw()
            c3.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,14))

            # (4) Correlation heatmap — no cursors
            fig4 = Figure(figsize=(12,4.5), dpi=100, constrained_layout=True)
            ax4 = fig4.add_subplot(1,1,1)
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            im = ax4.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            tickers = list(correlation_matrix.index)
            ax4.set_xticks(range(len(tickers))); ax4.set_yticks(range(len(tickers)))
            ax4.set_xticklabels(tickers, rotation=45, ha='right', fontsize=9)
            ax4.set_yticklabels(tickers, fontsize=9)
            for i in range(len(tickers)):
                for j in range(len(tickers)):
                    ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                            ha='center', va='center', fontsize=8)
            ax4.set_title('Correlation Matrix', fontweight='bold')
            fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04).set_label('Correlation')
            c4 = FigureCanvasTkAgg(fig4, self.chart_inner); c4.draw()
            c4.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,10))

        except Exception as e:
            ttk.Label(self.chart_inner, text=f"Error generating charts: {str(e)}", foreground="red").pack(expand=True, pady=10)

    # ==========================
    # Populate Ratios Table
    # ==========================
    def populate_financial_ratios(self):
        if not self.analysis_results: return
        for i in self.fund_tree.get_children(): self.fund_tree.delete(i)
        self._fund_rows_cache = []

        tickers = self.analysis_results['tickers']
        daily_rf = self.analysis_results['sharpe_metrics']['risk_free_rate'] / 252.0
        indiv_ret = self.analysis_results['individual_returns']

        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="1y")
                close = hist['Close']
                day = last_day_change(close)
                from_low = pct_from_52w_low(close)
                vol = annualized_vol_from_returns(close.pct_change().dropna().iloc[-60:])
                rsi_val = compute_rsi(close, period=14)
                try:
                    info = yf.Ticker(t).get_info()
                except Exception:
                    info = {}
                sector = info.get('sector', 'N/A')
                insider = info.get('heldPercentInsiders', np.nan)
                insider_pct = f"{insider*100:.2f}%" if pd.notnull(insider) else "N/A"
                if t in indiv_ret.columns and not indiv_ret[t].empty:
                    sr_indiv = sharpe_from_returns(indiv_ret[t], daily_rf)
                else:
                    sr_indiv = sharpe_from_returns(close.pct_change().dropna(), daily_rf)
                row = (t, f"{day:.2f}%", f"{from_low:.2f}%", f"{vol*100:.2f}%", sector, f"{rsi_val:.1f}", insider_pct, f"{sr_indiv:.3f}")
            except Exception:
                row = (t, "N/A","N/A","N/A","N/A","N/A","N/A","N/A")
            self._fund_rows_cache.append(row)
            self.fund_tree.insert('', tk.END, values=row)

    def apply_fund_sort(self):
        self.sort_fund_table(self.fund_sort_key.get(), self.fund_sort_order_desc.get())

    def sort_fund_table(self, col, descending=False):
        cols = ["Ticker","Day %","% from 52w Low","Ann Vol","Sector","RSI(14)","Insider %","Sharpe (ann)"]
        idx = cols.index(col)
        def keyfunc(row):
            val = row[idx]
            if isinstance(val, str) and val.endswith('%'):
                try: return float(val.replace('%',''))
                except: return val
            try: return float(val)
            except: return val
        sorted_rows = sorted(self._fund_rows_cache, key=keyfunc, reverse=descending)
        for i in self.fund_tree.get_children(): self.fund_tree.delete(i)
        for r in sorted_rows: self.fund_tree.insert('', tk.END, values=r)

    # =========================
    # Suggestions Engine
    # =========================
    def refresh_suggestions(self):
        for i in self.suggestion_tree.get_children(): self.suggestion_tree.delete(i)
        if not self.analysis_results: return
        try:
            base_ret = self.analysis_results['portfolio_returns']
            base_sharpe = sharpe_from_returns(base_ret, self.analysis_results['sharpe_metrics']['risk_free_rate']/252)
            cur_tickers = self.analysis_results['tickers']
            cur_weights = self.analysis_results.get('weights', {t: 1.0/len(cur_tickers) for t in cur_tickers})
            cur_weights = normalize_weights(cur_weights)
            start = base_ret.index.min() - pd.Timedelta(days=5); end = base_ret.index.max() + pd.Timedelta(days=5)
            rows = []

            # Add + Replace ideas across universe
            for c in SUGGESTION_UNIVERSE:
                if c in cur_tickers: 
                    continue
                try:
                    pr = yf.Ticker(c).history(start=start, end=end)['Close'].pct_change().dropna()
                    df = pd.concat([base_ret, pr], axis=1).dropna()
                    if df.shape[0] < len(base_ret)*0.5:
                        continue

                    # ADD 10%
                    w_new = {**cur_weights}
                    for k in w_new: w_new[k] *= 0.90
                    w_new[c] = 0.10
                    indiv = self.analysis_results['individual_returns']
                    aligned = pd.concat([indiv, pr.rename(c)], axis=1).dropna()
                    port_new = (aligned * pd.Series({**w_new})).sum(axis=1)
                    new_sharpe = sharpe_from_returns(port_new, self.analysis_results['sharpe_metrics']['risk_free_rate']/252)
                    rows.append(("Add", c, new_sharpe - base_sharpe, "Add 10% weight, scale others"))

                    # REPLACE "worst" (corr * vol)
                    worst = None; worst_score = -1e9
                    for k in cur_tickers:
                        try:
                            r = indiv[k].reindex(aligned.index).dropna()
                            corr = np.corrcoef(r, base_ret.reindex(r.index).dropna())[0,1] if len(r)>10 else 0
                            score = corr * r.std()
                            if score > worst_score: worst_score = score; worst = k
                        except:
                            continue
                    if worst:
                        w_rep = {**cur_weights}; rep_w = w_rep.get(worst, 0.0)
                        if worst in w_rep: del w_rep[worst]
                        w_rep[c] = rep_w; w_rep = normalize_weights(w_rep)
                        aligned2 = pd.concat([indiv.drop(columns=[col for col in indiv.columns if col not in w_rep]), pr.rename(c)], axis=1).dropna()
                        port_rep = (aligned2 * pd.Series(w_rep)).sum(axis=1)
                        new_sharpe2 = sharpe_from_returns(port_rep, self.analysis_results['sharpe_metrics']['risk_free_rate']/252)
                        rows.append(("Replace", f"{worst}→{c}", new_sharpe2 - base_sharpe, f"Swap {worst} with {c}"))
                except Exception:
                    continue

            # Removals that improve Sharpe
            indiv = self.analysis_results['individual_returns']
            for drop_t in cur_tickers:
                try:
                    if drop_t not in cur_weights:
                        continue
                    w_drop = {k: v for k, v in cur_weights.items() if k != drop_t}
                    if not w_drop:
                        continue
                    w_drop = normalize_weights(w_drop)
                    kept_cols = [c for c in indiv.columns if c in w_drop]
                    aligned_drop = indiv[kept_cols].dropna()
                    if aligned_drop.empty:
                        continue
                    port_drop = (aligned_drop * pd.Series(w_drop)).sum(axis=1)
                    new_sharpe_drop = sharpe_from_returns(port_drop, self.analysis_results['sharpe_metrics']['risk_free_rate']/252)
                    delta_drop = new_sharpe_drop - base_sharpe
                    if delta_drop > 0:
                        rows.append(("Remove", f"{drop_t}", delta_drop, f"Drop {drop_t} and reweight others"))
                except Exception:
                    continue

            if not rows: 
                return
            df = pd.DataFrame(rows, columns=["Type","Ticker","DeltaSharpe","Note"]).sort_values("DeltaSharpe", ascending=False)
            self.suggestions_df = df
            for _, r in df.head(10).iterrows():
                self.suggestion_tree.insert("", tk.END, values=(r["Type"], r["Ticker"], f"{r['DeltaSharpe']:+.3f}", r["Note"]))
        except Exception as e:
            messagebox.showwarning("Suggestions", f"Could not compute suggestions: {e}")

    def apply_selected_suggestion(self):
        sel = self.suggestion_tree.selection()
        if not sel:
            messagebox.showinfo("Suggestions", "Select a suggestion first."); return
        item = self.suggestion_tree.item(sel[0])["values"]
        stype, field, _, _ = item
        if not self.analysis_results: return
        cur_tickers = self.analysis_results['tickers'][:]

        if stype == "Add":
            c = field
            new_tickers = sorted(list(set(cur_tickers + [c])))
        elif stype == "Remove":
            drop_t = field
            new_tickers = [t for t in cur_tickers if t != drop_t]
        else:  # Replace
            try:
                old, new = field.split("→")
            except:
                messagebox.showerror("Suggestions", "Invalid replacement format."); return
            new_tickers = sorted(list(set([t for t in cur_tickers if t != old] + [new])))

        self.ticker_entry.delete("1.0", tk.END)
        self.ticker_entry.insert("1.0", ", ".join(new_tickers))
        # No weight UI anymore — just re-run analysis, optimizer will recompute
        self.start_analysis()

    def clear_all(self):
        self.ticker_entry.delete("1.0", tk.END); self.weight_var.set("equal")
        for widget in self.weights_frame.winfo_children(): widget.destroy()
        self.analysis_results = None; self.portfolio_data = {}; self.portfolio_returns = pd.DataFrame()
        self.summary_text.delete(1.0, tk.END); self.summary_text.insert(tk.END, "Portfolio analysis results will appear here after running analysis.")
        self.covariance_text.delete(1.0, tk.END); self.covariance_text.insert(tk.END, "Detailed covariance analysis will appear here after running analysis.")
        for item in self.data_tree.get_children(): self.data_tree.delete(item)
        # clear stacked/scrollable areas
        if hasattr(self, 'chart_inner'):
            for w in self.chart_inner.winfo_children(): w.destroy()
            ttk.Label(self.chart_inner, text="Portfolio performance charts will appear here after analysis.", font=('Arial', 11), justify=tk.CENTER).pack(expand=True, pady=10)
        if hasattr(self, 'bh_inner'):
            for w in self.bh_inner.winfo_children(): w.destroy()
            ttk.Label(self.bh_inner, text="Buy & Hold portfolio chart will appear here.", font=('Arial', 11)).pack(expand=True, pady=10)
        for i in getattr(self, 'fund_tree', []).get_children() if hasattr(self, 'fund_tree') else []: self.fund_tree.delete(i)
        for pane in [getattr(self, 'beta_frame', None), getattr(self, 'rolling_beta_frame', None), getattr(self, 'arc_frame', None), getattr(self, 'cluster_pca_frame', None)]:
            if pane:
                for w in pane.winfo_children(): w.destroy()
        for i in getattr(self, 'suggestion_tree', []).get_children() if hasattr(self, 'suggestion_tree') else []: self.suggestion_tree.delete(i)
        self.status_label.configure(text="Ready to analyze portfolio", foreground="green")
    
    def save_results(self):
        if not self.analysis_results:
            messagebox.showwarning("No Results", "No analysis results to save. Run analysis first."); return
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                                                    title="Save Portfolio Analysis Results")
            if filename:
                summary_content = self.summary_text.get(1.0, tk.END)
                covariance_content = self.covariance_text.get(1.0, tk.END)
                with open(filename, 'w') as f:
                    f.write("PORTFOLIO ANALYSIS RESULTS\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("="*80 + "\n\n")
                    f.write("PORTFOLIO SUMMARY:\n")
                    f.write("-" * 40 + "\n")
                    f.write(summary_content)
                    f.write("\n\n")
                    f.write("COVARIANCE ANALYSIS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(covariance_content)
                messagebox.showinfo("Save Successful", f"Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving file: {str(e)}")

def main():
    root = tk.Tk()
    app = PortfolioGUI(root)
    root.mainloop()

if __name__ == "__main__":
    print("="*60)
    print("PORTFOLIO ANALYSIS GUI")
    print("="*60)
    print("\nTo run this GUI application:")
    print("1. Install deps: pip install tkinter yfinance pandas numpy matplotlib seaborn mplcursors")
    print("2. Run: python portfolio_gui.py")
    print("3. New UX:")
    print("   • Portfolio Charts & Buy & Hold tabs are scrollable with vertically stacked plots.")
    print("   • Buy & Hold line shows hover value tooltips.")
    print("   • Suggestions support Add / Replace / Remove.")
    print("="*60)
    main()
