import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ---------- Page config ----------
st.set_page_config(page_title="Solar ROI – WA", page_icon="☀️", layout="wide")

# ---------- Constants (edit as needed) ----------
c_swis_kw = 3000000 # Estimated total SWIS installed rooftop solar in 2024
monthly_share = {
    'January':0.0933,'February':0.0917,'March':0.0867,'April':0.0742,'May':0.0692,'June':0.0725,
    'July':0.0750,'August':0.0800,'September':0.0850,'October':0.0900,'November':0.0933,'December':0.0892
}
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September','October','November','December']
day_in_month = {
    'January':31,'February':28,'March':31,'April':30,'May':31,'June':30,
    'July':31,'August':31,'September':30,'October':31,'November':30,'December':31
}

# ---------- UI: Title ----------
st.title("☀️ Solar ROI (Western Australia Household)")
st.markdown("### Discover the savings and return on investment of your rooftop solar panels.")

with st.sidebar:
    st.markdown("📘 Need help? Read this **[user guide.](https://github.com/osuriyanto/capstone1_app/blob/main/README.md)**")

# --- UI: uploader + note ---
col1, col2 = st.columns(2)
with col1:
    # keep a key counter in session state
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "**Option (1)** Upload 1 year usage summary from your Synergy account, see user guide.", 
        type=["csv"],
        key=f"uploader_{st.session_state.uploader_key}",
        help="We do not save your file. It lives only in this session’s memory."
    )

    # Remove button bumps the key (creates a fresh uploader = cleared)
    if uploaded_file  and st.button("Remove uploaded file"):
        st.session_state.uploader_key += 1
        st.rerun()
    
with col2:
    # --- assign usage value based on uploaded csv or user input
    base_usage_kwh_yearly = st.number_input(
        "**Option (2)** Enter your estimated annual usage (kWh):",
        min_value=0, value=5000, step=100
    )

# --- function to prepare the uploaded usage csv
def _resolve_column(df, preferred, aliases=()):
    """Find a column by exact name or reasonable aliases (case/space/underscore insensitive)."""
    norm = {c.lower().replace("_"," ").strip(): c for c in df.columns}
    candidates = [preferred, *aliases]
    for name in candidates:
        key = name.lower().replace("_"," ").strip()
        if key in norm:
            return norm[key]
    return None

def process_usage_csv(file) -> tuple[float, pd.DataFrame]:
    """Return (annual_usage, cleaned_df) from an uploaded CSV-like object."""
    df = pd.read_csv(file)

    # Try to be robust to mild header variations
    days_col = _resolve_column(
        df,
        "Number of billing days",
        aliases=("billing days", "number of days", "days")
    )
    usage_col = _resolve_column(
        df,
        "Total usage for period",
        aliases=("total usage (kwh)", "usage (kwh)", "total kwh", "kwh")
    )

    if not days_col or not usage_col:
        raise ValueError(
            "Missing required columns. Expected headers like "
            "'Number of billing days' and 'Total usage for period'."
        )

    # Coerce to numeric and drop bad rows
    df[days_col] = pd.to_numeric(df[days_col], errors="coerce")
    df[usage_col] = pd.to_numeric(df[usage_col], errors="coerce")
    df = df.dropna(subset=[days_col, usage_col])

    total_days = df[days_col].sum()
    total_usage = df[usage_col].sum()

    if total_days <= 0:
        raise ValueError("Total billing days computed as 0. Please check the CSV content.")

    annual_usage = total_usage / total_days * 365.0
    return annual_usage, df

# --- function to get annual usage either from uploaded file or manual user entry
def get_annual_usage(base_usage_kwh_yearly: float) -> tuple[float, pd.DataFrame | None]:
    """
    If a CSV is uploaded, compute from CSV; otherwise fall back to user input.
    Returns (annual_usage, df_or_None).
    """
    if uploaded_file is not None:
        try:
            annual_usage, df = process_usage_csv(uploaded_file)
            st.success("CSV processed successfully.")
            with st.expander("Preview uploaded data (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)
            st.metric("Annual usage (kWh) computed from input CSV:", f"{annual_usage:,.0f}")
            return annual_usage, df
        except Exception as e:
            st.error(f"Could not process CSV: {e}")
            st.info("Falling back to manually entered estimate (option 2).")

    # Fallback: use manual/user input estimate
    st.metric("Estimated annual usage (kWh) entered:", f"{base_usage_kwh_yearly:,.0f}")
    return base_usage_kwh_yearly, None

annual_usage_kwh, uploaded_df = get_annual_usage(base_usage_kwh_yearly)

# ---------- Caching / Data layer ----------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.replace(' ', '_')
    if 'Trading_Date' in df.columns:
        df = df.drop(columns=['Trading_Date'])
    df['Trading_Interval'] = pd.to_datetime(df['Trading_Interval'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    df['Extracted_At'] = pd.to_datetime(df['Extracted_At'], format='%Y-%m-%d %H:%M:%S', errors='raise')
    df['month_name'] = df['Trading_Interval'].dt.month_name()
    df['month_period'] = df['Trading_Interval'].dt.to_period('M')
    df['hour'] = df['Trading_Interval'].dt.hour
    return df

@st.cache_data(show_spinner=False)
def load_and_prepare(source):
    """
    source can be:
      - a filename like 'distributed-pv-2025.csv'
      - a Path object
      - a file-like object from st.file_uploader (has .read())
    """
    # If user uploaded a file (BytesIO/StringIO), read it directly
    if hasattr(source, "read"):
        return _prepare(pd.read_csv(source))

    # Otherwise treat it as a path/name and try common locations
    name = str(source)
    candidates = [
        Path(name),                 # as-given (absolute or relative)
        HERE / name,               # alongside this script
        DATA_DIR / name,           # ./data/<name>
        Path.cwd() / name,         # current working directory
        Path.cwd() / "data" / name # CWD/data/<name>
    ]

    for fp in candidates:
        if fp.exists():
            return _prepare(pd.read_csv(fp))

    # Helpful diagnostics if nothing matched
    raise FileNotFoundError(
        f"Could not find '{name}'. Tried:\n  - " +
        "\n  - ".join(str(p) for p in candidates) +
        f"\nCWD: {Path.cwd()}\nScript dir: {HERE}"
    )

# ---------- Core computations ----------
# function to compute discounted payback and extract the dataframe

def discounted_payback_details(capex, *, rate=0.05, cashflows=None, annual_saving=None,
                               years=50, timing='end', round_map=None,percent_cols=("Discount Rate",),
                               currency_cols=("Nominal CF","Present Value","Cumulative PV"), return_formatted=False
                              ):
    """
    Returns:
      - payback_years
      - df_numeric  (always numeric columns)
      - df_display  (optional, pretty-formatted strings if return_formatted=True)

    df_numeric columns:
      ['Year','Nominal CF','Discount Rate','Discount Factor','Present Value','Cumulative PV','Reached Payback?']
    'Year' starts at 1 for 'end'/'mid' and at 0 for 'begin'.
    """

    if (cashflows is None) == (annual_saving is None):
        raise ValueError("Provide either `cashflows` OR `annual_saving` (but not both).")
    if cashflows is None:
        cashflows = [annual_saving] * years

    shift_map = {'end': 0.0, 'mid': 0.5, 'begin': 1.0}
    if timing not in shift_map:
        raise ValueError("timing must be 'end', 'mid', or 'begin'")
    shift = shift_map[timing]

    rows = []
    cumulative_pv = 0.0
    payback_years = None

    # Choose how to label the first year shown
    first_year_label = 0 if timing == 'begin' else 1

    for t, cf in enumerate(cashflows, start=1):  # discount exponent uses t - shift
        disc_factor = 1 / ((1 + rate) ** (t - shift))
        pv = cf * disc_factor
        prev = cumulative_pv
        cumulative_pv += pv

        # Year label for display
        year_label = first_year_label + (t - 1)

        rows.append({
            'Year': year_label,
            'Nominal CF': float(cf),
            'Discount Rate': float(rate),
            'Discount Factor': float(disc_factor),
            'Present Value': float(pv),
            'Cumulative PV': float(cumulative_pv),
        })

        # detect payback crossing
        if payback_years is None and cumulative_pv >= capex:
            need = capex - prev
            frac = need / pv if pv > 0 else 1.0
            # For 'begin', first CF is at year 0, so shift back by 1
            payback_years = ((t - 2) + frac) if timing == 'begin' else ((t - 1) + frac)

    df = pd.DataFrame(rows)

    # Optional: mark the first row where payback is reached
    if payback_years is not None:
        # find the crossing row index
        cross_idx = df.index[df['Cumulative PV'] >= capex][0]
        df.loc[cross_idx, 'Reached Payback?'] = 'Yes'
        df.loc[:cross_idx-1, 'Reached Payback?'] = ''
        df.loc[cross_idx+1:, 'Reached Payback?'] = ''

    # ---- NEW: numeric rounding inside the dataframe (preserves dtype) ----
    # sensible defaults if none provided
    if round_map is None:
        round_map = {
            'Year': 0,
            'Nominal CF': 2,
            'Discount Rate': 4,      # keep as fraction (e.g., 0.0500) – show as % in display df
            'Discount Factor': 4,
            'Present Value': 2,
            'Cumulative PV': 2,
        }
    # only round existing columns
    df = df.round({k: v for k, v in round_map.items() if k in df.columns})

    if not return_formatted:
        return payback_years, df

    # ---- Optional: pretty display copy with $ and % strings ----
    def _fmt_currency(x): return f"${x:,.2f}"
    def _fmt_percent(x):  return f"{x*100:.2f}%"

    df_display = df.copy()
    for c in currency_cols:
        if c in df_display:
            df_display[c] = df_display[c].map(_fmt_currency)
    for c in percent_cols:
        if c in df_display:
            df_display[c] = df_display[c].map(_fmt_percent)

    # Year as integer-like string if you prefer
    if 'Year' in df_display:
        df_display['Year'] = df_display['Year'].map(lambda v: f"{int(v)}")

    return payback_years, df, df_display

# function to compute npv
def npv(rate, cashflows):
    """NPV for equally spaced periods t = 0..T."""
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))

# function to compute IRR with simple intrapolation
def irr_bisection(cashflows, low=-0.9999, high=10.0, tol=1e-7, max_iter=200):
    """
    Robust IRR via bisection.
    Requires NPV(low) and NPV(high) to have opposite signs.
    """
    f_low = npv(low, cashflows)
    f_high = npv(high, cashflows)
    if f_low == 0: return low
    if f_high == 0: return high
    if f_low * f_high > 0:
        raise ValueError("IRR not bracketed: NPV(low) and NPV(high) have same sign.")

    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = npv(mid, cashflows)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2  # best effort

# function to compute IRR with newton-raphson iteration
def irr_newton(cashflows, guess=0.1, tol=1e-8, max_iter=100):
    """
    IRR via Newton–Raphson. Needs a reasonable guess and a mostly monotone NPV.
    """
    r = guess
    for _ in range(max_iter):
        # f(r) and f'(r)
        f = 0.0
        fp = 0.0
        for t, cf in enumerate(cashflows):
            denom = (1 + r) ** t
            f += cf / denom
            if t > 0:
                fp += -t * cf / ((1 + r) ** (t + 1))
        if abs(f) < tol:
            return r
        if fp == 0:
            break
        r -= f / fp
    raise RuntimeError("Newton did not converge")

# plot bar chart of total usage, pv generation, and pv self consumption 
# added line plot for the bills
def plot_usage_pv_with_costs(
    df, *, title = None,
    month_col='month_name',
    usage_col='monthly_usage_kwh',
    pv_col='pv_generation_kwh',
    self_usage_fraction=0.25,         # replace number with the variable name when apply the function
    base_cost_col='monthly_base_$',
    solar_cost_col='monthly_solar_$',
    month_order=('January','February','March','April','May','June',
                 'July','August','September','October','November','December'),
    height = 520,
    compact=False,
    stack_bars=False
):
    df = df.copy()
    # Order months nicely if present
    if month_col in df.columns:
        df[month_col] = pd.Categorical(df[month_col], categories=month_order, ordered=True)
        df = df.sort_values(month_col)

    # --- Sanitize inputs ---
    # clip negatives to 0 for plotting sanity
    usage = df[usage_col].astype(float).clip(lower=0.0)
    pv    = df[pv_col].astype(float).clip(lower=0.0)
    
    # Build the self-consumption fraction (scalar or per-row)
    if isinstance(self_usage_fraction, str):
        # column-based fraction
        frac = df[self_usage_fraction].astype(float).fillna(0.0)
        frac = frac.clip(lower=0.0, upper=1.0)        # pandas Series.clip
    else:
        # scalar fraction
        frac = float(self_usage_fraction)
        frac = float(np.clip(frac, 0.0, 1.0))         # numpy clip for scalar

    # Raw self-use request vs. capped by load
    y3_raw   = pv * frac
    y3_capped = np.minimum(y3_raw, usage)
    capped_flag = (y3_raw > usage)

    # Text hints for hover
    hover_tag = np.where(capped_flag, " (capped to usage)", "")
    # Optional stripe pattern to highlight capped bars (requires Plotly ≥5)
    pattern_shape = np.where(capped_flag, "/", "")  # "/" for stripes, "" for solid

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Bars (primary y) ---
    fig.add_trace(
        go.Bar(
            x=df[month_col], y=df[usage_col], name="Total Usage (kWh)",
            marker=dict(color='#90CAF9'),
            hovertemplate="Month: %{x}<br>Usage: %{y:.2f} kWh<extra></extra>"
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Bar(
            x=df[month_col], y=df[pv_col], name="PV generation (kWh)",
            marker=dict(color='#1E88E5'),
            hovertemplate="Month: %{x}<br>PV gen: %{y:.2f} kWh<extra></extra>"
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Bar(
            x=df[month_col], y=y3_capped, name="PV self-used (kWh)",
            text=hover_tag,  # used by hovertemplate
            marker=dict(
                color='#0D47A1',
                pattern=dict(shape=pattern_shape)  # stripes on capped months
            ),
            hovertemplate="Month: %{x}<br>PV self-used: %{y:.2f} kWh%{text}<extra></extra>"
        ),
        secondary_y=False
    )

    # --- Lines (secondary y) ---
    if base_cost_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[month_col], y=df[base_cost_col],
                name="Bill w/o Solar ($)", mode="lines+markers",
                marker=dict(color='#E57373'),
                hovertemplate="Month: %{x}<br>Base bill: $%{y:.2f}<extra></extra>"
            ),
            secondary_y=True
        )
    if solar_cost_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[month_col], y=df[solar_cost_col],
                name="Bill with Solar ($)", mode="lines+markers",
                marker=dict(color='#1B5E20'),
                hovertemplate="Month: %{x}<br>Solar bill: $%{y:.2f}<extra></extra>"
            ),
            secondary_y=True
        )

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.4),
        barmode='relative' if stack_bars else 'group',
        xaxis_title="",
        yaxis_title="Energy (kWh)",
        legend_title="",
        margin=dict(t=48, r=16, b=64, l=56),
        hovermode="x" if compact else "x unified",
        height=420 if compact else height,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.22, yanchor="top",     # push legend below plot
            font=dict(size=10 if compact else 12),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant"
        )
    )
    fig.update_xaxes(tickangle=-30, automargin=True)
    # Secondary axis formatting
    fig.update_yaxes(
        title_text="Bill ($)",
        secondary_y=True,
        tickprefix="$",
        separatethousands=True
    )

    return fig

# function to plot the savings
def plot_monthly_savings_breakdown(
    df, *, title = None,
    month_col='month_name',
    s_self_col='savings_pv_self_use_$',
    s_peak_col='peak_rebate_$',
    s_offpeak_col='off_peak_rebate_$',
    total_col='total_savings_$',
    month_order=('January','February','March','April','May','June',
                 'July','August','September','October','November','December'),
    barmode='stack',  # 'stack' or 'group'
    height = 520,
    compact=False,
    legend_bottom=True,
    short_names=True
):
    d = df.copy()
    # Order months
    d[month_col] = pd.Categorical(d[month_col], categories=month_order, ordered=True)
    d = d.sort_values(month_col)

    # Shorter labels for small screens
    name_self    = "Self-used" if short_names else "Savings: PV self-consumption"
    name_peak    = "Peak"      if short_names else "Savings: Peak rebate"
    name_offpeak = "Off-peak"  if short_names else "Savings: Off-peak rebate"
    name_total   = "Total"     if short_names else "Total savings"

    fig = go.Figure()

    # Bars (all share the same $ axis)
    fig.add_bar(
        x=d[month_col], y=d[s_self_col], name=name_self,
        marker=dict(color='#2E7D32'),
        hovertemplate="Month: %{x}<br>Self-used: $%{y:.2f}<extra></extra>",
        marker_line_width=0
    )
    fig.add_bar(
        x=d[month_col], y=d[s_peak_col], name=name_peak,
        marker=dict(color='#66BB6A'),
        hovertemplate="Month: %{x}<br>Peak rebate: $%{y:.2f}<extra></extra>",
        marker_line_width=0
    )
    fig.add_bar(
        x=d[month_col], y=d[s_offpeak_col], name=name_offpeak,
        marker=dict(color='#A5D6A7'),
        hovertemplate="Month: %{x}<br>Off-peak rebate: $%{y:.2f}<extra></extra>",
        marker_line_width=0
    )

    # Line (same y-axis)
    fig.add_trace(
        go.Scatter(
            x=d[month_col], y=d[total_col],
            mode="lines+markers", name=name_total,
            marker=dict(size=6 if compact else 7, color='#1B5E20'),
            line=dict(width=2),
            hovertemplate="Month: %{x}<br>Total: $%{y:.2f}<extra></extra>"
        )
    )

    # zero saving reference line
    fig.add_hline(y=0, line_width=3, line_dash="dash", line_color="#E57373")
    fig.add_annotation(
        xref="paper", x=1.0, y=0, xanchor="right", yanchor="bottom",
        text=f"Zero savings", showarrow=False,
        font=dict(size=12, color="#E57373"), bgcolor="rgba(255,255,255,0.6)"
    )

    fig.update_layout(
        title = dict(text = title, x = 0.4),
        barmode=barmode,
        xaxis_title="",
        yaxis_title="Savings ($)",
        legend_title="",
        margin=dict(t=48, r=16, b=64, l=56),
        hovermode="x" if compact else "x unified",
        height=420 if compact else height,
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.22 if legend_bottom else 1.02,
            yanchor="top" if legend_bottom else "bottom",
            font=dict(size=10 if compact else 12),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant"
        ),
        uniformtext_minsize=8 if compact else 0,
        uniformtext_mode="hide"
    )
    fig.update_xaxes(tickangle=-30, automargin=True)
    fig.update_yaxes(tickprefix="$", separatethousands=True)

    # Extra compact tweaks
    if compact:
        fig.update_traces(hoverlabel_namelength=0)
        # Slight transparency helps stacked bars on small screens
        fig.update_traces(opacity=0.95, selector=dict(type="bar"))

    return fig

# function to plot pie chart
def plot_savings_piechart(df_display, *, font_size=15):
    # keep only the savings columns
    colors = ['#1B5E20','#43A047','#A5D6A7']
    cols = ['savings_pv_self_use_$', 'peak_rebate_$', 'off_peak_rebate_$']
    cols = [c for c in cols if c in df_display.columns]
    if not cols:
        # empty chart
        return go.Figure().update_layout(
            title="Annual Savings ($)",
            annotations=[dict(text="No savings columns found", showarrow=False)]
        )

    s = df_display[cols].select_dtypes(include='number').sum()
    s = s.clip(lower=0)  # avoid negative slices for a “savings” chart
    s = s[s > 0]
    if s.empty:
        return go.Figure().update_layout(
            title="Annual Savings ($)",
            annotations=[dict(text="No positive values", showarrow=False)]
        )

    df = s.reset_index()
    df.columns = ['source', 'value']

    fig = go.Figure(
        go.Pie(
            labels=df['source'],
            values=df['value'],
            marker=dict(colors=colors),
            pull=[0.08] + [0] * (len(df) - 1),
            texttemplate='%{label}<br>$%{value:,.0f} (%{percent})',
            textposition='outside',
            textfont_size=font_size,
            hovertemplate='%{label}: $%{value:,.0f} (%{percent})<extra></extra>',
            hole=0.0,          # set to 0.0 for full pie
            sort=False,
            showlegend=False
        )
    )
    fig.update_layout(
        title='Annual Savings ($)',
        margin=dict(l=30, r=60, t=60, b=30),
        uniformtext_minsize=font_size,
        uniformtext_mode='show'
    )
    
    return fig

# function to plot cumulative discounted cashflow and capex
def st_cum_pv_chart(
    df,
    capex,
    *,
    year_col="Year",
    cum_col="Cumulative PV",
    payback_year=None,   # pass your computed year; if None we'll infer the first >= capex
    title="Cumulative Discounted Cashflow & CAPEX",
    use_container_width=True,
    height=None,
    download_filename="cum_pv_vs_capex"
):
    # --- infer payback year only if not provided ---
    if payback_year is None:
        mask = df[cum_col] >= capex
        payback_year = df.loc[mask.idxmax(), year_col] if mask.any() else None

    # --- colors ---
    before_color    = "#A0AEC0"
    after_color     = "#CBD5E0"
    highlight_color = "#2E7D32"

    colors = []
    for _, row in df.iterrows():
        y = row[year_col]
        if payback_year is None:
            colors.append(before_color)
        elif y == payback_year:
            colors.append(highlight_color)
        elif y < payback_year:
            colors.append(before_color)
        else:
            colors.append(after_color)

    # --- figure ---
    fig = go.Figure([
        go.Bar(
            x=df[year_col],
            y=df[cum_col],
            marker=dict(color=colors),
            name="Cumulative PV",
            hovertemplate=f"{cum_col}: $%{{y:,.0f}}<extra></extra>",
        )
    ])

    # CAPEX reference line + label
    fig.add_hline(y=capex, line_width=2, line_dash="dash", line_color="#E53E3E")
    fig.add_annotation(
        xref="paper", x=0.1, y=capex, xanchor="right", yanchor="bottom",
        text=f"CAPEX = ${capex:,.0f}", showarrow=False,
        font=dict(size=12, color="#E53E3E"), bgcolor="rgba(255,255,255,0.6)"
    )

    # Payback year annotation
    if payback_year is not None and (df[year_col] == payback_year).any():
        y_val = df.loc[df[year_col] == payback_year, cum_col].iloc[0]
        fig.add_annotation(
            x=payback_year, y=y_val, yshift=10,
            text=f"Payback year: {payback_year}",
            showarrow=False, font=dict(size=12, color=highlight_color)
        )

    fig.update_layout(
        title=title,
        xaxis_title=year_col,
        yaxis_title="$",
        margin=dict(l=50, r=30, t=60, b=40),
        bargap=0.15,
        hovermode="x unified",
        height=height
    )

    st.plotly_chart(
        fig,
        use_container_width=use_container_width,
        config={"toImageButtonOptions": {"filename": download_filename}}
    )
    return fig

# ---------- Sidebar (user inputs) ----------
with st.sidebar:
    st.header("Inputs")
    pv_size_kw      = st.number_input("PV size (kW)", 0.5, 20.0, 2.2, step=0.1)
    pr      = st.number_input("PV performance ratio", 0.7, 0.9, 0.85, step=0.05)
    pv_capex_aud       = st.number_input("PV system cost ($)", 1000.0, 20000.0, pv_size_kw*1000, step=100.0)
    discount_rate = st.number_input("Discount rate", 0.04, 0.1, 0.05, step=0.001, format="%.3f")
    pv_self_consumption_fraction  = st.slider("Self-consumption proportion", 0.05, 1.0, 0.25, step=0.05)
    #tariff_cents_per_kwh   = st.number_input("Import tariff c/kWh", 0.0, 40.0, 32.372, step=0.1, format="%.3f")
    #supply_cents_per_day   = st.number_input("Supply charge c/day", 0.0, 200.0, 116.050, step=0.1, format="%.3f")
    #rebate_peak_cents_per_kwh     = st.number_input("Export peak c/kWh", 0.0, 100.0, 10.0, step=0.1, format="%.3f")
    #rebate_off_peak_cents_per_kwh = st.number_input("Export off peak c/kWh", 0.0, 100.0, 2.0, step=0.1, format="%.3f")
    #peak_export_fraction  = st.slider("Peak export proportion (optional)", 0.0, 1.0, 0.2, step=0.05)
    st.caption("Tip: Optimum self-consumption is above 10-15%. Set self-consumption to reflect appliance timing and household profile.")

# ---------load DPV data
# Load only the year in use to prevent FileNotFoundError & speed startup.
distributed_dpv_2024 = load_and_prepare('distributed-pv-2024.csv')

# ---------Compute household PV generation (kwh), monthly consumption (kwh) and savings, $
df = distributed_dpv_2024.copy()

# Compute household pv generation 
df['pt_kw_per_kw']= df['Estimated_DPV_Generation_(MW)']*1000/c_swis_kw
df['pv_generation_kwh']=df['pt_kw_per_kw']*pv_size_kw*pr*0.5

# aggregate monthly
month_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
df_monthly = (df.groupby('month_name', as_index=False)['pv_generation_kwh']
                .sum()
                .rename(columns={'pv_generation_kwh':'pv_generation_kwh'}))
df_monthly['pv_generation_kwh'] = df_monthly['pv_generation_kwh'].round(2)
df_monthly = df_monthly.set_index('month_name').reindex(month_order).reset_index()

# usage profile
df_monthly['usage_share']=df_monthly['month_name'].map(monthly_share)
df_monthly['monthly_usage_kwh']=df_monthly['usage_share']*annual_usage_kwh

# input tariff
tariff_cents_per_kwh = 32.372
supply_cents_per_day = 116.050
rebate_peak_cents_per_kwh = 10.0
rebate_off_peak_cents_per_kwh = 2.0
peak_export_fraction = 0.2

# Sanitize fractions
sc = float(np.clip(pv_self_consumption_fraction, 0.0, 1.0))
pf = float(np.clip(peak_export_fraction, 0.0, 1.0))
pv   = df_monthly['pv_generation_kwh'].astype(float)
use  = df_monthly['monthly_usage_kwh'].astype(float)

# kWh actually self-consumed: cap by load
self_use_kwh = np.minimum(pv * sc, use)

# kWh exported: whatever PV remains after self-consumption
export_kwh = (pv - self_use_kwh).clip(lower=0.0)

# $ calculations (input cents/kWh -> $)
savings_self = self_use_kwh * (tariff_cents_per_kwh / 100.0)
peak_rebate  = export_kwh * pf * (rebate_peak_cents_per_kwh / 100.0)
off_rebate   = export_kwh * (1.0 - pf) * (rebate_off_peak_cents_per_kwh / 100.0)

df_monthly['savings_pv_self_use_$'] = savings_self.round(2)
df_monthly['peak_rebate_$']         = peak_rebate.round(2)
df_monthly['off_peak_rebate_$']     = off_rebate.round(2)
df_monthly['total_savings_$']       = (savings_self + peak_rebate + off_rebate).round(2)

# ---------Compute household bills, before & after pv installation
# base case bill
df_monthly['day']=df_monthly['month_name'].map(day_in_month)
df_monthly['monthly_base_$']=round((df_monthly['monthly_usage_kwh']*tariff_cents_per_kwh/100 + df_monthly['day']*supply_cents_per_day/100),2)

# solar case bill
df_monthly['monthly_solar_$']=round((df_monthly['monthly_base_$']-df_monthly['total_savings_$']),2)

# ---------Compute payback time
# compute simple payback year
annual_savings = float(df_monthly['total_savings_$'].sum())
annual_savings_self_use = float(df_monthly['savings_pv_self_use_$'].sum())
percentage_savings_self_use = (annual_savings_self_use / annual_savings * 100) if annual_savings > 0 else 0.0
payback_year_simple = pv_capex_aud/annual_savings if annual_savings > 0 else float('inf')
try:
    _yrs = int(round(payback_year_simple)) if (payback_year_simple != float('inf')) else 50
except Exception:
    _yrs = 50
_yrs = max(50, _yrs * 2)
dpb_year, dpb_table = discounted_payback_details(
    pv_capex_aud,
    rate=discount_rate,
    annual_saving=annual_savings,
    years=_yrs,
    timing='end'
)

# ---------Compute Internal Rate of Return
irr_horizon_years = 30
cfs = [-pv_capex_aud] + [annual_savings]*irr_horizon_years
irr = None
if annual_savings > 0:
    try:
        irr = irr_newton(cfs)
    except Exception:
        try:
            irr = irr_bisection(cfs)
        except Exception:
            irr = None  # not bracketed or other issue

# ---------- UI Layout ----------
# Warn if PV is oversized relative to load given the chosen self-consumption fraction
oversized = (pv * sc) > use
if oversized.any():
    st.info(f"Possibly oversized PV. {oversized.sum()} month(s) with estimated PV self consumption higher than estimated monthly usage.")

colA, colB, colC, colD = st.columns(4)
colA.metric("Annual Savings ($)", f"{annual_savings:,.0f}")
colB.metric("Savings from PV Self-Use", f"{percentage_savings_self_use:,.0f}%")
colC.metric("Discounted Payback (yr)", f"{dpb_year:.1f}" if dpb_year is not None else "> horizon")
colD.metric("Internal Rate of Return", f"{irr*100:.1f}%" if irr is not None else "n/a")

with st.expander("Assumptions", expanded=False):
    st.write({
        "import tariff (cents/kwh) from Synergy's A1 Home plan": tariff_cents_per_kwh, "peak export rebate (cents/kwh)": rebate_peak_cents_per_kwh,
        "off-peak export rebate (cents/kwh)": rebate_off_peak_cents_per_kwh, "peak export fraction":  peak_export_fraction, "total installed PV in WA(kw)":c_swis_kw, "monthly usage profile": monthly_share})

# Tabs
T1, T2, T3 = st.tabs(["Charts","Payback Time", "Monthly View (Year 1)"])

with T1:
    # top chart
    fig_pie = plot_savings_piechart(df_monthly, font_size=12)   # same df
    st.plotly_chart(
        fig_pie,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["toImage"],
            "toImageButtonOptions": {
                "format": "png",                  # png | svg | jpeg | webp
                "filename": "Annual_Savings_PieChart",   
                "height": 600,
                "width": 800,
                "scale": 2,
            },
        },
    )
    
    # middle chart
    fig_saving = plot_monthly_savings_breakdown(
            df_monthly,
            title="Monthly Savings Breakdown",
            barmode="stack",       # stack is friendlier on narrow screens
            compact=True,          # mobile preset
            legend_bottom=True,
            short_names=True,
            height=520)
    st.plotly_chart(
    fig_saving,
    use_container_width=True,
    config={
        "displayModeBar": True,            # show the toolbar
        "displaylogo": False,              # hide Plotly logo
        "modeBarButtonsToAdd": ["toImage"],# ensure toImage is present
        "toImageButtonOptions": {
            "format": "png",               # "png", "svg", "jpeg", "webp"
            "filename": "Monthly_Savings",
            "height": 600,
            "width": 1000,
            "scale": 2
        }
    },
)

    # bottom chart
    fig_usage = plot_usage_pv_with_costs(df_monthly, self_usage_fraction=pv_self_consumption_fraction, title = "Consumption and Bills", compact=True)
    st.plotly_chart(
    fig_usage,
    use_container_width=True,
    config={
        "displayModeBar": True,            # show the toolbar
        "displaylogo": False,              # hide Plotly logo
        "modeBarButtonsToAdd": ["toImage"],# ensure toImage is present
        "toImageButtonOptions": {
            "format": "png",               # "png", "svg", "jpeg", "webp"
            "filename": "Monthly_Usage_Bills",
            "height": 600,
            "width": 1000,
            "scale": 2
        }
    },
)
   
    
print_upto = (math.ceil(dpb_year) + 1) if dpb_year is not None else 1
payback_table = dpb_table.head(print_upto)
with T2:
    # bar chart
    st_cum_pv_chart(payback_table, capex=pv_capex_aud)
    # table    
    st.dataframe(
        payback_table,
        use_container_width=True)

with T3:
    st.subheader("Monthly details – Year 1")
    # Rebuild first-year monthly DF for display / plots
    df_display = df_monthly[['month_name', 'pv_generation_kwh', 'savings_pv_self_use_$',
       'peak_rebate_$', 'off_peak_rebate_$', 'total_savings_$',
       'monthly_usage_kwh', 'monthly_base_$','monthly_solar_$']]
    st.dataframe(df_display, use_container_width=True)
        
# ---------- Exports ----------
@st.cache_data(show_spinner=False)
def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download payback estimate (CSV)", data=to_csv_download(payback_table), file_name="estimate_payback.csv", mime="text/csv")
with col2:
    st.download_button("Download Year1 monthly (CSV)", data=to_csv_download(df_display), file_name="monthly_year1.csv", mime="text/csv")
