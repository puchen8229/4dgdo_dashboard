# -*- coding: utf-8 -*-
"""
Global Development Indicators Dashboard
- Scalar (country–year)
- Bilateral (reporter–partner–year)
- Input–Output (industry–industry, OECD TiVA)
"""

import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
from datetime import datetime
import time

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global Development Indicators Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Paths with fallback
# ---------------------------------------------------------
project_root = Path(__file__).parent.parent
db_path = project_root / "warehouse" / "db" / "global.duckdb"

# Check if database exists, if not use fallback to Parquet files
DB_AVAILABLE = db_path.exists()

# ---------------------------------------------------------
# Database connection helper with fallback
# ---------------------------------------------------------
def get_db_connection(max_retries=3, retry_delay=2):
    if not DB_AVAILABLE:
        return None
        
    for attempt in range(max_retries):
        try:
            return duckdb.connect(str(db_path))
        except Exception as e:
            if "cannot access the file" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None
    return None

# ---------------------------------------------------------
# Fallback data loaders using Parquet files directly
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def load_scalar_meta_fallback():
    """Fallback when database is not available"""
    try:
        # Load dimensions from Parquet files
        dim_country_path = project_root / "warehouse" / "dims" / "dim_country.parquet"
        dim_indicator_path = project_root / "warehouse" / "dims" / "dim_indicator.parquet"
        dim_unit_path = project_root / "warehouse" / "dims" / "dim_unit.parquet"
        fact_path = project_root / "warehouse" / "facts" / "fact_annual" / "fact_annual.parquet"
        
        if not all(path.exists() for path in [dim_country_path, dim_indicator_path, fact_path]):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        indicators = pd.read_parquet(dim_indicator_path)
        countries = pd.read_parquet(dim_country_path)
        
        # Load facts to get stats
        facts = pd.read_parquet(fact_path)
        stats = pd.DataFrame([{
            'min_year': facts['year'].min(),
            'max_year': facts['year'].max(), 
            'country_count': facts['iso3'].nunique(),
            'indicator_count': facts['indicator_code'].nunique()
        }])
        
        # Merge units if available
        if dim_unit_path.exists():
            units = pd.read_parquet(dim_unit_path)
            indicators = indicators.merge(units, on='unit_code', how='left')
        else:
            indicators['unit_name'] = indicators['unit_code']
            
        return indicators, countries, stats
        
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()







# ---------------------------------------------------------
# Meta loaders (cached) with fallback
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def load_scalar_meta():
    # Try database first
    con = get_db_connection()
    if con is None:
        return load_scalar_meta_fallback()
        
    try:
        indicators = con.execute(
            """
            SELECT DISTINCT 
                indicator_code, indicator_name, domain, subdomain, unit_code,
                source_provider AS source_name
            FROM dim_indicator_v
            WHERE status = 'active'
            ORDER BY domain, subdomain, indicator_name
            """
        ).df()
        countries = con.execute(
            "SELECT iso3, country_name, region, income_group FROM dim_country_v ORDER BY country_name"
        ).df()
        units = con.execute("SELECT unit_code, unit_name FROM dim_unit_v").df()
        stats = con.execute(
            """
            SELECT MIN(year) AS min_year, MAX(year) AS max_year,
                   COUNT(DISTINCT iso3) AS country_count,
                   COUNT(DISTINCT indicator_code) AS indicator_count
            FROM fact_annual_v
            """
        ).df()
        con.close()

        if not indicators.empty and not units.empty:
            indicators = indicators.merge(units, on="unit_code", how="left")
        else:
            if not indicators.empty and "unit_name" not in indicators.columns:
                indicators["unit_name"] = indicators["unit_code"]
        return indicators, countries, stats
        
    except Exception as e:
        if con:
            con.close()
        return load_scalar_meta_fallback()

@st.cache_data(ttl=300)
def load_bilateral_meta():
    con = get_db_connection()
    if con is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        bilat_inds = con.execute(
            """
            SELECT
                f.indicator_code,
                COALESCE(MAX(i.indicator_name), f.indicator_code) AS indicator_name,
                COALESCE(MAX(i.unit_code),      MAX(f.unit_code)) AS unit_code,
                COALESCE(MAX(i.source_provider),MAX(f.source_id)) AS source_name
            FROM fact_bilateral_v AS f
            LEFT JOIN dim_indicator_v AS i
              ON f.indicator_code = i.indicator_code
            GROUP BY f.indicator_code
            ORDER BY indicator_name
            """
        ).df()
        reporters = con.execute("SELECT DISTINCT reporter AS iso3 FROM fact_bilateral_v ORDER BY 1").df()
        partners  = con.execute("SELECT DISTINCT partner  AS iso3 FROM fact_bilateral_v ORDER BY 1").df()
        yrs       = con.execute("SELECT MIN(year) AS min_year, MAX(year) AS max_year FROM fact_bilateral_v").df()
        con.close()
        return bilat_inds, reporters, partners, yrs
    except Exception as e:
        if con:
            con.close()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300)
def load_io_meta():
    con = get_db_connection()
    if con is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False
    named_ok = True
    try:
        con.execute("SELECT 1 FROM fact_io_named_v LIMIT 1")
    except Exception:
        named_ok = False
    view = "fact_io_named_v" if named_ok else "fact_io_v"
    inds = con.execute(f"SELECT DISTINCT indicator_code FROM {view} ORDER BY 1").df()
    reps = con.execute(f"SELECT DISTINCT reporter AS iso3 FROM {view} ORDER BY 1").df()
    pars = con.execute(f"SELECT DISTINCT partner  AS iso3 FROM {view} ORDER BY 1").df()
    yrs  = con.execute(f"SELECT MIN(year) AS min_year, MAX(year) AS max_year FROM {view}").df()
    con.close()
    return inds, reps, pars, yrs, named_ok

# ---------------------------------------------------------
# Plot helpers (Scalar + Bilateral) - KEEP YOUR ORIGINAL FUNCTIONS
# ---------------------------------------------------------
def plot_scalar_timeseries(df, title, unit_label):
    fig = px.line(df, x="year", y="value", color="country_name",
                  labels={"value": unit_label, "year":"Year"},
                  title=title)
    fig.update_layout(hovermode="x unified", height=520)
    return fig

def make_reporter_partner_heatmap(df, title, zlabel):
    if df.empty:
        return go.Figure()
    mat = df.pivot_table(index="partner", columns="reporter", values="value", aggfunc="sum")
    fig = px.imshow(mat.sort_index(), labels=dict(x="Reporter", y="Partner", color=zlabel),
                    title=title, aspect="auto")
    fig.update_layout(height=700)
    return fig

def make_global_matrix_heatmap(df, title, zlabel):
    if df.empty:
        return go.Figure()
    mat = df.pivot_table(index="partner", columns="reporter", values="value", aggfunc="sum").fillna(0.0)
    fig = px.imshow(mat, labels=dict(x="Reporter", y="Partner", color=zlabel),
                    title=title, aspect="auto")
    fig.update_layout(height=750)
    return fig

def make_sankey(df, title):
    if df.empty:
        return go.Figure()
    reporters = sorted(df["reporter"].unique().tolist())
    partners  = sorted(df["partner"].unique().tolist())
    nodes = reporters + partners
    idx = {n:i for i,n in enumerate(nodes)}
    sources = df["reporter"].map(idx).tolist()
    targets = df["partner"].map(idx).tolist()
    values  = df["value"].astype(float).tolist()
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=12, thickness=14),
        link=dict(source=sources, target=targets, value=values),
        arrangement="snap"
    ))
    fig.update_layout(title=title, height=700)
    return fig

def make_network(df, title, top_edges=300):
    if df.empty:
        return go.Figure()
    dff = df.sort_values("value", ascending=False).head(top_edges)
    G = nx.Graph()
    for _, r in dff.iterrows():
        w = float(r["value"])
        if w <= 0:
            continue
        G.add_edge(r["reporter"], r["partner"], weight=w)
    if G.number_of_edges() == 0:
        return go.Figure()
    pos = nx.spring_layout(G, k=0.6, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_tr = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none", opacity=0.5)
    node_x, node_y, labels = [], [], []
    for n,(x,y) in pos.items():
        node_x.append(x); node_y.append(y); labels.append(n)
    node_tr = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center",
                         marker=dict(size=10, line=dict(width=0.5)))
    fig = go.Figure(data=[edge_tr, node_tr])
    fig.update_layout(title=title, showlegend=False, height=750, margin=dict(l=20,r=20,t=60,b=20))
    return fig

# ---------------------------------------------------------
# Tabs - UPDATED WITH FALLBACK LOGIC
# ---------------------------------------------------------
def render_scalar_tab():
    indicators_df, countries_df, stats_df = load_scalar_meta()
    if indicators_df.empty:
        st.error("Could not load scalar metadata.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        country_count = len(countries_df) if stats_df.empty else int(stats_df['country_count'].iloc[0])
        st.metric("Countries", f"{country_count:,}")
    with c2: 
        indicator_count = len(indicators_df) if stats_df.empty else int(stats_df['indicator_count'].iloc[0])
        st.metric("Indicators", f"{indicator_count:,}")
    with c3: 
        if not stats_df.empty:
            min_year = int(stats_df['min_year'].iloc[0])
            max_year = int(stats_df['max_year'].iloc[0])
            st.metric("Time Range", f"{min_year}–{max_year}")
        else:
            st.metric("Time Range", "N/A")
    with c4: st.metric("Data Points", "—")

    st.markdown("---")

    # Filters
    domains = sorted(indicators_df["domain"].dropna().unique())
    selected_domain = st.sidebar.selectbox("Domain", domains, index=0) if domains else None
    domain_indicators = indicators_df[indicators_df["domain"] == selected_domain] if selected_domain else indicators_df

    subdomains = sorted(domain_indicators["subdomain"].fillna("").unique())
    if len(subdomains) > 1:
        selected_subdomain = st.sidebar.selectbox("Subdomain", subdomains, index=0)
        sub_df = (domain_indicators[domain_indicators["subdomain"].isna()]
                  if selected_subdomain == ""
                  else domain_indicators[domain_indicators["subdomain"] == selected_subdomain])
    else:
        sub_df = domain_indicators

    indicator_options = sub_df[["indicator_code","indicator_name","unit_code"]].drop_duplicates()
    sel_ind = st.sidebar.selectbox(
        "Indicator",
        indicator_options["indicator_code"],
        format_func=lambda x: indicator_options.loc[indicator_options["indicator_code"]==x, "indicator_name"].iloc[0]
    )

    # Countries
    country_options = countries_df[["iso3","country_name"]].drop_duplicates()
    all_choice = [("ALL", "🌍 All Countries")]
    choice_pairs = all_choice + list(zip(country_options["iso3"], country_options["country_name"]))
    sel_codes = st.sidebar.multiselect(
        "Countries",
        options=[c[0] for c in choice_pairs],
        default=["ALL"],
        format_func=lambda x: dict(choice_pairs)[x]
    )
    selected_countries = country_options["iso3"].tolist() if "ALL" in sel_codes else sel_codes

    # Years
    min_y = int(stats_df["min_year"].iloc[0]) if not stats_df.empty else 2000
    max_y = int(stats_df["max_year"].iloc[0]) if not stats_df.empty else 2024
    y_start, y_end = st.sidebar.slider("Year Range", min_value=min_y, max_value=max_y,
                                       value=(max(min_y, 2000), max_y))

    viz = st.sidebar.radio("Visualization", ["Time Series", "Choropleth Map", "Dynamic Heatmap", "Data Table"])

    # Query data with fallback
    if not DB_AVAILABLE:
        # Use Parquet files directly
        fact_path = project_root / "warehouse" / "facts" / "fact_annual" / "fact_annual.parquet"
        if fact_path.exists():
            facts_df = pd.read_parquet(fact_path)
            df = facts_df[
                (facts_df['indicator_code'] == sel_ind) & 
                (facts_df['iso3'].isin(selected_countries)) &
                (facts_df['year'].between(y_start, y_end))
            ]
            # Merge with country names
            df = df.merge(countries_df[['iso3', 'country_name']], on='iso3', how='left')
            # Get indicator name
            indicator_name = indicators_df.loc[indicators_df['indicator_code'] == sel_ind, 'indicator_name'].iloc[0] if not indicators_df.empty else sel_ind
            df['indicator_name'] = indicator_name
        else:
            df = pd.DataFrame()
    else:
        # Use database
        con = get_db_connection()
        if con:
            q = f"""
                SELECT f.iso3, dc.country_name, f.year, f.value,
                       di.indicator_name, di.unit_code
                FROM fact_annual_v f
                JOIN dim_country_v  dc ON f.iso3 = dc.iso3
                JOIN dim_indicator_v di ON f.indicator_code = di.indicator_code
                WHERE f.indicator_code = ?
                  AND f.iso3 IN ({','.join(['?']*len(selected_countries))})
                  AND f.year BETWEEN ? AND ?
                ORDER BY f.year, dc.country_name
            """
            params = [sel_ind] + list(selected_countries) + [y_start, y_end]
            df = con.execute(q, params).df()
            con.close()
        else:
            df = pd.DataFrame()

    if df.empty:
        st.warning("No data for the selected criteria.")
        return

    unit_label = indicator_options.loc[indicator_options["indicator_code"]==sel_ind, "unit_code"].iloc[0]

    if viz == "Time Series":
        st.plotly_chart(plot_scalar_timeseries(df, df["indicator_name"].iloc[0], unit_label), use_container_width=True)
    elif viz == "Choropleth Map":
        years = sorted(df["year"].unique())
        sel_year = st.selectbox("Year", years, index=len(years)-1)
        mdf = df[df["year"] == sel_year]
        fig = px.choropleth(mdf, locations="iso3", color="value", hover_name="country_name",
                            hover_data={"value":":.2f", "iso3": False},
                            title=f"{df['indicator_name'].iloc[0]} ({unit_label}) – {sel_year}",
                            color_continuous_scale="Viridis",
                            labels={"value": unit_label})
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    elif viz == "Dynamic Heatmap":
        fig = px.choropleth(df, locations="iso3", color="value", hover_name="country_name",
                            hover_data={"value":":.2f", "iso3": False},
                            animation_frame="year",
                            title=f"{df['indicator_name'].iloc[0]} ({unit_label}) – over time",
                            color_continuous_scale="Viridis",
                            labels={"value": unit_label})
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df[["iso3","country_name","year","value"]].sort_values(["iso3","year"]), use_container_width=True)
        st.download_button("Download CSV",
                           df[["iso3","country_name","year","value"]].to_csv(index=False),
                           file_name=f"{sel_ind}_{y_start}_{y_end}.csv",
                           mime="text/csv")

# KEEP YOUR ORIGINAL bilateral and io tab functions - they should work as is



def render_bilateral_tab():
    st.header("🌐 Bilateral Data (Reporter-Partner-Year)")
    
    # Load bilateral data
    bilat_fact_path = project_root / "warehouse" / "facts" / "fact_bilateral" / "fact_bilateral.parquet"
    
    if not bilat_fact_path.exists():
        st.error("Bilateral data file not found")
        return

    try:
        bilat_data = pd.read_parquet(bilat_fact_path)
    except Exception as e:
        st.error(f"Could not load bilateral Parquet file: {e}")
        return

    # Simple column renaming - Create a new dataframe with the correct column names
    bilat_data_renamed = bilat_data.rename(columns={
        'reporter_iso3': 'reporter',
        'partner_iso3': 'partner',
        'year': 'year', 
        'value': 'value',
        'indicator_code': 'indicator_code'
    })
    
    # Use the renamed dataframe for the rest of the function
    bilat_data = bilat_data_renamed

    # Create metadata from the data
    bilat_inds = bilat_data[['indicator_code']].drop_duplicates().reset_index(drop=True)
    bilat_inds['indicator_name'] = bilat_inds['indicator_code']
    bilat_inds['unit_code'] = 'USD'
    bilat_inds['source_name'] = 'Bilateral Data'

    reporters = pd.DataFrame({'iso3': bilat_data['reporter'].unique()})
    partners = pd.DataFrame({'iso3': bilat_data['partner'].unique()})
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Reporters", f"{reporters['iso3'].nunique():,}")
    with c2: st.metric("Partners", f"{partners['iso3'].nunique():,}")
    with c3: st.metric("Indicators", f"{bilat_inds['indicator_code'].nunique():,}")
    with c4: st.metric("Data Points", f"{len(bilat_data):,}")

    # Controls
    sel_ind = st.sidebar.selectbox(
        "Bilateral Indicator",
        bilat_inds["indicator_code"],
        format_func=lambda x: bilat_inds.loc[bilat_inds["indicator_code"]==x, "indicator_name"].iloc[0]
    )
    
    sel_unit = bilat_inds.loc[bilat_inds["indicator_code"]==sel_ind, "unit_code"].iloc[0]
    sel_source = bilat_inds.loc[bilat_inds["indicator_code"]==sel_ind, "source_name"].iloc[0]

    rep_opts = reporters["iso3"].tolist()
    rep_idx = rep_opts.index("AUS") if "AUS" in rep_opts else 0
    sel_rep = st.sidebar.selectbox("Reporter (ISO3)", rep_opts, index=rep_idx)

    # Year range
    y_min = int(bilat_data['year'].min())
    y_max = int(bilat_data['year'].max())
    y_range = st.sidebar.slider("Year Range", min_value=y_min, max_value=y_max,
                                value=(max(y_min, 2000), y_max))

    topn = st.sidebar.number_input("Top N partners (latest-year ranking)", min_value=3, max_value=50, value=10)

    viz = st.sidebar.radio(
        "Visualization (Bilateral)",
        ["Time Series (by partner)", "Top Partners (bar, year)",
         "Reporter×Partner Heatmap (year)", "Global Matrix Heatmap (year)",
         "Sankey (reporter→partner, year)", "Global Network (year)"],
        index=0
    )

    # Filter data based on selections
    base_df = bilat_data[
        (bilat_data['indicator_code'] == sel_ind) &
        (bilat_data['reporter'] == sel_rep) &
        (bilat_data['year'].between(y_range[0], y_range[1]))
    ].copy()

    # Get latest year data for snapshots
    latest_y = bilat_data['year'].max()
    single_year = st.sidebar.number_input("Single Year (for snapshot)", min_value=y_min, max_value=y_max, value=latest_y)
    snap_df = bilat_data[
        (bilat_data['indicator_code'] == sel_ind) &
        (bilat_data['year'] == single_year)
    ].copy()

    # Get partner list for selection
    available_partners = base_df['partner'].unique().tolist() if not base_df.empty else []
    if not available_partners and not snap_df.empty:
        available_partners = snap_df['partner'].unique().tolist()

    sel_parts = st.sidebar.multiselect(
        "Partners (ISO3)", 
        options=available_partners, 
        default=available_partners[:min(10, len(available_partners))]
    )

    # Filter base_df by selected partners if any are selected
    if sel_parts and not base_df.empty:
        base_df = base_df[base_df['partner'].isin(sel_parts)]

    st.subheader(bilat_inds.loc[bilat_inds["indicator_code"]==sel_ind, "indicator_name"].iloc[0])
    st.caption(f"Unit: {sel_unit} | Source: {sel_source} | Reporter: {sel_rep}")

    # Visualization logic
    if viz == "Time Series (by partner)":
        if base_df.empty:
            st.warning("No data for the chosen criteria.")
        else:
            fig = px.line(base_df, x="year", y="value", color="partner",
                          title=f"{sel_ind} — {sel_rep} (by partner)",
                          labels={"value": sel_unit, "year": "Year", "partner": "Partner"},
                          hover_data={"value": ":.2f"})
            fig.update_layout(hovermode="x unified", height=520)
            st.plotly_chart(fig, use_container_width=True)

    elif viz == "Top Partners (bar, year)":
        if snap_df.empty:
            st.warning("No data for selected year.")
        else:
            top = (snap_df[snap_df['reporter'] == sel_rep]
                   .groupby("partner", as_index=False)["value"].sum()
                   .sort_values("value", ascending=False).head(int(topn)))
            if not top.empty:
                fig = px.bar(top, x="partner", y="value",
                             title=f"Top {int(topn)} partners — {sel_rep} — {int(single_year)}",
                             labels={"value": sel_unit, "partner": "Partner"},
                             hover_data={"value":":.2f"})
                fig.update_layout(height=520)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data for selected reporter and year")

    elif viz == "Reporter×Partner Heatmap (year)":
        one = snap_df[snap_df['reporter'] == sel_rep] if not snap_df.empty else pd.DataFrame()
        if not one.empty:
            st.plotly_chart(make_reporter_partner_heatmap(one, f"{sel_ind} — {sel_rep} — {int(single_year)}", sel_unit),
                            use_container_width=True)
        else:
            st.warning("No data for heatmap")

    elif viz == "Global Matrix Heatmap (year)":
        if snap_df.empty:
            st.warning("No data for selected year.")
        else:
            K = int(st.sidebar.number_input("Edges (top K)", min_value=100, max_value=5000, value=500, step=50))
            top_edges = snap_df.sort_values("value", ascending=False).head(K)
            if not top_edges.empty:
                st.plotly_chart(make_global_matrix_heatmap(top_edges, f"{sel_ind} — Global — {int(single_year)} (top {K} flows)", sel_unit),
                                use_container_width=True)
            else:
                st.warning("No data for global matrix")

    elif viz == "Sankey (reporter→partner, year)":
        if snap_df.empty:
            st.warning("No data for selected year.")
        else:
            sank = (snap_df[snap_df['reporter'] == sel_rep]
                    .groupby(["reporter","partner"], as_index=False)["value"].sum()
                    .sort_values("value", ascending=False).head(int(topn)))
            if not sank.empty:
                st.plotly_chart(make_sankey(sank, f"Sankey — {sel_rep} → partners — {int(single_year)}"),
                                use_container_width=True)
            else:
                st.warning("No data for Sankey diagram")

    else:  # Global Network
        if snap_df.empty:
            st.warning("No data for selected year.")
        else:
            E = int(st.sidebar.number_input("Edges in network (top E)", min_value=50, max_value=5000, value=300, step=50))
            edges = snap_df.sort_values("value", ascending=False).head(E)[["reporter","partner","value"]]
            if not edges.empty:
                st.plotly_chart(make_network(edges, f"Global Network — {sel_ind} — {int(single_year)} (top {E} flows)"),
                                use_container_width=True)
            else:
                st.warning("No data for network diagram")

    # Data table
    st.subheader("Bilateral Data Table (current selection)")
    if not base_df.empty:
        st.dataframe(base_df[['reporter', 'partner', 'year', 'value', 'indicator_code']], use_container_width=True)
        st.download_button("Download CSV", base_df[['reporter', 'partner', 'year', 'value', 'indicator_code']].to_csv(index=False),
                           file_name=f"{sel_ind}_{sel_rep}_{y_range[0]}_{y_range[1]}.csv",
                           mime="text/csv")
    else:
        st.warning("No data to display for current selection")









def render_io_tab():
    st.header("📊 Input–Output (Industry × Industry)")
    inds, reporters, partners, yrs, named_ok = load_io_meta()
    
    if inds.empty:
        st.warning("No IO data found.")
        
        # Try to load IO data directly from Parquet files
        io_fact_path = project_root / "warehouse" / "facts" / "fact_io"
        if io_fact_path.exists():
            # Try to find any IO parquet files
            io_files = list(io_fact_path.glob("**/*.parquet"))
            if io_files:
                st.info("Loading IO data directly from Parquet files...")
                # Load the first available IO file
                io_data = pd.read_parquet(io_files[0])
                
                if not io_data.empty:
                    inds = pd.DataFrame({'indicator_code': io_data['indicator_code'].unique()})
                    reporters = pd.DataFrame({'iso3': io_data['reporter'].unique()})
                    partners = pd.DataFrame({'iso3': io_data['partner'].unique()})
                    yrs = pd.DataFrame({
                        'min_year': [io_data['year'].min()],
                        'max_year': [io_data['year'].max()]
                    })
                    named_ok = 'from_industry_name' in io_data.columns
                else:
                    return
            else:
                return
        else:
            return

    if inds.empty:
        st.error("No IO data available")
        return

    sel_ind = st.sidebar.selectbox("IO Indicator", inds["indicator_code"].tolist(), index=0)
    rep_opts = reporters["iso3"].tolist() if not reporters.empty else []
    par_opts = partners["iso3"].tolist() if not partners.empty else []
    
    if not rep_opts or not par_opts:
        st.error("No reporter or partner data available")
        return
        
    rep_idx  = rep_opts.index("AUS") if "AUS" in rep_opts else 0
    par_idx  = par_opts.index("CHN") if "CHN" in par_opts else 0
    sel_rep  = st.sidebar.selectbox("Reporter (ISO3)", rep_opts, index=rep_idx)
    sel_par  = st.sidebar.selectbox("Partner (ISO3)",  par_opts, index=par_idx)

    y_min = int(yrs["min_year"].iloc[0]) if not yrs.empty else 1995
    y_max = int(yrs["max_year"].iloc[0]) if not yrs.empty else 2022
    if y_min >= y_max:
        st.sidebar.write(f"Available year: **{y_min}**")
        sel_year = y_min
    else:
        sel_year = st.sidebar.slider("Year (snapshot)", min_value=y_min, max_value=y_max, value=min(y_max, 2015))

    topn = st.sidebar.number_input("Top sector flows", 5, 100, 20, 1)

    # Load IO data with fallback
    if not DB_AVAILABLE:
        # Use Parquet files directly
        io_fact_path = project_root / "warehouse" / "facts" / "fact_io"
        if io_fact_path.exists():
            # Find and load IO data
            io_files = list(io_fact_path.glob("**/*.parquet"))
            if io_files:
                # For simplicity, load the first file that matches our criteria
                df_list = []
                for io_file in io_files:
                    try:
                        temp_df = pd.read_parquet(io_file)
                        # Filter for our selection
                        filtered_df = temp_df[
                            (temp_df['indicator_code'] == sel_ind) &
                            (temp_df['reporter'] == sel_rep) &
                            (temp_df['partner'] == sel_par) &
                            (temp_df['year'] == sel_year)
                        ]
                        if not filtered_df.empty:
                            df_list.append(filtered_df)
                    except:
                        continue
                
                if df_list:
                    df = pd.concat(df_list, ignore_index=True)
                else:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
    else:
        # Use database
        con = get_db_connection()
        if con:
            view = "fact_io_named_v" if named_ok else "fact_io_v"
            df = con.execute(
                f"""
                SELECT *
                FROM {view}
                WHERE indicator_code = ? AND reporter = ? AND partner = ? AND year = ?
                """,
                [sel_ind, sel_rep, sel_par, int(sel_year)],
            ).df()
            con.close()
        else:
            df = pd.DataFrame()

    if df.empty:
        st.warning("No IO data for this selection.")
        return

    st.caption(f"Indicator: {sel_ind} | Reporter: {sel_rep} → Partner: {sel_par} | Year: {sel_year}")

    f_lab = "from_industry_name" if named_ok and "from_industry_name" in df.columns else "from_industry_code"
    t_lab = "to_industry_name"   if named_ok and "to_industry_name"   in df.columns else "to_industry_code"

    # Heatmap
    st.subheader("Heatmap (from → to industry)")
    if not df.empty:
        mat = pd.pivot_table(df, index=f_lab, columns=t_lab, values="value", aggfunc="sum").fillna(0.0)
        fig_hm = px.imshow(mat, labels=dict(x="To industry", y="From industry", color="Value"),
                           title=f"{sel_rep} → {sel_par} — {sel_ind} — {sel_year}", aspect="auto")
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.warning("No data for heatmap")

    # Top flows
    st.subheader("Top sector flows")
    if not df.empty:
        top = (df.groupby([f_lab, t_lab], as_index=False)["value"].sum()
                 .sort_values("value", ascending=False).head(int(topn)))
        if not top.empty:
            top["flow"] = top[f_lab] + " → " + top[t_lab]
            fig_b = px.bar(top, x="flow", y="value",
                           labels={"value":"Value","flow":"Flow"},
                           title=f"Top {int(topn)} flows — {sel_rep}→{sel_par} — {sel_year}")
            fig_b.update_layout(xaxis_tickangle=-35, height=500)
            st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.warning("No data for top flows")

    # Sankey
    st.subheader("Sankey (from → to industry)")
    if not df.empty:
        top = (df.groupby([f_lab, t_lab], as_index=False)["value"].sum()
                 .sort_values("value", ascending=False).head(int(topn)))
        if not top.empty:
            nodes = pd.Index(sorted(set(top[f_lab]).union(set(top[t_lab]))))
            idx = {n:i for i,n in enumerate(nodes)}
            s = top[f_lab].map(idx).tolist()
            t = top[t_lab].map(idx).tolist()
            v = top["value"].astype(float).tolist()
            fig_s = go.Figure(go.Sankey(node=dict(label=list(nodes), pad=15, thickness=14),
                                        link=dict(source=s, target=t, value=v)))
            fig_s.update_layout(height=650, title=f"Sankey — {sel_rep}→{sel_par} — {sel_year}")
            st.plotly_chart(fig_s, use_container_width=True)
    else:
        st.warning("No data for Sankey diagram")

    # Data table
    st.subheader("Data table")
    if not df.empty:
        show_cols = ["from_industry_code","to_industry_code","value","unit_code"]
        if named_ok and "from_industry_name" in df.columns:
            show_cols = ["from_industry_name","to_industry_name","value","unit_code","from_industry_code","to_industry_code"]
        st.dataframe(df[show_cols].sort_values("value", ascending=False), use_container_width=True)
        st.download_button(
            "Download CSV",
            df[show_cols].to_csv(index=False),
            file_name=f"IO_{sel_ind}_{sel_rep}_{sel_par}_{sel_year}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data to display")
# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    st.title("🌍 Global Development Indicators Dashboard")

    st.sidebar.title("🔧 Dashboard Controls")
    if st.sidebar.button("🔄 Refresh data / clear cache"):
        st.cache_data.clear()
        st.experimental_rerun()

    data_type = st.sidebar.radio(
        "Data Type",
        ["Scalar (country–year)", "Bilateral (reporter–partner–year)", "Input–Output (industry–industry)"],
        index=0
    )

    if data_type.startswith("Scalar"):
        render_scalar_tab()
    elif data_type.startswith("Bilateral"):
        render_bilateral_tab()
    else:
        render_io_tab()

    st.markdown("---")
    st.caption(f"Database: {'DuckDB' if DB_AVAILABLE else 'Parquet Files'} | Updated {datetime.now():%Y-%m-%d}")

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    main()