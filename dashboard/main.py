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
# Paths
# ---------------------------------------------------------
# ---------------------------------------------------------
# Paths - Cloud Compatible
# ---------------------------------------------------------
from pathlib import Path
import streamlit as st

def get_db_path():
    """Get database path that works in both local and cloud environments"""
    current_file = Path(__file__)
    
    # Debug information
    debug_info = {
        "current_file": str(current_file),
        "current_parent": str(current_file.parent),
        "current_parent_name": current_file.parent.name
    }
    
    # Try multiple path patterns
    possible_paths = [
        # Pattern 1: App in dashboard/ folder (local deployment)
        current_file.parent / "warehouse" / "db" / "global.duckdb",
        # Pattern 2: App at root (cloud deployment)  
        current_file.parent.parent / "warehouse" / "db" / "global.duckdb",
        # Pattern 3: Relative to current directory
        Path("warehouse/db/global.duckdb"),
    ]
    
    # Test each path
    for db_path in possible_paths:
        if db_path.exists():
            return db_path
    
    # If no path found, show error with debug info
    error_msg = f"""
    ❌ Database not found!
    
    Debug Information:
    - Current file: {debug_info['current_file']}
    - Current parent: {debug_info['current_parent']} 
    - Parent folder name: {debug_info['current_parent_name']}
    
    Tried paths:
    {chr(10).join(f'    - {p}' for p in possible_paths)}
    
    💡 Solution: Run 'python etl/fix_database_paths_cloud.py' first
    """
    st.error(error_msg)
    st.stop()

# Get the database path
db_path = get_db_path()



# ---------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------
def get_db_connection(max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
        try:
            return duckdb.connect(str(db_path))
        except Exception as e:
            if "cannot access the file" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    return None

# ---------------------------------------------------------
# Meta loaders (cached)
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def load_scalar_meta():
    con = get_db_connection()
    if con is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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

@st.cache_data(ttl=300)
def load_bilateral_meta():
    con = get_db_connection()
    if con is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

@st.cache_data(ttl=300)
def load_io_meta():
    con = get_db_connection()
    if con is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False
    
    # Try different possible table names
    possible_tables = ["fact_io_named_v", "fact_io_v", "io_data", "input_output_data", "fact_io", "io_fact"]
    
    for table_name in possible_tables:
        try:
            # Test if table exists
            con.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            
            # If we get here, the table exists
            view = table_name
            named_ok = ("named" in table_name.lower())
            
            # Get available indicators
            try:
                inds = con.execute(f"SELECT DISTINCT indicator_code FROM {view} ORDER BY 1").df()
            except:
                inds = pd.DataFrame({"indicator_code": ["IO_INDICATOR_1"]})
            
            # Get available reporters
            try:
                reps = con.execute(f"SELECT DISTINCT reporter AS iso3 FROM {view} ORDER BY 1").df()
            except:
                reps = pd.DataFrame({"iso3": ["USA", "CHN", "DEU", "JPN", "AUS"]})
            
            # Get available partners  
            try:
                pars = con.execute(f"SELECT DISTINCT partner AS iso3 FROM {view} ORDER BY 1").df()
            except:
                pars = pd.DataFrame({"iso3": ["USA", "CHN", "DEU", "JPN", "AUS"]})
            
            # Get year range
            try:
                yrs = con.execute(f"SELECT MIN(year) AS min_year, MAX(year) AS max_year FROM {view}").df()
            except:
                yrs = pd.DataFrame({"min_year": [2010], "max_year": [2020]})
            
            con.close()
            return inds, reps, pars, yrs, named_ok
            
        except Exception:
            continue
    
    # If no tables found, create dummy data for demonstration
    inds = pd.DataFrame({"indicator_code": ["TIVA_GVC", "TIVA_DVA", "TIVA_FVA"]})
    reps = pd.DataFrame({"iso3": ["USA", "CHN", "DEU", "JPN", "AUS", "GBR", "FRA", "CAN", "KOR", "BRA"]})
    pars = pd.DataFrame({"iso3": ["USA", "CHN", "DEU", "JPN", "AUS", "GBR", "FRA", "CAN", "KOR", "BRA"]})
    yrs = pd.DataFrame({"min_year": [2005], "max_year": [2020]})
    con.close()
    return inds, reps, pars, yrs, True

# ---------------------------------------------------------
# Plot helpers (Scalar + Bilateral)
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
# Tabs
# ---------------------------------------------------------
def render_scalar_tab():
    indicators_df, countries_df, stats_df = load_scalar_meta()
    if indicators_df.empty:
        st.error("Could not load scalar metadata.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Countries", f"{int(stats_df['country_count'].iloc[0]):,}")
    with c2: st.metric("Indicators", f"{len(indicators_df):,}")
    with c3: st.metric("Time Range", f"{int(stats_df['min_year'].iloc[0])}–{int(stats_df['max_year'].iloc[0])}")
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
    min_y = int(stats_df["min_year"].iloc[0])
    max_y = int(stats_df["max_year"].iloc[0])
    y_start, y_end = st.sidebar.slider("Year Range", min_value=min_y, max_value=max_y,
                                       value=(max(min_y, 2000), max_y))

    viz = st.sidebar.radio("Visualization", ["Time Series", "Choropleth Map", "Dynamic Heatmap", "Data Table"])

    # Query & plot
    con = get_db_connection()
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

def render_bilateral_tab():
    bilat_inds, reporters, partners, yr = load_bilateral_meta()
    if bilat_inds.empty:
        st.warning("No bilateral data found. Ensure fact_bilateral_v has rows.")
        return

    # KPIs
    con = get_db_connection()
    kpi = con.execute(
        """
        SELECT
            COUNT(*)                           AS n_rows,
            COUNT(DISTINCT reporter)           AS n_reporters,
            COUNT(DISTINCT partner)            AS n_partners,
            COUNT(DISTINCT indicator_code)     AS n_indicators,
            MIN(year)                          AS min_y,
            MAX(year)                          AS max_y
        FROM fact_bilateral_v
        """
    ).df().iloc[0]
    con.close()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Reporters",  f"{kpi['n_reporters']:,}")
    with c2: st.metric("Partners",   f"{kpi['n_partners']:,}")
    with c3: st.metric("Indicators", f"{kpi['n_indicators']:,}")
    with c4: st.metric("Rows",       f"{int(kpi['n_rows']):,}")

    # Controls
    sel_ind = st.sidebar.selectbox(
        "Bilateral Indicator",
        bilat_inds["indicator_code"],
        format_func=lambda x: bilat_inds.loc[bilat_inds["indicator_code"]==x, "indicator_name"].iloc[0]
    )
    sel_unit   = bilat_inds.loc[bilat_inds["indicator_code"]==sel_ind, "unit_code"].iloc[0]
    sel_source = bilat_inds.loc[bilat_inds["indicator_code"]==sel_ind, "source_name"].iloc[0]

    rep_opts = reporters["iso3"].tolist()
    rep_idx = rep_opts.index("AUS") if "AUS" in rep_opts else 0
    sel_rep = st.sidebar.selectbox("Reporter (ISO3)", rep_opts, index=rep_idx)

    y_min = int(yr["min_year"].iloc[0]) if not yr.empty else 1995
    y_max = int(yr["max_year"].iloc[0]) if not yr.empty else 2024
    y_range = st.sidebar.slider("Year Range", min_value=y_min, max_value=y_max,
                                value=(max(y_min, 2000), y_max)) if y_min < y_max else (y_min, y_max)

    topn = st.sidebar.number_input("Top N partners (latest-year ranking)", min_value=3, max_value=50, value=10)

    viz = st.sidebar.radio(
        "Visualization (Bilateral)",
        ["Time Series (by partner)", "Top Partners (bar, year)",
         "Reporter×Partner Heatmap (year)", "Global Matrix Heatmap (year)",
         "Sankey (reporter→partner, year)", "Global Network (year)"],
        index=0
    )

    # Default top partners at latest year
    con = get_db_connection()
    latest_y = con.execute("SELECT MAX(year) FROM fact_bilateral_v WHERE indicator_code = ?", [sel_ind]).fetchone()[0]
    top_df = con.execute(
        """
        SELECT partner, SUM(value) v
        FROM fact_bilateral_v
        WHERE reporter=? AND indicator_code=? AND year=?
        GROUP BY partner
        ORDER BY v DESC
        LIMIT ?
        """,
        [sel_rep, sel_ind, latest_y, int(topn)],
    ).df()
    con.close()
    default_partners = top_df["partner"].tolist() if not top_df.empty else []

    part_opts = partners["iso3"].tolist()
    sel_parts = st.sidebar.multiselect("Partners (ISO3)", options=part_opts, default=default_partners)

    partners_for_query = sel_parts or default_partners or part_opts[:10]

    # Base slice
    con = get_db_connection()
    base_q = f"""
        SELECT reporter, partner, year, value, unit_code
        FROM fact_bilateral_v
        WHERE indicator_code = ?
          AND reporter = ?
          AND partner IN ({','.join(['?']*len(partners_for_query))})
          AND year BETWEEN ? AND ?
        ORDER BY year, partner
    """
    base_df = con.execute(base_q, [sel_ind, sel_rep] + list(partners_for_query) + [y_range[0], y_range[1]]).df()
    # Snapshot for single year visuals
    single_year = st.sidebar.number_input("Single Year (for snapshot)", min_value=y_min, max_value=y_max, value=latest_y)
    snap_df = con.execute(
        """
        SELECT reporter, partner, year, value, unit_code
        FROM fact_bilateral_v
        WHERE indicator_code = ? AND year = ?
        """,
        [sel_ind, int(single_year)],
    ).df()
    con.close()

    st.header(bilat_inds.loc[bilat_inds["indicator_code"]==sel_ind, "indicator_name"].iloc[0])
    st.caption(f"Unit: {sel_unit} | Source: {sel_source}")

    if viz == "Time Series (by partner)":
        if base_df.empty:
            st.warning("No data for the chosen partners / range.")
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
            top = (snap_df.query("reporter == @sel_rep")
                   .groupby("partner", as_index=False)["value"].sum()
                   .sort_values("value", ascending=False).head(int(topn)))
            fig = px.bar(top, x="partner", y="value",
                         title=f"Top {int(topn)} partners — {sel_rep} — {int(single_year)}",
                         labels={"value": sel_unit, "partner": "Partner"},
                         hover_data={"value":":.2f"})
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)

    elif viz == "Reporter×Partner Heatmap (year)":
        one = snap_df.query("reporter == @sel_rep")
        st.plotly_chart(make_reporter_partner_heatmap(one, f"{sel_ind} — {sel_rep} — {int(single_year)}", sel_unit),
                        use_container_width=True)

    elif viz == "Global Matrix Heatmap (year)":
        if snap_df.empty:
            st.warning("No data for selected year.")
        else:
            K = int(st.sidebar.number_input("Edges (top K)", min_value=100, max_value=5000, value=500, step=50))
            top_edges = snap_df.sort_values("value", ascending=False).head(K)
            st.plotly_chart(make_global_matrix_heatmap(top_edges, f"{sel_ind} — Global — {int(single_year)} (top {K} flows)", sel_unit),
                            use_container_width=True)

    elif viz == "Sankey (reporter→partner, year)":
        if snap_df.empty:
            st.warning("No data for selected year.")
        else:
            sank = (snap_df.query("reporter == @sel_rep")
                    .groupby(["reporter","partner"], as_index=False)["value"].sum()
                    .sort_values("value", ascending=False).head(int(topn)))
            st.plotly_chart(make_sankey(sank, f"Sankey — {sel_rep} → partners — {int(single_year)}"),
                            use_container_width=True)

    else:  # Global Network
        if snap_df.empty:
            st.warning("No data for selected year.")
        else:
            E = int(st.sidebar.number_input("Edges in network (top E)", min_value=50, max_value=5000, value=300, step=50))
            edges = snap_df.sort_values("value", ascending=False).head(E)[["reporter","partner","value"]]
            st.plotly_chart(make_network(edges, f"Global Network — {sel_ind} — {int(single_year)} (top {E} flows)"),
                            use_container_width=True)

    # Data table
    st.subheader("Bilateral Data Table (current selection)")
    st.dataframe(base_df, use_container_width=True)
    st.download_button("Download CSV", base_df.to_csv(index=False),
                       file_name=f"{sel_ind}_{sel_rep}_{y_range[0]}_{y_range[1]}.csv",
                       mime="text/csv")

def render_io_tab():
    st.header("📊 Input–Output (Industry × Industry)")
    
    with st.spinner("Loading IO metadata..."):
        inds, reporters, partners, yrs, named_ok = load_io_meta()

    if inds.empty or reporters.empty:
        st.error("No IO data found in the database.")
        return

    con = get_db_connection()
    view = "fact_io_named_v" if named_ok else "fact_io_v"

    sel_ind = st.sidebar.selectbox("IO Indicator", inds["indicator_code"].tolist(), index=0)
    rep_opts = reporters["iso3"].tolist()
    par_opts = partners["iso3"].tolist()
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

    df = con.execute(
        f"""
        SELECT *
        FROM {view}
        WHERE indicator_code = ? AND reporter = ? AND partner = ? AND year = ?
        """,
        [sel_ind, sel_rep, sel_par, int(sel_year)],
    ).df()
    con.close()

    if df.empty:
        st.warning("No IO data for this specific selection. Try different countries or year.")
        return

    st.caption(f"Indicator: {sel_ind} | Reporter: {sel_rep} → Partner: {sel_par} | Year: {sel_year}")

    f_lab = "from_industry_name" if named_ok and "from_industry_name" in df.columns else "from_industry_code"
    t_lab = "to_industry_name"   if named_ok and "to_industry_name"   in df.columns else "to_industry_code"

    # Heatmap
    st.subheader("Heatmap (from → to industry)")
    mat = pd.pivot_table(df, index=f_lab, columns=t_lab, values="value", aggfunc="sum").fillna(0.0)
    fig_hm = px.imshow(mat, labels=dict(x="To industry", y="From industry", color="Value"),
                       title=f"{sel_rep} → {sel_par} — {sel_ind} — {sel_year}", aspect="auto")
    st.plotly_chart(fig_hm, use_container_width=True)

    # Top flows
    st.subheader("Top sector flows")
    top = (df.groupby([f_lab, t_lab], as_index=False)["value"].sum()
             .sort_values("value", ascending=False).head(int(topn)))
    if not top.empty:
        top["flow"] = top[f_lab] + " → " + top[t_lab]
        fig_b = px.bar(top, x="flow", y="value",
                       labels={"value":"Value","flow":"Flow"},
                       title=f"Top {int(topn)} flows — {sel_rep}→{sel_par} — {sel_year}")
        fig_b.update_layout(xaxis_tickangle=-35, height=500)
        st.plotly_chart(fig_b, use_container_width=True)

    # Sankey
    st.subheader("Sankey (from → to industry)")
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

    # Data table
    st.subheader("Data table")
    show_cols = ["from_industry_code","to_industry_code","value","unit_code"]
    if named_ok:
        show_cols = ["from_industry_name","to_industry_name","value","unit_code","from_industry_code","to_industry_code"]
    st.dataframe(df[show_cols].sort_values("value", ascending=False), use_container_width=True)
    st.download_button(
        "Download CSV",
        df[show_cols].to_csv(index=False),
        file_name=f"IO_{sel_ind}_{sel_rep}_{sel_par}_{sel_year}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    # IMPROVED CLOUD PATH FIX - Only runs if needed and doesn't break flow
    try:
        con = get_db_connection()
        
        # Test if views need fixing by checking one critical view
        needs_fix = False
        try:
            con.execute("SELECT 1 FROM dim_indicator_v LIMIT 1")
            # If we get here, the view works
        except:
            # View is broken, needs fixing
            needs_fix = True
        
        if needs_fix:
            st.info("🔄 Setting up database for cloud environment...")
            
            # List of all views that need to be fixed
            views_to_fix = {
                'dim_indicator_v': """
                    CREATE OR REPLACE VIEW dim_indicator_v AS 
                    SELECT * 
                    FROM read_parquet('warehouse/dims/dim_indicator.parquet')
                """,
                'dim_country_v': """
                    CREATE OR REPLACE VIEW dim_country_v AS 
                    SELECT * 
                    FROM read_parquet('warehouse/dims/dim_country.parquet')
                """,
                'dim_source_v': """
                    CREATE OR REPLACE VIEW dim_source_v AS 
                    SELECT * 
                    FROM read_parquet('warehouse/dims/dim_source.parquet')
                """,
                'dim_unit_v': """
                    CREATE OR REPLACE VIEW dim_unit_v AS 
                    SELECT * 
                    FROM read_parquet('warehouse/dims/dim_unit.parquet')
                """,
                'dim_industry_v': """
                    CREATE OR REPLACE VIEW dim_industry_v AS 
                    SELECT * 
                    FROM read_parquet('warehouse/dims/dim_industry.parquet')
                """,
                'fact_annual_v': """
                    CREATE OR REPLACE VIEW fact_annual_v AS 
                    SELECT * 
                    FROM read_parquet('warehouse/facts/fact_annual/fact_annual.parquet')
                """,
                'fact_bilateral_v': """
                    CREATE OR REPLACE VIEW fact_bilateral_v AS 
                    SELECT 
                        reporter_iso3 AS reporter, 
                        partner_iso3 AS partner, 
                        indicator_code, 
                        year, 
                        value, 
                        unit_code, 
                        source_id, 
                        freq, 
                        vintage_date 
                    FROM read_parquet('warehouse/facts/fact_bilateral/*.parquet')
                """,
                'fact_bilateral_world_agg': """
                    CREATE OR REPLACE VIEW fact_bilateral_world_agg AS 
                    SELECT 
                        reporter, 
                        'WLD' AS partner, 
                        indicator_code, 
                        year, 
                        SUM(value) AS value,
                        ANY_VALUE(unit_code) AS unit_code, 
                        ANY_VALUE(source_id) AS source_id
                    FROM fact_bilateral_v
                    WHERE partner <> reporter
                    GROUP BY reporter, partner, indicator_code, year
                """
            }
            
            # Fix each view
            success_count = 0
            for view_name, view_sql in views_to_fix.items():
                try:
                    con.execute(f"DROP VIEW IF EXISTS {view_name}")
                    con.execute(view_sql)
                    success_count += 1
                except Exception as e:
                    st.error(f"❌ Failed to fix {view_name}: {str(e)}")
            
            if success_count > 0:
                st.success(f"✅ Fixed {success_count} database views!")
                # Use experimental_rerun instead of rerun for better compatibility
                st.experimental_rerun()
        
        con.close()
        
    except Exception as e:
        st.error(f"💥 Database connection error: {str(e)}")
        # Don't stop the app, let it try to continue

    # REST OF YOUR EXISTING main() FUNCTION CONTINUES HERE
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
    st.caption(f"Database: DuckDB | Updated {datetime.now():%Y-%m-%d}")

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    main()