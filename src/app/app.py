import streamlit as st
import pandas as pd
from optimize import Optimizer
from forecast import Forecast
import networkx as nx
import matplotlib.pyplot as plt

HOLDOUT_WEEKS = 12
FORECAST_HORIZON = 52
INPUT_SALES = r"../../data/sales_history.csv"
SITE_SPECS = r"../../data/site_specs.csv"
TRAVEL = r"../../data/travel_times.csv"
BARGE = r"../../data/barge_specs.csv"

st.title("Weekly Barge Delivery Optimizer")
st.markdown("""
This app helps you plan barge deliveries efficiently using forecasted demand and 
route optimization. Steps:
1. Select the week to optimize deliveries.
2. Click **Run Forecast & Optimize**.
3. View the forecasted demand, optimized delivery sequence, and a route map.
""")

week_start = st.text_input("Select week to optimize", value="2026-04-13")

# --- Forecast caching to avoid reruns ---
@st.cache_data
def generate_forecast():
    forecaster = Forecast(INPUT_SALES, HOLDOUT_WEEKS, FORECAST_HORIZON)
    return forecaster.run()

# --- Run pipeline ---
if st.button("Run Forecast & Optimize"):

    # --- Forecast ---
    with st.spinner("Generating forecasts..."):
        forecast_df = generate_forecast()
        st.success("Forecast complete!")

    st.subheader("Forecast Table")
    st.markdown("""
    Shows the predicted weekly demand per site and product.
    - `site_id`: delivery location.
    - `product_id`: product.
    - `week_start`: start date of the week.
    - `forecast_units`: predicted demand.
    """)
    st.dataframe(forecast_df.tail(20))

    # --- Optimization ---
    with st.spinner("Running optimizer..."):
        optimizer = Optimizer(forecast_df, SITE_SPECS, TRAVEL, BARGE)
        route_dict = optimizer.run(week_start=str(week_start))

    st.success("Optimization complete!")

    if route_dict:
        # --- Flatten dictionary into DataFrame ---
        route_rows = []
        for barge_id, stops in route_dict.items():
            for stop in stops:
                route_rows.append({**stop, 'barge_id': barge_id})
        df_route = pd.DataFrame(route_rows)

        if not df_route.empty:
            df_route['cumulative_qty'] = df_route.groupby('barge_id')['qty'].cumsum()
            df_route['visit_order'] = df_route.groupby('barge_id')['order'].rank(method='first').astype(int)

            st.subheader(f"Optimized Delivery Routes")
            st.markdown("""
            - `visit_order`: sequence of deliveries.
            - `site_id`: delivery site.
            - `qty`: units to deliver at that site.
            - `cumulative_qty`: running total (helps track barge capacity usage).
            - `arrival_min` / `departure_min`: estimated arrival and departure times in minutes since start of week.
            """)

            # --- Show per-barge route tables ---
            for barge_id, stops in df_route.groupby('barge_id'):
                st.subheader(f"Barge {barge_id} Route")
                st.dataframe(stops[['visit_order', 'site_id', 'qty', 'cumulative_qty', 'arrival_min', 'departure_min']])

            st.subheader(f"Route Maps")
            st.markdown("Visual representation of the optimized delivery route from the depot.")
           
            # --- Draw per-barge route maps ---
            for barge_id, stops in df_route.groupby('barge_id'):
                st.subheader(f"Barge {barge_id} Route Map")
                G = nx.DiGraph()
                if len(stops) > 0:
                    edges = [('PORT0', stops.iloc[0]['site_id'])] + \
                            [(stops.iloc[i]['site_id'], stops.iloc[i+1]['site_id']) for i in range(len(stops)-1)] + \
                            [(stops.iloc[-1]['site_id'], 'PORT0')]
                    G.add_edges_from(edges)
                    pos = {site: (i, i % 2) for i, site in enumerate(['PORT0'] + list(stops['site_id']))}
                    plt.figure(figsize=(8, 4))
                    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', arrows=True)
                    st.pyplot(plt.gcf())
        else:
            st.warning("No feasible stops for any barge this week.")
    else:
        st.warning("No feasible route found for this week.")
