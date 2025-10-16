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

# --- Forecast caching to avoid reruns (run Optimizer) ---
@st.cache_data
def generate_forecast():
    """Run forecast once and cache results."""
    forecaster = Forecast(INPUT_SALES, HOLDOUT_WEEKS, FORECAST_HORIZON)
    forecast_df = forecaster.run()
    return forecast_df


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
        route = optimizer.run(week_start=str(week_start))

    st.success("Optimization complete!")

    # --- Show route table ---
    if route:
        df_route = pd.DataFrame(route)
        df_route['cumulative_qty'] = df_route['qty'].cumsum()
        df_route['visit_order'] = df_route['order'] + 1

        st.subheader("Optimized Delivery Route")
        st.markdown("""
        - `visit_order`: sequence of deliveries.
        - `site_id`: delivery site.
        - `qty`: units to deliver at that site.
        - `cumulative_qty`: running total (helps track barge capacity usage).
        """)
        st.dataframe(df_route[['visit_order', 'site_id', 'qty', 'cumulative_qty']])

        # --- Draw route map ---
        st.subheader("Route Map")
        st.markdown("Visual representation of the optimized delivery route from the depot.")
        G = nx.DiGraph()
        edges = [('PORT0', route[0]['site_id'])] + [(route[i]['site_id'], route[i+1]['site_id']) for i in range(len(route)-1)] + [(route[-1]['site_id'], 'PORT0')]
        G.add_edges_from(edges)

        pos = {site: (i, i % 2) for i, site in enumerate(['PORT0'] + [r['site_id'] for r in route])}
        plt.figure(figsize=(8, 4))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', arrows=True)
        st.pyplot(plt.gcf())

    else:
        st.warning("No feasible route found for this week.")
