# Global Development Data Observatory Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://4dgdodashboard.streamlit.app/)

This repository contains the source code for the **Global Development Data Observatory (4DGDO) Dashboard**, a Streamlit web application designed for interactive exploration of global development indicators. The dashboard provides a user-friendly interface to visualize and analyze a wide range of economic and social development data from various international sources.

## Features

The 4DGDO Dashboard offers a rich set of features for data exploration and visualization:

- **Multiple Data Types:** Explore scalar (country-year), bilateral (reporter-partner-year), and input-output data.
- **Interactive Visualizations:** Choose from time series plots, choropleth maps, and dynamic heatmaps to visualize the data.
- **Customizable Queries:** Filter data by domain, subdomain, indicator, country, and year range.
- **Rich Data Catalog:** Access over 170 indicators and data for 212 countries, with a time range spanning from 1750 to 2030.
- **Data Caching:** The application caches data to ensure fast and efficient performance.

## Data Sources

The dashboard integrates data from a variety of reputable international organizations, including:

- World Bank (WDI)
- International Monetary Fund (IMF, WEO)
- Our World in Data (OWID)
- Organisation for Economic Co-operation and Development (OECD, TiVA)

## Technical Architecture

The project is built with Python and leverages a modern data stack:

- **Frontend:** [Streamlit](https://streamlit.io/) for the interactive web interface.
- **Data Backend:** [DuckDB](https://duckdb.org/) for efficient data storage and querying.
- **Data Processing:** [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation.
- **Visualization:** [Plotly](https://plotly.com/python/) for creating interactive charts and maps.
- **Project Structure:** The repository is organized into `dashboard`, `registry`, and `warehouse` directories, separating the application logic, data catalog, and data storage.

## Installation and Usage

To run the dashboard locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/puchen8229/4dgdo_dashboard.git
    cd 4dgdo_dashboard
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**

    ```bash
    streamlit run streamlit_app.py
    ```

## Deployment

The dashboard is deployed on [Streamlit Cloud](https://streamlit.io/cloud) and is publicly accessible at:

[https://4dgdodashboard.streamlit.app/](https://4dgdodashboard.streamlit.app/)

## Contributing

Contributions to the project are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
