# Dashboard Documentation

## Overview

The Swiss German ASR Dashboard is a Streamlit-based web application for visualizing and analyzing Swiss German Automatic Speech Recognition model performance data.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
    - [Requirements](#requirements)
    - [Quick Start](#quick-start)
- [Screenshots](#screenshots)
    - [Main Dashboard View](#main-dashboard-view)
    - [Model Comparison](#model-comparison)
    - [Dialect Breakdown](#dialect-breakdown)
    - [Data Table](#data-table)
- [Usage Guide](#usage-guide)
    - [Navigation](#navigation)
    - [Using Filters](#using-filters)
    - [Interpreting Charts](#interpreting-charts)
- [Components](#components)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
    - [Dashboard won't start](#dashboard-wont-start)
    - [Data not loading](#data-not-loading)
    - [Performance issues](#performance-issues)
    - [Port conflicts](#port-conflicts)
    - [Browser compatibility](#browser-compatibility)

## Installation

### Requirements

- Docker
- Docker Compose

### Quick Start

1. Navigate to the project root directory:
```bash
cd /path/to/cp-swiss-german-asr
```

2. Start the dashboard:
```bash
docker compose up dashboard
```

   **Alternative:** To run as an isolated service with manual port mapping:
   ```bash
   docker compose run --rm -p 8501:8501 dashboard
   ```

3. Open your browser and navigate to `http://localhost:8501`

4. **Explore the dashboard:**
   - Select a model from the sidebar dropdown (e.g., "whisper-small")
   - Use filters to narrow results:
     - **Models:** Compare multiple models
     - **Dialects:** Focus on specific Swiss German variants
     - **Metric:** Switch between WER, CER, or BLEU
   - Navigate tabs:
     - **Overview:** Aggregate metrics and summary statistics
     - **Dialect Analysis:** Per-canton performance breakdown

5. To stop the dashboard, press `CTRL + C` in the terminal

## Screenshots

### Main Dashboard View
*[Screenshot placeholder - Add main dashboard overview]*

### Model Comparison
*[Screenshot placeholder - Add model comparison chart]*

### Dialect Breakdown
*[Screenshot placeholder - Add dialect breakdown visualization]*

### Data Table
*[Screenshot placeholder - Add data table view]*

## Usage Guide

### Navigation

The dashboard consists of a sidebar for filtering and multiple visualization components:

- **Sidebar**: Contains filters and configuration options
- **Statistics Panel**: Displays key metrics and summary statistics
- **Model Comparison**: Compare performance across different models
- **Dialect Breakdown**: Analyze results by Swiss German dialect
- **Data Table**: View detailed tabular data

### Using Filters

1. Use the sidebar to select specific models, dialects, or date ranges
2. Filters apply globally across all visualizations
3. Reset filters using the reset button in the sidebar

### Interpreting Charts

- **Model Comparison**: Shows comparative metrics (WER, CER, accuracy) across models
- **Dialect Breakdown**: Displays performance distribution by dialect region
- Hover over chart elements for detailed tooltips

## Components

The dashboard is built with modular components:

- `components/sidebar.py` - Filter controls and navigation
- `components/statistics_panel.py` - Summary metrics display
- `components/model_comparison.py` - Model performance comparison charts
- `components/dialect_breakdown.py` - Dialect-based analysis
- `components/data_table.py` - Tabular data view
- `utils/data_loader.py` - Data loading and processing utilities

## Known Limitations

- Large datasets may cause slower initial load times
- Real-time data updates require manual refresh
- Some visualizations may not render optimally on mobile devices
- Maximum file upload size is limited by Streamlit defaults (provisional feature)

## Troubleshooting

### Dashboard won't start

**Issue**: Docker container fails to start

**Solution**:
- Ensure Docker is running: `docker --version`
- Check if port 8501 is already in use: `lsof -i :8501`
- Review logs: `docker compose logs dashboard`

### Data not loading

**Issue**: No data appears in visualizations

**Solution**:
- Verify data files are mounted correctly in `docker-compose.yml`
- Check data file format and structure
- Review application logs for errors

### Performance issues

**Issue**: Dashboard is slow or unresponsive

**Solution**:
- Reduce the date range or filter dataset size
- Restart the container: `docker compose restart dashboard`
- Increase Docker memory allocation in Docker settings

### Port conflicts

**Issue**: Port 8501 already in use

**Solution**:
- Modify the port mapping in `docker-compose.yml`
- Stop other services using the port
- Use an alternative port: `docker compose run --rm -p 8502:8501 dashboard`

### Browser compatibility

**Issue**: Dashboard doesn't display correctly

**Solution**:
- Use a modern browser (Chrome, Firefox, Edge, Safari)
- Clear browser cache
- Disable browser extensions that may interfere