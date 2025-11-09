# Dashboard Documentation

## Overview

The Swiss German ASR Dashboard is a Streamlit-based web application for visualizing and analyzing Swiss German Automatic Speech Recognition model performance data.

## Installation

### Requirements

- Docker
- Docker Compose

### Running the Dashboard

1. Navigate to the project root directory:
```bash
cd /path/to/cp-swiss-german-asr
```

2. Start the dashboard using Docker Compose:
```bash
docker compose run --rm -p 8501:8501 dashboard
```
You can also use:
```bash
docker compose up
```
to start all services defined in the `docker-compose.yml`.

3. Access the dashboard in your browser at `http://localhost:8501`

4. To stop the dashboard:
- Press `CTRL + C` in the terminal where the dashboard is running

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
- Review logs: `docker compose logs frontend`

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
- Restart the container: `docker compose restart frontend`
- Increase Docker memory allocation in Docker settings

### Port conflicts

**Issue**: Port 8501 already in use

**Solution**:
- Modify the port mapping in `docker-compose.yml`
- Stop other services using the port
- Use an alternative port: `docker compose up frontend -p 8502:8501`

### Browser compatibility

**Issue**: Dashboard doesn't display correctly

**Solution**:
- Use a modern browser (Chrome, Firefox, Edge, Safari)
- Clear browser cache
- Disable browser extensions that may interfere