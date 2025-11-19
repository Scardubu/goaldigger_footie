# GoalDiggers Dashboard Documentation

## Overview
The GoalDiggers Dashboard is an AI-powered football betting insights platform that provides users with comprehensive match analysis, predictions, and value betting opportunities. The dashboard features a modern, responsive UI with intuitive navigation and real-time data updates.

## Features

- **Unified Material+Glassmorphism UI**: All dashboards use a unified design system with Material Design principles and glassmorphic effects for a visually cohesive, modern, and accessible experience.
- **Match Predictions**: AI-driven predictions for upcoming football matches
- **Value Betting**: Identification of value betting opportunities with edge calculations
- **Performance Tracking**: Historical record of prediction accuracy and betting performance
- **System Status Monitoring**: Real-time monitoring of API keys, scrapers, and system health

## Technical Architecture


### Frontend
- **Streamlit**: Primary UI framework
- **Plotly**: Interactive data visualizations
- **UnifiedDesignSystem**: Centralized Python module for all CSS, color, and component styling. Injects Material+glassmorphism CSS and provides robust fallbacks for all dashboards.
- **Custom CSS**: Responsive design with mobile optimization, Material shadows, and glassmorphic backgrounds

### Backend
- **Data Integration**: Asynchronous data acquisition from multiple sources
- **AI Models**: Integration with various AI providers for match analysis
- **Performance Monitoring**: Tracking of system metrics and prediction accuracy

## Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/goaldiggers.git

# Navigate to the project directory
cd goaldiggers

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard
```bash
python -m streamlit run dashboard/app.py
```

## Dashboard Components

### Match Details
The match details component displays comprehensive information about a selected match, including:
- Team information and match details
- Prediction probabilities with confidence indicators
- Bookmaker odds and implied probabilities
- Value betting opportunities with edge calculations
- Context features used in the prediction
- AI-generated match analysis with tactical insights

### System Status
The system status component monitors the health of various system components:
- API key availability and validity
- Scraper performance and health metrics
- Cache hit/miss statistics
- System resource utilization

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines for Python code
- Use snake_case for functions and variables
- Use PascalCase for classes
- Include comprehensive docstrings for all modules and functions

### Testing
- Unit tests should be written for all core functionality
- Integration tests should validate the end-to-end data pipeline
- Run tests before submitting pull requests:
```bash
pytest tests/
```


### Accessibility & Fallbacks
The dashboard is designed with accessibility and robust fallback logic:
- ARIA attributes for screen reader support
- Sufficient color contrast for readability
- Responsive design for various screen sizes
- Keyboard navigation support
- Focus rings and accessible micro-interactions (Material Design)
- Fallback logic for missing data, API errors, or unsupported browsers

## Performance Optimizations

- Data caching for frequently accessed information
- Asynchronous data loading to prevent UI blocking
- Lazy loading of visualizations for faster initial load
- Resource monitoring to identify bottlenecks
- Unified CSS injection for fast, consistent UI rendering

## Troubleshooting
Common issues and their solutions:
- **API Connection Errors**: Check API key validity and rate limits
- **Slow Performance**: Review cache settings and reduce unnecessary data requests
- **Visualization Issues**: Ensure Plotly and Streamlit versions are compatible

## Contributing
Guidelines for contributing to the project:
1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with a detailed description of changes

## License
This project is licensed under the MIT License - see the LICENSE file for details.
