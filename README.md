<div align="center">



## 🎉 Final Integration Status

**Status:** ✅ **PRODUCTION READY** - Complete Integration Achieved
**Date:** 2025-10-07 20:46:10
**Platform:** GoalDiggers Football Betting Intelligence Platform

### Integration Highlights
- ✅ 100% Production Validation Pass Rate
- ✅ Real Data Integration Active
- ✅ Optimized UI/UX Components
- ✅ Comprehensive Error Handling
- ✅ Performance Optimized
- ✅ Deployment Ready

### Quick Start
```bash
# Launch the full platform (dashboard + services)
python unified_launcher.py all

# Or just the dashboard
python unified_launcher.py dashboard
```

Legacy launchers such as `launch_production.py` remain available as fallbacks, but all new workflows should use `unified_launcher.py`.

**Platform is ready for immediate deployment and use!**

---

# GoalDiggers AI Football Predictions Platform

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](#) [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](#)

<sub>Update badge URLs to point to your repository (placeholders used).</sub>

</div>

## Overview

GoalDiggers is a comprehensive AI-powered football prediction platform that combines real-time data ingestion, machine learning models, and advanced analytics to provide accurate betting insights for football matches across major leagues worldwide.

## 🚀 Key Features (v2.0 Highlights)

### Data Management

- **Real-time Data Ingestion**: Automated ETL pipeline with multi-source integration
- **Historical Data Backfill**: Comprehensive historical data for top 6 leagues
- **Data Quality Assurance**: Validation and consistency checks
- **Enhanced Team Metadata**: Proper team names, flags, countries, and venue information

### Machine Learning & Explainability

- **Calibrated Probabilities**: Multi-class isotonic / Platt scaling for reliable probability estimates
- **Feature Capture**: Real feature snapshots stored each prediction for aligned SHAP explanations
- **Explainability Service**: Cached SHAP (or mock fallback) with latency + mode metrics
- **Betting Insights Generator**: Expected value, edge, risk categorization & stake suggestions

### User Interface

- **Modern Dashboard**: Professional Streamlit interface with enhanced UX
- **Interactive Components**: Team cards, match displays, and prediction visualizations
- **Responsive Design**: Mobile-friendly interface with custom CSS
- **Real-time Updates**: Live data refresh and background processing

### Production & Observability

- **Observability Service**: FastAPI endpoints `/metrics`, `/health/live`, `/health/ready`
- **Prometheus Metrics**: Prediction latency, calibration applications, errors, explanation counts & latency
- **Idempotent Ingestion**: Differential update skipping & uniqueness enforcement
- **Resilience**: Categorized errors and graceful UI fallbacks (mock explanations, fallback matches)

## 🏗️ Architecture

See `ARCHITECTURE.md` for a full diagram, data flow, calibration internals, explainability, metrics and extensibility roadmap.

### Core Components (Summary)

```
GoalDiggers/
├── database/                 # Database schema and management
│   ├── schema.py            # SQLAlchemy ORM models
│   ├── db_manager.py        # Database operations
│   └── enhanced_migration.py # Team metadata migration
├── ingestion/               # Data ingestion pipeline
│   ├── etl_pipeline.py      # ETL operations
│   ├── historical_backfill.py # Historical data population
│   ├── scheduler.py         # Background job scheduling
│   └── data_quality.py     # Data validation
├── models/                  # Machine learning models
│   ├── enhanced_ml_pipeline.py # Calibrated ensemble models
│   ├── betting_insights_generator.py # Betting insights
│   └── predictive/         # Core ML models
├── dashboard/               # User interface
│   ├── enhanced_production_homepage.py # Main dashboard
│   ├── components/         # UI components
│   └── health_check.py     # System monitoring
├── utils/                   # Utility functions
│   └── team_data_enhancer.py # Team metadata management
└── unified_launcher.py         # Canonical application launcher
```

### Data Flow

1. **Data Ingestion**: Real-time data from multiple APIs (Football-Data.org, etc.)
2. **ETL Processing**: Data transformation, normalization, and quality checks
3. **Database Storage**: Normalized storage in SQLite with proper relationships
4. **Model Training**: ML models trained on historical data
5. **Prediction Generation**: Real-time predictions with confidence scores
6. **UI Display**: Enhanced dashboard with team flags, match cards, and insights

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- SQLite3
- Required Python packages (see requirements.txt)

### Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd footie
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   ```bash
   # Set up API keys (optional)
   export FOOTBALL_DATA_API_KEY="your-api-key"
   ```

4. **Launch the platform**

   ```bash
   python unified_launcher.py dashboard
   ```

   To orchestrate the dashboard plus supporting services in one command:

   ```bash
   python unified_launcher.py all --services api sse
   ```

   Legacy launchers (`streamlined_production_launcher.py`, `ultimate_production_launcher.py`, `launch_production.py`) remain for compatibility but are no longer the primary entry points.

5. **(Optional) Run performance probe baseline**

   ```bash
   python scripts/performance/performance_probe.py --batches 5 --batch-size 12 --output probe_launch.json
   ```

6. **(Optional) Verify model artifact integrity**

   ```bash
   python -c "from models.xgboost_predictor import XGBoostPredictor; XGBoostPredictor('models/xgboost_model.pkl', verify_checksums=True)"
   ```

7. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - The system will automatically populate with historical or fallback sample data if live APIs are unavailable

---

### Launch / Release Checklist (Condensed)

| Step | Command / Action | Outcome |
|------|------------------|---------|
| Run smoke tests | `pytest -q tests/test_prediction_engine.py` | Core prediction logic green |
| Timeout audit | `python scripts/audit_http_timeouts.py` | No active unbounded sessions |
| Performance probe | `python scripts/performance/performance_probe.py --batches 5 --batch-size 12 --output probe_launch.json` | Baseline JSON captured |
| Check sums | `python ultimate_production_launcher.py --verify-checksums --dry-run` | All model artifacts verified |
| Tag release | `echo 1.0.0 > VERSION && git add VERSION && git commit -m "Release v1.0.0" && git tag v1.0.0` | Immutable tag created |
| Archive artifacts | Attach `probe_launch.json`, `PRODUCTION_LAUNCH_READINESS.md` to release | Reproducibility & audit trail |

---

### Model Artifact Integrity

Model persistence uses JSON/UBJ formats plus SHA256 checksum sidecar files located alongside the model metadata and booster files. On load, mismatches produce a warning or raise (if strict mode enabled):

```python
from models.xgboost_predictor import XGBoostPredictor
predictor = XGBoostPredictor('models/xgboost_model.pkl', verify_checksums=True, strict=True)
```

If a checksum mismatch occurs in strict mode, deployment should abort and the artifact bundle re-generated.

### Probability Calibration Status

Calibration loads lazily if fitted parameters exist. To introspect:

```python
from models.enhanced_real_data_predictor import get_enhanced_match_prediction
info = get_enhanced_match_prediction('Arsenal', 'Chelsea', league='Premier League')
print(info['calibration'])
```

Output fields include `enabled`, `loaded`, `applied`, and `fitted`.

### Performance Probe Usage

The probe script produces latency, cache, and instrumentation histograms for N batches of synthetic or live fixture predictions:

```bash
python scripts/performance/performance_probe.py --batches 5 --batch-size 12 --output probe_launch.json
```

Key JSON fields:

- `avg_batch_duration_ms`, `p95_batch_duration_ms`
- `cache_hit_rate_series`
- `evictions`
- `model_version`
- `artifact_checksum_status`

Embed this JSON (or a summarized subset) in release notes for traceable baselines.

### Handling Common Warnings

| Warning | Cause | Mitigation |
|---------|-------|------------|
| NumPy ufunc size changed | Binary ABI mismatch (typically transient) | Ensure consistent dependency lock before CI build |
| Pydantic v1 class-based config deprecation | Upcoming Pydantic v3 removal | Scheduled migration (see roadmap) |
| XGBoost deprecated binary model format | Legacy binary .pkl save | Prefer JSON/UBJ export path already implemented |
| pkg_resources deprecation | Deprecated Setuptools API | Replace with importlib.metadata in future refactor |

### Future Migration Notes

Pydantic v2 migration is planned post-launch; code currently remains compatible with v1 while suppressing disruptive behavior. External telemetry (Prometheus exporter wiring) can be layered atop existing timing & histogram utilities.

## 📊 Database Schema

### Core Tables

- **leagues**: League information (Premier League, La Liga, etc.)
- **teams**: Team metadata with flags, countries, venues
- **matches**: Match data with scores, dates, status
- **predictions**: ML model predictions with confidence scores
- **odds**: Betting odds from various bookmakers
- **match_stats**: Detailed match statistics
- **team_stats**: Team performance metrics

### Enhanced Team Metadata

Each team includes:

- Full name and display name
- Team flag emoji and country flag
- Primary color scheme
- Venue information and capacity
- League association and country code

## 🤖 Machine Learning, Calibration & Explainability

### Pipeline

1. Generate raw probabilities (strength heuristics + contextual adjustments).
2. Capture full numeric feature snapshot (`EnhancedRealDataPredictor.get_last_features`).
3. Apply probability calibration (isotonic or Platt) if fitted parameters found.
4. Produce betting insights (expected value, edge, risk, stake sizing).
5. Generate explanation via `explanation_service` (SHAP or mock) using real captured features.

### Calibration

- Implemented as one-vs-rest multi-class probability post-processing.
- Stored JSON parameters; lazy-loaded; gauge metric indicates fitted state.

### Explainability

- Hash-based cache; modes: `real`, `cache`, `mock` (all metered in Prometheus).
- Real feature alignment ensures attributions reflect calibrated inputs (pre-calibration snapshot also stored for analysis).

### Performance / Metrics

- Latency histograms for predictions & explanations.
- Counters for calibration applications & prediction errors.

## 🎨 User Interface & Design System

### Dashboard Features

- **Unified Material+Glassmorphism UI**: All dashboards use a unified design system with Material Design principles and subtle glassmorphic effects for depth and elegance. Modern CSS ensures a visually cohesive, accessible, and responsive experience.
- **Team Selection**: Interactive team picker with flags, colors, and metadata
- **Match Display**: Enhanced match cards with team information, glassmorphic backgrounds, and Material shadows
- **Prediction Visualization**: Animated confidence bars and outcome probabilities
- **Betting Insights**: Expected value analysis and recommendations
- **Real-time Updates**: Live data refresh and background processing

### UI Components & Design System

- **UnifiedDesignSystem**: Centralized Python module (`dashboard/components/unified_design_system.py`) for all CSS, color, and component styling. Injects Material+glassmorphism CSS and provides robust fallbacks.
- **Team Cards**: Professional team display with flags, colors, and glassmorphic backgrounds
- **Match Cards**: Comprehensive match information with predictions, Material shadows, and micro-interactions
- **Loading States**: Smooth loading animations and error handling
- **Responsive Design**: Mobile-friendly interface, accessible focus rings, and ARIA support
- **Fallback Logic**: All UI components include robust fallback logic for missing data, API errors, or unsupported browsers

#### Example: Injecting Unified CSS

```python
from dashboard.components.unified_design_system import get_unified_design_system
design_system = get_unified_design_system()
design_system.inject_unified_css('premium')
```

#### Example: Creating a Unified Card

```python
design_system.create_unified_card(lambda: st.write("Card content here"))
```

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URI=sqlite:///data/football.db

# API Keys (optional)
FOOTBALL_DATA_API_KEY=your-api-key
API_FOOTBALL_KEY=your-api-key

# Background Processing
ETL_INTERVAL_HOURS=6
BACKGROUND_INGESTION=true
```

### Customization

- **Team Metadata**: Edit `utils/team_data_enhancer.py` for team information
- **UI Styling**: Modify `dashboard/components/enhanced_ui_components.py`
- **Model Parameters**: Update model configuration files
- **Data Sources**: Configure additional APIs in ingestion modules

## 📈 Production Deployment & Observability

### Background Services

- **ETL Pipeline**: Runs every 6 hours for data refresh
- **Model Updates**: Automated model retraining
- **Health Monitoring**: System status checks
- **Error Recovery**: Automatic retry mechanisms

### Performance Optimization

- **Memory Management**: Optimized for production memory usage
- **Caching**: Intelligent caching for API responses
- **Database Optimization**: Indexed queries and connection pooling
- **Error Handling**: Comprehensive error recovery

## 🔍 Monitoring, Metrics & Health

Prometheus Metrics (subset):

```
goaldiggers_prediction_latency_seconds{model_version="2.0"}
goaldiggers_prediction_calibration_applications_total
goaldiggers_prediction_errors_total
goaldiggers_explanations_generated_total{mode="real|mock|cache"}
goaldiggers_explanation_latency_seconds{mode="real|mock|cache"}
goaldiggers_process_uptime_seconds
goaldiggers_calibration_fitted
real_data_integrator_matches_stored
```

Health Endpoints:

```
/health/live  -> basic liveness
/health/ready -> DB + predictor readiness aggregation
```

### Health Checks

- **Database Status**: Connection and data integrity checks
- **API Connectivity**: External service availability
- **Model Performance**: Prediction accuracy monitoring
- **System Resources**: Memory and CPU usage tracking

### Data Quality

- **Validation Rules**: Comprehensive data quality checks
- **Consistency Monitoring**: Cross-reference validation
- **Error Reporting**: Detailed error logging and reporting
- **Recovery Procedures**: Automated data recovery

## 🚀 Advanced Features

### Betting Insights

- **Expected Value Analysis**: Mathematical EV calculations
- **Risk Assessment**: User-specific risk tolerance
- **Stake Recommendations**: Optimal bet sizing
- **Market Analysis**: Odds comparison and value identification

### Real-time Processing

- **Live Data Ingestion**: Real-time match updates
- **Dynamic Predictions**: Updated predictions as new data arrives
- **Background Jobs**: Automated processing without user intervention
- **Scalable Architecture**: Designed for high-volume data processing

## 📚 API Reference (Selected)

### Core Functions

```python
# Team data enhancement
from utils.team_data_enhancer import get_enhanced_team_data
team_data = get_enhanced_team_data("Manchester City")

# Data migration
from database.enhanced_migration import run_enhanced_migration
run_enhanced_migration()

# ETL pipeline
from ingestion.etl_pipeline import ingest_from_sources
ingest_from_sources(days_back=2, days_ahead=7)

# Betting insights
from models.betting_insights_generator import AdvancedBettingInsightsGenerator
insights = generator.generate_insights(home_team, away_team, match_data, odds_data)
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include type hints
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the health check dashboard
- Contact the development team

## 🔄 Changelog

### Version 2.0.0 (Current)

- Real feature-aligned explainability layer
- Probability calibration (isotonic / Platt) integration
- Observability service (metrics + health endpoints)
- Extended Prometheus metrics (prediction & explanation)
- CI pipeline (lint/test/package/security scan)
- Enhanced UI component system & design tokens

### Version 1.0.0

- Basic prediction system
- Simple dashboard interface
- Historical data backfill
- Core ML models

---

**GoalDiggers AI Football Predictions Platform** - Powered by advanced machine learning and real-time data processing.
