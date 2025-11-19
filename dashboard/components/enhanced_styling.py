"""
Enhanced CSS Styles for GoalDiggers Platform
Production-ready styling with modern design principles
"""

GOALDIGGERS_CSS = """
<style>
/* ===== GOALDIGGERS PRODUCTION THEME ===== */

/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root Variables */
:root {
    --primary-color: #1e3a8a;
    --secondary-color: #3b82f6;
    --accent-color: #10b981;
    --success-color: #059669;
    --warning-color: #f59e0b;
    --error-color: #dc2626;
    --text-primary: #1f2937;
    --text-secondary: #6b7280;
    --background-primary: #ffffff;
    --background-secondary: #f8fafc;
    --border-color: #e5e7eb;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    --border-radius: 8px;
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Global Styles */
.stApp {
    font-family: var(--font-family);
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Container */
.main > div {
    padding-top: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}

/* ===== HEADER STYLES ===== */
.goaldiggers-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    padding: 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    text-align: center;
}

.goaldiggers-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.goaldiggers-header .subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    margin-top: 0.5rem;
}

/* ===== CARD STYLES ===== */
.prediction-card {
    background: var(--background-primary);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.prediction-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.card-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent-color);
}

/* ===== TEAM SELECTION STYLES ===== */
.team-selector {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 1rem;
    align-items: center;
    margin: 1.5rem 0;
}

.team-card {
    background: var(--background-secondary);
    border: 2px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
}

.team-card:hover {
    border-color: var(--secondary-color);
    background: var(--background-primary);
}

.vs-indicator {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-secondary);
    padding: 0.5rem;
    background: var(--background-primary);
    border-radius: 50%;
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px solid var(--border-color);
}

/* ===== PREDICTION RESULTS ===== */
.prediction-results {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.prediction-item {
    background: var(--background-primary);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    text-align: center;
    border: 2px solid var(--border-color);
    transition: all 0.3s ease;
}

.prediction-item:hover {
    border-color: var(--secondary-color);
    transform: translateY(-2px);
}

.prediction-label {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.prediction-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
}

.prediction-percentage {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
}

/* ===== BUTTON STYLES ===== */
.stButton > button {
    background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
    filter: brightness(1.1);
}

.stButton > button:active {
    transform: translateY(0);
}

/* ===== STATUS INDICATORS ===== */
.status-indicator {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-success {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success-color);
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-warning {
    background: rgba(245, 158, 11, 0.1);
    color: var(--warning-color);
    border: 1px solid rgba(245, 158, 11, 0.2);
}

.status-error {
    background: rgba(220, 38, 38, 0.1);
    color: var(--error-color);
    border: 1px solid rgba(220, 38, 38, 0.2);
}

/* ===== METRICS DASHBOARD ===== */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: var(--background-primary);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    border: 1px solid var(--border-color);
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-color);
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ===== LOADING ANIMATIONS ===== */
.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--border-color);
    border-radius: 50%;
    border-top-color: var(--secondary-color);
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ===== SIDEBAR STYLES ===== */
.sidebar .sidebar-content {
    background: var(--background-primary);
    padding: 1rem;
}

/* ===== RESPONSIVE DESIGN ===== */
@media (max-width: 768px) {
    .goaldiggers-header h1 {
        font-size: 2rem;
    }
    
    .team-selector {
        grid-template-columns: 1fr;
        gap: 0.5rem;
    }
    
    .vs-indicator {
        width: 40px;
        height: 40px;
        font-size: 1.2rem;
    }
    
    .prediction-results {
        grid-template-columns: 1fr;
    }
    
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

/* ===== ACCESSIBILITY ===== */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* ===== UTILITY CLASSES ===== */
.text-center { text-align: center; }
.text-left { text-align: left; }
.text-right { text-align: right; }

.mb-1 { margin-bottom: 0.25rem; }
.mb-2 { margin-bottom: 0.5rem; }
.mb-3 { margin-bottom: 1rem; }
.mb-4 { margin-bottom: 1.5rem; }

.mt-1 { margin-top: 0.25rem; }
.mt-2 { margin-top: 0.5rem; }
.mt-3 { margin-top: 1rem; }
.mt-4 { margin-top: 1.5rem; }

.p-1 { padding: 0.25rem; }
.p-2 { padding: 0.5rem; }
.p-3 { padding: 1rem; }
.p-4 { padding: 1.5rem; }

.font-weight-light { font-weight: 300; }
.font-weight-normal { font-weight: 400; }
.font-weight-medium { font-weight: 500; }
.font-weight-semibold { font-weight: 600; }
.font-weight-bold { font-weight: 700; }

.text-primary { color: var(--text-primary); }
.text-secondary { color: var(--text-secondary); }
.text-accent { color: var(--accent-color); }
.text-success { color: var(--success-color); }
.text-warning { color: var(--warning-color); }
.text-error { color: var(--error-color); }

</style>
"""

def inject_css():
    """Inject the GoalDiggers CSS into Streamlit."""
    import streamlit as st
    st.markdown(GOALDIGGERS_CSS, unsafe_allow_html=True)

def create_header(title="⚽ GoalDiggers", subtitle="AI-Powered Football Betting Intelligence"):
    """Create a professional header with GoalDiggers branding."""
    import streamlit as st
    
    header_html = f"""
    <div class="goaldiggers-header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def create_card(content, header=None):
    """Create a styled card component."""
    import streamlit as st
    
    card_html = f"""
    <div class="prediction-card">
        {f'<div class="card-header">{header}</div>' if header else ''}
        {content}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_team_selector_visual(home_team, away_team):
    """Create a visual team selector."""
    import streamlit as st
    
    selector_html = f"""
    <div class="team-selector">
        <div class="team-card">
            <h3>{home_team}</h3>
            <div class="text-secondary">Home</div>
        </div>
        <div class="vs-indicator">VS</div>
        <div class="team-card">
            <h3>{away_team}</h3>
            <div class="text-secondary">Away</div>
        </div>
    </div>
    """
    st.markdown(selector_html, unsafe_allow_html=True)

def create_prediction_results(predictions):
    """Create styled prediction results."""
    import streamlit as st
    
    if len(predictions) >= 3:
        results_html = f"""
        <div class="prediction-results">
            <div class="prediction-item">
                <div class="prediction-label">Home Win</div>
                <div class="prediction-value">{predictions[0]:.1%}</div>
                <div class="prediction-percentage">Probability</div>
            </div>
            <div class="prediction-item">
                <div class="prediction-label">Draw</div>
                <div class="prediction-value">{predictions[1]:.1%}</div>
                <div class="prediction-percentage">Probability</div>
            </div>
            <div class="prediction-item">
                <div class="prediction-label">Away Win</div>
                <div class="prediction-value">{predictions[2]:.1%}</div>
                <div class="prediction-percentage">Probability</div>
            </div>
        </div>
        """
        st.markdown(results_html, unsafe_allow_html=True)

def create_status_indicator(status, text):
    """Create a status indicator."""
    import streamlit as st
    
    status_class = f"status-{status}"
    indicator_html = f"""
    <span class="status-indicator {status_class}">
        {text}
    </span>
    """
    st.markdown(indicator_html, unsafe_allow_html=True)

def create_metrics_grid(metrics):
    """Create a metrics dashboard grid."""
    import streamlit as st
    
    metrics_html = '<div class="metrics-grid">'
    for label, value in metrics.items():
        metrics_html += f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """
    metrics_html += '</div>'
    
    st.markdown(metrics_html, unsafe_allow_html=True)

def show_loading(text="Loading..."):
    """Show a loading indicator."""
    import streamlit as st
    
    loading_html = f"""
    <div class="text-center p-3">
        <span class="loading-spinner"></span>
        <div class="mt-2">{text}</div>
    </div>
    """
    st.markdown(loading_html, unsafe_allow_html=True)
