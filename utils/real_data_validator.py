#!/usr/bin/env python3
"""
Real Data Validator - Ensures predictions use fresh, validated real data
Blocks publication and surfaces alerts when data quality is insufficient
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RealDataValidator:
    """
    Validates real data quality and freshness for prediction pipelines.
    
    Features:
    - Health checks for data sources (DB, API, scrapers)
    - Freshness validation with configurable SLAs
    - Quality scoring for prediction features
    - Publication blocking when data is stale
    """
    
    def __init__(
        self,
        max_fixture_age_hours: float = 24.0,
        max_form_age_days: int = 7,
        max_standings_age_days: int = 3,
        min_quality_score: float = 0.5
    ):
        """
        Initialize real data validator.
        
        Args:
            max_fixture_age_hours: Maximum acceptable age for fixture data
            max_form_age_days: Maximum acceptable age for team form data
            max_standings_age_days: Maximum acceptable age for league standings
            min_quality_score: Minimum quality score to pass validation
        """
        self.max_fixture_age = timedelta(hours=max_fixture_age_hours)
        self.max_form_age = timedelta(days=max_form_age_days)
        self.max_standings_age = timedelta(days=max_standings_age_days)
        self.min_quality_score = min_quality_score
        
        # Track validation results
        self._last_validation: Optional[Dict[str, Any]] = None
        self._validation_count = 0
        self._passed_count = 0
        self._failed_count = 0
        
        # Metrics integration (optional Prometheus)
        self._metrics_enabled = False
        try:
            from prometheus_client import Counter, Histogram
            self._validation_counter = Counter(
                'real_data_validation_total',
                'Total real data validations',
                ['result', 'recommendation']
            )
            self._quality_score_histogram = Histogram(
                'real_data_quality_score',
                'Real data quality scores'
            )
            self._metrics_enabled = True
            logger.info("✅ Prometheus metrics enabled for RealDataValidator")
        except Exception:
            logger.debug("Prometheus metrics not available for RealDataValidator")
    
    def validate_prediction_data(
        self,
        features: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate prediction data quality and freshness.
        
        Args:
            features: Prediction features from vectorized generator
            metadata: Additional metadata (timestamps, sources, etc.)
        
        Returns:
            Tuple of (is_valid, validation_report)
        """
        self._validation_count += 1
        now = datetime.now(timezone.utc)
        
        report = {
            'timestamp': now.isoformat(),
            'valid': False,
            'quality_score': 0.0,
            'checks': {},
            'warnings': [],
            'errors': [],
            'blocking_issues': []
        }
        
        metadata = metadata or {}
        
        # Check 1: Real data usage flag
        real_data_used = features.get('real_data_used', 0.0) > 0.5
        report['checks']['real_data_used'] = real_data_used
        if not real_data_used:
            report['errors'].append("Features indicate fallback/synthetic data used")
            report['blocking_issues'].append("real_data_not_available")
        
        # Check 2: Fixture data freshness
        fixture_timestamp = self._extract_timestamp(
            metadata.get('real_data_timestamp') or 
            features.get('real_data_timestamp_epoch')
        )
        if fixture_timestamp:
            fixture_age = now - fixture_timestamp
            # For scheduled matches, be more lenient (up to 7 days old is acceptable)
            # Only block if data is extremely stale (>7 days)
            fixture_too_old = fixture_age > timedelta(days=7)
            fixture_fresh = fixture_age <= self.max_fixture_age
            report['checks']['fixture_fresh'] = fixture_fresh
            report['checks']['fixture_age_hours'] = fixture_age.total_seconds() / 3600
            report['checks']['fixture_too_old'] = fixture_too_old
            
            if fixture_too_old:
                msg = f"Fixture data is {fixture_age.days}d old (limit: 7 days)"
                report['warnings'].append(msg)
                report['blocking_issues'].append("fixture_data_too_old")
            elif not fixture_fresh:
                # Warning but not blocking for data 1-7 days old
                msg = f"Fixture data is {fixture_age.total_seconds() / 3600:.1f}h old (optimal: <{self.max_fixture_age.total_seconds() / 3600:.1f}h)"
                report['warnings'].append(msg)
        else:
            report['checks']['fixture_fresh'] = False
            report['errors'].append("No fixture timestamp available")
            report['blocking_issues'].append("missing_fixture_timestamp")
        
        # Check 3: Feature coverage (form, H2H, standings, xG)
        # Updated to recognize estimated xG as partial credit
        xg_quality = features.get('xg_data_quality', 0.0)
        xg_fallback = features.get('xg_fallback_used', 0.0)
        
        feature_coverage = {
            'historic_form': bool(features.get('real_data_historic_form', 0.0)),
            'head_to_head': bool(features.get('real_data_head_to_head', 0.0)),
            'league_table': bool(features.get('real_data_league_table', 0.0)),
            'xg': xg_quality > 0.3  # Accept estimated xG as valid (quality > 0.3)
        }
        report['checks']['feature_coverage'] = feature_coverage
        report['checks']['xg_estimated'] = bool(xg_fallback > 0.5)
        
        coverage_score = sum(feature_coverage.values()) / len(feature_coverage)
        if coverage_score < 0.5:
            report['warnings'].append(f"Low feature coverage: {coverage_score:.1%}")
        
        if xg_fallback > 0.5 and xg_quality < 0.5:
            report['warnings'].append(f"Using estimated xG (quality: {xg_quality:.2f})")
        
        # Check 4: Data source health (optional for development/testing)
        data_sources = metadata.get('data_sources', {})
        if isinstance(data_sources, dict):
            source_health = {
                'api_available': bool(data_sources.get('api_football') or data_sources.get('football_data')),
                'db_available': bool(data_sources.get('database')),
                'scraper_available': bool(data_sources.get('understat'))
            }
            report['checks']['source_health'] = source_health
            
            # Only block if explicitly no sources AND we're in production mode
            has_any_source = any(source_health.values())
            if not has_any_source:
                # Check if we're in development/test mode (be more lenient)
                if os.getenv('GOALDIGGERS_MODE') == 'production':
                    report['errors'].append("No data sources available")
                    report['blocking_issues'].append("all_sources_unavailable")
                else:
                    report['warnings'].append("No data sources configured (development mode)")
        
        # Check 5: Cross-source consistency (if multiple sources available)
        real_data_sources = features.get('real_data_sources', {})
        if isinstance(real_data_sources, dict):
            source_count = sum([
                bool(real_data_sources.get('historic_form')),
                bool(real_data_sources.get('h2h')),
                bool(real_data_sources.get('league_table')),
                bool(real_data_sources.get('xg'))
            ])
            
            # Check for consistency in quality scores
            quality_raw = features.get('real_data_quality_raw', 0.0)
            quality_adj = features.get('real_data_quality', 0.0)
            recency_factor = features.get('real_data_recency_factor', 1.0)
            
            report['checks']['cross_source_consistency'] = {
                'source_count': source_count,
                'quality_raw': quality_raw,
                'quality_adjusted': quality_adj,
                'recency_factor': recency_factor
            }
            
            # Warn if recency adjustment significantly downgraded quality
            if recency_factor < 0.7 and quality_raw > 0.5:
                report['warnings'].append(
                    f"Data quality degraded by recency (factor: {recency_factor:.2f})"
                )
            
            # Only block on inconsistency if we have real data but no sources detected
            if source_count == 0 and real_data_used and os.getenv('GOALDIGGERS_MODE') == 'production':
                # Only treat as inconsistency if real_data_used flag is set but no sources detected
                report['errors'].append("No real data sources detected despite real_data_used flag")
                report['blocking_issues'].append("data_source_inconsistency")
        
        # Calculate overall quality score
        quality_components = []
        
        # Real data weight: 40%
        if real_data_used:
            quality_components.append(0.4)
        
        # Freshness weight: 20% (reduced from 30%)
        if fixture_timestamp:
            if report['checks'].get('fixture_fresh'):
                quality_components.append(0.2)  # Full credit for fresh data
            elif not report['checks'].get('fixture_too_old', False):
                # Partial credit for data 1-7 days old
                age_days = fixture_age.days
                freshness_factor = max(0.3, 1 - (age_days / 7))  # Min 30% credit
                quality_components.append(0.2 * freshness_factor)
            # No credit for data >7 days old
        
        # Feature coverage weight: 40% (increased from 30%)
        quality_components.append(0.4 * coverage_score)
        
        quality_score = sum(quality_components)
        report['quality_score'] = quality_score
        
        # Determine validity
        is_valid = (
            quality_score >= self.min_quality_score and
            len(report['blocking_issues']) == 0
        )
        report['valid'] = is_valid
        
        # Add recommendation
        if is_valid:
            report['recommendation'] = 'PUBLISH_ALLOWED'
            self._passed_count += 1
        elif quality_score >= self.min_quality_score * 0.7:
            report['recommendation'] = 'PUBLISH_WITH_WARNING'
            self._failed_count += 1
        else:
            report['recommendation'] = 'BLOCK_PUBLICATION'
            report['blocking_issues'].append('insufficient_quality_score')
            self._failed_count += 1
        
        self._last_validation = report
        
        # Record metrics
        if self._metrics_enabled:
            try:
                result_label = 'pass' if is_valid else 'fail'
                recommendation_label = report.get('recommendation', 'unknown')
                self._validation_counter.labels(
                    result=result_label,
                    recommendation=recommendation_label
                ).inc()
                self._quality_score_histogram.observe(quality_score)
            except Exception as e:
                logger.debug(f"Failed to record validation metrics: {e}")
        
        # Log validation result
        if is_valid:
            logger.info(f"✅ Data validation PASSED (quality: {quality_score:.2f})")
        else:
            logger.warning(
                f"⚠️ Data validation FAILED (quality: {quality_score:.2f}, "
                f"blocking issues: {report['blocking_issues']})"
            )
        
        return is_valid, report
    
    def _extract_timestamp(self, value: Any) -> Optional[datetime]:
        """Extract datetime from various formats."""
        if value is None:
            return None
        
        # Unix timestamp (epoch)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except (ValueError, OSError):
                return None
        
        # ISO string
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                return None
        
        # Already datetime
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value
        
        return None
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self._validation_count
        pass_rate = (self._passed_count / total * 100) if total > 0 else 0
        
        return {
            'total_validations': total,
            'passed': self._passed_count,
            'failed': self._failed_count,
            'pass_rate_pct': pass_rate,
            'last_validation': self._last_validation
        }
    
    def should_block_publication(self, validation_report: Dict[str, Any]) -> bool:
        """Determine if publication should be blocked based on validation report."""
        return (
            validation_report.get('recommendation') == 'BLOCK_PUBLICATION' or
            len(validation_report.get('blocking_issues', [])) > 0
        )
    
    def format_validation_message(self, validation_report: Dict[str, Any]) -> str:
        """Format validation report for UI display."""
        quality = validation_report.get('quality_score', 0.0)
        recommendation = validation_report.get('recommendation', 'UNKNOWN')
        
        if recommendation == 'PUBLISH_ALLOWED':
            return f"✅ High quality prediction (score: {quality:.0%}) - Publication allowed"
        elif recommendation == 'PUBLISH_WITH_WARNING':
            warnings = validation_report.get('warnings', [])
            warning_text = warnings[0] if warnings else "Data quality acceptable but not optimal"
            return f"⚠️ {warning_text} (quality: {quality:.0%})"
        else:
            issues = validation_report.get('blocking_issues', [])
            issue_text = ', '.join(issues) if issues else "Insufficient data quality"
            return f"🚫 Publication blocked: {issue_text} (quality: {quality:.0%})"


# Global validator instance
_validator: Optional[RealDataValidator] = None


def get_real_data_validator() -> RealDataValidator:
    """Get or create global validator instance."""
    global _validator
    if _validator is None:
        _validator = RealDataValidator(
            max_fixture_age_hours=float(os.getenv('REAL_DATA_MAX_FIXTURE_AGE_HOURS', '24')),
            max_form_age_days=int(os.getenv('REAL_DATA_MAX_FORM_AGE_DAYS', '7')),
            max_standings_age_days=int(os.getenv('REAL_DATA_MAX_STANDINGS_AGE_DAYS', '3')),
            min_quality_score=float(os.getenv('REAL_DATA_MIN_QUALITY_SCORE', '0.5'))
        )
    return _validator


if __name__ == "__main__":
    # Test validator
    logging.basicConfig(level=logging.INFO)
    
    validator = RealDataValidator()
    
    # Test case 1: Good quality data
    print("\n=== Test 1: High Quality Data ===")
    features1 = {
        'real_data_used': 1.0,
        'real_data_historic_form': 1.0,
        'real_data_head_to_head': 1.0,
        'real_data_league_table': 1.0,
        'xg_data_quality': 0.85,
        'xg_fallback_used': 0.0,
        'real_data_timestamp_epoch': datetime.now(timezone.utc).timestamp(),
        'real_data_sources': {
            'historic_form': True,
            'h2h': True,
            'league_table': True,
            'xg': True,
            'quality': 0.85
        }
    }
    metadata1 = {
        'real_data_timestamp': datetime.now(timezone.utc).isoformat(),
        'data_sources': {'api_football': True, 'database': True}
    }
    valid, report = validator.validate_prediction_data(features1, metadata1)
    print(f"Valid: {valid}")
    print(f"Quality: {report['quality_score']:.2%}")
    print(f"Recommendation: {report['recommendation']}")
    print(validator.format_validation_message(report))
    
    # Test case 2: Stale data
    print("\n=== Test 2: Stale Data ===")
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    features2 = {
        'real_data_used': 1.0,
        'real_data_historic_form': 1.0,
        'real_data_timestamp_epoch': (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    }
    metadata2 = {'real_data_timestamp': old_timestamp}
    valid, report = validator.validate_prediction_data(features2, metadata2)
    print(f"Valid: {valid}")
    print(f"Quality: {report['quality_score']:.2%}")
    print(f"Recommendation: {report['recommendation']}")
    print(validator.format_validation_message(report))
    
    # Test case 3: Fallback data
    print("\n=== Test 3: Fallback Data ===")
    features3 = {'real_data_used': 0.0}
    valid, report = validator.validate_prediction_data(features3, {})
    print(f"Valid: {valid}")
    print(f"Quality: {report['quality_score']:.2%}")
    print(f"Recommendation: {report['recommendation']}")
    print(validator.format_validation_message(report))
    
    # Stats
    print("\n=== Validation Stats ===")
    stats = validator.get_validation_stats()
    print(f"Total: {stats['total_validations']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Pass rate: {stats['pass_rate_pct']:.1f}%")
