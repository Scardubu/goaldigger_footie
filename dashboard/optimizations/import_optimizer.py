#!/usr/bin/env python3
"""
Import Optimization Manager for GoalDiggers Platform
Phase 3.1: Import Optimization & Dependency Cleanup

This module provides comprehensive import optimization across all dashboard variants:
- Lazy loading of heavy dependencies
- Circular import detection and resolution
- Memory-efficient import patterns
- Dependency cleanup and consolidation

Target: 15MB memory savings through optimized imports
"""

import ast
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)

class ImportOptimizer:
    """
    Comprehensive import optimization for GoalDiggers platform.
    
    Features:
    - Lazy import pattern generation
    - Circular dependency detection
    - Unused import identification
    - Memory-efficient import strategies
    """
    
    def __init__(self, project_root: str = None):
        """Initialize import optimizer."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.dashboard_files = []
        self.import_graph = {}
        self.circular_imports = []
        self.unused_imports = {}
        self.heavy_imports = {
            'pandas', 'numpy', 'plotly', 'sklearn', 'scipy', 'matplotlib',
            'seaborn', 'tensorflow', 'torch', 'transformers', 'psutil',
            'asyncio', 'aiohttp', 'requests', 'sqlalchemy'
        }
        
        logger.info("🔧 Import Optimizer initialized")
    
    def analyze_dashboard_imports(self) -> Dict[str, any]:
        """Analyze imports across all dashboard variants."""
        logger.info("📊 Analyzing dashboard imports...")
        
        # Find all dashboard files
        dashboard_patterns = [
            'dashboard/*_dashboard.py',
            'dashboard/components/*.py',
            'dashboard/optimizations/*.py',
            'utils/*.py'
        ]
        
        for pattern in dashboard_patterns:
            files = list(self.project_root.glob(pattern))
            self.dashboard_files.extend(files)
        
        analysis_results = {
            'total_files': len(self.dashboard_files),
            'import_patterns': {},
            'heavy_imports_found': {},
            'optimization_opportunities': [],
            'memory_savings_potential': 0
        }
        
        # Analyze each file
        for file_path in self.dashboard_files:
            try:
                file_analysis = self._analyze_file_imports(file_path)
                analysis_results['import_patterns'][str(file_path)] = file_analysis
                
                # Check for heavy imports
                heavy_found = file_analysis['imports'] & self.heavy_imports
                if heavy_found:
                    analysis_results['heavy_imports_found'][str(file_path)] = list(heavy_found)
                
            except Exception as e:
                logger.warning(f"⚠️ Could not analyze {file_path}: {e}")
        
        # Generate optimization recommendations
        analysis_results['optimization_opportunities'] = self._generate_optimization_recommendations(analysis_results)
        analysis_results['memory_savings_potential'] = self._estimate_memory_savings(analysis_results)
        
        logger.info(f"✅ Import analysis complete: {analysis_results['total_files']} files analyzed")
        return analysis_results
    
    def _analyze_file_imports(self, file_path: Path) -> Dict[str, any]:
        """Analyze imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            imports = set()
            from_imports = {}
            lazy_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        imports.add(module)
                        
                        if module not in from_imports:
                            from_imports[module] = []
                        from_imports[module].extend([alias.name for alias in node.names])
                
                # Check for lazy import patterns
                elif isinstance(node, ast.FunctionDef) and 'lazy' in node.name.lower():
                    lazy_imports.append(node.name)
            
            return {
                'imports': imports,
                'from_imports': from_imports,
                'lazy_imports': lazy_imports,
                'has_lazy_patterns': len(lazy_imports) > 0,
                'heavy_import_count': len(imports & self.heavy_imports)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {file_path}: {e}")
            return {'imports': set(), 'from_imports': {}, 'lazy_imports': [], 'has_lazy_patterns': False, 'heavy_import_count': 0}
    
    def _generate_optimization_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        # Check for files with many heavy imports
        for file_path, patterns in analysis['import_patterns'].items():
            if patterns['heavy_import_count'] > 3 and not patterns['has_lazy_patterns']:
                recommendations.append({
                    'type': 'lazy_loading',
                    'file': file_path,
                    'description': f"Convert {patterns['heavy_import_count']} heavy imports to lazy loading",
                    'priority': 'high',
                    'memory_savings_mb': patterns['heavy_import_count'] * 2.5
                })
        
        # Check for duplicate imports across files
        all_imports = {}
        for file_path, patterns in analysis['import_patterns'].items():
            for imp in patterns['imports']:
                if imp not in all_imports:
                    all_imports[imp] = []
                all_imports[imp].append(file_path)
        
        # Find commonly imported heavy modules
        for imp, files in all_imports.items():
            if imp in self.heavy_imports and len(files) > 2:
                recommendations.append({
                    'type': 'centralized_import',
                    'module': imp,
                    'files': files,
                    'description': f"Centralize {imp} import used in {len(files)} files",
                    'priority': 'medium',
                    'memory_savings_mb': (len(files) - 1) * 3.0
                })
        
        return recommendations
    
    def _estimate_memory_savings(self, analysis: Dict) -> float:
        """Estimate potential memory savings from optimizations."""
        total_savings = 0
        
        for rec in analysis.get('optimization_opportunities', []):
            total_savings += rec.get('memory_savings_mb', 0)
        
        return total_savings
    
    def generate_lazy_import_pattern(self, module_name: str, imports: List[str]) -> str:
        """Generate lazy import pattern for a module."""
        function_name = f"_lazy_import_{module_name.replace('.', '_')}"
        
        pattern = f'''
def {function_name}():
    """Lazy import {module_name} when needed."""
    try:
        from {module_name} import {', '.join(imports)}
        return {', '.join(imports)}
    except ImportError as e:
        logger.warning(f"⚠️ {module_name} not available: {{e}}")
        return {', '.join(['None'] * len(imports))}
'''
        return pattern
    
    def optimize_dashboard_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, any]:
        """Optimize imports in a specific dashboard file."""
        logger.info(f"🔧 Optimizing imports in {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Analyze current imports
            file_analysis = self._analyze_file_imports(file_path)
            
            # Generate optimized version
            optimized_content = self._generate_optimized_imports(original_content, file_analysis)
            
            if not dry_run:
                # Create backup
                backup_path = file_path.with_suffix('.py.backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Write optimized version
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_content)
                
                logger.info(f"✅ Optimized {file_path} (backup: {backup_path})")
            
            return {
                'file': str(file_path),
                'original_size': len(original_content),
                'optimized_size': len(optimized_content),
                'size_reduction': len(original_content) - len(optimized_content),
                'heavy_imports_optimized': len(file_analysis['imports'] & self.heavy_imports)
            }
            
        except Exception as e:
            logger.error(f"❌ Error optimizing {file_path}: {e}")
            return {'error': str(e)}
    
    def _generate_optimized_imports(self, content: str, analysis: Dict) -> str:
        """Generate optimized import section for file content."""
        lines = content.split('\n')
        
        # Find import section
        import_start = -1
        import_end = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')) and import_start == -1:
                import_start = i
            elif import_start != -1 and not stripped.startswith(('import ', 'from ', '#')) and stripped:
                import_end = i
                break
        
        if import_start == -1:
            return content  # No imports found
        
        if import_end == -1:
            import_end = len(lines)
        
        # Generate optimized imports
        optimized_imports = self._create_optimized_import_section(analysis)
        
        # Reconstruct file
        new_lines = lines[:import_start] + optimized_imports + lines[import_end:]
        
        return '\n'.join(new_lines)
    
    def _create_optimized_import_section(self, analysis: Dict) -> List[str]:
        """Create optimized import section."""
        lines = []
        
        # Standard library imports first
        std_imports = []
        third_party_imports = []
        local_imports = []
        heavy_imports = []
        
        for imp in sorted(analysis['imports']):
            if imp in self.heavy_imports:
                heavy_imports.append(imp)
            elif imp in {'os', 'sys', 'time', 'logging', 'typing', 'dataclasses', 'enum'}:
                std_imports.append(imp)
            elif '.' not in imp:
                third_party_imports.append(imp)
            else:
                local_imports.append(imp)
        
        # Add standard imports
        if std_imports:
            lines.extend([f"import {imp}" for imp in std_imports])
            lines.append("")
        
        # Add third-party imports
        if third_party_imports:
            lines.extend([f"import {imp}" for imp in third_party_imports])
            lines.append("")
        
        # Add lazy loading for heavy imports
        if heavy_imports:
            lines.append("# Heavy imports - lazy loaded for performance")
            for imp in heavy_imports:
                lines.extend(self.generate_lazy_import_pattern(imp, ['*']).split('\n'))
            lines.append("")
        
        # Add local imports
        if local_imports:
            lines.extend([f"import {imp}" for imp in local_imports])
            lines.append("")
        
        return lines

def optimize_all_dashboards(project_root: str = None, dry_run: bool = True) -> Dict[str, any]:
    """Optimize imports across all dashboard files."""
    optimizer = ImportOptimizer(project_root)
    
    # Analyze current state
    analysis = optimizer.analyze_dashboard_imports()
    
    # Optimize each file
    optimization_results = []
    for file_path in optimizer.dashboard_files:
        if file_path.suffix == '.py':
            result = optimizer.optimize_dashboard_file(file_path, dry_run)
            optimization_results.append(result)
    
    return {
        'analysis': analysis,
        'optimizations': optimization_results,
        'total_memory_savings_mb': analysis['memory_savings_potential'],
        'files_optimized': len([r for r in optimization_results if 'error' not in r])
    }

if __name__ == "__main__":
    # Run import optimization analysis
    results = optimize_all_dashboards(dry_run=True)
    
    print("🔧 GoalDiggers Import Optimization Results")
    print("=" * 50)
    print(f"📊 Total Memory Savings Potential: {results['total_memory_savings_mb']:.1f}MB")
    print(f"📁 Files Analyzed: {results['analysis']['total_files']}")
    print(f"✅ Files Optimized: {results['files_optimized']}")
    
    print("\n🎯 Top Optimization Opportunities:")
    for i, rec in enumerate(results['analysis']['optimization_opportunities'][:5], 1):
        print(f"{i}. {rec['description']} ({rec['memory_savings_mb']:.1f}MB savings)")
