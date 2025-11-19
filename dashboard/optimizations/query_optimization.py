"""
Database Query Optimization Module

This module provides utilities for optimizing database queries in the GoalDiggers dashboard:

1. Query batching to reduce database round trips
2. Eager loading of related data to avoid N+1 query problems
3. Query result transformations with caching
4. Parameterized query templates with validation
"""

import logging
import time
import functools
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import sqlite3

from dashboard.error_log import ErrorLog
from dashboard.optimizations.caching import QueryCache

logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="query_optimization")

# Initialize query cache
query_cache = QueryCache(max_size=200, default_ttl=600)  # 10 minutes default TTL

def optimized_query(cache_ttl: int = 600, track_performance: bool = True):
    """
    Decorator for optimizing database query functions.
    
    Args:
        cache_ttl: Cache TTL in seconds
        track_performance: Whether to track query performance
        
    Returns:
        Decorated function with query optimization
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name, args, and kwargs
            cache_key = _create_query_cache_key(func.__name__, args, kwargs)
            
            # Check if result is in cache
            cached_result = query_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Query cache hit for {func.__name__}")
                return cached_result
            
            # Track performance if enabled
            start_time = time.time() if track_performance else None
            
            # Execute the query function
            try:
                result = func(*args, **kwargs)
                
                # Cache the result
                query_cache.set(cache_key, result, ttl=cache_ttl)
                
                # Log performance metrics if tracking is enabled
                if track_performance:
                    execution_time = time.time() - start_time
                    _log_query_performance(func.__name__, execution_time, args, kwargs)
                
                return result
            except Exception as e:
                error_log.log(
                    "query_error",
                    f"Error executing optimized query {func.__name__}: {str(e)}",
                    exception=e,
                    source="optimized_query"
                )
                raise
        
        return wrapper
    
    return decorator

def _create_query_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a cache key for a query function call."""
    # Convert args and kwargs to a string representation
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items()))
    
    # Create a hash from the function name and arguments
    key = f"{func_name}:{args_str}:{kwargs_str}"
    return hashlib.md5(key.encode()).hexdigest()

def _log_query_performance(func_name: str, execution_time: float, args: tuple, kwargs: dict):
    """Log query performance metrics."""
    # Log slow queries with warning
    if execution_time > 1.0:
        logger.warning(f"Slow query {func_name} took {execution_time:.2f}s")
    else:
        logger.debug(f"Query {func_name} took {execution_time:.2f}s")
    
    # Store performance metrics (could be sent to a monitoring system)
    # This is a placeholder - in a real system, this might write to a database or metrics service
    query_info = {
        "function": func_name,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat(),
        "args": str(args)[:100],  # Truncate for logging
        "kwargs": str(kwargs)[:100]  # Truncate for logging
    }
    
    # For now, just log it
    if execution_time > 0.1:  # Only log queries taking more than 100ms
        logger.info(f"Query performance: {json.dumps(query_info)}")

class BatchLoader:
    """
    Utility for loading related data in batches to avoid N+1 query problems.
    """
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.batch_size = 50  # Default batch size
    
    def load_related_items(self, 
                           item_ids: List[str], 
                           query_template: str, 
                           id_param_name: str = "ids"):
        """
        Load related items in batches.
        
        Args:
            item_ids: List of IDs to load related data for
            query_template: Query template with {id_list} placeholder
            id_param_name: Name of the ID parameter in the query
            
        Returns:
            Dictionary mapping IDs to related items
        """
        if not item_ids:
            return {}
        
        # Deduplicate IDs
        unique_ids = list(set(item_ids))
        
        # Process in batches
        all_results = {}
        for i in range(0, len(unique_ids), self.batch_size):
            batch_ids = unique_ids[i:i + self.batch_size]
            
            # Create placeholders for the SQL query
            placeholders = ", ".join("?" for _ in batch_ids)
            
            # Replace the placeholder in the query template
            query = query_template.format(id_list=placeholders)
            
            # Execute the query with the batch of IDs
            try:
                results = self.db_manager.fetchall(query, tuple(batch_ids), use_cache=True)
                
                # Process results into a dictionary
                for row in results:
                    item_id = str(row[0])  # Assuming first column is the ID
                    if item_id not in all_results:
                        all_results[item_id] = []
                    all_results[item_id].append(row[1:])  # Rest of the columns
            except Exception as e:
                error_log.log(
                    "batch_query_error",
                    f"Error executing batch query: {str(e)}",
                    exception=e,
                    source="BatchLoader.load_related_items"
                )
                logger.error(f"Error in batch query: {str(e)}")
        
        return all_results

class QueryOptimizer:
    """
    Utilities for optimizing SQLite queries in the GoalDiggers dashboard.
    """
    @staticmethod
    def analyze_query(query: str) -> Dict[str, Any]:
        """
        Analyze an SQLite query to identify potential optimizations.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "has_where_clause": "WHERE" in query.upper(),
            "has_join": "JOIN" in query.upper(),
            "has_order_by": "ORDER BY" in query.upper(),
            "has_group_by": "GROUP BY" in query.upper(),
            "has_limit": "LIMIT" in query.upper(),
            "table_count": query.upper().count("FROM") + query.upper().count("JOIN"),
            "potential_issues": []
        }
        
        # Check for potential issues
        if analysis["has_join"] and not analysis["has_where_clause"]:
            analysis["potential_issues"].append("JOIN without WHERE clause may return large result sets")
        
        if "SELECT *" in query.upper():
            analysis["potential_issues"].append("SELECT * may retrieve unnecessary columns")
        
        if analysis["has_order_by"] and not analysis["has_limit"]:
            analysis["potential_issues"].append("ORDER BY without LIMIT may be inefficient")
        
        if "LIKE '%" in query or "LIKE \"%" in query:
            analysis["potential_issues"].append("Leading wildcard in LIKE reduces index effectiveness")
        
        return analysis
    
    @staticmethod
    def suggest_optimizations(query: str) -> List[str]:
        """
        Suggest query optimizations based on analysis.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            List of optimization suggestions
        """
        analysis = QueryOptimizer.analyze_query(query)
        suggestions = []
        
        # Add suggestions based on analysis
        for issue in analysis["potential_issues"]:
            suggestions.append(issue)
        
        # Additional general suggestions
        if analysis["table_count"] > 2:
            suggestions.append("Consider denormalizing frequently joined tables for read-heavy operations")
        
        if not analysis["has_limit"]:
            suggestions.append("Add LIMIT to queries that don't need all results")
        
        # Add index suggestions
        if analysis["has_where_clause"]:
            # Extract tables and conditions from WHERE clause
            # This is a simplified approach - a real implementation would use SQL parsing
            where_part = query.upper().split("WHERE")[1].split("ORDER BY")[0].split("GROUP BY")[0].split("LIMIT")[0]
            columns = [
                col.strip() for col in where_part.replace("AND", ",").replace("OR", ",").split(",")
                if "=" in col or ">" in col or "<" in col
            ]
            
            for col in columns:
                # Extract column name (simplified approach)
                col_name = col.split("=")[0].strip() if "=" in col else col.split(">")[0].strip() if ">" in col else col.split("<")[0].strip()
                # Remove table prefix if present
                if "." in col_name:
                    col_name = col_name.split(".")[1]
                
                suggestions.append(f"Consider adding index on {col_name}")
        
        return suggestions
    
    @staticmethod
    def optimize_query(query: str) -> str:
        """
        Apply basic optimizations to an SQLite query.
        
        Args:
            query: SQL query to optimize
            
        Returns:
            Optimized query
        """
        # Replace SELECT * with explicit columns when possible
        if "SELECT *" in query.upper():
            # This is a placeholder - in a real system, we would extract table schema
            # and replace * with explicit column names
            logger.info("Query optimization: Consider replacing SELECT * with explicit columns")
        
        # Add LIMIT if not present
        if "LIMIT" not in query.upper():
            # Don't modify queries that need all results (e.g., aggregations)
            if "COUNT(" not in query.upper() and "SUM(" not in query.upper() and "GROUP BY" not in query.upper():
                query = f"{query} LIMIT 1000"
                logger.info("Query optimization: Added LIMIT 1000 to query")
        
        # Add query hints for SQLite
        if "ANALYZE" not in query.upper() and "EXPLAIN" not in query.upper():
            # SQLite specific optimizations
            if "ORDER BY" in query.upper() and "INDEXED BY" not in query.upper():
                # This is a placeholder - we would need to know the available indexes
                logger.info("Query optimization: Consider adding INDEXED BY for ORDER BY queries")
        
        return query
    
    @staticmethod
    def explain_query(db_manager, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Run EXPLAIN QUERY PLAN to get execution plan for an SQLite query.
        
        Args:
            db_manager: Database manager instance
            query: SQL query to explain
            params: Query parameters
            
        Returns:
            List of execution plan steps
        """
        explain_query = f"EXPLAIN QUERY PLAN {query}"
        
        try:
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(explain_query, params or ())
                
                # Format the explain output
                columns = [desc[0] for desc in cursor.description]
                plan = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                return plan
        except Exception as e:
            error_log.log(
                "explain_query_error",
                f"Error explaining query: {str(e)}",
                exception=e,
                source="QueryOptimizer.explain_query"
            )
            logger.error(f"Error explaining query: {str(e)}")
            return []

class LazyJoin:
    """
    Utility for efficient lazy joining of DataFrames with automatic caching.
    """
    def __init__(self, left_df: pd.DataFrame, cache_ttl: int = 300):
        self.left_df = left_df
        self.cache_ttl = cache_ttl
        self.joined_dfs = {}
        self._query_cache = QueryCache(max_size=50, default_ttl=cache_ttl)
    
    def join(self, 
             right_df: pd.DataFrame, 
             left_on: str, 
             right_on: str, 
             how: str = 'left', 
             suffixes: Tuple[str, str] = ('', '_right')) -> 'LazyJoin':
        """
        Join with another DataFrame and return self for chaining.
        
        Args:
            right_df: Right DataFrame to join with
            left_on: Column from left DataFrame to join on
            right_on: Column from right DataFrame to join on
            how: Join type (left, right, inner, outer)
            suffixes: Suffixes for overlapping columns
            
        Returns:
            Self for method chaining
        """
        # Create a cache key for this join
        key = f"join_{left_on}_{right_on}_{how}_{suffixes}"
        
        # Check if we've already done this join
        cached_result = self._query_cache.get(key)
        if cached_result is not None:
            self.left_df = cached_result
            return self
        
        # Perform the join
        try:
            # Convert object columns to string for safer joining
            if self.left_df[left_on].dtype == 'object':
                self.left_df[left_on] = self.left_df[left_on].astype(str)
            
            if right_df[right_on].dtype == 'object':
                right_df[right_on] = right_df[right_on].astype(str)
            
            # Perform the join
            self.left_df = self.left_df.merge(
                right_df, 
                left_on=left_on, 
                right_on=right_on, 
                how=how, 
                suffixes=suffixes
            )
            
            # Cache the result
            self._query_cache.set(key, self.left_df.copy())
            
        except Exception as e:
            error_log.log(
                "lazy_join_error",
                f"Error in LazyJoin: {str(e)}",
                exception=e,
                source="LazyJoin.join"
            )
            logger.error(f"Error in LazyJoin: {str(e)}")
        
        return self
    
    def result(self) -> pd.DataFrame:
        """Get the resulting DataFrame after all joins."""
        return self.left_df

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize a DataFrame's memory usage by downcasting numeric types and
    converting object columns to categorical when appropriate.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    if df.empty:
        return df
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    logger.debug(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
    
    # Copy the DataFrame to avoid modifying the original
    result = df.copy()
    
    # Process each column
    for col in result.columns:
        col_type = result[col].dtype
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_type):
            # Integers
            if pd.api.types.is_integer_dtype(col_type):
                # Get min and max values
                c_min = result[col].min()
                c_max = result[col].max()
                
                # Determine the smallest possible integer type
                if c_min >= 0:
                    if c_max < 256:
                        result[col] = result[col].astype(np.uint8)
                    elif c_max < 65536:
                        result[col] = result[col].astype(np.uint16)
                    elif c_max < 4294967296:
                        result[col] = result[col].astype(np.uint32)
                    else:
                        result[col] = result[col].astype(np.uint64)
                else:
                    if c_min > -128 and c_max < 128:
                        result[col] = result[col].astype(np.int8)
                    elif c_min > -32768 and c_max < 32768:
                        result[col] = result[col].astype(np.int16)
                    elif c_min > -2147483648 and c_max < 2147483648:
                        result[col] = result[col].astype(np.int32)
                    else:
                        result[col] = result[col].astype(np.int64)
                        
            # Floats
            elif pd.api.types.is_float_dtype(col_type):
                # Try to downcast to float32 if possible
                result[col] = pd.to_numeric(result[col], downcast='float')
        
        # Object columns (strings)
        elif col_type == 'object':
            # Check if it's a good candidate for categorical
            num_unique = result[col].nunique()
            num_total = len(result[col])
            
            # If column has low cardinality relative to its size, convert to categorical
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                result[col] = result[col].astype('category')
    
    # Check memory savings
    end_mem = result.memory_usage(deep=True).sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    logger.info(f"DataFrame optimized: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return result
