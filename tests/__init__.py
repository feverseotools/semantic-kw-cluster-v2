"""
Semantic Keyword Clustering - Test Suite Initialization

This module sets up the testing environment, configures logging,
and provides utility functions for testing across the project.
"""

import logging
import os
import sys
from typing import Dict, Any, Optional

# Ensure the project root is in the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a module-level logger
logger = logging.getLogger(__name__)

def setup_test_environment() -> Dict[str, Any]:
    """
    Set up the testing environment with necessary configurations.
    
    Returns:
        Dict containing test environment configuration
    """
    test_config = {
        'project_root': PROJECT_ROOT,
        'data_dir': os.path.join(PROJECT_ROOT, 'data', 'samples'),
        'test_data_dir': os.path.join(PROJECT_ROOT, 'tests', 'test_data'),
        'logging_level': logging.INFO
    }
    
    # Ensure test data directories exist
    os.makedirs(test_config['test_data_dir'], exist_ok=True)
    
    return test_config

def create_test_sample_data(sample_type: str) -> Optional[str]:
    """
    Create sample data files for testing.
    
    Args:
        sample_type: Type of sample data to create
    
    Returns:
        Path to the created sample data file, or None if creation fails
    """
    test_config = setup_test_environment()
    
    try:
        if sample_type == 'keywords':
            sample_path = os.path.join(test_config['test_data_dir'], 'sample_keywords.csv')
            
            # Create a sample keywords CSV
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write("keyword,search_volume,competition\n")
                sample_keywords = [
                    "digital marketing strategy,5400,0.75",
                    "seo best practices,3600,0.82",
                    "social media marketing,4200,0.68",
                    "content marketing tips,2900,0.61",
                    "email marketing automation,3100,0.79"
                ]
                f.write("\n".join(sample_keywords))
            
            logger.info(f"Created sample keywords file: {sample_path}")
            return sample_path
        
        elif sample_type == 'empty':
            sample_path = os.path.join(test_config['test_data_dir'], 'empty.csv')
            
            # Create an empty file
            open(sample_path, 'w').close()
            
            logger.info(f"Created empty sample file: {sample_path}")
            return sample_path
        
        else:
            logger.warning(f"Unsupported sample type: {sample_type}")
            return None
    
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return None

def cleanup_test_data() -> None:
    """
    Clean up test data files after testing.
    """
    test_config = setup_test_environment()
    
    try:
        import shutil
        
        # Remove test data directory
        if os.path.exists(test_config['test_data_dir']):
            shutil.rmtree(test_config['test_data_dir'])
            logger.info("Cleaned up test data directory")
        
    except Exception as e:
        logger.error(f"Error during test data cleanup: {e}")

def get_test_dependencies() -> Dict[str, str]:
    """
    List the test dependencies and their minimum required versions.
    
    Returns:
        Dictionary of test dependencies
    """
    return {
        'pytest': '>=7.0.0',
        'pytest-cov': '>=3.0.0',
        'hypothesis': '>=6.0.0',
        'mock': '>=4.0.0',
        'faker': '>=13.0.0'
    }

# Automatically set up the test environment when the module is imported
TEST_CONFIG = setup_test_environment()

# Optional: Add a custom exception for testing
class TestSetupError(Exception):
    """
    Custom exception for test setup failures.
    """
    pass

# Export key functions and classes
__all__ = [
    'setup_test_environment',
    'create_test_sample_data',
    'cleanup_test_data',
    'get_test_dependencies',
    'TEST_CONFIG',
    'TestSetupError'
]
