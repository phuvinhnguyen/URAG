"""
RAG Systems Registry

This module provides a centralized way to access different RAG system implementations.
Auto-discovers systems from the systems/ directory using importlib.
"""

import os
import importlib
import inspect
from typing import Dict, Any
from .abstract import AbstractRAGSystem

def _discover_systems() -> Dict[str, Any]:
    """
    Auto-discover RAG systems from the systems directory.
    
    Returns:
        Dictionary mapping system names to their classes
    """
    systems = {}
    current_dir = os.path.dirname(__file__)
    
    # Iterate through all Python files in the systems directory
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and filename not in ['__init__.py', 'abstract.py']:
            module_name = filename[:-3]  # Remove .py extension
            
            try:
                # Import the module
                module = importlib.import_module(f'systems.{module_name}')
                
                # Find classes that inherit from AbstractRAGSystem
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, AbstractRAGSystem) and 
                        obj != AbstractRAGSystem and 
                        obj.__module__ == module.__name__):
                        
                        # Use the module name as the system name
                        systems[module_name] = obj
                        break
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
            
    return systems

# Auto-discover available systems
AVAILABLE_SYSTEMS = _discover_systems()


def get_system(system_name: str, **kwargs) -> AbstractRAGSystem:
    """
    Get a system instance by name.
    
    Args:
        system_name: Name of the system to instantiate
        **kwargs: Arguments to pass to the system constructor
        
    Returns:
        Initialized system instance
        
    Raises:
        ValueError: If system_name is not found
    """
    system_name = system_name.lower()
    
    if system_name not in AVAILABLE_SYSTEMS:
        available = ", ".join(AVAILABLE_SYSTEMS.keys())
        raise ValueError(f"Unknown system '{system_name}'. Available systems: {available}")
    
    system_class = AVAILABLE_SYSTEMS[system_name]
    return system_class(**kwargs)


def list_systems() -> Dict[str, str]:
    """
    List all available systems with descriptions.
    
    Returns:
        Dictionary mapping system names to descriptions
    """
    descriptions = {}
    for name, system_class in AVAILABLE_SYSTEMS.items():
        doc = system_class.__doc__ or "No description available"
        # Extract first line of docstring
        first_line = doc.split('\n')[0].strip()
        descriptions[name] = first_line
    
    return descriptions


# Export main classes and functions
__all__ = [
    'AbstractRAGSystem',
    'SimpleLLMSystem',
    'SimpleRAGSystem',
    'RaptorLLMSystem',
    'RaptorRAGSystem',
    'GraphLLMSystem',
    'GraphRAGSystem',
    'get_system',
    'list_systems',
    'AVAILABLE_SYSTEMS'
]
