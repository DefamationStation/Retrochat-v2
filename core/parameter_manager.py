"""
Unified parameter management system for chat providers.

This module eliminates duplicate parameter handling code across providers by providing
a common interface for parameter validation, conversion, and persistence.
"""

from typing import Any, Dict, Set, Callable, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from utils.console import console


@dataclass
class ParameterDefinition:
    """Definition of a parameter with validation and conversion rules."""
    name: str
    param_type: type
    default_value: Any
    description: str = ""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[Set[Any]] = None
    converter: Optional[Callable[[Any], Any]] = None
    validator: Optional[Callable[[Any], bool]] = None
    aliases: Set[str] = field(default_factory=set)
    is_list: bool = False
    list_separator: str = ","


class ParameterConverter:
    """Handles parameter type conversions."""
    
    @staticmethod
    def to_int(value: Any) -> int:
        """Convert value to integer."""
        if isinstance(value, str):
            value = value.strip()
        return int(value)
    
    @staticmethod 
    def to_float(value: Any) -> float:
        """Convert value to float."""
        if isinstance(value, str):
            value = value.strip()
        return float(value)
    
    @staticmethod
    def to_bool(value: Any) -> bool:
        """Convert value to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
    
    @staticmethod
    def to_list(value: Any, separator: str = ",") -> list:
        """Convert value to list."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in value.split(separator) if item.strip()]
        return [value]
    
    @staticmethod
    def to_string(value: Any) -> str:
        """Convert value to string."""
        return str(value)


class ParameterValidator:
    """Handles parameter validation."""
    
    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]], 
                      max_val: Optional[Union[int, float]]) -> bool:
        """Validate value is within range."""
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    
    @staticmethod
    def validate_allowed_values(value: Any, allowed: Set[Any]) -> bool:
        """Validate value is in allowed set."""
        return value in allowed


class UnifiedParameterManager:
    """Unified parameter management for chat providers."""
    
    def __init__(self, history_manager):
        self.history_manager = history_manager
        self.parameters: Dict[str, Any] = {}
        self.parameter_definitions: Dict[str, ParameterDefinition] = {}
        self.converter = ParameterConverter()
        self.validator = ParameterValidator()
        
        # Define common parameters
        self._setup_common_parameters()
    
    def _setup_common_parameters(self):
        """Setup common parameters used across all providers."""
        common_params = [
            ParameterDefinition(
                name="temperature",
                param_type=float,
                default_value=0.8,
                description="Controls randomness in response generation",
                min_value=0.0,
                max_value=2.0
            ),
            ParameterDefinition(
                name="max_tokens",
                param_type=int,
                default_value=8192,
                description="Maximum number of tokens to generate",
                min_value=1,
                max_value=200000
            ),
            ParameterDefinition(
                name="top_p",
                param_type=float,
                default_value=1.0,
                description="Nucleus sampling parameter",
                min_value=0.0,
                max_value=1.0
            ),
            ParameterDefinition(
                name="frequency_penalty",
                param_type=float,
                default_value=1.1,
                description="Penalty for token frequency",
                min_value=-2.0,
                max_value=2.0,
                aliases={"repeat_penalty"}
            ),
            ParameterDefinition(
                name="presence_penalty", 
                param_type=float,
                default_value=0.0,
                description="Penalty for token presence",
                min_value=-2.0,
                max_value=2.0
            ),
            ParameterDefinition(
                name="stop",
                param_type=list,
                default_value=[],
                description="Stop sequences for generation",
                is_list=True
            ),
            ParameterDefinition(
                name="verbose",
                param_type=bool,
                default_value=False,
                description="Enable verbose output with token counts"
            ),
            ParameterDefinition(
                name="use_markdown",
                param_type=bool,
                default_value=True,
                description="Format responses with markdown"
            )
        ]
        
        for param in common_params:
            self.register_parameter(param)
    
    def register_parameter(self, definition: ParameterDefinition):
        """Register a parameter definition."""
        self.parameter_definitions[definition.name] = definition
        
        # Register aliases
        for alias in definition.aliases:
            self.parameter_definitions[alias] = definition
        
        # Set default value if not already set
        if definition.name not in self.parameters:
            self.parameters[definition.name] = definition.default_value
    
    def set_parameter(self, param: str, value: Any) -> bool:
        """
        Set a parameter with validation and conversion.
        
        Args:
            param: Parameter name (or alias)
            value: Parameter value
            
        Returns:
            bool: True if parameter was set successfully
        """
        # Find parameter definition (handle aliases)
        definition = self.parameter_definitions.get(param)
        if not definition:
            console.print(f"Invalid parameter: {param}", style="bold red")
            return False
        
        # Use the canonical name
        canonical_name = definition.name
        
        try:
            # Convert value to correct type
            converted_value = self._convert_value(value, definition)
            
            # Validate value
            if not self._validate_value(converted_value, definition):
                return False
            
            # Store old value for comparison
            old_value = self.parameters.get(canonical_name)
            
            # Set the parameter
            self.parameters[canonical_name] = converted_value
            
            # Handle special aliases (like repeat_penalty -> frequency_penalty)
            if param in definition.aliases:
                # For aliases, ensure the canonical parameter is updated
                self.parameters[canonical_name] = converted_value
            
            # Special handling for repeat_penalty/frequency_penalty sync
            if canonical_name == "frequency_penalty":
                # Always keep both in sync for backward compatibility
                self.parameters["repeat_penalty"] = converted_value
            elif param == "repeat_penalty":
                # If setting via alias, update the canonical parameter too
                self.parameters["frequency_penalty"] = converted_value
            
            # Save to persistent storage
            self.history_manager.save_parameters(self.parameters)
            
            # Print confirmation (skip for verbose or if value unchanged)
            if param != "verbose" or converted_value:
                if old_value != converted_value:
                    console.print(f"Parameter '{param}' set to {converted_value}", style="cyan")
                    
                    # Special messages for certain parameters
                    if canonical_name == "max_tokens":
                        console.print("(This will be sent as max_output_tokens to some APIs)", style="yellow")
            
            return True
            
        except (ValueError, TypeError) as e:
            console.print(f"Error setting parameter '{param}': {str(e)}", style="bold red")
            return False
    
    def _convert_value(self, value: Any, definition: ParameterDefinition) -> Any:
        """Convert value according to parameter definition."""
        if definition.converter:
            return definition.converter(value)
        
        if definition.is_list:
            return self.converter.to_list(value, definition.list_separator)
        elif definition.param_type == int:
            return self.converter.to_int(value)
        elif definition.param_type == float:
            return self.converter.to_float(value)
        elif definition.param_type == bool:
            return self.converter.to_bool(value)
        elif definition.param_type == str:
            return self.converter.to_string(value)
        else:
            return value
    
    def _validate_value(self, value: Any, definition: ParameterDefinition) -> bool:
        """Validate value according to parameter definition."""
        # Custom validator
        if definition.validator and not definition.validator(value):
            console.print(f"Value {value} failed custom validation for {definition.name}", style="bold red")
            return False
        
        # Range validation for numeric types
        if isinstance(value, (int, float)) and (definition.min_value is not None or definition.max_value is not None):
            if not self.validator.validate_range(value, definition.min_value, definition.max_value):
                console.print(
                    f"Value {value} out of range [{definition.min_value}, {definition.max_value}] for {definition.name}",
                    style="bold red"
                )
                return False
        
        # Allowed values validation
        if definition.allowed_values and not self.validator.validate_allowed_values(value, definition.allowed_values):
            console.print(
                f"Value {value} not in allowed values {definition.allowed_values} for {definition.name}",
                style="bold red"
            )
            return False
        
        return True
    
    def get_parameter(self, param: str, default: Any = None) -> Any:
        """Get parameter value."""
        definition = self.parameter_definitions.get(param)
        if definition:
            return self.parameters.get(definition.name, default)
        return self.parameters.get(param, default)
    
    def show_parameters(self):
        """Display current parameters."""
        console.print("Current Parameters:", style="cyan")
        
        # Group parameters by category
        shown_params = set()
        
        for param_name, definition in self.parameter_definitions.items():
            # Skip aliases to avoid duplication
            if definition.name != param_name:
                continue
                
            if definition.name in shown_params:
                continue
                
            current_value = self.parameters.get(definition.name, definition.default_value)
            
            # Special display for max_tokens
            if definition.name == "max_tokens":
                console.print(f"max_tokens: {current_value} (sent as max_output_tokens to some APIs)", style="green")
            # Special handling for frequency_penalty (show both names for clarity)
            elif definition.name == "frequency_penalty":
                console.print(f"frequency_penalty: {current_value}", style="green")
                console.print(f"repeat_penalty: {current_value} (alias for frequency_penalty)", style="green")
            else:
                console.print(f"{definition.name}: {current_value}", style="green")
            
            shown_params.add(definition.name)
        
        # Show any custom parameters not in definitions
        for param_name, value in self.parameters.items():
            if param_name not in shown_params and param_name != "repeat_penalty":  # Skip repeat_penalty since it's shown above
                console.print(f"{param_name}: {value}", style="green")
    
    def load_parameters(self) -> Dict[str, Any]:
        """Load parameters from persistent storage."""
        try:
            saved_params = self.history_manager.load_parameters()
            self.parameters.update(saved_params)
            return self.parameters
        except Exception as e:
            console.print(f"Error loading parameters: {e}", style="bold red")
            return {}
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all current parameters."""
        return self.parameters.copy()
    
    def reset_parameter(self, param: str) -> bool:
        """Reset parameter to default value."""
        definition = self.parameter_definitions.get(param)
        if not definition:
            console.print(f"Unknown parameter: {param}", style="bold red")
            return False
        
        return self.set_parameter(param, definition.default_value)
    
    def get_parameter_info(self, param: str) -> Optional[ParameterDefinition]:
        """Get information about a parameter."""
        return self.parameter_definitions.get(param)
