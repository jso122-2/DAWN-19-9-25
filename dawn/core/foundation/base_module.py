"""
Base Module Interface
====================

Foundational interface that all DAWN modules must implement.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModuleState(Enum):
    """Standard module states"""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"

@dataclass
class ModuleCapability:
    """Represents a module capability"""
    name: str
    description: str
    version: str = "1.0.0"

class BaseModule(ABC):
    """Base interface for all DAWN modules."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.state = ModuleState.UNINITIALIZED
        self.creation_time = datetime.now()
    
    @abstractmethod
    async def initialize(self) -> bool:
        pass
    
    def get_state(self) -> ModuleState:
        return self.state
