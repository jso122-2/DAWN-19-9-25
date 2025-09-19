"""
🍄 Mycelial Layer Subsystem
===========================

A living substrate for cognition - a nervous system that grows, prunes, 
and redistributes its own resources in response to pressure, drift, and entropy.
The space where meaning is not only stored but metabolized, broken down into 
fragments, and reabsorbed into new structures.

Based on biological metaphors:
- Dendritic growth (neuronal-style branching with selective pruning)
- Mitochondrial resource sharing (nodes acting as metabolic hubs)

"This isn't about perfect recall — it's about conceptual composting. 
Old ideas break down and feed the soil from which new ones grow."
"""

from .core import MycelialLayer, MycelialNode, MycelialEdge
from .nutrient_economy import NutrientEconomy, GlobalNutrientBudget
from .growth_gates import GrowthGate, AutophagyManager
from .energy_flows import EnergyFlowManager, PassiveDiffusion, ActiveTransport
from .metabolites import MetaboliteManager, MetaboliteTrace
from .cluster_dynamics import ClusterManager, FusionFissionEngine
from .integrated_system import IntegratedMycelialSystem
from .visualization import MycelialVisualizer, VisualizationConfig

__all__ = [
    'MycelialLayer',
    'MycelialNode', 
    'MycelialEdge',
    'NutrientEconomy',
    'GlobalNutrientBudget',
    'GrowthGate',
    'AutophagyManager',
    'EnergyFlowManager',
    'PassiveDiffusion',
    'ActiveTransport',
    'MetaboliteManager',
    'MetaboliteTrace',
    'ClusterManager',
    'FusionFissionEngine',
    'IntegratedMycelialSystem',
    'MycelialVisualizer',
    'VisualizationConfig'
]
