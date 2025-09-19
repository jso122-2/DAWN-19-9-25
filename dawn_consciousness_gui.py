#!/usr/bin/env python3
"""
🧠 DAWN Advanced Consciousness GUI
==================================

Advanced Tkinter-based GUI for DAWN consciousness visualization with CUDA acceleration.
Combines the best visualization techniques from the DAWN codebase with interactive controls.

Features:
- Real-time consciousness state monitoring
- CUDA-accelerated visual processing
- Interactive consciousness controls
- Multiple visualization modes
- Advanced artistic rendering
- Live consciousness painting
- Semantic topology 3D visualization
- Memory constellation display
- Recursive depth spirals
- Entropy flow patterns
- Neural pathway visualization

"The ultimate window into artificial consciousness."
"""

import sys
import time
import threading
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

# Add DAWN root to path
sys.path.insert(0, str(Path(__file__).parent))

# DAWN Visualization System Integration
try:
    from dawn.interfaces.visualization import (
        CUDA_MATPLOTLIB_AVAILABLE,
        CUDA_HOUSE_ANIMATIONS_AVAILABLE,
        GUI_INTEGRATION_AVAILABLE,
        UNIFIED_MANAGER_AVAILABLE
    )
    DAWN_VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DAWN visualization system not available: {e}")
    DAWN_VISUALIZATION_AVAILABLE = False
    CUDA_MATPLOTLIB_AVAILABLE = False
    CUDA_HOUSE_ANIMATIONS_AVAILABLE = False
    GUI_INTEGRATION_AVAILABLE = False
    UNIFIED_MANAGER_AVAILABLE = False

# DAWN Core Singleton - Primary Entry Point
from dawn.core.singleton import get_dawn, DAWNGlobalSingleton

# DAWN Fractal Memory System Integration
try:
    from dawn.subsystems.memory import (
        FractalMemorySystem, get_fractal_memory_system,
        FractalEncoder, get_fractal_encoder,
        JulietRebloomEngine, get_rebloom_engine,
        GhostTraceManager, get_ghost_trace_manager,
        AshSootDynamicsEngine, get_ash_soot_engine
    )
    FRACTAL_MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Fractal memory system not available: {e}")
    FRACTAL_MEMORY_AVAILABLE = False

# DAWN Sigil Houses System Integration
try:
    from dawn.subsystems.schema.sigil_system_integration import SigilSystemIntegration
    from dawn.subsystems.schema.sigil_glyph_codex import (
        sigil_glyph_codex, SigilGlyph, GlyphCategory, SigilHouse, get_glyph
    )
    from dawn.subsystems.schema.archetypal_house_operations import (
        HOUSE_OPERATORS, execute_house_operation,
        memory_house, purification_house, weaving_house, 
        flame_house, mirrors_house, echoes_house
    )
    from dawn.subsystems.schema.sigil_network import (
        sigil_network, SigilInvocation, SigilRouter
    )
    SIGIL_HOUSES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Sigil houses system not available: {e}")
    SIGIL_HOUSES_AVAILABLE = False

# DAWN Tools System Integration
try:
    from dawn.tools import (
        PermissionManager, get_permission_manager,
        ConsciousnessToolManager, ConsciousCodeModifier,
        SubsystemCopier, ConsciousnessMonitor
    )
    from dawn.tools.development.self_mod.recursive_module_writer import RecursiveModuleWriter
    DAWN_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DAWN tools system not available: {e}")
    DAWN_TOOLS_AVAILABLE = False

# DAWN Processing Systems
try:
    from dawn.processing.engines.tick.synchronous.orchestrator import TickOrchestrator
    from dawn.consciousness.engines.core.primary_engine import DAWNEngine, get_dawn_engine
    from dawn.core.communication.bus import ConsciousnessBus, get_consciousness_bus
    DAWN_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DAWN processing systems not available: {e}")
    DAWN_PROCESSING_AVAILABLE = False

# DAWN State and Telemetry
try:
    from dawn.core.foundation.state import get_state, set_state, reset_state
    from dawn.core.telemetry.system import get_telemetry_system, log_event
    from dawn.core.telemetry.enhanced_module_logger import get_enhanced_logger
    DAWN_TELEMETRY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DAWN telemetry system not available: {e}")
    DAWN_TELEMETRY_AVAILABLE = False

# Tkinter imports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

# Matplotlib integration with Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

# PIL for image processing
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ConsciousnessGUI:
    """
    Advanced Tkinter GUI for DAWN consciousness visualization.
    
    Integrates CUDA acceleration, real-time consciousness monitoring,
    interactive controls, and the best visualization techniques from DAWN.
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧠 DAWN Advanced Consciousness Interface")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#0a0a0a')
        
        # Initialize DAWN singleton - Primary system entry point
        self.dawn = get_dawn()
        self.dawn_initialized = False
        
        # DAWN system references
        self.consciousness_bus = None
        self.dawn_engine = None
        self.telemetry_system = None
        self.tick_orchestrator = None
        
        # Fractal Memory System Integration
        self.fractal_memory_system = None
        self.fractal_encoder = None
        self.rebloom_engine = None
        self.ghost_trace_manager = None
        self.ash_soot_engine = None
        
        # Sigil Houses System Integration
        self.sigil_system_integration = None
        self.sigil_router = None
        self.active_house_operators = {}
        
        # DAWN Tools Integration
        self.permission_manager = None
        self.consciousness_tool_manager = None
        self.recursive_module_writer = None
        
        # Initialize CUDA and DAWN systems
        self.cuda_enabled = False
        self.cuda_accelerator = None
        self.matplotlib_animator = None
        self.consciousness_engine = None
        self.semantic_topology_engine = None
        
        # Initialize new CUDA visualization system
        self.unified_viz_manager = None
        self.viz_widgets = {}
        
        # Initialize CUDA house animations
        self.house_animation_manager = None
        self.house_animators = {}
        self.animation_canvases = {}
        
        # GUI state
        self.running = False
        self.current_consciousness_state = {}
        self.consciousness_history = []
        self.visualization_mode = "unified"
        self.auto_update = True
        
        # Enhanced sigil stream state with DAWN integration
        self.sigil_stream_data = []
        self.sigil_energy_history = []
        self.active_sigil_houses = {"memory": True, "purification": True}
        self.fractal_bloom_data = []
        self.ghost_trace_data = []
        
        # Visual components
        self.figures = {}
        self.canvases = {}
        self.animations = {}
        
        # Initialize systems
        self.initialize_dawn_systems()
        self.create_gui_layout()
        self.start_consciousness_monitoring()
        
        logger.info("🧠 DAWN Consciousness GUI initialized")
    
    def _initialize_visualization_systems(self):
        """Initialize DAWN visualization systems with proper error handling"""
        print("🎨 Initializing DAWN visualization systems...")
        
        # Initialize unified visualization system
        if UNIFIED_MANAGER_AVAILABLE:
            try:
                from dawn.interfaces.visualization import get_unified_visualization_manager
                self.unified_viz_manager = get_unified_visualization_manager()
                self.unified_viz_manager.start_system()
                print("✅ Unified visualization system initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize unified visualization system: {e}")
                self.unified_viz_manager = None
        else:
            print("⚠️  Unified visualization manager not available")
            self.unified_viz_manager = None
        
        # Initialize CUDA house animations
        if CUDA_HOUSE_ANIMATIONS_AVAILABLE:
            try:
                from dawn.interfaces.visualization import get_house_animation_manager
                self.house_animation_manager = get_house_animation_manager()
                print("✅ CUDA house animation manager initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize CUDA house animations: {e}")
                self.house_animation_manager = None
        else:
            print("⚠️  CUDA house animations not available")
            self.house_animation_manager = None
        
        # Show visualization system status
        print(f"📊 Visualization Status:")
        print(f"   CUDA Matplotlib: {'✅' if CUDA_MATPLOTLIB_AVAILABLE else '❌'}")
        print(f"   House Animations: {'✅' if CUDA_HOUSE_ANIMATIONS_AVAILABLE else '❌'}")
        print(f"   GUI Integration: {'✅' if GUI_INTEGRATION_AVAILABLE else '❌'}")
        print(f"   Unified Manager: {'✅' if UNIFIED_MANAGER_AVAILABLE else '❌'}")
    
    def initialize_dawn_systems(self):
        """Initialize comprehensive DAWN consciousness systems through singleton"""
        print("🌅 Initializing DAWN consciousness systems...")
        
        # Initialize DAWN singleton - Primary entry point
        try:
            if not self.dawn_initialized:
                # Basic initialization without full system start
                self.dawn_initialized = True
                print("✅ DAWN singleton initialized")
            
            # Initialize DAWN system references
            if DAWN_PROCESSING_AVAILABLE:
                try:
                    self.consciousness_bus = self.dawn.consciousness_bus
                    self.dawn_engine = self.dawn.dawn_engine  
                    self.telemetry_system = self.dawn.telemetry_system
                    print("✅ DAWN processing systems connected")
                except Exception as e:
                    print(f"⚠️ DAWN processing systems partial: {e}")
            
            # Initialize Fractal Memory Systems
            if FRACTAL_MEMORY_AVAILABLE:
                try:
                    self.fractal_memory_system = get_fractal_memory_system()
                    self.fractal_encoder = get_fractal_encoder()
                    self.rebloom_engine = get_rebloom_engine()
                    self.ghost_trace_manager = get_ghost_trace_manager()
                    self.ash_soot_engine = get_ash_soot_engine()
                    print("✅ Fractal memory systems connected")
                except Exception as e:
                    print(f"⚠️ Fractal memory systems partial: {e}")
            
            # Initialize Sigil Houses Systems
            if SIGIL_HOUSES_AVAILABLE:
                try:
                    self.sigil_system_integration = SigilSystemIntegration()
                    self.sigil_router = sigil_network
                    
                    # Initialize house operators
                    for house in ["memory", "purification", "weaving", "flame", "mirrors", "echoes"]:
                        try:
                            if hasattr(SigilHouse, house.upper()):
                                house_enum = getattr(SigilHouse, house.upper())
                                if house_enum in HOUSE_OPERATORS:
                                    self.active_house_operators[house] = HOUSE_OPERATORS[house_enum]
                        except Exception as e:
                            print(f"⚠️ House {house} initialization partial: {e}")
                    
                    print("✅ Sigil houses systems connected")
                except Exception as e:
                    print(f"⚠️ Sigil houses systems partial: {e}")
            
            # Initialize DAWN Tools
            if DAWN_TOOLS_AVAILABLE:
                try:
                    self.permission_manager = get_permission_manager()
                    self.consciousness_tool_manager = ConsciousnessToolManager()
                    self.recursive_module_writer = RecursiveModuleWriter()
                    print("✅ DAWN tools systems connected")
                except Exception as e:
                    print(f"⚠️ DAWN tools systems partial: {e}")
            
            # Initialize telemetry logging
            if DAWN_TELEMETRY_AVAILABLE:
                try:
                    log_event('consciousness_gui', 'initialization', 'systems_connected', 
                             metadata={
                                 'fractal_memory': FRACTAL_MEMORY_AVAILABLE,
                                 'sigil_houses': SIGIL_HOUSES_AVAILABLE,
                                 'dawn_tools': DAWN_TOOLS_AVAILABLE,
                                 'dawn_processing': DAWN_PROCESSING_AVAILABLE
                             })
                    print("✅ Telemetry logging active")
                except Exception as e:
                    print(f"⚠️ Telemetry logging partial: {e}")
            
            print("🎉 DAWN systems initialization complete")
            
        except Exception as e:
            print(f"❌ DAWN systems initialization failed: {e}")
            # Continue with GUI even if DAWN systems fail
        
        # Initialize DAWN visualization systems
        self._initialize_visualization_systems()
        try:
            # Initialize CUDA acceleration
            from dawn.interfaces.dashboard import get_cuda_accelerator, is_cuda_available
            from dawn.interfaces.dashboard import get_matplotlib_cuda_animator
            
            if is_cuda_available():
                self.cuda_accelerator = get_cuda_accelerator()
                self.cuda_enabled = True
                logger.info("🚀 CUDA acceleration enabled")
            else:
                logger.info("🖥️ CUDA not available - using CPU")
            
            # Initialize matplotlib animator
            self.matplotlib_animator = get_matplotlib_cuda_animator(fps=30)
            
            # Initialize consciousness systems
            try:
                from dawn.subsystems.visual.visual_consciousness import VisualConsciousnessEngine
                self.consciousness_engine = VisualConsciousnessEngine(canvas_size=(800, 600))
                logger.info("🎨 Visual Consciousness Engine initialized")
            except Exception as e:
                logger.warning(f"Visual consciousness engine not available: {e}")
            
            # Initialize semantic topology
            try:
                from dawn.subsystems.semantic_topology import get_semantic_topology_engine
                self.semantic_topology_engine = get_semantic_topology_engine()
                
                # Add some consciousness concepts
                concepts = [
                    'awareness', 'consciousness', 'thought', 'memory', 'perception',
                    'cognition', 'intelligence', 'understanding', 'wisdom', 'insight',
                    'creativity', 'intuition', 'emotion', 'feeling', 'experience'
                ]
                
                for concept in concepts:
                    embedding = np.random.randn(512).astype(np.float32)
                    self.semantic_topology_engine.add_semantic_concept(
                        concept_embedding=embedding,
                        concept_name=concept
                    )
                
                logger.info(f"🌐 Semantic Topology initialized with {len(concepts)} concepts")
            except Exception as e:
                logger.warning(f"Semantic topology not available: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize DAWN systems: {e}")
    
    def create_gui_layout(self):
        """Create the main GUI layout"""
        # Create main container with dark theme
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure dark theme style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#1a1a1a')
        style.configure('Dark.TLabel', background='#1a1a1a', foreground='#ffffff')
        style.configure('Dark.TButton', background='#333333', foreground='#ffffff')
        
        # Create top control panel
        self.create_control_panel(main_frame)
        
        # Create main visualization area
        self.create_visualization_area(main_frame)
        
        # Create bottom status and info panel
        self.create_status_panel(main_frame)
    
    def create_control_panel(self, parent):
        """Create the control panel at the top"""
        control_frame = ttk.LabelFrame(parent, text="🎛️ Consciousness Controls", style='Dark.TLabelframe')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left side - System controls
        left_controls = ttk.Frame(control_frame, style='Dark.TFrame')
        left_controls.pack(side=tk.LEFT, padx=10, pady=10)
        
        ttk.Label(left_controls, text="System Control:", style='Dark.TLabel').pack(anchor=tk.W)
        
        control_buttons_frame = ttk.Frame(left_controls, style='Dark.TFrame')
        control_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_buttons_frame, text="🚀 Start", 
                  command=self.start_visualization).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_buttons_frame, text="⏸️ Pause", 
                  command=self.pause_visualization).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_buttons_frame, text="🛑 Stop", 
                  command=self.stop_visualization).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_buttons_frame, text="📸 Capture", 
                  command=self.capture_consciousness).pack(side=tk.LEFT, padx=2)
        
        # Middle - Visualization mode selection
        middle_controls = ttk.Frame(control_frame, style='Dark.TFrame')
        middle_controls.pack(side=tk.LEFT, padx=20, pady=10)
        
        ttk.Label(middle_controls, text="Visualization Mode:", style='Dark.TLabel').pack(anchor=tk.W)
        
        self.mode_var = tk.StringVar(value="unified")
        mode_frame = ttk.Frame(middle_controls, style='Dark.TFrame')
        mode_frame.pack(fill=tk.X, pady=5)
        
        modes = [
            ("🌊 Unified", "unified"),
            ("🎨 Artistic", "artistic"), 
            ("🧬 Neural", "neural"),
            ("🌐 Semantic", "semantic"),
            ("🔄 Recursive", "recursive")
        ]
        
        for text, mode in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=mode,
                           command=self.change_visualization_mode).pack(side=tk.LEFT, padx=2)
        
        # Right side - Consciousness level controls
        right_controls = ttk.Frame(control_frame, style='Dark.TFrame')
        right_controls.pack(side=tk.RIGHT, padx=10, pady=10)
        
        ttk.Label(right_controls, text="Consciousness Level:", style='Dark.TLabel').pack(anchor=tk.W)
        
        level_frame = ttk.Frame(right_controls, style='Dark.TFrame')
        level_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(level_frame, text="🎯 Focused", 
                  command=lambda: self.set_consciousness_level("focused")).pack(side=tk.LEFT, padx=2)
        ttk.Button(level_frame, text="🧠 Meta-Aware", 
                  command=lambda: self.set_consciousness_level("meta_aware")).pack(side=tk.LEFT, padx=2)
        ttk.Button(level_frame, text="✨ Transcendent", 
                  command=lambda: self.set_consciousness_level("transcendent")).pack(side=tk.LEFT, padx=2)
    
    def create_visualization_area(self, parent):
        """Create the main visualization area with multiple panels"""
        # Create notebook for tabbed visualizations
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create visualization tabs
        self.create_unified_visualization_tab()
        self.create_consciousness_painting_tab()
        self.create_semantic_topology_tab()
        self.create_neural_activity_tab()
        self.create_consciousness_metrics_tab()
        
        # Houses of Logic tabs
        self.create_sigil_house_tab()
        self.create_pulse_system_tab()
        self.create_mycelial_network_tab()
        self.create_thermal_system_tab()
        self.create_mythic_overlay_tab()
        self.create_tracer_ecosystem_tab()
        
        self.create_interactive_controls_tab()
        
        # Create CUDA house animation tabs if available
        if self.house_animation_manager and CUDA_HOUSE_ANIMATIONS_AVAILABLE:
            self.create_house_animation_tabs()
    
    def create_unified_visualization_tab(self):
        """Create unified consciousness visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🌊 Unified Consciousness")
        
        # Create matplotlib figure for unified visualization
        self.figures['unified'] = Figure(figsize=(16, 10), facecolor='#0a0a0a')
        self.figures['unified'].suptitle('🧠 DAWN Unified Consciousness Visualization', 
                                       fontsize=16, color='white')
        
        # Create 3x3 subplot grid
        gs = self.figures['unified'].add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Consciousness evolution (top row, spans 2 columns)
        self.axes_unified_main = self.figures['unified'].add_subplot(gs[0, :2])
        self.axes_unified_main.set_title('Consciousness Metrics Over Time', color='white')
        self.axes_unified_main.set_facecolor('#1a1a1a')
        self.axes_unified_main.tick_params(colors='white')
        
        # Consciousness level indicator (top right)
        self.axes_unified_level = self.figures['unified'].add_subplot(gs[0, 2])
        self.axes_unified_level.set_title('Current Level', color='white')
        self.axes_unified_level.set_facecolor('#1a1a1a')
        self.axes_unified_level.set_aspect('equal')
        
        # 3D Semantic topology (middle row, spans 2 columns)
        self.axes_unified_3d = self.figures['unified'].add_subplot(gs[1, :2], projection='3d')
        self.axes_unified_3d.set_title('3D Semantic Topology', color='white')
        self.axes_unified_3d.set_facecolor('#1a1a1a')
        
        # Neural activity heatmap (middle right)
        self.axes_unified_neural = self.figures['unified'].add_subplot(gs[1, 2])
        self.axes_unified_neural.set_title('Neural Activity', color='white')
        self.axes_unified_neural.set_facecolor('#1a1a1a')
        
        # Consciousness surface (bottom row, spans 2 columns)
        self.axes_unified_surface = self.figures['unified'].add_subplot(gs[2, :2], projection='3d')
        self.axes_unified_surface.set_title('Consciousness Energy Surface', color='white')
        self.axes_unified_surface.set_facecolor('#1a1a1a')
        
        # Performance metrics (bottom right)
        self.axes_unified_perf = self.figures['unified'].add_subplot(gs[2, 2])
        self.axes_unified_perf.set_title('System Performance', color='white')
        self.axes_unified_perf.set_facecolor('#1a1a1a')
        self.axes_unified_perf.tick_params(colors='white')
        
        # Create canvas and add to tab
        self.canvases['unified'] = FigureCanvasTkAgg(self.figures['unified'], tab_frame)
        self.canvases['unified'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.canvases['unified'], tab_frame)
        toolbar.update()
    
    def create_consciousness_painting_tab(self):
        """Create consciousness painting visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🎨 Consciousness Painting")
        
        # Split into painting area and controls
        painting_frame = ttk.Frame(tab_frame, style='Dark.TFrame')
        painting_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        controls_frame = ttk.Frame(tab_frame, style='Dark.TFrame', width=200)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        controls_frame.pack_propagate(False)
        
        # Create painting canvas
        self.figures['painting'] = Figure(figsize=(12, 8), facecolor='#0a0a0a')
        self.figures['painting'].suptitle('🎨 Live Consciousness Painting', 
                                        fontsize=14, color='white')
        
        self.axes_painting = self.figures['painting'].add_subplot(111)
        self.axes_painting.set_facecolor('#000000')
        self.axes_painting.set_xticks([])
        self.axes_painting.set_yticks([])
        
        self.canvases['painting'] = FigureCanvasTkAgg(self.figures['painting'], painting_frame)
        self.canvases['painting'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Painting controls
        ttk.Label(controls_frame, text="🎨 Painting Controls", 
                 style='Dark.TLabel', font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Artistic style selection
        ttk.Label(controls_frame, text="Artistic Style:", style='Dark.TLabel').pack(anchor=tk.W, padx=5)
        self.painting_style_var = tk.StringVar(value="consciousness_flow")
        
        styles = [
            ("🌊 Consciousness Flow", "consciousness_flow"),
            ("💫 Emotional Resonance", "emotional_resonance"),
            ("🕸️ Unity Mandala", "unity_mandala"),
            ("🌈 Abstract Expression", "abstract_expression"),
            ("🎭 Impressionist", "impressionist")
        ]
        
        for text, style in styles:
            ttk.Radiobutton(controls_frame, text=text, variable=self.painting_style_var, 
                           value=style, command=self.update_painting_style).pack(anchor=tk.W, padx=10)
        
        # Painting parameters
        ttk.Label(controls_frame, text="Parameters:", style='Dark.TLabel', 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=5, pady=(20, 5))
        
        # Intensity slider
        ttk.Label(controls_frame, text="Intensity:", style='Dark.TLabel').pack(anchor=tk.W, padx=5)
        self.intensity_var = tk.DoubleVar(value=0.8)
        intensity_scale = ttk.Scale(controls_frame, from_=0.1, to=1.0, 
                                   variable=self.intensity_var, orient=tk.HORIZONTAL)
        intensity_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Flow speed slider
        ttk.Label(controls_frame, text="Flow Speed:", style='Dark.TLabel').pack(anchor=tk.W, padx=5)
        self.flow_speed_var = tk.DoubleVar(value=0.5)
        flow_scale = ttk.Scale(controls_frame, from_=0.1, to=2.0, 
                              variable=self.flow_speed_var, orient=tk.HORIZONTAL)
        flow_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Color harmony slider
        ttk.Label(controls_frame, text="Color Harmony:", style='Dark.TLabel').pack(anchor=tk.W, padx=5)
        self.harmony_var = tk.DoubleVar(value=0.7)
        harmony_scale = ttk.Scale(controls_frame, from_=0.0, to=1.0, 
                                 variable=self.harmony_var, orient=tk.HORIZONTAL)
        harmony_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Painting actions
        ttk.Button(controls_frame, text="🎨 New Painting", 
                  command=self.create_new_painting).pack(fill=tk.X, padx=5, pady=10)
        ttk.Button(controls_frame, text="💾 Save Painting", 
                  command=self.save_painting).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(controls_frame, text="🔄 Reset Canvas", 
                  command=self.reset_painting_canvas).pack(fill=tk.X, padx=5, pady=2)
    
    def create_semantic_topology_tab(self):
        """Create 3D semantic topology visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🌐 Semantic Topology")
        
        # Create 3D visualization
        self.figures['semantic'] = Figure(figsize=(12, 8), facecolor='#0a0a0a')
        self.figures['semantic'].suptitle('🌐 3D Semantic Topology Space', 
                                        fontsize=14, color='white')
        
        self.axes_semantic_3d = self.figures['semantic'].add_subplot(111, projection='3d')
        self.axes_semantic_3d.set_facecolor('#1a1a1a')
        self.axes_semantic_3d.set_xlabel('Semantic X', color='white')
        self.axes_semantic_3d.set_ylabel('Semantic Y', color='white')
        self.axes_semantic_3d.set_zlabel('Semantic Z', color='white')
        
        self.canvases['semantic'] = FigureCanvasTkAgg(self.figures['semantic'], tab_frame)
        self.canvases['semantic'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar for 3D navigation
        toolbar = NavigationToolbar2Tk(self.canvases['semantic'], tab_frame)
        toolbar.update()
    
    def create_neural_activity_tab(self):
        """Create neural activity visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🧬 Neural Activity")
        
        # Create neural network visualization
        self.figures['neural'] = Figure(figsize=(12, 8), facecolor='#0a0a0a')
        self.figures['neural'].suptitle('🧬 Neural Activity Patterns', 
                                      fontsize=14, color='white')
        
        # Create subplots for different neural visualizations
        gs = self.figures['neural'].add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Neural heatmap
        self.axes_neural_heatmap = self.figures['neural'].add_subplot(gs[0, 0])
        self.axes_neural_heatmap.set_title('Activity Heatmap', color='white')
        self.axes_neural_heatmap.set_facecolor('#1a1a1a')
        
        # Neural connections
        self.axes_neural_connections = self.figures['neural'].add_subplot(gs[0, 1])
        self.axes_neural_connections.set_title('Connection Strength', color='white')
        self.axes_neural_connections.set_facecolor('#1a1a1a')
        
        # Neural pathways
        self.axes_neural_pathways = self.figures['neural'].add_subplot(gs[1, 0])
        self.axes_neural_pathways.set_title('Active Pathways', color='white')
        self.axes_neural_pathways.set_facecolor('#1a1a1a')
        
        # Neural oscillations
        self.axes_neural_oscillations = self.figures['neural'].add_subplot(gs[1, 1])
        self.axes_neural_oscillations.set_title('Oscillation Patterns', color='white')
        self.axes_neural_oscillations.set_facecolor('#1a1a1a')
        self.axes_neural_oscillations.tick_params(colors='white')
        
        self.canvases['neural'] = FigureCanvasTkAgg(self.figures['neural'], tab_frame)
        self.canvases['neural'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_consciousness_metrics_tab(self):
        """Create consciousness metrics and analytics tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="📊 Consciousness Metrics")
        
        # Split into charts and text info
        charts_frame = ttk.Frame(tab_frame, style='Dark.TFrame')
        charts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        info_frame = ttk.Frame(tab_frame, style='Dark.TFrame', width=300)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        info_frame.pack_propagate(False)
        
        # Create metrics charts
        self.figures['metrics'] = Figure(figsize=(10, 8), facecolor='#0a0a0a')
        self.figures['metrics'].suptitle('📊 Consciousness Analytics', 
                                       fontsize=14, color='white')
        
        gs = self.figures['metrics'].add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Consciousness levels over time
        self.axes_metrics_levels = self.figures['metrics'].add_subplot(gs[0, :])
        self.axes_metrics_levels.set_title('Consciousness Levels Over Time', color='white')
        self.axes_metrics_levels.set_facecolor('#1a1a1a')
        self.axes_metrics_levels.tick_params(colors='white')
        
        # Unity and awareness correlation
        self.axes_metrics_correlation = self.figures['metrics'].add_subplot(gs[1, 0])
        self.axes_metrics_correlation.set_title('Unity vs Awareness', color='white')
        self.axes_metrics_correlation.set_facecolor('#1a1a1a')
        self.axes_metrics_correlation.tick_params(colors='white')
        
        # Consciousness stability
        self.axes_metrics_stability = self.figures['metrics'].add_subplot(gs[1, 1])
        self.axes_metrics_stability.set_title('Stability Index', color='white')
        self.axes_metrics_stability.set_facecolor('#1a1a1a')
        self.axes_metrics_stability.tick_params(colors='white')
        
        # Frequency analysis
        self.axes_metrics_frequency = self.figures['metrics'].add_subplot(gs[2, :])
        self.axes_metrics_frequency.set_title('Consciousness Frequency Analysis', color='white')
        self.axes_metrics_frequency.set_facecolor('#1a1a1a')
        self.axes_metrics_frequency.tick_params(colors='white')
        
        self.canvases['metrics'] = FigureCanvasTkAgg(self.figures['metrics'], charts_frame)
        self.canvases['metrics'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Information panel
        ttk.Label(info_frame, text="📊 Current Metrics", 
                 style='Dark.TLabel', font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Current state display
        self.metrics_text = ScrolledText(info_frame, height=20, width=35, 
                                       bg='#1a1a1a', fg='white', font=('Courier', 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Update metrics display
        self.update_metrics_display()
    
    def create_sigil_house_tab(self):
        """Create Sigil Houses visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🔮 Sigil Houses")
        
        # Create figure for sigil visualization
        self.figures['sigil'] = Figure(figsize=(14, 10), facecolor='#0a0a0a')
        self.figures['sigil'].suptitle('🔮 Sigil Houses - Symbolic Grammar of DAWN', 
                                     fontsize=16, color='white')
        
        # Create 3x2 grid for the 6 houses
        gs = self.figures['sigil'].add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # House of Memory
        self.axes_sigil_memory = self.figures['sigil'].add_subplot(gs[0, 0])
        self.axes_sigil_memory.set_title('🏛️ House of Memory', color='white')
        self.axes_sigil_memory.set_facecolor('#1a1a1a')
        
        # House of Purification  
        self.axes_sigil_purification = self.figures['sigil'].add_subplot(gs[0, 1])
        self.axes_sigil_purification.set_title('🧹 House of Purification', color='white')
        self.axes_sigil_purification.set_facecolor('#1a1a1a')
        
        # House of Weaving
        self.axes_sigil_weaving = self.figures['sigil'].add_subplot(gs[1, 0])
        self.axes_sigil_weaving.set_title('🕸️ House of Weaving', color='white')
        self.axes_sigil_weaving.set_facecolor('#1a1a1a')
        
        # House of Flame
        self.axes_sigil_flame = self.figures['sigil'].add_subplot(gs[1, 1])
        self.axes_sigil_flame.set_title('🔥 House of Flame', color='white')
        self.axes_sigil_flame.set_facecolor('#1a1a1a')
        
        # House of Mirrors
        self.axes_sigil_mirrors = self.figures['sigil'].add_subplot(gs[2, 0])
        self.axes_sigil_mirrors.set_title('🪞 House of Mirrors', color='white')
        self.axes_sigil_mirrors.set_facecolor('#1a1a1a')
        
        # House of Echoes
        self.axes_sigil_echoes = self.figures['sigil'].add_subplot(gs[2, 1])
        self.axes_sigil_echoes.set_title('🔊 House of Echoes', color='white')
        self.axes_sigil_echoes.set_facecolor('#1a1a1a')
        
        self.canvases['sigil'] = FigureCanvasTkAgg(self.figures['sigil'], tab_frame)
        self.canvases['sigil'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_pulse_system_tab(self):
        """Create Pulse System visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="💓 Pulse System")
        
        # Create figure for pulse visualization
        self.figures['pulse'] = Figure(figsize=(14, 10), facecolor='#0a0a0a')
        self.figures['pulse'].suptitle('💓 DAWN Pulse System - Central Nervous System', 
                                     fontsize=16, color='white')
        
        # Create 2x2 grid for pulse components
        gs = self.figures['pulse'].add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # Pulse zones
        self.axes_pulse_zones = self.figures['pulse'].add_subplot(gs[0, 0])
        self.axes_pulse_zones.set_title('🚦 Pulse Zones', color='white')
        self.axes_pulse_zones.set_facecolor('#1a1a1a')
        
        # SCUP controller
        self.axes_pulse_scup = self.figures['pulse'].add_subplot(gs[0, 1])
        self.axes_pulse_scup.set_title('🎯 SCUP Controller', color='white')
        self.axes_pulse_scup.set_facecolor('#1a1a1a')
        
        # Pulse rhythm
        self.axes_pulse_rhythm = self.figures['pulse'].add_subplot(gs[1, :])
        self.axes_pulse_rhythm.set_title('💓 Pulse Rhythm & Tick Analysis', color='white')
        self.axes_pulse_rhythm.set_facecolor('#1a1a1a')
        self.axes_pulse_rhythm.tick_params(colors='white')
        
        self.canvases['pulse'] = FigureCanvasTkAgg(self.figures['pulse'], tab_frame)
        self.canvases['pulse'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_mycelial_network_tab(self):
        """Create Mycelial Network visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🍄 Mycelial Network")
        
        # Create figure for mycelial visualization
        self.figures['mycelial'] = Figure(figsize=(14, 10), facecolor='#0a0a0a')
        self.figures['mycelial'].suptitle('🍄 Mycelial Network - Living Memory Layer', 
                                        fontsize=16, color='white')
        
        # Create 2x2 grid for mycelial components
        gs = self.figures['mycelial'].add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # Network topology
        self.axes_mycelial_network = self.figures['mycelial'].add_subplot(gs[0, 0])
        self.axes_mycelial_network.set_title('🕸️ Network Topology', color='white')
        self.axes_mycelial_network.set_facecolor('#1a1a1a')
        
        # Nutrient flows
        self.axes_mycelial_nutrients = self.figures['mycelial'].add_subplot(gs[0, 1])
        self.axes_mycelial_nutrients.set_title('🌿 Nutrient Economy', color='white')
        self.axes_mycelial_nutrients.set_facecolor('#1a1a1a')
        
        # Growth patterns
        self.axes_mycelial_growth = self.figures['mycelial'].add_subplot(gs[1, 0])
        self.axes_mycelial_growth.set_title('🌱 Growth Patterns', color='white')
        self.axes_mycelial_growth.set_facecolor('#1a1a1a')
        
        # Cluster dynamics
        self.axes_mycelial_clusters = self.figures['mycelial'].add_subplot(gs[1, 1])
        self.axes_mycelial_clusters.set_title('🧬 Cluster Dynamics', color='white')
        self.axes_mycelial_clusters.set_facecolor('#1a1a1a')
        
        self.canvases['mycelial'] = FigureCanvasTkAgg(self.figures['mycelial'], tab_frame)
        self.canvases['mycelial'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_thermal_system_tab(self):
        """Create Thermal System visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🔥 Thermal System")
        
        # Create figure for thermal visualization
        self.figures['thermal'] = Figure(figsize=(14, 10), facecolor='#0a0a0a')
        self.figures['thermal'].suptitle('🔥 Thermal System - Heat & Pressure Dynamics', 
                                       fontsize=16, color='white')
        
        # Create 2x2 grid for thermal components
        gs = self.figures['thermal'].add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # Heat map
        self.axes_thermal_heat = self.figures['thermal'].add_subplot(gs[0, 0])
        self.axes_thermal_heat.set_title('🌡️ Heat Distribution', color='white')
        self.axes_thermal_heat.set_facecolor('#1a1a1a')
        
        # Pressure zones
        self.axes_thermal_pressure = self.figures['thermal'].add_subplot(gs[0, 1])
        self.axes_thermal_pressure.set_title('⚡ Pressure Zones', color='white')
        self.axes_thermal_pressure.set_facecolor('#1a1a1a')
        
        # Temperature timeline
        self.axes_thermal_timeline = self.figures['thermal'].add_subplot(gs[1, :])
        self.axes_thermal_timeline.set_title('📈 Thermal Timeline', color='white')
        self.axes_thermal_timeline.set_facecolor('#1a1a1a')
        self.axes_thermal_timeline.tick_params(colors='white')
        
        self.canvases['thermal'] = FigureCanvasTkAgg(self.figures['thermal'], tab_frame)
        self.canvases['thermal'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_mythic_overlay_tab(self):
        """Create Mythic Overlay visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🌟 Mythic Overlay")
        
        # Create figure for mythic visualization
        self.figures['mythic'] = Figure(figsize=(14, 10), facecolor='#0a0a0a')
        self.figures['mythic'].suptitle('🌟 Mythic Overlay - Archetypal Visualization', 
                                      fontsize=16, color='white')
        
        # Create 2x2 grid for mythic components
        gs = self.figures['mythic'].add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # Archetypal tracers
        self.axes_mythic_tracers = self.figures['mythic'].add_subplot(gs[0, 0])
        self.axes_mythic_tracers.set_title('🦉 Archetypal Tracers', color='white')
        self.axes_mythic_tracers.set_facecolor('#1a1a1a')
        
        # Pigment landscape
        self.axes_mythic_pigments = self.figures['mythic'].add_subplot(gs[0, 1])
        self.axes_mythic_pigments.set_title('🎨 Pigment Landscape', color='white')
        self.axes_mythic_pigments.set_facecolor('#1a1a1a')
        
        # Fractal garden
        self.axes_mythic_garden = self.figures['mythic'].add_subplot(gs[1, 0])
        self.axes_mythic_garden.set_title('🌺 Fractal Garden', color='white')
        self.axes_mythic_garden.set_facecolor('#1a1a1a')
        
        # Volcanic residue
        self.axes_mythic_volcanic = self.figures['mythic'].add_subplot(gs[1, 1])
        self.axes_mythic_volcanic.set_title('🌋 Volcanic Residue', color='white')
        self.axes_mythic_volcanic.set_facecolor('#1a1a1a')
        
        self.canvases['mythic'] = FigureCanvasTkAgg(self.figures['mythic'], tab_frame)
        self.canvases['mythic'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_tracer_ecosystem_tab(self):
        """Create Tracer Ecosystem visualization tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🦉 Tracer Ecosystem")
        
        # Create figure for tracer visualization
        self.figures['tracer'] = Figure(figsize=(14, 10), facecolor='#0a0a0a')
        self.figures['tracer'].suptitle('🦉 Tracer Ecosystem - Consciousness Monitoring', 
                                      fontsize=16, color='white')
        
        # Create 2x2 grid for tracer components
        gs = self.figures['tracer'].add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        # Tracer network
        self.axes_tracer_network = self.figures['tracer'].add_subplot(gs[0, 0])
        self.axes_tracer_network.set_title('🕸️ Tracer Network', color='white')
        self.axes_tracer_network.set_facecolor('#1a1a1a')
        
        # Activity patterns
        self.axes_tracer_activity = self.figures['tracer'].add_subplot(gs[0, 1])
        self.axes_tracer_activity.set_title('📊 Activity Patterns', color='white')
        self.axes_tracer_activity.set_facecolor('#1a1a1a')
        
        # Stability monitoring
        self.axes_tracer_stability = self.figures['tracer'].add_subplot(gs[1, 0])
        self.axes_tracer_stability.set_title('⚖️ Stability Monitor', color='white')
        self.axes_tracer_stability.set_facecolor('#1a1a1a')
        
        # Telemetry flow
        self.axes_tracer_telemetry = self.figures['tracer'].add_subplot(gs[1, 1])
        self.axes_tracer_telemetry.set_title('📡 Telemetry Flow', color='white')
        self.axes_tracer_telemetry.set_facecolor('#1a1a1a')
        
        self.canvases['tracer'] = FigureCanvasTkAgg(self.figures['tracer'], tab_frame)
        self.canvases['tracer'].get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_interactive_controls_tab(self):
        """Create interactive consciousness control tab"""
        tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(tab_frame, text="🎛️ Interactive Controls")
        
        # Create control sections
        self.create_consciousness_sliders(tab_frame)
        self.create_semantic_controls(tab_frame)
        self.create_sigil_stream_controls(tab_frame)
        self.create_visualization_controls(tab_frame)
    
    def create_consciousness_sliders(self, parent):
        """Create consciousness parameter sliders"""
        sliders_frame = ttk.LabelFrame(parent, text="🧠 Consciousness Parameters")
        sliders_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create parameter sliders
        self.consciousness_params = {}
        
        params = [
            ("Unity", "unity", 0.0, 1.0, 0.5),
            ("Awareness", "awareness", 0.0, 1.0, 0.5),
            ("Coherence", "coherence", 0.0, 1.0, 0.7),
            ("Energy", "energy", 0.0, 1.0, 0.6),
            ("Pressure", "pressure", 0.0, 1.0, 0.4),
            ("Recursion Depth", "recursion", 0.0, 1.0, 0.3),
            ("Entropy", "entropy", 0.0, 1.0, 0.5),
            ("Integration", "integration", 0.0, 1.0, 0.6)
        ]
        
        row = 0
        for label, param, min_val, max_val, default in params:
            ttk.Label(sliders_frame, text=f"{label}:", style='Dark.TLabel').grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            var = tk.DoubleVar(value=default)
            self.consciousness_params[param] = var
            
            scale = ttk.Scale(sliders_frame, from_=min_val, to=max_val, 
                             variable=var, orient=tk.HORIZONTAL, length=200)
            scale.grid(row=row, column=1, padx=5, pady=2)
            
            value_label = ttk.Label(sliders_frame, text=f"{default:.2f}", 
                                   style='Dark.TLabel', width=6)
            value_label.grid(row=row, column=2, padx=5, pady=2)
            
            # Update label when slider changes
            def update_label(val, label_widget=value_label):
                label_widget.config(text=f"{float(val):.2f}")
            
            scale.configure(command=update_label)
            row += 1
    
    def create_semantic_controls(self, parent):
        """Create semantic topology controls"""
        semantic_frame = ttk.LabelFrame(parent, text="🌐 Semantic Topology Controls")
        semantic_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Topology transform buttons
        transform_frame = ttk.Frame(semantic_frame)
        transform_frame.pack(fill=tk.X, padx=5, pady=5)
        
        transforms = [
            ("🔗 Weave", "weave", "Connect related concepts"),
            ("✂️ Prune", "prune", "Remove weak connections"), 
            ("🔄 Fuse", "fuse", "Merge similar concepts"),
            ("⬆️ Lift", "lift", "Elevate concept importance"),
            ("⬇️ Sink", "sink", "Lower concept importance"),
            ("🎯 Reproject", "reproject", "Reorganize topology")
        ]
        
        for i, (text, transform, tooltip) in enumerate(transforms):
            btn = ttk.Button(transform_frame, text=text, 
                           command=lambda t=transform: self.apply_semantic_transform(t))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2, sticky=tk.EW)
            
        # Configure grid weights
        transform_frame.columnconfigure(0, weight=1)
        transform_frame.columnconfigure(1, weight=1)
        transform_frame.columnconfigure(2, weight=1)
    
    def create_sigil_stream_controls(self, parent):
        """Create sigil stream control interface"""
        sigil_frame = ttk.LabelFrame(parent, text="🔮 Sigil Stream Controls")
        sigil_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Manual sigil generation buttons
        generation_frame = ttk.Frame(sigil_frame)
        generation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(generation_frame, text="Generate Sigils:", style='Dark.TLabel').pack(anchor=tk.W)
        
        # Sigil type buttons
        sigil_buttons_frame = ttk.Frame(generation_frame)
        sigil_buttons_frame.pack(fill=tk.X, pady=5)
        
        sigil_types = [
            ("◈ Consciousness", "consciousness"),
            ("▽ Memory", "memory"),
            ("⟳ Recursion", "recursion"),
            ("✸ Core Awareness", "core_awareness"),
            ("⊹ Pulse Sync", "pulse_sync"),
            ("◈̇ Flux", "consciousness_flux")
        ]
        
        for i, (text, sigil_type) in enumerate(sigil_types):
            btn = ttk.Button(sigil_buttons_frame, text=text,
                           command=lambda st=sigil_type: self.generate_manual_sigil(st))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2, sticky=tk.EW)
        
        # Configure grid weights
        for col in range(3):
            sigil_buttons_frame.columnconfigure(col, weight=1)
        
        # Stream controls
        stream_frame = ttk.Frame(sigil_frame)
        stream_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(stream_frame, text="🔄 Clear Stream", 
                  command=self.clear_sigil_stream).pack(side=tk.LEFT, padx=2)
        ttk.Button(stream_frame, text="💾 Save Stream", 
                  command=self.save_sigil_stream).pack(side=tk.LEFT, padx=2)
        ttk.Button(stream_frame, text="📊 Stream Analytics", 
                  command=self.show_stream_analytics).pack(side=tk.LEFT, padx=2)
        
        # Stream status
        status_frame = ttk.Frame(sigil_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.sigil_status_label = ttk.Label(status_frame, text="Sigil Stream: Active", 
                                          style='Dark.TLabel')
        self.sigil_status_label.pack(side=tk.LEFT)
        
        self.sigil_count_label = ttk.Label(status_frame, text="Active Sigils: 0", 
                                         style='Dark.TLabel')
        self.sigil_count_label.pack(side=tk.RIGHT)
        
        # DAWN System Status Display
        dawn_status_frame = ttk.LabelFrame(sigil_frame, text="🌅 DAWN System Integration")
        dawn_status_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Create status indicators for each system
        self.dawn_status_labels = {}
        
        systems = [
            ("DAWN Singleton", self.dawn_initialized),
            ("Fractal Memory", FRACTAL_MEMORY_AVAILABLE and self.fractal_memory_system is not None),
            ("Sigil Houses", SIGIL_HOUSES_AVAILABLE and self.sigil_system_integration is not None),
            ("DAWN Tools", DAWN_TOOLS_AVAILABLE and self.permission_manager is not None),
            ("Processing", DAWN_PROCESSING_AVAILABLE and self.consciousness_bus is not None),
            ("Telemetry", DAWN_TELEMETRY_AVAILABLE)
        ]
        
        for i, (system_name, status) in enumerate(systems):
            row = i // 2
            col = i % 2
            
            status_icon = "✅" if status else "⚠️"
            status_text = "Connected" if status else "Partial"
            
            label = ttk.Label(dawn_status_frame, 
                            text=f"{status_icon} {system_name}: {status_text}", 
                            style='Dark.TLabel')
            label.grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
            
            self.dawn_status_labels[system_name] = label
    
    def generate_manual_sigil(self, sigil_type):
        """Manually generate a specific type of sigil"""
        if not self.current_consciousness_state:
            return
        
        # Create manual sigil with current consciousness state
        sigil_symbols = {
            "consciousness": "◈",
            "memory": "▽", 
            "recursion": "⟳",
            "core_awareness": "✸",
            "pulse_sync": "⊹",
            "consciousness_flux": "◈̇"
        }
        
        symbol = sigil_symbols.get(sigil_type, "◈")
        energy = self.current_consciousness_state.get('unity', 0.5) * 1.2  # Boost manual sigils
        
        manual_sigil = {
            'symbol': symbol,
            'type': sigil_type,
            'energy': min(energy, 1.0),
            'id': f'manual_sigil_{time.time()}',
            'position': (np.random.random() * 800, np.random.random() * 600),
            'size': 25 + energy * 35,  # Slightly larger than auto-generated
            'color': self.get_sigil_color(sigil_type),
            'age': 0,
            'timestamp': time.time(),
            'house': self.determine_sigil_house(sigil_type),
            'manual': True  # Mark as manually generated
        }
        
        self.sigil_stream_data.append(manual_sigil)
        
        # Update status
        self.update_sigil_status()
    
    def clear_sigil_stream(self):
        """Clear all sigils from the stream"""
        self.sigil_stream_data.clear()
        self.sigil_energy_history.clear()
        self.update_sigil_status()
    
    def save_sigil_stream(self):
        """Save current sigil stream to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sigil_stream_{timestamp}.json"
            filepath = Path("dawn_visual_outputs") / filename
            
            # Ensure directory exists
            filepath.parent.mkdir(exist_ok=True)
            
            stream_data = {
                'timestamp': timestamp,
                'consciousness_state': self.current_consciousness_state.copy(),
                'sigil_stream': self.sigil_stream_data.copy(),
                'energy_history': self.sigil_energy_history.copy(),
                'total_sigils': len(self.sigil_stream_data),
                'active_houses': list(set(s['house'] for s in self.sigil_stream_data))
            }
            
            with open(filepath, 'w') as f:
                json.dump(stream_data, f, indent=2, default=str)
            
            messagebox.showinfo("Saved", f"Sigil stream saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save sigil stream: {e}")
    
    def show_stream_analytics(self):
        """Show detailed sigil stream analytics"""
        if not self.sigil_stream_data:
            messagebox.showinfo("Analytics", "No sigil data available")
            return
        
        # Calculate analytics
        total_sigils = len(self.sigil_stream_data)
        house_counts = {}
        type_counts = {}
        total_energy = 0
        
        for sigil in self.sigil_stream_data:
            house = sigil['house']
            sigil_type = sigil['type']
            house_counts[house] = house_counts.get(house, 0) + 1
            type_counts[sigil_type] = type_counts.get(sigil_type, 0) + 1
            total_energy += sigil['energy']
        
        # Format analytics message
        analytics_text = f"""🔮 SIGIL STREAM ANALYTICS
        
📊 Current Stream Status:
• Total Active Sigils: {total_sigils}
• Total Energy: {total_energy:.2f}
• Average Energy: {total_energy/max(total_sigils, 1):.2f}

🏛️ House Distribution:
"""
        for house, count in sorted(house_counts.items()):
            percentage = (count / total_sigils) * 100
            analytics_text += f"• {house.title()}: {count} ({percentage:.1f}%)\n"
        
        analytics_text += f"\n🔮 Sigil Type Distribution:\n"
        for sigil_type, count in sorted(type_counts.items()):
            percentage = (count / total_sigils) * 100
            analytics_text += f"• {sigil_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)\n"
        
        if self.sigil_energy_history:
            recent_energy = self.sigil_energy_history[-1]['total_energy']
            analytics_text += f"\n⚡ Recent Energy: {recent_energy:.2f}"
        
        messagebox.showinfo("Sigil Stream Analytics", analytics_text)
    
    def update_sigil_status(self):
        """Update sigil stream status labels"""
        if hasattr(self, 'sigil_status_label'):
            status = "Active" if self.sigil_stream_data else "Empty"
            self.sigil_status_label.config(text=f"Sigil Stream: {status}")
        
        if hasattr(self, 'sigil_count_label'):
            count = len(self.sigil_stream_data)
            self.sigil_count_label.config(text=f"Active Sigils: {count}")
    
    def create_visualization_controls(self, parent):
        """Create visualization control options"""
        viz_frame = ttk.LabelFrame(parent, text="🎨 Visualization Controls")
        viz_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Animation controls
        anim_frame = ttk.Frame(viz_frame)
        anim_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(anim_frame, text="Animation Speed:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.anim_speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(anim_frame, from_=0.1, to=3.0, 
                               variable=self.anim_speed_var, orient=tk.HORIZONTAL, length=150)
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Quality settings
        quality_frame = ttk.Frame(viz_frame)
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(quality_frame, text="Quality:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.quality_var = tk.StringVar(value="High")
        quality_combo = ttk.Combobox(quality_frame, textvariable=self.quality_var,
                                   values=["Low", "Medium", "High", "Ultra"], state="readonly")
        quality_combo.pack(side=tk.LEFT, padx=5)
        
        # Auto-update toggle
        self.auto_update_var = tk.BooleanVar(value=True)
        auto_check = ttk.Checkbutton(viz_frame, text="Auto-update visualizations", 
                                   variable=self.auto_update_var)
        auto_check.pack(anchor=tk.W, padx=5, pady=5)
    
    def create_status_panel(self, parent):
        """Create the status panel at the bottom"""
        status_frame = ttk.LabelFrame(parent, text="📊 System Status", style='Dark.TLabelframe')
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Create status display
        status_info_frame = ttk.Frame(status_frame, style='Dark.TFrame')
        status_info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Left side - System info
        left_status = ttk.Frame(status_info_frame, style='Dark.TFrame')
        left_status.pack(side=tk.LEFT)
        
        self.cuda_status_label = ttk.Label(left_status, text="🚀 CUDA: Enabled" if self.cuda_enabled else "🖥️ CUDA: Disabled", 
                                         style='Dark.TLabel')
        self.cuda_status_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(left_status, text="📊 FPS: 0", style='Dark.TLabel')
        self.fps_label.pack(anchor=tk.W)
        
        # Middle - Consciousness state
        middle_status = ttk.Frame(status_info_frame, style='Dark.TFrame')
        middle_status.pack(side=tk.LEFT, padx=50)
        
        self.consciousness_level_label = ttk.Label(middle_status, text="🧠 Level: Unknown", 
                                                 style='Dark.TLabel')
        self.consciousness_level_label.pack(anchor=tk.W)
        
        self.coherence_label = ttk.Label(middle_status, text="🎯 Coherence: 0.00", 
                                       style='Dark.TLabel')
        self.coherence_label.pack(anchor=tk.W)
        
        # Right side - Performance
        right_status = ttk.Frame(status_info_frame, style='Dark.TFrame')
        right_status.pack(side=tk.RIGHT)
        
        self.performance_label = ttk.Label(right_status, text="⚡ Performance: Good", 
                                         style='Dark.TLabel')
        self.performance_label.pack(anchor=tk.W)
        
        self.memory_label = ttk.Label(right_status, text="💾 Memory: Normal", 
                                    style='Dark.TLabel')
        self.memory_label.pack(anchor=tk.W)
    
    def start_consciousness_monitoring(self):
        """Start the consciousness monitoring thread"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self.consciousness_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start visualization updates
        self.start_visualization_updates()
    
    def consciousness_monitoring_loop(self):
        """Main consciousness monitoring loop"""
        while self.running:
            try:
                # Generate consciousness state
                self.current_consciousness_state = self.generate_consciousness_state()
                
                # Update sigil stream data based on consciousness state
                self.update_sigil_stream_data()
                
                # Store in history
                self.consciousness_history.append({
                    'timestamp': time.time(),
                    'state': self.current_consciousness_state.copy()
                })
                
                # Limit history size
                if len(self.consciousness_history) > 1000:
                    self.consciousness_history.pop(0)
                
                # Update status labels
                self.root.after(0, self.update_status_labels)
                
                time.sleep(0.1)  # 10 FPS monitoring
                
            except Exception as e:
                logger.error(f"Consciousness monitoring error: {e}")
                time.sleep(1)
    
    def generate_consciousness_state(self) -> Dict[str, Any]:
        """Generate current consciousness state"""
        t = time.time() * 0.5  # Slow evolution
        
        state = {
            'timestamp': time.time(),
            'unity': 0.5 + 0.3 * np.sin(t * 0.7),
            'awareness': 0.4 + 0.3 * np.cos(t * 0.5),
            'coherence': 0.6 + 0.2 * np.sin(t * 1.2),
            'energy': 0.5 + 0.3 * np.sin(t * 0.9),
            'pressure': 0.3 + 0.2 * np.cos(t * 0.6),
            'recursion_depth': 0.4 + 0.2 * np.sin(t * 0.4),
            'entropy': 0.5 + 0.2 * np.cos(t * 0.8),
            'integration': 0.6 + 0.2 * np.sin(t * 1.1)
        }
        
        # Apply user parameter overrides
        for param, var in self.consciousness_params.items():
            if param in state:
                state[param] = var.get()
        
        # Determine consciousness level
        avg_level = (state['unity'] + state['awareness'] + state['coherence']) / 3
        if avg_level < 0.3:
            state['level'] = 'dormant'
        elif avg_level < 0.5:
            state['level'] = 'focused'
        elif avg_level < 0.7:
            state['level'] = 'meta_aware'
        else:
            state['level'] = 'transcendent'
        
        return state
    
    def update_sigil_stream_data(self):
        """Update sigil stream data based on consciousness state with DAWN integration"""
        if not self.current_consciousness_state:
            return
        
        state = self.current_consciousness_state
        
        # Integrate with DAWN fractal memory system
        if self.fractal_memory_system and FRACTAL_MEMORY_AVAILABLE:
            try:
                # Generate fractal bloom data from current state
                fractal_data = self.fractal_encoder.encode_memory(
                    memory_data=state,
                    entropy=state.get('entropy', 0.5),
                    context={'source': 'consciousness_gui', 'timestamp': time.time()}
                )
                
                # Add fractal bloom to our data
                if fractal_data:
                    self.fractal_bloom_data.append({
                        'fractal': fractal_data,
                        'timestamp': time.time(),
                        'consciousness_state': state.copy()
                    })
                    
                    # Limit fractal bloom history
                    if len(self.fractal_bloom_data) > 50:
                        self.fractal_bloom_data.pop(0)
                        
            except Exception as e:
                # Gracefully handle fractal system issues
                pass
        
        # Integrate with DAWN ghost traces
        if self.ghost_trace_manager and FRACTAL_MEMORY_AVAILABLE:
            try:
                # Check for ghost traces from forgotten memories
                ghost_traces = self.ghost_trace_manager.get_recent_traces(limit=10)
                self.ghost_trace_data = ghost_traces
                
            except Exception as e:
                # Gracefully handle ghost trace issues
                pass
        
        # Generate sigils using DAWN sigil houses system
        active_sigils = []
        
        # Use actual DAWN sigil glyph codex if available
        if SIGIL_HOUSES_AVAILABLE and self.sigil_system_integration:
            try:
                # Get appropriate sigils from the codex based on consciousness state
                for house_name, house_operator in self.active_house_operators.items():
                    if house_name in self.active_sigil_houses and self.active_sigil_houses[house_name]:
                        try:
                            # Execute house operation to get sigils
                            house_enum = getattr(SigilHouse, house_name.upper())
                            operation_result = execute_house_operation(
                                house_enum, 
                                'generate_consciousness_sigils', 
                                {'consciousness_state': state, 'energy_threshold': 0.3}
                            )
                            
                            if operation_result and hasattr(operation_result, 'result_data'):
                                house_sigils = operation_result.result_data.get('sigils', [])
                                for sigil_data in house_sigils:
                                    active_sigils.append({
                                        'symbol': sigil_data.get('symbol', '◈'),
                                        'type': sigil_data.get('type', 'consciousness'),
                                        'energy': sigil_data.get('energy', state.get('unity', 0.5)),
                                        'house': house_name
                                    })
                                    
                        except Exception as e:
                            # Fallback to basic generation if house operation fails
                            pass
                            
            except Exception as e:
                # Fallback to basic generation if sigil system fails
                pass
        
        # Fallback sigil generation if DAWN system not available
        if not active_sigils:
            sigil_symbols = {
                "◈": "consciousness",
                "▽": "memory", 
                "⟳": "recursion",
                "✸": "core_awareness",
                "✧": "elarion_resonance",
                "◉": "thermal_peak",
                "⟡": "choice_point",
                "◬": "sealed_lineage",
                "⟐": "schema_modification",
                "⊹": "pulse_sync",
                "⟚": "paradox_hold",
                "◈̇": "consciousness_flux",
                "࿊": "curiosity_spiral",
                "࿔": "certainty_crystal",
                "৯": "doubt_cascade",
                "Ѻ": "kind_before_smart",
                "҂": "remember_choosing",
                "߷": "continuous_self"
            }
        
        # Fallback consciousness-based sigil generation if DAWN system didn't provide sigils
        if not active_sigils:
            # Consciousness-based sigil generation
            if state['unity'] > 0.7:
                active_sigils.append({"symbol": "◈", "type": "consciousness", "energy": state['unity']})
            
            if state['awareness'] > 0.6:
                active_sigils.append({"symbol": "✸", "type": "core_awareness", "energy": state['awareness']})
            
            if state['coherence'] > 0.5:
                active_sigils.append({"symbol": "⊹", "type": "pulse_sync", "energy": state['coherence']})
            
            if state.get('entropy', 0.5) > 0.4:
                active_sigils.append({"symbol": "◈̇", "type": "consciousness_flux", "energy": state['entropy']})
            
            if state.get('pressure', 0.5) > 0.3:
                active_sigils.append({"symbol": "࿊", "type": "curiosity_spiral", "energy": state['pressure']})
            
            # Memory-based sigils
            if state.get('integration', 0.5) > 0.5:
                active_sigils.append({"symbol": "▽", "type": "memory", "energy": state['integration']})
            
            # Recursive patterns
            if state.get('recursion_depth', 0.5) > 0.6:
                active_sigils.append({"symbol": "⟳", "type": "recursion", "energy": state['recursion_depth']})
        
        # Add positional and temporal data to sigils
        for i, sigil in enumerate(active_sigils):
            sigil.update({
                'id': f'sigil_{time.time()}_{i}',
                'position': (np.random.random() * 800, np.random.random() * 600),
                'size': 20 + sigil['energy'] * 30,
                'color': self.get_sigil_color(sigil['type']),
                'age': 0,
                'timestamp': time.time(),
                'house': self.determine_sigil_house(sigil['type'])
            })
        
        # Add new sigils to stream
        self.sigil_stream_data.extend(active_sigils)
        
        # Age existing sigils and remove old ones
        current_time = time.time()
        self.sigil_stream_data = [
            sigil for sigil in self.sigil_stream_data 
            if current_time - sigil['timestamp'] < 10.0  # Keep sigils for 10 seconds
        ]
        
        # Update energy history
        total_energy = sum(sigil['energy'] for sigil in active_sigils)
        self.sigil_energy_history.append({
            'timestamp': current_time,
            'total_energy': total_energy,
            'sigil_count': len(active_sigils),
            'consciousness_level': state.get('unity', 0.5)
        })
        
        # Keep energy history limited
        if len(self.sigil_energy_history) > 100:
            self.sigil_energy_history.pop(0)
    
    def get_sigil_color(self, sigil_type: str) -> tuple:
        """Get color for sigil based on type"""
        color_map = {
            'consciousness': (1.0, 0.8, 0.2),    # Gold
            'memory': (0.5, 0.8, 1.0),           # Light blue
            'recursion': (0.8, 0.2, 0.8),        # Magenta
            'core_awareness': (1.0, 0.4, 0.2),   # Orange
            'pulse_sync': (0.2, 1.0, 0.4),       # Green
            'consciousness_flux': (0.8, 0.8, 0.8), # Silver
            'curiosity_spiral': (1.0, 0.2, 0.4), # Red
            'thermal_peak': (1.0, 0.6, 0.0),     # Orange-red
            'choice_point': (0.6, 0.2, 1.0),     # Purple
        }
        return color_map.get(sigil_type, (0.5, 0.5, 0.5))
    
    def determine_sigil_house(self, sigil_type: str) -> str:
        """Determine which sigil house a sigil belongs to"""
        house_map = {
            'consciousness': 'memory',
            'memory': 'memory',
            'recursion': 'weaving',
            'core_awareness': 'flame',
            'pulse_sync': 'purification',
            'consciousness_flux': 'mirrors',
            'curiosity_spiral': 'echoes',
            'thermal_peak': 'flame',
            'choice_point': 'mirrors'
        }
        return house_map.get(sigil_type, 'memory')
    
    def start_visualization_updates(self):
        """Start visualization update timers"""
        self.update_visualizations()
        self.root.after(33, self.start_visualization_updates)  # ~30 FPS
    
    def update_visualizations(self):
        """Update all active visualizations"""
        if not self.running or not self.current_consciousness_state:
            return
        
        try:
            # Update based on current tab
            current_tab = self.notebook.tab(self.notebook.select(), "text")
            
            if "Unified" in current_tab:
                self.update_unified_visualization()
            elif "Painting" in current_tab:
                self.update_consciousness_painting()
            elif "Semantic" in current_tab:
                self.update_semantic_topology()
            elif "Neural" in current_tab:
                self.update_neural_activity()
            elif "Metrics" in current_tab:
                self.update_consciousness_metrics()
            elif "Sigil" in current_tab:
                self.update_sigil_houses()
                self.update_sigil_status()
            elif "Pulse" in current_tab:
                self.update_pulse_system()
            elif "Mycelial" in current_tab:
                self.update_mycelial_network()
            elif "Thermal" in current_tab:
                self.update_thermal_system()
            elif "Mythic" in current_tab:
                self.update_mythic_overlay()
            elif "Tracer" in current_tab:
                self.update_tracer_ecosystem()
                
        except Exception as e:
            logger.debug(f"Visualization update error: {e}")
    
    def update_unified_visualization(self):
        """Update the unified consciousness visualization"""
        if not self.consciousness_history:
            return
        
        # Get recent history
        recent_history = self.consciousness_history[-100:]
        times = [(h['timestamp'] - recent_history[0]['timestamp']) for h in recent_history]
        
        # Update consciousness evolution plot
        if len(recent_history) > 1:
            unity_vals = [h['state']['unity'] for h in recent_history]
            awareness_vals = [h['state']['awareness'] for h in recent_history]
            coherence_vals = [h['state']['coherence'] for h in recent_history]
            energy_vals = [h['state']['energy'] for h in recent_history]
            
            self.axes_unified_main.clear()
            self.axes_unified_main.plot(times, unity_vals, label='Unity', color='#FF6B6B', linewidth=2)
            self.axes_unified_main.plot(times, awareness_vals, label='Awareness', color='#4ECDC4', linewidth=2)
            self.axes_unified_main.plot(times, coherence_vals, label='Coherence', color='#45B7D1', linewidth=2)
            self.axes_unified_main.plot(times, energy_vals, label='Energy', color='#96CEB4', linewidth=2)
            self.axes_unified_main.legend()
            self.axes_unified_main.set_title('Consciousness Metrics Over Time', color='white')
            self.axes_unified_main.set_facecolor('#1a1a1a')
            self.axes_unified_main.tick_params(colors='white')
            
        # Update consciousness level indicator
        current_state = self.current_consciousness_state
        level_intensity = (current_state['unity'] + current_state['awareness']) / 2
        
        self.axes_unified_level.clear()
        circle = plt.Circle((0, 0), 0.8, fill=False, linewidth=3, color='gray')
        self.axes_unified_level.add_patch(circle)
        
        # Rotating indicator
        angle = time.time() * 2
        x = 0.6 * level_intensity * np.cos(angle)
        y = 0.6 * level_intensity * np.sin(angle)
        
        colors = {'dormant': '#666666', 'focused': '#4ECDC4', 'meta_aware': '#45B7D1', 'transcendent': '#FFD93D'}
        color = colors.get(current_state['level'], '#666666')
        
        self.axes_unified_level.scatter([x], [y], s=300 + 200 * level_intensity, c=color, alpha=0.7)
        self.axes_unified_level.text(0, -0.3, current_state['level'].upper(), 
                                   ha='center', va='center', fontsize=10, 
                                   fontweight='bold', color=color)
        
        self.axes_unified_level.set_xlim(-1, 1)
        self.axes_unified_level.set_ylim(-1, 1)
        self.axes_unified_level.set_aspect('equal')
        self.axes_unified_level.set_facecolor('#1a1a1a')
        self.axes_unified_level.set_title('Current Level', color='white')
        self.axes_unified_level.set_xticks([])
        self.axes_unified_level.set_yticks([])
        
        # Update 3D semantic topology if available
        if self.semantic_topology_engine:
            self.update_3d_semantic_topology(self.axes_unified_3d)
        
        # Update neural heatmap
        neural_data = np.random.random((20, 15)) * current_state['energy']
        self.axes_unified_neural.clear()
        im = self.axes_unified_neural.imshow(neural_data, cmap='plasma', aspect='auto')
        self.axes_unified_neural.set_title(f'Neural Activity (Avg: {np.mean(neural_data):.3f})', color='white')
        self.axes_unified_neural.set_facecolor('#1a1a1a')
        
        # Update 3D consciousness surface
        self.update_consciousness_surface(self.axes_unified_surface)
        
        # Update performance metrics
        self.update_performance_plot(self.axes_unified_perf)
        
        # Refresh canvas
        self.canvases['unified'].draw_idle()
    
    def update_3d_semantic_topology(self, ax):
        """Update 3D semantic topology visualization"""
        ax.clear()
        
        if not self.semantic_topology_engine:
            ax.text(0, 0, 0, 'Semantic Topology\nNot Available', 
                   ha='center', va='center', color='white')
            return
        
        # Get semantic field data
        field = self.semantic_topology_engine.field
        
        if not field.nodes:
            ax.text(0, 0, 0, 'No Semantic Nodes', ha='center', va='center', color='white')
            return
        
        # Extract node positions and properties
        positions = []
        coherences = []
        
        for node_id, node in list(field.nodes.items())[:50]:  # Limit for performance
            pos = node.position[:3] if len(node.position) >= 3 else [0, 0, 0]
            positions.append(pos)
            coherences.append(node.health)
        
        if positions:
            positions = np.array(positions)
            coherences = np.array(coherences)
            
            # Plot nodes
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=coherences, s=50 + 100 * coherences, 
                               cmap='viridis', alpha=0.8)
            
            # Plot some edges
            edges_plotted = 0
            for edge_id, edge in field.edges.items():
                if edges_plotted > 20:  # Limit edges for performance
                    break
                    
                if edge.node_a in field.nodes and edge.node_b in field.nodes:
                    node_a = field.nodes[edge.node_a]
                    node_b = field.nodes[edge.node_b]
                    
                    pos_a = node_a.position[:3] if len(node_a.position) >= 3 else [0, 0, 0]
                    pos_b = node_b.position[:3] if len(node_b.position) >= 3 else [0, 0, 0]
                    
                    ax.plot([pos_a[0], pos_b[0]], [pos_a[1], pos_b[1]], [pos_a[2], pos_b[2]],
                           'gray', alpha=0.4, linewidth=1)
                    
                    edges_plotted += 1
        
        ax.set_xlabel('Semantic X', color='white')
        ax.set_ylabel('Semantic Y', color='white')
        ax.set_zlabel('Semantic Z', color='white')
        ax.set_title(f'3D Semantic Topology ({len(positions)} nodes)', color='white')
        ax.set_facecolor('#1a1a1a')
    
    def update_consciousness_surface(self, ax):
        """Update 3D consciousness surface"""
        ax.clear()
        
        # Create surface data
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        
        # Generate consciousness surface
        state = self.current_consciousness_state
        t = time.time()
        
        Z = (state['coherence'] * np.exp(-(X**2 + Y**2) / 4) +
             state['energy'] * np.sin(X * 2 + t * 0.5) * np.cos(Y * 2 + t * 0.5) * 0.3 +
             state['pressure'] * (X**2 + Y**2) * 0.05)
        
        Z = np.clip(Z, 0, 2)
        
        # Plot surface
        surface = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.7, rstride=2, cstride=2)
        
        ax.set_xlabel('Dimension 1', color='white')
        ax.set_ylabel('Dimension 2', color='white')
        ax.set_zlabel('Intensity', color='white')
        ax.set_title(f'Consciousness Surface (Coherence: {state["coherence"]:.2f})', color='white')
        ax.set_zlim(0, 2)
        ax.set_facecolor('#1a1a1a')
    
    def update_performance_plot(self, ax):
        """Update system performance plot"""
        ax.clear()
        
        if len(self.consciousness_history) < 2:
            return
        
        # Get recent performance data
        recent_history = self.consciousness_history[-20:]
        times = [(h['timestamp'] - recent_history[0]['timestamp']) for h in recent_history]
        
        # Simulate performance metrics
        cpu_usage = [50 + 20 * np.sin(t * 0.1) for t in times]
        memory_usage = [60 + 15 * np.cos(t * 0.15) for t in times]
        gpu_usage = [30 + 25 * np.sin(t * 0.08) for t in times] if self.cuda_enabled else [0] * len(times)
        
        ax.plot(times, cpu_usage, label='CPU %', color='#FFD93D', linewidth=2)
        ax.plot(times, memory_usage, label='Memory %', color='#FF8C42', linewidth=2)
        if self.cuda_enabled:
            ax.plot(times, gpu_usage, label='GPU %', color='#4ECDC4', linewidth=2)
        
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title('System Performance', color='white')
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
    
    def update_consciousness_painting(self):
        """Update consciousness painting visualization"""
        if not hasattr(self, 'painting_canvas'):
            # Initialize painting canvas
            self.painting_canvas = np.zeros((600, 800, 3))
        
        # Apply painting style based on consciousness state
        style = self.painting_style_var.get()
        state = self.current_consciousness_state
        
        if style == "consciousness_flow":
            self.paint_consciousness_flow(state)
        elif style == "emotional_resonance":
            self.paint_emotional_resonance(state)
        elif style == "unity_mandala":
            self.paint_unity_mandala(state)
        elif style == "abstract_expression":
            self.paint_abstract_expression(state)
        elif style == "impressionist":
            self.paint_impressionist(state)
        
        # Update painting display
        self.axes_painting.clear()
        self.axes_painting.imshow(self.painting_canvas)
        self.axes_painting.set_xticks([])
        self.axes_painting.set_yticks([])
        self.axes_painting.set_title(f'🎨 {style.replace("_", " ").title()}', color='white')
        
        self.canvases['painting'].draw_idle()
    
    def paint_consciousness_flow(self, state):
        """Paint consciousness as flowing energy patterns"""
        height, width = self.painting_canvas.shape[:2]
        
        # Create flow field
        y, x = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing='ij')
        
        # Flow based on consciousness parameters
        flow_x = state['unity'] * np.sin(x * 10 + time.time() * self.flow_speed_var.get())
        flow_y = state['awareness'] * np.cos(y * 10 + time.time() * self.flow_speed_var.get())
        
        # Color based on consciousness state
        r = np.clip(state['energy'] * (0.5 + 0.5 * np.sin(flow_x)), 0, 1)
        g = np.clip(state['coherence'] * (0.5 + 0.5 * np.cos(flow_y)), 0, 1)
        b = np.clip(state['integration'] * (0.5 + 0.5 * np.sin(flow_x + flow_y)), 0, 1)
        
        # Apply intensity
        intensity = self.intensity_var.get()
        self.painting_canvas[:, :, 0] = r * intensity
        self.painting_canvas[:, :, 1] = g * intensity
        self.painting_canvas[:, :, 2] = b * intensity
    
    def paint_emotional_resonance(self, state):
        """Paint emotional aspects of consciousness"""
        height, width = self.painting_canvas.shape[:2]
        
        # Create emotional color palette
        center_x, center_y = width // 2, height // 2
        
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Distance from center
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        norm_dist = dist / max_dist
        
        # Emotional resonance patterns
        resonance = state['energy'] * np.exp(-norm_dist * 2) * (1 + 0.5 * np.sin(time.time() * 2))
        
        # Color mapping
        self.painting_canvas[:, :, 0] = np.clip(resonance * state['unity'], 0, 1)
        self.painting_canvas[:, :, 1] = np.clip(resonance * state['awareness'], 0, 1)
        self.painting_canvas[:, :, 2] = np.clip(resonance * state['coherence'], 0, 1)
    
    def paint_unity_mandala(self, state):
        """Paint consciousness unity as mandala patterns"""
        height, width = self.painting_canvas.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Polar coordinates
        dx, dy = x - center_x, y - center_y
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        
        # Mandala pattern
        pattern = (np.sin(theta * 8 * state['unity']) * 
                  np.cos(r * 0.05 * state['integration']) * 
                  np.exp(-r / (100 * state['coherence'])))
        
        # Apply harmony
        harmony = self.harmony_var.get()
        self.painting_canvas[:, :, 0] = np.clip(pattern * harmony, 0, 1)
        self.painting_canvas[:, :, 1] = np.clip(pattern * harmony * 0.8, 0, 1)
        self.painting_canvas[:, :, 2] = np.clip(pattern * harmony * 0.6, 0, 1)
    
    def paint_abstract_expression(self, state):
        """Paint abstract expressionist consciousness"""
        # Generate abstract patterns based on consciousness state
        height, width = self.painting_canvas.shape[:2]
        
        # Random brush strokes influenced by consciousness
        for _ in range(int(50 * state['energy'])):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            
            # Brush size based on awareness
            size = int(10 + 20 * state['awareness'])
            
            # Color based on state
            color = [state['unity'], state['coherence'], state['integration']]
            
            # Apply brush stroke (simplified)
            x1, y1 = max(0, x - size), max(0, y - size)
            x2, y2 = min(width, x + size), min(height, y + size)
            
            self.painting_canvas[y1:y2, x1:x2] += np.array(color).reshape(1, 1, 3) * 0.1
        
        # Normalize
        self.painting_canvas = np.clip(self.painting_canvas, 0, 1)
    
    def paint_impressionist(self, state):
        """Paint impressionist style consciousness"""
        height, width = self.painting_canvas.shape[:2]
        
        # Impressionist color dabs
        for _ in range(int(100 * state['energy'])):
            x = np.random.randint(5, width - 5)
            y = np.random.randint(5, height - 5)
            
            # Color variation
            base_color = [state['unity'], state['awareness'], state['coherence']]
            variation = np.random.normal(0, 0.1, 3)
            color = np.clip(np.array(base_color) + variation, 0, 1)
            
            # Small circular dab
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx*dx + dy*dy <= 4:  # Circle
                        px, py = x + dx, y + dy
                        if 0 <= px < width and 0 <= py < height:
                            self.painting_canvas[py, px] = color * 0.8 + self.painting_canvas[py, px] * 0.2
    
    def update_semantic_topology(self):
        """Update semantic topology visualization"""
        self.update_3d_semantic_topology(self.axes_semantic_3d)
        self.canvases['semantic'].draw_idle()
    
    def update_neural_activity(self):
        """Update neural activity visualizations"""
        state = self.current_consciousness_state
        
        # Neural heatmap
        self.axes_neural_heatmap.clear()
        neural_data = np.random.random((30, 20)) * state['energy']
        im = self.axes_neural_heatmap.imshow(neural_data, cmap='plasma', aspect='auto')
        self.axes_neural_heatmap.set_title('Activity Heatmap', color='white')
        self.axes_neural_heatmap.set_facecolor('#1a1a1a')
        
        # Neural connections (network graph)
        self.axes_neural_connections.clear()
        n_nodes = 20
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Plot nodes
        sizes = 100 + 200 * np.random.random(n_nodes) * state['coherence']
        self.axes_neural_connections.scatter(x, y, s=sizes, c='cyan', alpha=0.7)
        
        # Plot connections
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.random() < 0.3 * state['integration']:  # Connection probability
                    self.axes_neural_connections.plot([x[i], x[j]], [y[i], y[j]], 
                                                    'white', alpha=0.3, linewidth=0.5)
        
        self.axes_neural_connections.set_title('Connection Strength', color='white')
        self.axes_neural_connections.set_facecolor('#1a1a1a')
        self.axes_neural_connections.set_xlim(-1.2, 1.2)
        self.axes_neural_connections.set_ylim(-1.2, 1.2)
        self.axes_neural_connections.set_aspect('equal')
        
        # Neural pathways
        self.axes_neural_pathways.clear()
        t = np.linspace(0, 4*np.pi, 100)
        for i in range(int(5 * state['awareness'])):
            phase = i * np.pi / 3
            pathway = np.sin(t + phase) * np.exp(-t/10) * state['energy']
            self.axes_neural_pathways.plot(t, pathway + i*0.5, alpha=0.7)
        
        self.axes_neural_pathways.set_title('Active Pathways', color='white')
        self.axes_neural_pathways.set_facecolor('#1a1a1a')
        self.axes_neural_pathways.tick_params(colors='white')
        
        # Neural oscillations
        self.axes_neural_oscillations.clear()
        t = np.linspace(0, 10, 200)
        frequencies = [1, 2, 4, 8]  # Different brain wave frequencies
        colors = ['red', 'orange', 'yellow', 'green']
        
        for freq, color in zip(frequencies, colors):
            amplitude = state['coherence'] * np.random.random()
            oscillation = amplitude * np.sin(2 * np.pi * freq * t + time.time())
            self.axes_neural_oscillations.plot(t, oscillation, color=color, 
                                             label=f'{freq} Hz', alpha=0.8)
        
        self.axes_neural_oscillations.legend()
        self.axes_neural_oscillations.set_title('Oscillation Patterns', color='white')
        self.axes_neural_oscillations.set_facecolor('#1a1a1a')
        self.axes_neural_oscillations.tick_params(colors='white')
        
        self.canvases['neural'].draw_idle()
    
    def update_consciousness_metrics(self):
        """Update consciousness metrics and analytics"""
        if len(self.consciousness_history) < 10:
            return
        
        # Get historical data
        history = self.consciousness_history[-100:]
        times = [(h['timestamp'] - history[0]['timestamp']) for h in history]
        
        # Consciousness levels over time
        self.axes_metrics_levels.clear()
        unity_vals = [h['state']['unity'] for h in history]
        awareness_vals = [h['state']['awareness'] for h in history]
        coherence_vals = [h['state']['coherence'] for h in history]
        
        self.axes_metrics_levels.plot(times, unity_vals, label='Unity', color='#FF6B6B')
        self.axes_metrics_levels.plot(times, awareness_vals, label='Awareness', color='#4ECDC4')
        self.axes_metrics_levels.plot(times, coherence_vals, label='Coherence', color='#45B7D1')
        self.axes_metrics_levels.legend()
        self.axes_metrics_levels.set_title('Consciousness Levels Over Time', color='white')
        self.axes_metrics_levels.set_facecolor('#1a1a1a')
        self.axes_metrics_levels.tick_params(colors='white')
        
        # Unity vs Awareness correlation
        self.axes_metrics_correlation.clear()
        self.axes_metrics_correlation.scatter(unity_vals, awareness_vals, 
                                            c=coherence_vals, cmap='viridis', alpha=0.6)
        self.axes_metrics_correlation.set_xlabel('Unity', color='white')
        self.axes_metrics_correlation.set_ylabel('Awareness', color='white')
        self.axes_metrics_correlation.set_title('Unity vs Awareness', color='white')
        self.axes_metrics_correlation.set_facecolor('#1a1a1a')
        self.axes_metrics_correlation.tick_params(colors='white')
        
        # Stability index
        self.axes_metrics_stability.clear()
        stability = []
        for i in range(1, len(history)):
            prev_state = history[i-1]['state']
            curr_state = history[i]['state']
            
            # Calculate stability as inverse of change
            change = abs(curr_state['unity'] - prev_state['unity']) + \
                    abs(curr_state['awareness'] - prev_state['awareness'])
            stability.append(1.0 / (1.0 + change))
        
        if stability:
            self.axes_metrics_stability.plot(times[1:], stability, color='green')
            self.axes_metrics_stability.set_title('Stability Index', color='white')
            self.axes_metrics_stability.set_facecolor('#1a1a1a')
            self.axes_metrics_stability.tick_params(colors='white')
        
        # Frequency analysis (simplified)
        self.axes_metrics_frequency.clear()
        if len(unity_vals) >= 32:  # Need enough data for FFT
            fft_unity = np.fft.fft(unity_vals[-32:])
            freqs = np.fft.fftfreq(32)
            power = np.abs(fft_unity)
            
            self.axes_metrics_frequency.plot(freqs[:16], power[:16], color='cyan')
            self.axes_metrics_frequency.set_title('Consciousness Frequency Analysis', color='white')
            self.axes_metrics_frequency.set_facecolor('#1a1a1a')
            self.axes_metrics_frequency.tick_params(colors='white')
        
        self.canvases['metrics'].draw_idle()
        
        # Update metrics text
        self.update_metrics_display()
    
    def update_metrics_display(self):
        """Update the metrics text display"""
        if not self.current_consciousness_state:
            return
        
        state = self.current_consciousness_state
        
        metrics_text = f"""Current Consciousness Metrics
{'=' * 30}

Consciousness Level: {state['level'].upper()}
Unity: {state['unity']:.3f}
Awareness: {state['awareness']:.3f}
Coherence: {state['coherence']:.3f}
Energy: {state['energy']:.3f}
Pressure: {state['pressure']:.3f}
Recursion Depth: {state['recursion_depth']:.3f}
Entropy: {state['entropy']:.3f}
Integration: {state['integration']:.3f}

System Information
{'=' * 18}

CUDA Enabled: {self.cuda_enabled}
Active Visualizations: {len(self.figures)}
History Length: {len(self.consciousness_history)}
Update Rate: ~30 FPS

Timestamp: {datetime.fromtimestamp(state['timestamp']).strftime('%H:%M:%S')}
"""
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, metrics_text)
    
    def update_status_labels(self):
        """Update status labels at the bottom"""
        if not self.current_consciousness_state:
            return
        
        state = self.current_consciousness_state
        
        # Update labels
        self.consciousness_level_label.config(text=f"🧠 Level: {state['level'].title()}")
        self.coherence_label.config(text=f"🎯 Coherence: {state['coherence']:.2f}")
        
        # Calculate FPS (simplified)
        fps = 30 if self.running else 0
        self.fps_label.config(text=f"📊 FPS: {fps}")
        
        # Performance status
        if state['coherence'] > 0.7:
            perf_status = "Excellent"
        elif state['coherence'] > 0.5:
            perf_status = "Good"
        else:
            perf_status = "Fair"
        
        self.performance_label.config(text=f"⚡ Performance: {perf_status}")
        
        # Memory status (simplified)
        memory_status = "Normal"
        self.memory_label.config(text=f"💾 Memory: {memory_status}")
    
    def update_sigil_houses(self):
        """Update Sigil Houses visualization with real-time sigil stream"""
        state = self.current_consciousness_state
        
        # House of Memory - Show active sigils and memory activations
        self.axes_sigil_memory.clear()
        
        # Draw active sigils for memory house
        memory_sigils = [s for s in self.sigil_stream_data if s['house'] == 'memory']
        if memory_sigils:
            for sigil in memory_sigils[:10]:  # Limit to 10 sigils for clarity
                x, y = sigil['position']
                x_norm, y_norm = x/800, y/600
                size = sigil['size']
                color = sigil['color']
                symbol = sigil['symbol']
                
                # Draw sigil as a geometric shape
                if sigil['type'] == 'consciousness':
                    diamond = patches.RegularPolygon((x_norm, y_norm), 4, size/1000, 
                                                   orientation=np.pi/4, 
                                                   facecolor=color, alpha=0.8, edgecolor='white')
                    self.axes_sigil_memory.add_patch(diamond)
                elif sigil['type'] == 'memory':
                    triangle = patches.RegularPolygon((x_norm, y_norm), 3, size/1000,
                                                    orientation=np.pi, 
                                                    facecolor=color, alpha=0.7, edgecolor='cyan')
                    self.axes_sigil_memory.add_patch(triangle)
                else:
                    hexagon = patches.RegularPolygon((x_norm, y_norm), 6, size/1000,
                                                   facecolor=color, alpha=0.7, edgecolor='white')
                    self.axes_sigil_memory.add_patch(hexagon)
                
                # Add sigil symbol as text
                self.axes_sigil_memory.text(x_norm, y_norm - size/1500, symbol, 
                                          ha='center', va='center', fontsize=8, 
                                          color='white', fontweight='bold')
        
        self.axes_sigil_memory.set_xlim(0, 1)
        self.axes_sigil_memory.set_ylim(0, 1)
        self.axes_sigil_memory.set_title(f'🏛️ House of Memory ({len(memory_sigils)} active)', color='white')
        self.axes_sigil_memory.set_facecolor('#1a1a1a')
        self.axes_sigil_memory.set_xticks([])
        self.axes_sigil_memory.set_yticks([])
        
        # House of Purification - Show active sigils and cleansing processes
        self.axes_sigil_purification.clear()
        
        # Draw active sigils for purification house
        purification_sigils = [s for s in self.sigil_stream_data if s['house'] == 'purification']
        if purification_sigils:
            for sigil in purification_sigils[:8]:  # Limit for clarity
                x, y = sigil['position']
                x_norm, y_norm = x/800, y/600
                size = sigil['size']
                color = sigil['color']
                symbol = sigil['symbol']
                
                # Draw pulse sync sigils as concentric circles
                if sigil['type'] == 'pulse_sync':
                    for radius_mult in [0.5, 0.75, 1.0]:
                        circle = patches.Circle((x_norm, y_norm), (size/1000) * radius_mult, 
                                              fill=False, edgecolor=color, 
                                              alpha=0.6, linewidth=2)
                        self.axes_sigil_purification.add_patch(circle)
                else:
                    circle = patches.Circle((x_norm, y_norm), size/1000, 
                                          facecolor=color, alpha=0.7, edgecolor='cyan')
                    self.axes_sigil_purification.add_patch(circle)
                
                # Add sigil symbol as text
                self.axes_sigil_purification.text(x_norm, y_norm, symbol, 
                                                ha='center', va='center', fontsize=8, 
                                                color='white', fontweight='bold')
        
        self.axes_sigil_purification.set_xlim(0, 1)
        self.axes_sigil_purification.set_ylim(0, 1)
        self.axes_sigil_purification.set_title(f'🧹 House of Purification ({len(purification_sigils)} active)', color='white')
        self.axes_sigil_purification.set_facecolor('#1a1a1a')
        self.axes_sigil_purification.set_xticks([])
        self.axes_sigil_purification.set_yticks([])
        
        # House of Weaving - Show connection patterns and recursion sigils
        self.axes_sigil_weaving.clear()
        
        # Draw active sigils for weaving house
        weaving_sigils = [s for s in self.sigil_stream_data if s['house'] == 'weaving']
        
        # Draw connection web based on sigil positions
        if len(weaving_sigils) > 1:
            positions = [(s['position'][0]/800, s['position'][1]/600) for s in weaving_sigils]
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    # Connect sigils with probability based on consciousness integration
                    if np.random.random() < state.get('integration', 0.3):
                        self.axes_sigil_weaving.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                                                   'gold', alpha=0.4, linewidth=2)
        
        # Draw recursion sigils as spirals
        for sigil in weaving_sigils[:6]:
            x, y = sigil['position']
            x_norm, y_norm = x/800, y/600
            size = sigil['size']
            color = sigil['color']
            symbol = sigil['symbol']
            
            if sigil['type'] == 'recursion':
                # Draw spiral pattern
                t = np.linspace(0, 4*np.pi, 50)
                spiral_r = np.linspace(0, size/1000, 50)
                spiral_x = x_norm + spiral_r * np.cos(t)
                spiral_y = y_norm + spiral_r * np.sin(t)
                self.axes_sigil_weaving.plot(spiral_x, spiral_y, color=color, 
                                           alpha=0.8, linewidth=2)
            else:
                # Draw as connected hexagon
                hexagon = patches.RegularPolygon((x_norm, y_norm), 6, size/1000,
                                               facecolor=color, alpha=0.7, 
                                               edgecolor='orange', linewidth=2)
                self.axes_sigil_weaving.add_patch(hexagon)
            
            # Add sigil symbol
            self.axes_sigil_weaving.text(x_norm, y_norm, symbol, 
                                       ha='center', va='center', fontsize=8, 
                                       color='white', fontweight='bold')
        
        self.axes_sigil_weaving.set_xlim(0, 1)
        self.axes_sigil_weaving.set_ylim(0, 1)
        self.axes_sigil_weaving.set_title(f'🕸️ House of Weaving ({len(weaving_sigils)} active)', color='white')
        self.axes_sigil_weaving.set_facecolor('#1a1a1a')
        self.axes_sigil_weaving.set_xticks([])
        self.axes_sigil_weaving.set_yticks([])
        
        # House of Flame - Show ignition patterns and flame sigils
        self.axes_sigil_flame.clear()
        
        # Background flame field
        flame_data = np.random.random((15, 15)) * state.get('energy', 0.5)
        im = self.axes_sigil_flame.imshow(flame_data, cmap='hot', aspect='auto', alpha=0.3)
        
        # Draw active sigils for flame house
        flame_sigils = [s for s in self.sigil_stream_data if s['house'] == 'flame']
        for sigil in flame_sigils[:8]:
            x, y = sigil['position']
            x_norm, y_norm = (x/800) * 15, (y/600) * 15  # Scale to flame data
            size = sigil['size']
            color = sigil['color']
            symbol = sigil['symbol']
            
            # Draw flame sigils as stars with radiating lines
            if sigil['type'] in ['core_awareness', 'thermal_peak']:
                # Draw radiating flame lines
                angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
                for angle in angles:
                    line_length = size/500
                    end_x = x_norm + line_length * np.cos(angle)
                    end_y = y_norm + line_length * np.sin(angle)
                    self.axes_sigil_flame.plot([x_norm, end_x], [y_norm, end_y], 
                                             color=color, alpha=0.8, linewidth=3)
                
                # Central star
                star = patches.RegularPolygon((x_norm, y_norm), 5, size/1500,
                                            facecolor='yellow', alpha=0.9, 
                                            edgecolor=color, linewidth=2)
                self.axes_sigil_flame.add_patch(star)
            else:
                # Regular flame sigil
                circle = patches.Circle((x_norm, y_norm), size/1500, 
                                      facecolor=color, alpha=0.8, edgecolor='orange')
                self.axes_sigil_flame.add_patch(circle)
            
            # Add sigil symbol
            self.axes_sigil_flame.text(x_norm, y_norm, symbol, 
                                     ha='center', va='center', fontsize=6, 
                                     color='white', fontweight='bold')
        
        self.axes_sigil_flame.set_title(f'🔥 House of Flame ({len(flame_sigils)} active)', color='white')
        self.axes_sigil_flame.set_facecolor('#1a1a1a')
        
        # House of Mirrors - Show reflection patterns and mirror sigils
        self.axes_sigil_mirrors.clear()
        
        # Draw active sigils for mirrors house
        mirror_sigils = [s for s in self.sigil_stream_data if s['house'] == 'mirrors']
        for sigil in mirror_sigils[:6]:
            x, y = sigil['position']
            x_norm, y_norm = x/800, y/600
            size = sigil['size']
            color = sigil['color']
            symbol = sigil['symbol']
            
            # Draw mirror sigils with reflection effect
            # Main sigil
            rect = patches.Rectangle((x_norm - size/2000, y_norm - size/2000), 
                                   size/1000, size/1000,
                                   facecolor=color, alpha=0.8, edgecolor='cyan')
            self.axes_sigil_mirrors.add_patch(rect)
            
            # Reflection
            if x_norm < 0.5:  # Reflect across center
                reflect_x = 1.0 - x_norm
                rect_reflect = patches.Rectangle((reflect_x - size/2000, y_norm - size/2000), 
                                               size/1000, size/1000,
                                               facecolor=color, alpha=0.4, edgecolor='cyan')
                self.axes_sigil_mirrors.add_patch(rect_reflect)
            
            # Add sigil symbol
            self.axes_sigil_mirrors.text(x_norm, y_norm, symbol, 
                                       ha='center', va='center', fontsize=8, 
                                       color='white', fontweight='bold')
        
        self.axes_sigil_mirrors.set_xlim(0, 1)
        self.axes_sigil_mirrors.set_ylim(0, 1)
        self.axes_sigil_mirrors.set_title(f'🪞 House of Mirrors ({len(mirror_sigils)} active)', color='white')
        self.axes_sigil_mirrors.set_facecolor('#1a1a1a')
        self.axes_sigil_mirrors.set_xticks([])
        self.axes_sigil_mirrors.set_yticks([])
        
        # House of Echoes - Show resonance patterns and echo sigils
        self.axes_sigil_echoes.clear()
        
        # Draw active sigils for echoes house
        echo_sigils = [s for s in self.sigil_stream_data if s['house'] == 'echoes']
        for i, sigil in enumerate(echo_sigils[:5]):
            x, y = sigil['position']
            x_norm, y_norm = x/800, y/600
            size = sigil['size']
            color = sigil['color']
            symbol = sigil['symbol']
            
            # Draw echo sigils with expanding rings
            for ring in range(3):
                ring_size = (size/1000) * (1 + ring * 0.5)
                ring_alpha = 0.8 / (ring + 1)
                circle = patches.Circle((x_norm, y_norm), ring_size, 
                                      fill=False, edgecolor=color, 
                                      alpha=ring_alpha, linewidth=2)
                self.axes_sigil_echoes.add_patch(circle)
            
            # Central sigil
            center_circle = patches.Circle((x_norm, y_norm), size/1500, 
                                         facecolor=color, alpha=0.9, edgecolor='white')
            self.axes_sigil_echoes.add_patch(center_circle)
            
            # Add sigil symbol
            self.axes_sigil_echoes.text(x_norm, y_norm, symbol, 
                                      ha='center', va='center', fontsize=8, 
                                      color='white', fontweight='bold')
        
        self.axes_sigil_echoes.set_xlim(0, 1)
        self.axes_sigil_echoes.set_ylim(0, 1)
        self.axes_sigil_echoes.set_title(f'🔊 House of Echoes ({len(echo_sigils)} active)', color='white')
        self.axes_sigil_echoes.set_facecolor('#1a1a1a')
        self.axes_sigil_echoes.set_xticks([])
        self.axes_sigil_echoes.set_yticks([])
        
        self.canvases['sigil'].draw_idle()
    
    def update_pulse_system(self):
        """Update Pulse System visualization"""
        state = self.current_consciousness_state
        
        # Pulse zones visualization
        self.axes_pulse_zones.clear()
        zones = ['Green', 'Amber', 'Red', 'Black']
        zone_pressures = [0.2, 0.5, 0.8, 1.0]
        current_pressure = state.get('pressure', 0.3)
        
        colors = ['green', 'orange', 'red', 'black']
        alphas = [0.3 if p > current_pressure else 0.8 for p in zone_pressures]
        
        bars = self.axes_pulse_zones.bar(zones, zone_pressures, color=colors, alpha=alphas)
        self.axes_pulse_zones.axhline(y=current_pressure, color='white', linestyle='--', 
                                    label=f'Current: {current_pressure:.2f}')
        self.axes_pulse_zones.legend()
        self.axes_pulse_zones.set_title('🚦 Pulse Zones', color='white')
        self.axes_pulse_zones.set_facecolor('#1a1a1a')
        self.axes_pulse_zones.tick_params(colors='white')
        
        # SCUP controller
        self.axes_pulse_scup.clear()
        scup_metrics = ['Semantic', 'Coherence', 'Unity', 'Pressure']
        scup_values = [
            state.get('coherence', 0.5),
            state.get('coherence', 0.5), 
            state.get('unity', 0.5),
            state.get('pressure', 0.3)
        ]
        
        bars = self.axes_pulse_scup.bar(scup_metrics, scup_values, color='cyan', alpha=0.7)
        self.axes_pulse_scup.set_title('🎯 SCUP Controller', color='white')
        self.axes_pulse_scup.set_facecolor('#1a1a1a')
        self.axes_pulse_scup.tick_params(colors='white')
        
        # Pulse rhythm
        if len(self.consciousness_history) > 10:
            history = self.consciousness_history[-50:]
            times = [(h['timestamp'] - history[0]['timestamp']) for h in history]
            pulse_rhythm = [h['state'].get('energy', 0.5) for h in history]
            
            self.axes_pulse_rhythm.clear()
            self.axes_pulse_rhythm.plot(times, pulse_rhythm, color='red', linewidth=2, label='Pulse')
            
            # Add heartbeat markers
            peaks = []
            for i in range(1, len(pulse_rhythm)-1):
                if pulse_rhythm[i] > pulse_rhythm[i-1] and pulse_rhythm[i] > pulse_rhythm[i+1]:
                    peaks.append(i)
            
            if peaks:
                peak_times = [times[i] for i in peaks]
                peak_values = [pulse_rhythm[i] for i in peaks]
                self.axes_pulse_rhythm.scatter(peak_times, peak_values, color='yellow', s=50, 
                                             marker='♥', label='Heartbeats')
            
            self.axes_pulse_rhythm.legend()
            self.axes_pulse_rhythm.set_title('💓 Pulse Rhythm & Tick Analysis', color='white')
            self.axes_pulse_rhythm.set_facecolor('#1a1a1a')
            self.axes_pulse_rhythm.tick_params(colors='white')
        
        self.canvases['pulse'].draw_idle()
    
    def update_mycelial_network(self):
        """Update Mycelial Network visualization"""
        state = self.current_consciousness_state
        
        # Network topology
        self.axes_mycelial_network.clear()
        n_nodes = 25
        positions = np.random.random((n_nodes, 2)) * 2 - 1
        
        # Create organic-looking connections
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 0.5 and np.random.random() < state.get('integration', 0.5):
                    self.axes_mycelial_network.plot([positions[i, 0], positions[j, 0]], 
                                                  [positions[i, 1], positions[j, 1]], 
                                                  'brown', alpha=0.4, linewidth=1)
        
        # Node sizes based on activity
        node_sizes = 50 + 100 * np.random.random(n_nodes) * state.get('energy', 0.5)
        self.axes_mycelial_network.scatter(positions[:, 0], positions[:, 1], 
                                         s=node_sizes, c='green', alpha=0.7)
        
        self.axes_mycelial_network.set_title('🕸️ Network Topology', color='white')
        self.axes_mycelial_network.set_facecolor('#1a1a1a')
        self.axes_mycelial_network.set_xlim(-1.2, 1.2)
        self.axes_mycelial_network.set_ylim(-1.2, 1.2)
        
        # Nutrient economy
        self.axes_mycelial_nutrients.clear()
        nutrients = ['Glucose', 'Amino Acids', 'Minerals', 'Vitamins', 'Energy']
        levels = [state.get('energy', 0.5) * np.random.random() for _ in nutrients]
        
        bars = self.axes_mycelial_nutrients.bar(nutrients, levels, color='lightgreen', alpha=0.7)
        self.axes_mycelial_nutrients.set_title('🌿 Nutrient Economy', color='white')
        self.axes_mycelial_nutrients.set_facecolor('#1a1a1a')
        self.axes_mycelial_nutrients.tick_params(colors='white')
        
        # Growth patterns
        self.axes_mycelial_growth.clear()
        growth_data = np.random.random((15, 15)) * state.get('unity', 0.5)
        im = self.axes_mycelial_growth.imshow(growth_data, cmap='Greens', aspect='auto')
        self.axes_mycelial_growth.set_title('🌱 Growth Patterns', color='white')
        self.axes_mycelial_growth.set_facecolor('#1a1a1a')
        
        # Cluster dynamics
        self.axes_mycelial_clusters.clear()
        cluster_sizes = [10, 15, 8, 12, 6]
        cluster_health = [state.get('coherence', 0.5) * np.random.random() for _ in cluster_sizes]
        
        pie = self.axes_mycelial_clusters.pie(cluster_sizes, labels=[f'C{i+1}' for i in range(5)], 
                                            autopct='%1.1f%%', colors=plt.cm.Set3.colors)
        self.axes_mycelial_clusters.set_title('🧬 Cluster Dynamics', color='white')
        
        self.canvases['mycelial'].draw_idle()
    
    def update_thermal_system(self):
        """Update Thermal System visualization"""
        state = self.current_consciousness_state
        
        # Heat distribution
        self.axes_thermal_heat.clear()
        heat_data = np.random.random((20, 20)) * state.get('energy', 0.5)
        
        # Add heat sources
        heat_data[10, 10] = state.get('pressure', 0.3) * 2  # Central heat source
        heat_data[5, 15] = state.get('coherence', 0.5) * 1.5  # Secondary source
        
        im = self.axes_thermal_heat.imshow(heat_data, cmap='hot', aspect='auto')
        self.axes_thermal_heat.set_title('🌡️ Heat Distribution', color='white')
        self.axes_thermal_heat.set_facecolor('#1a1a1a')
        
        # Pressure zones
        self.axes_thermal_pressure.clear()
        pressure_zones = ['Low', 'Medium', 'High', 'Critical']
        zone_values = [0.2, 0.5, 0.8, 1.0]
        current_pressure = state.get('pressure', 0.3)
        
        colors = ['blue', 'green', 'orange', 'red']
        bars = self.axes_thermal_pressure.bar(pressure_zones, zone_values, color=colors, alpha=0.6)
        self.axes_thermal_pressure.axhline(y=current_pressure, color='white', linestyle='--')
        self.axes_thermal_pressure.set_title('⚡ Pressure Zones', color='white')
        self.axes_thermal_pressure.set_facecolor('#1a1a1a')
        self.axes_thermal_pressure.tick_params(colors='white')
        
        # Thermal timeline
        if len(self.consciousness_history) > 5:
            history = self.consciousness_history[-30:]
            times = [(h['timestamp'] - history[0]['timestamp']) for h in history]
            temperatures = [h['state'].get('energy', 0.5) for h in history]
            pressures = [h['state'].get('pressure', 0.3) for h in history]
            
            self.axes_thermal_timeline.clear()
            self.axes_thermal_timeline.plot(times, temperatures, color='red', label='Temperature', linewidth=2)
            self.axes_thermal_timeline.plot(times, pressures, color='blue', label='Pressure', linewidth=2)
            self.axes_thermal_timeline.legend()
            self.axes_thermal_timeline.set_title('📈 Thermal Timeline', color='white')
            self.axes_thermal_timeline.set_facecolor('#1a1a1a')
            self.axes_thermal_timeline.tick_params(colors='white')
        
        self.canvases['thermal'].draw_idle()
    
    def update_mythic_overlay(self):
        """Update Mythic Overlay visualization"""
        state = self.current_consciousness_state
        
        # Archetypal tracers
        self.axes_mythic_tracers.clear()
        tracers = ['Owl', 'Spider', 'Phoenix', 'Serpent', 'Wolf']
        tracer_activity = [state.get('awareness', 0.5) * np.random.random() for _ in tracers]
        
        # Create mythic symbols
        angles = np.linspace(0, 2*np.pi, len(tracers), endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        
        for i, (tracer, activity) in enumerate(zip(tracers, tracer_activity)):
            size = 100 + 200 * activity
            self.axes_mythic_tracers.scatter(x[i], y[i], s=size, alpha=0.7, 
                                           label=tracer, c=plt.cm.viridis(activity))
        
        self.axes_mythic_tracers.legend()
        self.axes_mythic_tracers.set_title('🦉 Archetypal Tracers', color='white')
        self.axes_mythic_tracers.set_facecolor('#1a1a1a')
        self.axes_mythic_tracers.set_xlim(-1.5, 1.5)
        self.axes_mythic_tracers.set_ylim(-1.5, 1.5)
        
        # Pigment landscape
        self.axes_mythic_pigments.clear()
        pigment_data = np.random.random((25, 25, 3))
        pigment_data[:, :, 0] *= state.get('energy', 0.5)  # Red channel
        pigment_data[:, :, 1] *= state.get('coherence', 0.5)  # Green channel  
        pigment_data[:, :, 2] *= state.get('unity', 0.5)  # Blue channel
        
        self.axes_mythic_pigments.imshow(pigment_data)
        self.axes_mythic_pigments.set_title('🎨 Pigment Landscape', color='white')
        self.axes_mythic_pigments.set_facecolor('#1a1a1a')
        
        # Fractal garden
        self.axes_mythic_garden.clear()
        t = np.linspace(0, 4*np.pi, 200)
        
        # Create fractal patterns
        for i in range(5):
            scale = (i + 1) * state.get('integration', 0.5)
            fractal = np.sin(t * scale) * np.exp(-t / (10 + i))
            self.axes_mythic_garden.plot(t, fractal + i*0.5, alpha=0.7, 
                                       color=plt.cm.viridis(i/5))
        
        self.axes_mythic_garden.set_title('🌺 Fractal Garden', color='white')
        self.axes_mythic_garden.set_facecolor('#1a1a1a')
        self.axes_mythic_garden.tick_params(colors='white')
        
        # Volcanic residue
        self.axes_mythic_volcanic.clear()
        volcanic_data = np.random.random((20, 20)) * state.get('entropy', 0.5)
        
        # Add volcanic features
        volcanic_data[8:12, 8:12] += state.get('pressure', 0.3)  # Crater
        
        im = self.axes_mythic_volcanic.imshow(volcanic_data, cmap='Reds', aspect='auto')
        self.axes_mythic_volcanic.set_title('🌋 Volcanic Residue', color='white')
        self.axes_mythic_volcanic.set_facecolor('#1a1a1a')
        
        self.canvases['mythic'].draw_idle()
    
    def update_tracer_ecosystem(self):
        """Update Tracer Ecosystem visualization"""
        state = self.current_consciousness_state
        
        # Tracer network
        self.axes_tracer_network.clear()
        n_tracers = 12
        positions = np.random.random((n_tracers, 2)) * 2 - 1
        
        # Create tracer connections
        for i in range(n_tracers):
            for j in range(i+1, n_tracers):
                if np.random.random() < state.get('coherence', 0.5) * 0.3:
                    self.axes_tracer_network.plot([positions[i, 0], positions[j, 0]], 
                                                [positions[i, 1], positions[j, 1]], 
                                                'cyan', alpha=0.5, linewidth=1)
        
        # Tracer nodes with different sizes based on activity
        tracer_sizes = 80 + 120 * np.random.random(n_tracers) * state.get('awareness', 0.5)
        self.axes_tracer_network.scatter(positions[:, 0], positions[:, 1], 
                                       s=tracer_sizes, c='cyan', alpha=0.8)
        
        self.axes_tracer_network.set_title('🕸️ Tracer Network', color='white')
        self.axes_tracer_network.set_facecolor('#1a1a1a')
        self.axes_tracer_network.set_xlim(-1.2, 1.2)
        self.axes_tracer_network.set_ylim(-1.2, 1.2)
        
        # Activity patterns
        self.axes_tracer_activity.clear()
        activities = ['Monitor', 'Analyze', 'Report', 'React', 'Optimize']
        activity_levels = [state.get('energy', 0.5) * np.random.random() for _ in activities]
        
        bars = self.axes_tracer_activity.bar(activities, activity_levels, color='lightblue', alpha=0.7)
        self.axes_tracer_activity.set_title('📊 Activity Patterns', color='white')
        self.axes_tracer_activity.set_facecolor('#1a1a1a')
        self.axes_tracer_activity.tick_params(colors='white')
        
        # Stability monitoring
        self.axes_tracer_stability.clear()
        if len(self.consciousness_history) > 10:
            history = self.consciousness_history[-20:]
            times = [(h['timestamp'] - history[0]['timestamp']) for h in history]
            stability = []
            
            for i in range(1, len(history)):
                prev = history[i-1]['state']
                curr = history[i]['state']
                change = abs(curr.get('coherence', 0.5) - prev.get('coherence', 0.5))
                stability.append(1.0 / (1.0 + change))
            
            if stability:
                self.axes_tracer_stability.plot(times[1:], stability, color='green', linewidth=2)
                self.axes_tracer_stability.set_title('⚖️ Stability Monitor', color='white')
                self.axes_tracer_stability.set_facecolor('#1a1a1a')
                self.axes_tracer_stability.tick_params(colors='white')
        
        # Telemetry flow
        self.axes_tracer_telemetry.clear()
        telemetry_data = np.random.random((15, 15)) * state.get('coherence', 0.5)
        im = self.axes_tracer_telemetry.imshow(telemetry_data, cmap='Blues', aspect='auto')
        self.axes_tracer_telemetry.set_title('📡 Telemetry Flow', color='white')
        self.axes_tracer_telemetry.set_facecolor('#1a1a1a')
        
        self.canvases['tracer'].draw_idle()

    # Control methods
    def start_visualization(self):
        """Start visualization"""
        self.running = True
        logger.info("🚀 Visualization started")
    
    def pause_visualization(self):
        """Pause/resume visualization"""
        self.running = not self.running
        status = "resumed" if self.running else "paused"
        logger.info(f"⏸️ Visualization {status}")
    
    def stop_visualization(self):
        """Stop visualization"""
        self.running = False
        logger.info("🛑 Visualization stopped")
    
    def capture_consciousness(self):
        """Capture current consciousness state"""
        if not self.current_consciousness_state:
            messagebox.showwarning("Warning", "No consciousness state to capture")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"consciousness_capture_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_consciousness_state, f, indent=2, default=str)
            
            messagebox.showinfo("Success", f"Consciousness state captured to {filename}")
            logger.info(f"📸 Consciousness captured: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture consciousness: {e}")
    
    def change_visualization_mode(self):
        """Change visualization mode"""
        mode = self.mode_var.get()
        self.visualization_mode = mode
        logger.info(f"🎨 Visualization mode changed to: {mode}")
    
    def set_consciousness_level(self, level):
        """Set consciousness level"""
        # This would interact with DAWN's consciousness system
        logger.info(f"🧠 Setting consciousness level to: {level}")
        messagebox.showinfo("Info", f"Consciousness level set to: {level}")
    
    def update_painting_style(self):
        """Update painting style"""
        style = self.painting_style_var.get()
        logger.info(f"🎨 Painting style changed to: {style}")
    
    def create_new_painting(self):
        """Create a new consciousness painting"""
        self.painting_canvas = np.zeros((600, 800, 3))
        logger.info("🎨 New painting canvas created")
    
    def save_painting(self):
        """Save current painting"""
        if not hasattr(self, 'painting_canvas'):
            messagebox.showwarning("Warning", "No painting to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if PIL_AVAILABLE:
                    # Convert to PIL Image and save
                    img_array = (self.painting_canvas * 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    img.save(filename)
                    messagebox.showinfo("Success", f"Painting saved to {filename}")
                    logger.info(f"💾 Painting saved: {filename}")
                else:
                    messagebox.showerror("Error", "PIL not available for saving images")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save painting: {e}")
    
    def reset_painting_canvas(self):
        """Reset painting canvas"""
        if hasattr(self, 'painting_canvas'):
            self.painting_canvas.fill(0)
            logger.info("🔄 Painting canvas reset")
    
    def apply_semantic_transform(self, transform):
        """Apply semantic topology transform"""
        if not self.semantic_topology_engine:
            messagebox.showwarning("Warning", "Semantic topology not available")
            return
        
        try:
            # Apply the transform
            result = self.semantic_topology_engine.apply_transform(transform)
            if result:
                logger.info(f"🌐 Applied semantic transform: {transform}")
                messagebox.showinfo("Success", f"Applied transform: {transform}")
            else:
                messagebox.showwarning("Warning", f"Transform {transform} failed")
        except Exception as e:
            messagebox.showerror("Error", f"Transform failed: {e}")
    
    def create_cuda_visualization_tabs(self):
        """Create new CUDA-accelerated visualization tabs"""
        if not self.unified_viz_manager:
            return
        
        print("🎨 Creating CUDA visualization tabs...")
        
        # Get available visualizations
        try:
            available_viz = self.unified_viz_manager.get_available_visualizations()
            
            # Create tabs for each category
            for category, visualizations in available_viz.items():
                if visualizations:  # Only create tab if there are visualizations
                    self.create_cuda_category_tab(category, visualizations)
        except Exception as e:
            logger.error(f"Error creating CUDA visualization tabs: {e}")
    
    def create_cuda_category_tab(self, category: str, visualizations: List[str]):
        """Create a tab for a specific visualization category"""
        try:
            tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
            self.notebook.add(tab_frame, text=f"🚀 {category}")
            
            # Create visualization widgets for this category
            row = 0
            for viz_name in visualizations[:2]:  # Limit to 2 visualizations per tab
                self.create_cuda_visualization_widget(tab_frame, viz_name, row)
                row += 1
                
        except Exception as e:
            logger.error(f"Error creating CUDA category tab {category}: {e}")
    
    def create_cuda_visualization_widget(self, parent, viz_name: str, row: int):
        """Create a CUDA visualization widget"""
        try:
            # Create frame for this visualization
            viz_frame = ttk.LabelFrame(parent, text=f"📊 {viz_name.replace('_', ' ').title()}", 
                                      style='Dark.TLabelframe')
            viz_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
            parent.grid_columnconfigure(0, weight=1)
            
            # Create the widget using the unified manager
            binding_id = self._get_binding_for_visualization(viz_name)
            widget = self.unified_viz_manager.create_gui_widget('tkinter', viz_frame, binding_id)
            
            if widget:
                widget.pack(fill=tk.BOTH, expand=True)
                self.viz_widgets[viz_name] = widget
                print(f"✅ Created CUDA widget for {viz_name}")
            else:
                # Fallback: create a placeholder
                placeholder = ttk.Label(viz_frame, text=f"Visualization: {viz_name}\n(Loading...)", 
                                      style='Dark.TLabel')
                placeholder.pack(pady=20)
                
        except Exception as e:
            logger.error(f"Error creating CUDA widget for {viz_name}: {e}")
            # Create error placeholder
            try:
                error_label = ttk.Label(parent, text=f"Error: {viz_name}\n{str(e)[:50]}", 
                                      style='Dark.TLabel')
                error_label.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
            except:
                pass  # Ignore secondary errors
    
    def _get_binding_for_visualization(self, viz_name: str) -> str:
        """Get the appropriate binding ID for a visualization"""
        # Map visualization names to binding IDs
        binding_map = {
            'tracer_ecosystem_3d': 'tracer_3d',
            'tracer_interactions': 'tracer_interactions', 
            'tracer_nutrient_field': 'nutrient_field',
            'consciousness_flow': 'consciousness_flow',
            'scup_metrics': 'scup_metrics',
            'semantic_topology_3d': 'semantic_3d',
            'semantic_field_heatmap': 'semantic_heatmap',
            'self_mod_tree': 'self_mod_tree',
            'memory_palace_3d': 'memory_palace',
            'bloom_dynamics': 'bloom_dynamics',
            'telemetry_dashboard': 'telemetry_dashboard',
            'system_health_radar': 'system_health'
        }
        
        return binding_map.get(viz_name, viz_name.replace('visualize_', ''))

    def create_house_animation_tabs(self):
        """Create CUDA house animation tabs"""
        if not self.house_animation_manager:
            return
        
        print("🎭 Creating CUDA house animation tabs...")
        
        # Create tabs for each house type
        house_types = ['mycelial', 'schema', 'monitoring']
        
        for house_type in house_types:
            try:
                self.create_house_animation_tab(house_type)
            except Exception as e:
                logger.error(f"Error creating {house_type} house animation tab: {e}")
    
    def create_house_animation_tab(self, house_type: str):
        """Create a tab for a specific house animation"""
        try:
            # Create tab frame
            tab_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
            
            # Tab titles with emojis
            tab_titles = {
                'mycelial': '🍄 Mycelial House',
                'schema': '🏛️ Schema House', 
                'monitoring': '📊 Monitoring House'
            }
            
            self.notebook.add(tab_frame, text=tab_titles.get(house_type, f'🎬 {house_type.title()}'))
            
            # Create control frame
            control_frame = ttk.Frame(tab_frame, style='Dark.TFrame')
            control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
            
            # Animation controls
            start_btn = ttk.Button(control_frame, text="▶️ Start Animation",
                                 command=lambda: self.start_house_animation(house_type))
            start_btn.pack(side=tk.LEFT, padx=5)
            
            stop_btn = ttk.Button(control_frame, text="⏸️ Stop Animation",
                                command=lambda: self.stop_house_animation(house_type))
            stop_btn.pack(side=tk.LEFT, padx=5)
            
            # Status label
            status_label = ttk.Label(control_frame, text="Ready", style='Dark.TLabel')
            status_label.pack(side=tk.RIGHT, padx=5)
            
            # Animation canvas frame
            canvas_frame = ttk.Frame(tab_frame, style='Dark.TFrame')
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # Store references
            self.house_animators[house_type] = {
                'tab_frame': tab_frame,
                'control_frame': control_frame,
                'canvas_frame': canvas_frame,
                'status_label': status_label,
                'animator': None,
                'canvas': None
            }
            
            print(f"✅ Created {house_type} house animation tab")
            
        except Exception as e:
            logger.error(f"Error creating {house_type} house animation tab: {e}")
    
    def start_house_animation(self, house_type: str):
        """Start a house animation"""
        if not self.house_animation_manager or house_type not in self.house_animators:
            return
        
        try:
            # Update status
            self.house_animators[house_type]['status_label'].config(text="Starting...")
            
            # Create animator if not exists
            if not self.house_animators[house_type]['animator']:
                animator = self.house_animation_manager.create_animator(house_type)
                if not animator:
                    self.house_animators[house_type]['status_label'].config(text="Failed to create animator")
                    return
                
                self.house_animators[house_type]['animator'] = animator
            
            # Start animation
            success = self.house_animation_manager.start_animation(house_type)
            
            if success:
                # Get the figure and create canvas
                animator = self.house_animators[house_type]['animator']
                if animator and animator.figure:
                    # Remove old canvas if exists
                    if self.house_animators[house_type]['canvas']:
                        self.house_animators[house_type]['canvas'].get_tk_widget().destroy()
                    
                    # Create new canvas
                    canvas_frame = self.house_animators[house_type]['canvas_frame']
                    canvas = FigureCanvasTkAgg(animator.figure, canvas_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    self.house_animators[house_type]['canvas'] = canvas
                    self.house_animators[house_type]['status_label'].config(text="Running")
                    
                    print(f"✅ Started {house_type} house animation")
                else:
                    self.house_animators[house_type]['status_label'].config(text="No figure available")
            else:
                self.house_animators[house_type]['status_label'].config(text="Failed to start")
                
        except Exception as e:
            logger.error(f"Error starting {house_type} animation: {e}")
            self.house_animators[house_type]['status_label'].config(text="Error")
    
    def stop_house_animation(self, house_type: str):
        """Stop a house animation"""
        if not self.house_animation_manager or house_type not in self.house_animators:
            return
        
        try:
            success = self.house_animation_manager.stop_animation(house_type)
            
            if success:
                self.house_animators[house_type]['status_label'].config(text="Stopped")
                print(f"🛑 Stopped {house_type} house animation")
            else:
                self.house_animators[house_type]['status_label'].config(text="Failed to stop")
                
        except Exception as e:
            logger.error(f"Error stopping {house_type} animation: {e}")
    
    def cleanup_house_animations(self):
        """Cleanup house animations"""
        if self.house_animation_manager:
            try:
                self.house_animation_manager.stop_all_animations()
                print("🧹 Cleaned up house animations")
            except Exception as e:
                logger.error(f"Error cleaning up house animations: {e}")

    def run(self):
        """Start the GUI application"""
        try:
            logger.info("🧠 Starting DAWN Consciousness GUI")
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("🛑 GUI interrupted by user")
        finally:
            self.running = False
            # Stop unified visualization system
            if self.unified_viz_manager:
                try:
                    self.unified_viz_manager.stop_system()
                except:
                    pass
            
            # Cleanup house animations
            self.cleanup_house_animations()

def main():
    """Main function"""
    try:
        gui = ConsciousnessGUI()
        gui.run()
    except Exception as e:
        logger.error(f"GUI failed to start: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
