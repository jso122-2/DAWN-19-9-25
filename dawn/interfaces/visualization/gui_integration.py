#!/usr/bin/env python3
"""
🚀 DAWN GUI Visualization Integration
====================================

Integration layer between DAWN's CUDA matplotlib visualization engine and
GUI systems. Provides seamless embedding of visualizations in Tkinter,
Qt, and other GUI frameworks.

Features:
- Tkinter canvas integration
- Qt widget integration
- Real-time visualization updates
- Interactive controls
- Multi-threaded rendering
- DAWN singleton integration

"Bridging consciousness visualization and user interfaces."
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import uuid
import queue

# DAWN core imports
from dawn.core.singleton import get_dawn
from .cuda_matplotlib_engine import get_cuda_matplotlib_engine, VisualizationConfig

logger = logging.getLogger(__name__)

# GUI framework imports
TKINTER_AVAILABLE = False
QT_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from tkinter.scrolledtext import ScrolledText
    TKINTER_AVAILABLE = True
    logger.info("✅ Tkinter available for GUI integration")
except ImportError:
    logger.debug("Tkinter not available")

try:
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
    from PyQt5.QtCore import QTimer, pyqtSignal
    QT_AVAILABLE = True
    logger.info("✅ Qt5 available for GUI integration")
except ImportError:
    try:
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        from PyQt6.QtCore import QTimer, pyqtSignal
        QT_AVAILABLE = True
        logger.info("✅ Qt6 available for GUI integration")
    except ImportError:
        logger.debug("Qt not available")

# Matplotlib GUI backends
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    MATPLOTLIB_TKINTER_AVAILABLE = True
except ImportError:
    MATPLOTLIB_TKINTER_AVAILABLE = False
    logger.debug("Matplotlib Tkinter backend not available")

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    MATPLOTLIB_QT_AVAILABLE = True
except ImportError:
    try:
        from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt6agg import NavigationToolbar2QT as NavigationToolbar
        MATPLOTLIB_QT_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_QT_AVAILABLE = False
        logger.debug("Matplotlib Qt backend not available")


@dataclass
class GUIVisualizationConfig:
    """Configuration for GUI-embedded visualizations"""
    update_interval_ms: int = 100  # Update interval in milliseconds
    enable_toolbar: bool = True
    enable_interactive_controls: bool = True
    auto_refresh: bool = True
    max_history: int = 100
    enable_export: bool = True
    default_size: Tuple[int, int] = (800, 600)


class DAWNVisualizationWidget:
    """
    Base class for DAWN visualization widgets that can be embedded in GUI frameworks.
    Provides common functionality for all GUI integrations.
    """
    
    def __init__(self, viz_name: str, config: Optional[GUIVisualizationConfig] = None):
        self.viz_name = viz_name
        self.config = config or GUIVisualizationConfig()
        self.widget_id = str(uuid.uuid4())
        
        # DAWN integration
        self.dawn = get_dawn()
        self.viz_engine = get_cuda_matplotlib_engine()
        
        # State management
        self.data_history: List[Dict[str, Any]] = []
        self.current_data: Optional[Dict[str, Any]] = None
        self.update_callback: Optional[Callable] = None
        
        # Threading
        self.update_thread: Optional[threading.Thread] = None
        self.update_running = False
        self._lock = threading.RLock()
        
        logger.info(f"🎨 DAWN Visualization Widget created: {viz_name} ({self.widget_id})")
    
    def set_data(self, data: Dict[str, Any]):
        """Set data for visualization"""
        with self._lock:
            self.current_data = data
            self.data_history.append(data)
            
            # Limit history size
            if len(self.data_history) > self.config.max_history:
                self.data_history.pop(0)
    
    def get_current_data(self) -> Optional[Dict[str, Any]]:
        """Get current visualization data"""
        with self._lock:
            return self.current_data
    
    def start_auto_update(self, data_source_callback: Optional[Callable] = None):
        """Start automatic data updates"""
        if self.update_running:
            return
        
        self.update_running = True
        self.update_thread = threading.Thread(
            target=self._auto_update_loop,
            args=(data_source_callback,),
            name=f"viz_update_{self.widget_id}",
            daemon=True
        )
        self.update_thread.start()
        logger.info(f"🔄 Started auto-update for {self.viz_name}")
    
    def stop_auto_update(self):
        """Stop automatic data updates"""
        if not self.update_running:
            return
        
        self.update_running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
        logger.info(f"🛑 Stopped auto-update for {self.viz_name}")
    
    def _auto_update_loop(self, data_source_callback: Optional[Callable]):
        """Auto-update loop running in background thread"""
        while self.update_running:
            try:
                # Get new data if callback provided
                if data_source_callback:
                    try:
                        new_data = data_source_callback()
                        if new_data:
                            self.set_data(new_data)
                            
                            # Trigger update if callback set
                            if self.update_callback:
                                self.update_callback()
                    except Exception as e:
                        logger.error(f"Error in data source callback: {e}")
                
                # Sleep for update interval
                time.sleep(self.config.update_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in auto-update loop: {e}")
                time.sleep(1.0)
    
    def export_visualization(self, filename: str, format: str = 'png'):
        """Export current visualization to file"""
        if not self.current_data:
            logger.warning("No data to export")
            return False
        
        try:
            # Create figure
            fig = self.viz_engine.create_figure(f"{self.viz_name}_export")
            
            # Generate visualization
            viz_func = self.viz_engine.visualization_registry.get(self.viz_name)
            if viz_func:
                viz_func(self.current_data, fig)
                
                # Save figure
                fig.savefig(filename, format=format, dpi=150, bbox_inches='tight')
                logger.info(f"📁 Exported visualization to {filename}")
                return True
            else:
                logger.error(f"Visualization function not found: {self.viz_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export visualization: {e}")
            return False


class TkinterVisualizationWidget(DAWNVisualizationWidget):
    """
    Tkinter-specific visualization widget.
    Embeds DAWN visualizations in Tkinter applications.
    """
    
    def __init__(self, parent_widget, viz_name: str, config: Optional[GUIVisualizationConfig] = None):
        super().__init__(viz_name, config)
        
        if not TKINTER_AVAILABLE or not MATPLOTLIB_TKINTER_AVAILABLE:
            raise RuntimeError("Tkinter or Matplotlib Tkinter backend not available")
        
        self.parent = parent_widget
        self.frame = None
        self.canvas = None
        self.toolbar = None
        self.figure = None
        
        self._create_widget()
        
        # Set up update callback for GUI thread
        self.update_callback = self._schedule_gui_update
    
    def _create_widget(self):
        """Create the Tkinter widget structure"""
        # Create main frame
        self.frame = ttk.Frame(self.parent)
        
        # Create matplotlib figure
        self.figure = self.viz_engine.create_figure(f"{self.viz_name}_tkinter")
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.draw()
        
        # Pack canvas
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Create toolbar if enabled
        if self.config.enable_toolbar:
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
            self.toolbar.update()
        
        # Create control frame if interactive controls enabled
        if self.config.enable_interactive_controls:
            self._create_controls()
        
        logger.info(f"✅ Created Tkinter visualization widget for {self.viz_name}")
    
    def _create_controls(self):
        """Create interactive control widgets"""
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # Refresh button
        refresh_btn = ttk.Button(control_frame, text="🔄 Refresh", command=self.refresh)
        refresh_btn.pack(side=tk.LEFT, padx=2)
        
        # Export button
        if self.config.enable_export:
            export_btn = ttk.Button(control_frame, text="📁 Export", command=self._export_dialog)
            export_btn.pack(side=tk.LEFT, padx=2)
        
        # Auto-refresh toggle
        if self.config.auto_refresh:
            self.auto_refresh_var = tk.BooleanVar(value=True)
            auto_refresh_cb = ttk.Checkbutton(
                control_frame, 
                text="Auto Refresh", 
                variable=self.auto_refresh_var,
                command=self._toggle_auto_refresh
            )
            auto_refresh_cb.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side=tk.RIGHT, padx=5)
    
    def _schedule_gui_update(self):
        """Schedule GUI update in main thread"""
        if self.parent and hasattr(self.parent, 'after'):
            self.parent.after(0, self.refresh)
    
    def refresh(self):
        """Refresh the visualization"""
        try:
            if not self.current_data:
                return
            
            # Clear figure
            self.figure.clear()
            
            # Get visualization function
            viz_func = self.viz_engine.visualization_registry.get(self.viz_name)
            if viz_func:
                # Generate visualization
                viz_func(self.current_data, self.figure)
                
                # Redraw canvas
                self.canvas.draw()
                
                # Update status
                if hasattr(self, 'status_label'):
                    self.status_label.config(text=f"Updated: {time.strftime('%H:%M:%S')}")
            else:
                logger.error(f"Visualization function not found: {self.viz_name}")
                
        except Exception as e:
            logger.error(f"Error refreshing visualization: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Error")
    
    def _export_dialog(self):
        """Show export dialog"""
        try:
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                title="Export Visualization",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            
            if filename:
                format = filename.split('.')[-1].lower()
                if self.export_visualization(filename, format):
                    messagebox.showinfo("Export", f"Visualization exported to {filename}")
                else:
                    messagebox.showerror("Export Error", "Failed to export visualization")
                    
        except Exception as e:
            logger.error(f"Error in export dialog: {e}")
            messagebox.showerror("Export Error", str(e))
    
    def _toggle_auto_refresh(self):
        """Toggle auto-refresh functionality"""
        if hasattr(self, 'auto_refresh_var'):
            if self.auto_refresh_var.get():
                self.start_auto_update()
            else:
                self.stop_auto_update()
    
    def get_widget(self) -> ttk.Frame:
        """Get the main widget frame"""
        return self.frame
    
    def pack(self, **kwargs):
        """Pack the widget"""
        self.frame.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid the widget"""
        self.frame.grid(**kwargs)


class QtVisualizationWidget(DAWNVisualizationWidget):
    """
    Qt-specific visualization widget.
    Embeds DAWN visualizations in Qt applications.
    """
    
    def __init__(self, parent_widget=None, viz_name: str = "", config: Optional[GUIVisualizationConfig] = None):
        super().__init__(viz_name, config)
        
        if not QT_AVAILABLE or not MATPLOTLIB_QT_AVAILABLE:
            raise RuntimeError("Qt or Matplotlib Qt backend not available")
        
        self.parent = parent_widget
        self.widget = None
        self.canvas = None
        self.toolbar = None
        self.figure = None
        self.timer = None
        
        self._create_widget()
        
        # Set up update callback
        self.update_callback = self._schedule_gui_update
    
    def _create_widget(self):
        """Create the Qt widget structure"""
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox
        from PyQt5.QtCore import QTimer
        
        # Create main widget
        self.widget = QWidget(self.parent)
        layout = QVBoxLayout(self.widget)
        
        # Create matplotlib figure
        self.figure = self.viz_engine.create_figure(f"{self.viz_name}_qt")
        
        # Create canvas
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Create toolbar if enabled
        if self.config.enable_toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self.widget)
            layout.addWidget(self.toolbar)
        
        # Create control layout if interactive controls enabled
        if self.config.enable_interactive_controls:
            control_layout = QHBoxLayout()
            
            # Refresh button
            refresh_btn = QPushButton("🔄 Refresh")
            refresh_btn.clicked.connect(self.refresh)
            control_layout.addWidget(refresh_btn)
            
            # Export button
            if self.config.enable_export:
                export_btn = QPushButton("📁 Export")
                export_btn.clicked.connect(self._export_dialog)
                control_layout.addWidget(export_btn)
            
            # Auto-refresh checkbox
            if self.config.auto_refresh:
                self.auto_refresh_cb = QCheckBox("Auto Refresh")
                self.auto_refresh_cb.setChecked(True)
                self.auto_refresh_cb.stateChanged.connect(self._toggle_auto_refresh)
                control_layout.addWidget(self.auto_refresh_cb)
            
            # Status label
            self.status_label = QLabel("Ready")
            control_layout.addWidget(self.status_label)
            
            control_layout.addStretch()
            layout.addLayout(control_layout)
        
        # Set up timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        
        logger.info(f"✅ Created Qt visualization widget for {self.viz_name}")
    
    def _schedule_gui_update(self):
        """Schedule GUI update using Qt timer"""
        if self.timer:
            self.timer.singleShot(0, self.refresh)
    
    def refresh(self):
        """Refresh the visualization"""
        try:
            if not self.current_data:
                return
            
            # Clear figure
            self.figure.clear()
            
            # Get visualization function
            viz_func = self.viz_engine.visualization_registry.get(self.viz_name)
            if viz_func:
                # Generate visualization
                viz_func(self.current_data, self.figure)
                
                # Redraw canvas
                self.canvas.draw()
                
                # Update status
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"Updated: {time.strftime('%H:%M:%S')}")
            else:
                logger.error(f"Visualization function not found: {self.viz_name}")
                
        except Exception as e:
            logger.error(f"Error refreshing Qt visualization: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.setText("Error")
    
    def _export_dialog(self):
        """Show Qt export dialog"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self.widget,
                "Export Visualization",
                "",
                "PNG files (*.png);;PDF files (*.pdf);;SVG files (*.svg);;All files (*.*)"
            )
            
            if filename:
                format = filename.split('.')[-1].lower()
                if self.export_visualization(filename, format):
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(self.widget, "Export", f"Visualization exported to {filename}")
                else:
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.critical(self.widget, "Export Error", "Failed to export visualization")
                    
        except Exception as e:
            logger.error(f"Error in Qt export dialog: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self.widget, "Export Error", str(e))
    
    def _toggle_auto_refresh(self):
        """Toggle auto-refresh functionality"""
        if hasattr(self, 'auto_refresh_cb'):
            if self.auto_refresh_cb.isChecked():
                self.start_auto_update()
            else:
                self.stop_auto_update()
    
    def get_widget(self) -> 'QWidget':
        """Get the main Qt widget"""
        return self.widget


class VisualizationGUIManager:
    """
    Manager for GUI-embedded visualizations across different frameworks.
    Provides unified interface for creating and managing visualization widgets.
    """
    
    def __init__(self):
        self.manager_id = str(uuid.uuid4())
        self.active_widgets: Dict[str, DAWNVisualizationWidget] = {}
        self.dawn = get_dawn()
        self.viz_engine = get_cuda_matplotlib_engine()
        
        logger.info(f"🎨 Visualization GUI Manager initialized: {self.manager_id}")
    
    def create_tkinter_widget(self, parent, viz_name: str, 
                             config: Optional[GUIVisualizationConfig] = None) -> Optional[TkinterVisualizationWidget]:
        """Create a Tkinter visualization widget"""
        if not TKINTER_AVAILABLE or not MATPLOTLIB_TKINTER_AVAILABLE:
            logger.error("Tkinter visualization not available")
            return None
        
        try:
            widget = TkinterVisualizationWidget(parent, viz_name, config)
            widget_id = f"tk_{viz_name}_{widget.widget_id}"
            self.active_widgets[widget_id] = widget
            
            logger.info(f"✅ Created Tkinter widget: {widget_id}")
            return widget
            
        except Exception as e:
            logger.error(f"Failed to create Tkinter widget: {e}")
            return None
    
    def create_qt_widget(self, parent=None, viz_name: str = "", 
                        config: Optional[GUIVisualizationConfig] = None) -> Optional[QtVisualizationWidget]:
        """Create a Qt visualization widget"""
        if not QT_AVAILABLE or not MATPLOTLIB_QT_AVAILABLE:
            logger.error("Qt visualization not available")
            return None
        
        try:
            widget = QtVisualizationWidget(parent, viz_name, config)
            widget_id = f"qt_{viz_name}_{widget.widget_id}"
            self.active_widgets[widget_id] = widget
            
            logger.info(f"✅ Created Qt widget: {widget_id}")
            return widget
            
        except Exception as e:
            logger.error(f"Failed to create Qt widget: {e}")
            return None
    
    def get_available_visualizations(self) -> List[str]:
        """Get list of available visualizations"""
        return self.viz_engine.get_available_visualizations()
    
    def update_all_widgets(self, subsystem: str, data: Dict[str, Any]):
        """Update all widgets for a specific subsystem"""
        for widget_id, widget in self.active_widgets.items():
            if subsystem in widget.viz_name or widget.viz_name in subsystem:
                widget.set_data(data)
    
    def cleanup_widgets(self):
        """Cleanup all active widgets"""
        for widget_id, widget in self.active_widgets.items():
            try:
                widget.stop_auto_update()
            except Exception as e:
                logger.error(f"Error stopping widget {widget_id}: {e}")
        
        self.active_widgets.clear()
        logger.info("🧹 Cleaned up all visualization widgets")
    
    def get_manager_summary(self) -> Dict[str, Any]:
        """Get manager status summary"""
        return {
            'manager_id': self.manager_id,
            'active_widgets': len(self.active_widgets),
            'widget_types': {
                'tkinter': len([w for w in self.active_widgets.keys() if w.startswith('tk_')]),
                'qt': len([w for w in self.active_widgets.keys() if w.startswith('qt_')])
            },
            'available_visualizations': len(self.get_available_visualizations()),
            'capabilities': {
                'tkinter_available': TKINTER_AVAILABLE and MATPLOTLIB_TKINTER_AVAILABLE,
                'qt_available': QT_AVAILABLE and MATPLOTLIB_QT_AVAILABLE
            }
        }


# Global manager instance
_global_gui_manager: Optional[VisualizationGUIManager] = None
_manager_lock = threading.Lock()


def get_visualization_gui_manager() -> VisualizationGUIManager:
    """Get the global visualization GUI manager instance"""
    global _global_gui_manager
    
    with _manager_lock:
        if _global_gui_manager is None:
            _global_gui_manager = VisualizationGUIManager()
    
    return _global_gui_manager


def create_visualization_widget(framework: str, parent, viz_name: str, 
                               config: Optional[GUIVisualizationConfig] = None):
    """
    Convenience function to create visualization widget for any framework.
    
    Args:
        framework: 'tkinter' or 'qt'
        parent: Parent widget
        viz_name: Name of visualization to create
        config: Optional configuration
        
    Returns:
        Visualization widget or None if failed
    """
    manager = get_visualization_gui_manager()
    
    if framework.lower() == 'tkinter':
        return manager.create_tkinter_widget(parent, viz_name, config)
    elif framework.lower() == 'qt':
        return manager.create_qt_widget(parent, viz_name, config)
    else:
        logger.error(f"Unknown GUI framework: {framework}")
        return None


if __name__ == "__main__":
    # Demo the GUI integration
    logging.basicConfig(level=logging.INFO)
    
    print("🚀" * 40)
    print("🧠 DAWN GUI VISUALIZATION INTEGRATION DEMO")
    print("🚀" * 40)
    
    # Get manager
    manager = get_visualization_gui_manager()
    
    # Show summary
    summary = manager.get_manager_summary()
    print(f"✅ Manager Summary: {summary}")
    
    # Test Tkinter if available
    if TKINTER_AVAILABLE and MATPLOTLIB_TKINTER_AVAILABLE:
        print("\n🎨 Testing Tkinter integration...")
        
        # Create simple Tkinter app
        root = tk.Tk()
        root.title("DAWN Visualization Demo")
        root.geometry("1000x700")
        
        # Create visualization widget
        widget = manager.create_tkinter_widget(root, 'consciousness_flow')
        
        if widget:
            widget.pack(fill=tk.BOTH, expand=True)
            
            # Set test data
            test_data = {
                'consciousness_history': [
                    {'coherence': 0.7, 'unity': 0.6, 'pressure': 0.4},
                    {'coherence': 0.8, 'unity': 0.7, 'pressure': 0.5},
                    {'coherence': 0.6, 'unity': 0.5, 'pressure': 0.6}
                ],
                'current_state': {
                    'coherence': 0.8, 'unity': 0.7, 'pressure': 0.5,
                    'entropy': 0.3, 'awareness': 0.9, 'integration': 0.8
                }
            }
            
            widget.set_data(test_data)
            widget.refresh()
            
            print("✅ Created Tkinter visualization widget")
            print("   Close the window to continue...")
            
            try:
                root.mainloop()
            except KeyboardInterrupt:
                pass
        
        root.destroy()
    
    else:
        print("⚠️  Tkinter not available - skipping Tkinter demo")
    
    # Cleanup
    manager.cleanup_widgets()
    
    print("\n🚀 GUI Integration demo complete!")
