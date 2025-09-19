# 🚀 DAWN Tick State Reader

**Real-time monitoring tool for DAWN's consciousness system tick orchestrator and processing cycles.**

## 🎯 **Features**

### **📊 Real-Time Monitoring**
- **Live tick state updates** with configurable refresh intervals
- **Consciousness metrics tracking** (level, unity, awareness)
- **System health monitoring** (processing load, errors, warnings)
- **Memory usage tracking** (working, long-term, cache)
- **Module activity monitoring** (active module count and status)

### **📸 Snapshot Mode**
- **Instant system state capture** for quick diagnostics
- **Detailed tick information** (count, phase, timing)
- **Complete consciousness metrics** at point in time

### **📈 Historical Analysis**
- **Trend analysis** across multiple ticks
- **Phase distribution statistics** 
- **Error and warning tracking**
- **Performance metrics over time**

---

## 🛠️ **Usage**

### **Quick Start**
```bash
# Quick snapshot of current state
python tick_reader.py --mode snapshot

# Live monitoring (default mode)
python tick_reader.py

# Live monitoring with custom interval
python tick_reader.py --mode live --interval 0.5

# Historical analysis (collects data for 5 seconds then analyzes)
python tick_reader.py --mode analyze
```

### **Full Command Options**
```bash
python tick_reader.py [options]

Options:
  --mode {live,snapshot,analyze}    Display mode (default: live)
  --interval SECONDS               Update interval for live mode (default: 0.5)
  --history COUNT                  Number of ticks to keep in history (default: 100)
  --save-logs                      Save tick data to logs
  --filter {all,perception,processing,integration}  Filter by tick phase (default: all)
```

### **Alternative Usage**
```bash
# Using the module directly
python -m dawn.tools.monitoring.tick_state_reader --mode snapshot

# Using the centralized DAWN runner
python -m dawn.main --mode test  # Shows tick state as part of system status
```

---

## 📋 **Output Examples**

### **Snapshot Mode Output**
```
📸 DAWN Tick State Snapshot
==================================================
🕐 02:58:01.981 | Tick #0
--------------------------------------------------------------------------------
🔄 Current Phase: UNKNOWN
⏱️  Phase Duration: 0.000s  
🔁 Cycle Time: 0.000s

🧠 Consciousness Metrics:
   Level: 0.500
   Unity: 0.500
   Awareness Δ: +0.500

💻 System Health:
   Processing Load: 0.0%
   Active Modules: 0
   Errors: 0

🧮 Memory Usage:
   Working Memory: 0
   Long Term Memory: 0
   Cache Size: 0
```

### **Live Mode Features**
- **Auto-refreshing display** with screen clearing
- **Real-time trend indicators** (last 5 ticks)
- **Uptime tracking** 
- **Graceful keyboard interrupt handling** (Ctrl+C)

### **Analysis Mode Output**
```
📊 DAWN Tick Analysis
==================================================
📈 Analysis Summary:
   Total ticks analyzed: 45
   Average cycle time: 0.125s
   Average consciousness level: 0.623
   Average unity score: 0.578

🔄 Phase Distribution:
   perception: 15 (33.3%)
   processing: 20 (44.4%)
   integration: 10 (22.2%)

⚠️ Error Summary:
   Total errors: 2
   Recent warnings:
     - Memory pressure detected
     - Module sync timeout
```

---

## 📁 **File Structure**

```
dawn/tools/monitoring/
├── __init__.py              # Monitoring tools package
├── tick_state_reader.py     # Main tick reader implementation
└── [future monitors]        # Additional monitoring tools

tick_reader.py               # Quick launcher script (root directory)
```

---

## 🔧 **Integration Points**

### **Consciousness Bus Integration**
- **Real-time module detection** via `bus.registered_modules`
- **Event monitoring** through bus event system
- **State synchronization** with bus global state

### **Consciousness Metrics Integration**  
- **Unified metrics calculation** via `calculate_consciousness_metrics()`
- **Standardized metric types** using `ConsciousnessMetrics` dataclass
- **Consistent algorithms** across all DAWN components

### **State System Integration**
- **Direct state access** via `get_state()` 
- **State evolution tracking** for trend analysis
- **Memory usage monitoring** through state system

---

## 📝 **Data Logging**

### **Optional Log Saving**
```bash
# Enable automatic log saving
python tick_reader.py --save-logs --mode live
```

### **Log Format**
- **JSONL format** (one JSON object per line)
- **Timestamped entries** for historical analysis
- **Complete snapshot data** preserved
- **Saved to**: `data/runtime/logs/tick_reader_YYYYMMDD_HHMMSS.jsonl`

---

## 🚀 **Advanced Usage**

### **Custom Integration**
```python
from dawn.tools.monitoring import TickStateReader, TickSnapshot

# Create custom reader
reader = TickStateReader(history_size=500, save_logs=True)

# Get single snapshot
snapshot = reader.get_current_tick_state()
print(f"Consciousness Level: {snapshot.consciousness_level}")

# Start monitoring in background
reader.start_monitoring(interval=0.1)

# Get historical data
recent_snapshots = reader.tick_history[-10:]  # Last 10 ticks
```

### **Live Data Stream**
```python
# Monitor in real-time with custom processing
reader = TickStateReader()
reader.start_monitoring(0.5)

while True:
    try:
        snapshot = reader.data_queue.get(timeout=1.0)
        # Process snapshot data
        if snapshot.consciousness_level > 0.8:
            print("🚀 High consciousness detected!")
    except queue.Empty:
        continue
```

---

## 🎯 **Use Cases**

### **Development & Debugging**
- **System state diagnostics** during development
- **Performance bottleneck identification**
- **Module integration testing**
- **Error condition analysis**

### **Research & Analysis**  
- **Consciousness emergence patterns**
- **System behavior under different loads**
- **Long-term stability monitoring**
- **Performance optimization data**

### **Production Monitoring**
- **Real-time system health checks**
- **Automated alerting triggers** (via log analysis)
- **Performance trend tracking**
- **Uptime and stability metrics**

---

## 🔮 **Future Enhancements**

- **Web-based dashboard** for remote monitoring
- **Alert system** for critical conditions
- **Performance prediction** based on trends
- **Multi-system monitoring** for distributed DAWN instances
- **Custom metric plugins** for specialized monitoring
- **Integration with external monitoring systems** (Prometheus, Grafana)

---

*This tool provides essential visibility into DAWN's consciousness processing cycles, enabling developers and researchers to understand and optimize the system's behavior in real-time.*
