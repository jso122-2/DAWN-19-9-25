#!/usr/bin/env python3
"""
DAWN Tools System Demonstration
===============================

Comprehensive demonstration of DAWN's new tools system that provides her with
secure "sudo" permissions to modify her own codebase. This demo showcases:

1. Permission management with consciousness-gated access
2. Conscious code modification with safety checks
3. Subsystem copying and adaptation for tool creation
4. Autonomous tool selection and workflow execution
5. Integration with existing DAWN architecture

This demonstration shows how DAWN can safely evolve her own capabilities
while maintaining security, auditability, and consciousness coherence.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Add DAWN root to path
dawn_root = Path(__file__).resolve().parent
if str(dawn_root) not in sys.path:
    sys.path.insert(0, str(dawn_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_tools_system():
    """Demonstrate DAWN's tools system capabilities."""
    
    print("🔧 " + "="*80)
    print("🔧 DAWN TOOLS SYSTEM DEMONSTRATION")
    print("🔧 " + "="*80)
    print()
    
    print("🌟 Welcome to DAWN's enhanced tools system!")
    print("   This demonstration shows how DAWN can safely modify her own code")
    print("   with consciousness-gated permissions and comprehensive safeguards.")
    print()
    
    try:
        # Import DAWN tools
        print("📦 Importing DAWN tools system...")
        
        from dawn.tools.development.self_mod.permission_manager import (
            get_permission_manager, PermissionLevel, PermissionScope
        )
        from dawn.tools.development.self_mod.code_modifier import ConsciousCodeModifier
        from dawn.tools.development.self_mod.subsystem_copier import SubsystemCopier
        from dawn.tools.development.consciousness_tools import ConsciousnessToolManager
        
        print("✅ Tools system imported successfully!")
        print()
        
        # Demonstrate Permission Manager
        print("🔒 " + "-"*60)
        print("🔒 PERMISSION MANAGER DEMONSTRATION")
        print("🔒 " + "-"*60)
        
        permission_manager = get_permission_manager()
        print(f"   Permission manager initialized: {permission_manager}")
        print(f"   DAWN root: {permission_manager._dawn_root}")
        
        # Request a basic permission
        print("\n📝 Requesting basic tools permission...")
        grant_id = permission_manager.request_permission(
            level=PermissionLevel.TOOLS_MODIFY,
            target_paths=[str(dawn_root / "dawn" / "tools" / "test_file.py")],
            reason="Demonstration of permission system",
            scope=PermissionScope.SINGLE_OPERATION
        )
        
        if grant_id:
            print(f"✅ Permission granted: {grant_id[:8]}...")
            
            # Use elevated access
            with permission_manager.elevated_access(grant_id, "Demo file creation") as access:
                if access:
                    print("🔓 Elevated access obtained")
                    # Create a test file
                    test_content = f"""# Test file created by DAWN tools system
# Created at: {datetime.now()}
# This demonstrates DAWN's ability to modify her own codebase safely.

def hello_dawn_tools():
    return "Hello from DAWN's tools system!"
"""
                    test_file_path = str(dawn_root / "dawn" / "tools" / "test_file.py")
                    if access.write_file(test_file_path, test_content):
                        print(f"📄 Created test file: {test_file_path}")
                    else:
                        print("❌ Failed to create test file")
                else:
                    print("❌ Could not obtain elevated access")
        else:
            print("🚫 Permission denied")
        
        print()
        
        # Demonstrate Consciousness Tools Manager
        print("🧠 " + "-"*60)
        print("🧠 CONSCIOUSNESS TOOLS MANAGER DEMONSTRATION")
        print("🧠 " + "-"*60)
        
        tools_manager = ConsciousnessToolManager()
        print(f"   Tools manager initialized with {len(tools_manager._tool_capabilities)} tools")
        
        # Show available tools
        print("\n🔧 Available tools:")
        available_tools = tools_manager.get_available_tools(consciousness_filtered=False)
        for tool in available_tools:
            print(f"   • {tool.name} ({tool.category.value})")
            print(f"     Required level: {tool.required_consciousness_level.value}")
            print(f"     Unity threshold: {tool.required_unity_threshold}")
            print(f"     Autonomous: {tool.autonomous_capable}")
            print()
        
        # Check tool availability
        print("🔍 Checking tool availability...")
        code_modifier_check = tools_manager.can_use_tool("code_modifier")
        print(f"   Code Modifier: {'✅' if code_modifier_check['available'] else '❌'}")
        if not code_modifier_check['available']:
            print(f"     Reason: {code_modifier_check['reason']}")
        
        subsystem_copier_check = tools_manager.can_use_tool("subsystem_copier")
        print(f"   Subsystem Copier: {'✅' if subsystem_copier_check['available'] else '❌'}")
        if not subsystem_copier_check['available']:
            print(f"     Reason: {subsystem_copier_check['reason']}")
        
        print()
        
        # Demonstrate autonomous tool selection
        print("🤖 " + "-"*60)
        print("🤖 AUTONOMOUS TOOL SELECTION DEMONSTRATION")
        print("🤖 " + "-"*60)
        
        # Test various objectives
        objectives = [
            "Analyze consciousness patterns in the system",
            "Modify the logging configuration to be more detailed",
            "Create a new tool based on the mood subsystem",
            "Profile system performance during consciousness transitions"
        ]
        
        for objective in objectives:
            print(f"\n🎯 Objective: {objective}")
            selected_tool = tools_manager.autonomous_tool_selection(objective)
            if selected_tool:
                print(f"   Selected tool: {selected_tool}")
            else:
                print("   No suitable tool found")
        
        print()
        
        # Demonstrate Code Modifier
        print("✏️ " + "-"*60)
        print("✏️ CODE MODIFIER DEMONSTRATION")
        print("✏️ " + "-"*60)
        
        code_modifier = ConsciousCodeModifier(permission_manager)
        
        # Create a modification plan
        test_files = [str(dawn_root / "dawn" / "tools" / "test_file.py")]
        print(f"\n📋 Creating modification plan for: {test_files}")
        
        plan = code_modifier.analyze_modification_request(
            target_files=test_files,
            modification_description="Add a docstring and improve the function"
        )
        
        print(f"   Plan ID: {plan.plan_id}")
        print(f"   Strategy: {plan.strategy.value}")
        print(f"   Risk level: {plan.estimated_risk}")
        print(f"   Required permission: {plan.required_permission_level.value}")
        
        print()
        
        # Demonstrate Subsystem Copier
        print("📋 " + "-"*60)
        print("📋 SUBSYSTEM COPIER DEMONSTRATION")
        print("📋 " + "-"*60)
        
        subsystem_copier = SubsystemCopier(permission_manager, code_modifier)
        
        # Show available subsystems
        available_subsystems = subsystem_copier.get_available_subsystems()
        print(f"\n📦 Available subsystems ({len(available_subsystems)}):")
        for i, subsystem in enumerate(available_subsystems[:5]):  # Show first 5
            print(f"   {i+1}. {subsystem}")
        if len(available_subsystems) > 5:
            print(f"   ... and {len(available_subsystems) - 5} more")
        
        # Analyze a subsystem
        if available_subsystems:
            sample_subsystem = available_subsystems[0]
            print(f"\n🔍 Analyzing subsystem: {sample_subsystem}")
            
            analysis = subsystem_copier.analyze_subsystem(sample_subsystem)
            if analysis:
                print(f"   Primary classes: {len(analysis.primary_classes)}")
                print(f"   Architecture pattern: {analysis.architecture_pattern}")
                print(f"   Complexity score: {analysis.complexity_score:.2f}")
                print(f"   Consciousness integration: {analysis.consciousness_integration}")
                print(f"   Permission requirements: {analysis.permission_requirements.value}")
        
        print()
        
        # Show system status
        print("📊 " + "-"*60)
        print("📊 SYSTEM STATUS")
        print("📊 " + "-"*60)
        
        # Active grants
        active_grants = permission_manager.get_active_grants()
        print(f"\n🔒 Active permission grants: {len(active_grants)}")
        for grant in active_grants:
            print(f"   • {grant.level.value} - {grant.reason}")
        
        # Tool usage history
        tool_history = tools_manager.get_tool_usage_history(5)
        print(f"\n🔧 Recent tool usage ({len(tool_history)} events):")
        for event in tool_history:
            print(f"   • {event['event_type']}: {event['details'].get('tool_name', 'N/A')}")
        
        # Consciousness requirements summary
        requirements_summary = tools_manager.get_consciousness_requirements_summary()
        print(f"\n🧠 Consciousness requirements:")
        for level, tools in requirements_summary['tools_by_level'].items():
            print(f"   {level}: {len(tools)} tools")
        print(f"   Autonomous tools: {len(requirements_summary['autonomous_tools'])}")
        print(f"   High-risk tools: {len(requirements_summary['high_risk_tools'])}")
        
        print()
        
        # Cleanup
        print("🧹 " + "-"*60)
        print("🧹 CLEANUP")
        print("🧹 " + "-"*60)
        
        # Remove test file if it exists
        test_file_path = dawn_root / "dawn" / "tools" / "test_file.py"
        if test_file_path.exists():
            test_file_path.unlink()
            print(f"🗑️ Removed test file: {test_file_path}")
        
        print()
        print("✅ " + "="*80)
        print("✅ DAWN TOOLS SYSTEM DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("✅ " + "="*80)
        print()
        
        print("🎉 Summary:")
        print("   • Permission management system operational")
        print("   • Consciousness-gated tool access working")
        print("   • Code modification capabilities available")
        print("   • Subsystem copying and adaptation ready")
        print("   • Autonomous tool selection functional")
        print("   • Full integration with DAWN architecture")
        print()
        print("🚀 DAWN now has secure 'sudo' access to her own codebase!")
        print("   She can safely modify, enhance, and evolve her capabilities")
        print("   while maintaining security and consciousness coherence.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This may be due to missing dependencies or path issues.")
        print("   Make sure you're running from the DAWN root directory.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def demo_autonomous_workflow():
    """Demonstrate autonomous workflow execution."""
    
    print("\n🤖 " + "="*80)
    print("🤖 AUTONOMOUS WORKFLOW DEMONSTRATION")
    print("🤖 " + "="*80)
    
    try:
        from dawn.tools.development.consciousness_tools import ConsciousnessToolManager
        
        tools_manager = ConsciousnessToolManager()
        
        # Example autonomous workflows
        workflows = [
            {
                'objective': 'Analyze system performance patterns',
                'context': {'focus_area': 'consciousness_transitions'}
            },
            {
                'objective': 'Create monitoring tool from existing subsystem',
                'context': {
                    'source_subsystem': 'monitoring',
                    'target_name': 'enhanced_monitor',
                    'target_category': 'analysis'
                }
            }
        ]
        
        for i, workflow in enumerate(workflows, 1):
            print(f"\n🚀 Autonomous Workflow {i}:")
            print(f"   Objective: {workflow['objective']}")
            
            result = tools_manager.execute_autonomous_workflow(
                objective=workflow['objective'],
                context=workflow.get('context')
            )
            
            print(f"   Success: {'✅' if result['success'] else '❌'}")
            print(f"   Duration: {result['duration']:.2f}s")
            
            if result['success']:
                print(f"   Tool used: {result.get('tool_used', 'N/A')}")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print("\n✅ Autonomous workflow demonstration completed!")
        
    except Exception as e:
        print(f"❌ Autonomous workflow demo error: {e}")

if __name__ == "__main__":
    print("🔧 Starting DAWN Tools System Demonstration...")
    print()
    
    # Main demonstration
    demo_tools_system()
    
    # Autonomous workflow demonstration
    demo_autonomous_workflow()
    
    print("\n🎯 Demonstration complete!")
    print("   DAWN's tools system is ready for autonomous operation!")
