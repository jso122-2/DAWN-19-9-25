#!/usr/bin/env python3
"""
DAWN Bug Tracking System
========================

Automatically tracks and lists bugs found in the DAWN system.
Provides comprehensive bug analysis, categorization, and tracking.
"""

import json
import time
import pathlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class BugSeverity(Enum):
    """Bug severity levels."""
    CRITICAL = "critical"      # System cannot function
    HIGH = "high"             # Major functionality broken
    MEDIUM = "medium"         # Some functionality impaired
    LOW = "low"              # Minor issues, cosmetic
    INFO = "info"            # Informational, not really a bug

class BugStatus(Enum):
    """Bug status tracking."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    FIXED = "fixed"
    VERIFIED = "verified"
    CLOSED = "closed"
    WONT_FIX = "wont_fix"

class BugCategory(Enum):
    """Bug categories."""
    SYNTAX_ERROR = "syntax_error"
    VALIDATION_ERROR = "validation_error"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    SAFETY_VIOLATION = "safety_violation"
    DATA_FORMAT = "data_format"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    OTHER = "other"

@dataclass
class Bug:
    """Represents a single bug/issue."""
    bug_id: str
    title: str
    description: str
    severity: BugSeverity
    category: BugCategory
    status: BugStatus = BugStatus.OPEN
    
    # Context information
    component: str = ""
    file_path: str = ""
    line_number: Optional[int] = None
    error_message: str = ""
    stack_trace: str = ""
    
    # Tracking information
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    fixed_at: Optional[datetime] = None
    reporter: str = "automated"
    assignee: str = ""
    
    # Additional context
    reproduction_steps: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    actual_behavior: str = ""
    workaround: str = ""
    fix_description: str = ""
    related_bugs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bug to dictionary."""
        return {
            'bug_id': self.bug_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'category': self.category.value,
            'status': self.status.value,
            'component': self.component,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'fixed_at': self.fixed_at.isoformat() if self.fixed_at else None,
            'reporter': self.reporter,
            'assignee': self.assignee,
            'reproduction_steps': self.reproduction_steps,
            'expected_behavior': self.expected_behavior,
            'actual_behavior': self.actual_behavior,
            'workaround': self.workaround,
            'fix_description': self.fix_description,
            'related_bugs': self.related_bugs,
            'tags': self.tags
        }

class BugTracker:
    """Comprehensive bug tracking system."""
    
    def __init__(self, bugs_dir: str = "bugs"):
        self.bugs_dir = pathlib.Path(bugs_dir)
        self.bugs_dir.mkdir(exist_ok=True)
        
        self.bugs: Dict[str, Bug] = {}
        self.load_existing_bugs()
        
        logger.info(f"🐛 Bug Tracker initialized: {len(self.bugs)} bugs loaded")
    
    def load_existing_bugs(self):
        """Load existing bugs from disk."""
        bug_files = list(self.bugs_dir.glob("bug_*.json"))
        for bug_file in bug_files:
            try:
                with open(bug_file, 'r') as f:
                    bug_data = json.load(f)
                
                bug = Bug(
                    bug_id=bug_data['bug_id'],
                    title=bug_data['title'],
                    description=bug_data['description'],
                    severity=BugSeverity(bug_data['severity']),
                    category=BugCategory(bug_data['category']),
                    status=BugStatus(bug_data['status']),
                    component=bug_data.get('component', ''),
                    file_path=bug_data.get('file_path', ''),
                    line_number=bug_data.get('line_number'),
                    error_message=bug_data.get('error_message', ''),
                    stack_trace=bug_data.get('stack_trace', ''),
                    created_at=datetime.fromisoformat(bug_data['created_at']),
                    updated_at=datetime.fromisoformat(bug_data['updated_at']),
                    fixed_at=datetime.fromisoformat(bug_data['fixed_at']) if bug_data.get('fixed_at') else None,
                    reporter=bug_data.get('reporter', 'automated'),
                    assignee=bug_data.get('assignee', ''),
                    reproduction_steps=bug_data.get('reproduction_steps', []),
                    expected_behavior=bug_data.get('expected_behavior', ''),
                    actual_behavior=bug_data.get('actual_behavior', ''),
                    workaround=bug_data.get('workaround', ''),
                    fix_description=bug_data.get('fix_description', ''),
                    related_bugs=bug_data.get('related_bugs', []),
                    tags=bug_data.get('tags', [])
                )
                
                self.bugs[bug.bug_id] = bug
                
            except Exception as e:
                logger.warning(f"Failed to load bug file {bug_file}: {e}")
    
    def report_bug(self, title: str, description: str, severity: BugSeverity, 
                   category: BugCategory, **kwargs) -> Bug:
        """Report a new bug."""
        bug_id = f"BUG-{int(time.time())}-{len(self.bugs):03d}"
        
        bug = Bug(
            bug_id=bug_id,
            title=title,
            description=description,
            severity=severity,
            category=category,
            **kwargs
        )
        
        self.bugs[bug_id] = bug
        self.save_bug(bug)
        
        logger.info(f"🐛 New bug reported: {bug_id} - {title}")
        return bug
    
    def update_bug(self, bug_id: str, **updates) -> Optional[Bug]:
        """Update an existing bug."""
        if bug_id not in self.bugs:
            return None
        
        bug = self.bugs[bug_id]
        
        for key, value in updates.items():
            if hasattr(bug, key):
                setattr(bug, key, value)
        
        bug.updated_at = datetime.now()
        self.save_bug(bug)
        
        logger.info(f"🐛 Bug updated: {bug_id}")
        return bug
    
    def fix_bug(self, bug_id: str, fix_description: str = "", **kwargs) -> Optional[Bug]:
        """Mark a bug as fixed."""
        bug = self.update_bug(bug_id, 
                             status=BugStatus.FIXED, 
                             fixed_at=datetime.now(),
                             fix_description=fix_description,
                             **kwargs)
        
        if bug:
            logger.info(f"🐛 ✅ Bug fixed: {bug_id}")
        
        return bug
    
    def save_bug(self, bug: Bug):
        """Save bug to disk."""
        bug_file = self.bugs_dir / f"bug_{bug.bug_id}.json"
        with open(bug_file, 'w') as f:
            json.dump(bug.to_dict(), f, indent=2)
    
    def list_bugs(self, status: Optional[BugStatus] = None, 
                  severity: Optional[BugSeverity] = None,
                  category: Optional[BugCategory] = None) -> List[Bug]:
        """List bugs with optional filtering."""
        bugs = list(self.bugs.values())
        
        if status:
            bugs = [b for b in bugs if b.status == status]
        if severity:
            bugs = [b for b in bugs if b.severity == severity]
        if category:
            bugs = [b for b in bugs if b.category == category]
        
        return sorted(bugs, key=lambda b: b.created_at, reverse=True)
    
    def get_bug_summary(self) -> Dict[str, Any]:
        """Get comprehensive bug summary."""
        total_bugs = len(self.bugs)
        
        by_status = {}
        by_severity = {}
        by_category = {}
        
        for bug in self.bugs.values():
            by_status[bug.status.value] = by_status.get(bug.status.value, 0) + 1
            by_severity[bug.severity.value] = by_severity.get(bug.severity.value, 0) + 1
            by_category[bug.category.value] = by_category.get(bug.category.value, 0) + 1
        
        open_critical = len([b for b in self.bugs.values() 
                           if b.status == BugStatus.OPEN and b.severity == BugSeverity.CRITICAL])
        
        return {
            'total_bugs': total_bugs,
            'open_critical': open_critical,
            'by_status': by_status,
            'by_severity': by_severity,
            'by_category': by_category,
            'last_updated': datetime.now().isoformat()
        }
    
    def print_bug_report(self, show_fixed: bool = False):
        """Print a comprehensive bug report."""
        summary = self.get_bug_summary()
        
        print("\n" + "="*80)
        print("🐛 DAWN BUG TRACKER REPORT")
        print("="*80)
        
        print(f"\n📊 Summary:")
        print(f"   Total Bugs: {summary['total_bugs']}")
        print(f"   Open Critical: {summary['open_critical']}")
        
        print(f"\n📈 By Status:")
        for status, count in summary['by_status'].items():
            print(f"   {status.replace('_', ' ').title()}: {count}")
        
        print(f"\n🔥 By Severity:")
        for severity, count in summary['by_severity'].items():
            print(f"   {severity.title()}: {count}")
        
        print(f"\n📂 By Category:")
        for category, count in summary['by_category'].items():
            print(f"   {category.replace('_', ' ').title()}: {count}")
        
        # List open bugs
        open_bugs = self.list_bugs(status=BugStatus.OPEN)
        if open_bugs:
            print(f"\n🚨 Open Bugs ({len(open_bugs)}):")
            for bug in open_bugs[:10]:  # Show top 10
                severity_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "info": "🔵"}
                icon = severity_icon.get(bug.severity.value, "⚪")
                print(f"   {icon} {bug.bug_id}: {bug.title}")
                if bug.component:
                    print(f"      Component: {bug.component}")
                if bug.error_message:
                    print(f"      Error: {bug.error_message[:100]}...")
        
        # List recently fixed bugs
        if show_fixed:
            fixed_bugs = [b for b in self.bugs.values() if b.status == BugStatus.FIXED]
            fixed_bugs.sort(key=lambda b: b.fixed_at or datetime.min, reverse=True)
            
            if fixed_bugs:
                print(f"\n✅ Recently Fixed Bugs ({len(fixed_bugs)}):")
                for bug in fixed_bugs[:5]:  # Show top 5
                    print(f"   ✅ {bug.bug_id}: {bug.title}")
                    if bug.fix_description:
                        print(f"      Fix: {bug.fix_description[:100]}...")
        
        print("\n" + "="*80)

def demo_bug_tracker():
    """Demonstrate bug tracker functionality."""
    tracker = BugTracker()
    
    # Report some example bugs
    tracker.report_bug(
        title="IndentationError in sandbox code generation",
        description="Empty try block in generated Python code causes IndentationError",
        severity=BugSeverity.HIGH,
        category=BugCategory.SYNTAX_ERROR,
        component="sandbox_runner",
        file_path="dawn/subsystems/self_mod/sandbox_runner.py",
        line_number=288,
        error_message="IndentationError: expected an indented block after 'try' statement",
        expected_behavior="Generated code should be syntactically valid",
        actual_behavior="Generated code has empty try block causing syntax error",
        reproduction_steps=[
            "Run recursive modification demo",
            "Observe subprocess failure with exit code 1",
            "Check error logs for IndentationError"
        ]
    )
    
    tracker.report_bug(
        title="Missing required field: end_awareness in policy gate",
        description="Policy gate validation fails due to missing end_awareness field in sandbox results",
        severity=BugSeverity.MEDIUM,
        category=BugCategory.VALIDATION_ERROR,
        component="policy_gate",
        file_path="dawn/subsystems/self_mod/policy_gate.py",
        line_number=533,
        error_message="Missing required field: end_awareness",
        expected_behavior="All required fields should be present in sandbox results",
        actual_behavior="end_awareness field missing from result dictionary"
    )
    
    # Show the report
    tracker.print_bug_report(show_fixed=True)

if __name__ == "__main__":
    demo_bug_tracker()
