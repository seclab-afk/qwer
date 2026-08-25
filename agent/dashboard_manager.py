# dashboard_manager.py
# Dashboard for real-time monitoring of harness generation process by category

import asyncio
import time
import os
import datetime
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
import json
from zoneinfo import ZoneInfo
from logger_config import disable_all_console_logging, enable_console_logging

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns
from rich import box

# Import parse_fuzzer_stats function from module.py
from module import parse_fuzzer_stats


class HarnessStage(Enum):
    """Each stage of the harness generation process"""
    WAITING = "Waiting"
    FLOW = "Flow Gen"
    GENERATING = "Harness Gen"
    BUILDING = "AFL Build"
    FUZZ_SHORT = "Short Fuzz"
    MEASURING = "Coverage"
    LLM_JUDGEMENT = "LLM Judge"
    FUZZ_LONG = "Long Fuzz"
    DOCKER_CLEANUP = "Cleanup"
    COMPLETED = "Completed"
    FAILED = "Failed"
    FIX = "FIX"


@dataclass
class CategoryStatus:
    """Status information per category"""
    category: str
    api_count: int
    current_stage: HarnessStage = HarnessStage.WAITING
    start_time: float = field(default_factory=time.time)
    stage_start_time: float = field(default_factory=time.time)
    progress: float = 0.0  # 0.0 ~ 1.0
    error_message: str = ""
    coverage: float = 0.0
    fuzz_duration: int = 0
    is_long_fuzz: bool = False
    fuzz_stats: Dict[str, Any] = field(default_factory=dict)  # fuzzer_stats info
    fix_retry_count: int = 0  # fix_harness retry count
    generate_retry_count: int = 0  # generate_harness retry count
    
    def get_elapsed_time(self) -> float:
        """Return total elapsed time"""
        return time.time() - self.start_time
    
    def get_stage_elapsed_time(self) -> float:
        """Return current stage elapsed time"""
        return time.time() - self.stage_start_time
    
    def update_stage(self, stage: HarnessStage, progress: float = 0.0, error: str = ""):
        """Update stage"""
        self.current_stage = stage
        self.stage_start_time = time.time()
        self.progress = progress
        self.error_message = error
    
    def update_fuzz_stats(self, stats: Dict[str, Any]):
        """Update fuzzing statistics"""
        self.fuzz_stats = stats

    def reset(self):
        """Reset category status"""
        self.category = ""
        self.api_count = 0
        self.current_stage = HarnessStage.WAITING
        self.start_time = time.time()
        self.stage_start_time = time.time()
        self.progress = 0.0
        self.error_message = ""
        self.coverage = 0.0
        self.fuzz_duration = 0
        self.is_long_fuzz = False
        self.fuzz_stats = {}
        self.fix_retry_count = 0
        self.generate_retry_count = 0


class DashboardManager:
    """Dashboard manager for real-time monitoring of harness generation process"""
    
    def __init__(self):
        self.console = Console()
        self.categories: Dict[str, CategoryStatus] = {}
        self.live: Optional[Live] = None
        self.total_categories = 0
        self.start_time = time.time()
        self._disabled_console_handlers = []  # Store disabled console handlers
        self.current_library = ""  # Currently processing library name
        self._running = False  # Dashboard running state (default: stopped)
        # Summary info (overall execution perspective)
        self.current_run_index: int = 0
        self.current_total_coverage: float = 0.0
        self.coverage_threshold: float = 0.0
        # Token usage tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.token_call_count: int = 0
        self._token_lock = Lock()
        
    def initialize_categories(self, categorized_apis: Dict[str, List[str]], library: str = "", run_index: int = None, coverage_threshold: float = None, total_coverage: float = None):
        """Initialize categories (clear previous execution's category list and reconstruct)"""

        self.categories.clear()
        self.total_categories = len(categorized_apis)
        self.current_library = library
        # Apply summary field initial values (optional)
        if run_index is not None:
            self.current_run_index = run_index
        if coverage_threshold is not None:
            self.coverage_threshold = coverage_threshold
        if total_coverage is not None:
            self.current_total_coverage = total_coverage
        for category, api_list in categorized_apis.items():
            self.categories[category] = CategoryStatus(
                category=category,
                api_count=len(api_list)
            )
    
    def update_category_stage(self, category: str, stage: HarnessStage, progress: float = 0.0, error: str = ""):
        """Update category stage"""
        if category in self.categories:
            self.categories[category].update_stage(stage, progress, error)

    # === Overall execution summary info update ===
    def update_run_index(self, run_index: int):
        self.current_run_index = run_index

    def update_total_coverage(self, total_coverage: float):
        self.current_total_coverage = total_coverage

    def update_coverage_threshold(self, threshold: float):
        self.coverage_threshold = threshold

    def update_token_usage(self, in_tokens: int, out_tokens: int, token_file: str = ""):
        """Accumulate token counts and optionally persist to JSON."""
        with self._token_lock:
            self.total_input_tokens  += in_tokens
            self.total_output_tokens += out_tokens
            self.token_call_count    += 1
            if token_file:
                self._persist_token_usage(token_file, in_tokens, out_tokens)

    def _persist_token_usage(self, token_file: str, last_in: int, last_out: int):
        """Write current cumulative token usage to JSON (for restart recovery)."""
        try:
            os.makedirs(os.path.dirname(token_file) or ".", exist_ok=True)
            data = {
                "total_input_tokens":  self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens":        self.total_input_tokens + self.total_output_tokens,
                "call_count":          self.token_call_count,
                "last_call_input_tokens":  last_in,
                "last_call_output_tokens": last_out,
                "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(token_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def load_token_snapshot(self, token_file: str):
        """Restore token counts from JSON file if it exists (called on startup)."""
        try:
            if os.path.exists(token_file):
                with open(token_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                with self._token_lock:
                    self.total_input_tokens  = data.get("total_input_tokens", 0)
                    self.total_output_tokens = data.get("total_output_tokens", 0)
                    self.token_call_count    = data.get("call_count", 0)
        except Exception:
            pass
    
    def update_category_coverage(self, category: str, coverage: float):
        """Update category coverage"""
        if category in self.categories:
            self.categories[category].coverage = coverage
    
    def update_category_fuzz_info(self, category: str, duration: int, is_long_fuzz: bool = False):
        """Update fuzzing info"""
        if category in self.categories:
            self.categories[category].fuzz_duration = duration
            self.categories[category].is_long_fuzz = is_long_fuzz
    
    def update_fix_retry_count(self, category: str, count: int):
        """Update fix_harness retry count"""
        if category in self.categories:
            self.categories[category].fix_retry_count = count
    
    def update_generate_retry_count(self, category: str, count: int):
        """Update generate_harness retry count"""
        if category in self.categories:
            self.categories[category].generate_retry_count = count
    
    def reset_categories(self):
        """Reset only category status (keep DashboardManager summary fields)."""
        for status in self.categories.values():
            status.reset()


    
    def get_overall_progress(self) -> float:
        """Calculate overall progress"""
        if not self.categories:
            return 0.0
        
        stage_weights = {
            HarnessStage.WAITING: 0.0,
            HarnessStage.FLOW: 0.1,
            HarnessStage.GENERATING: 0.2,
            HarnessStage.BUILDING: 0.3,
            HarnessStage.FUZZ_SHORT: 0.4,
            HarnessStage.MEASURING: 0.55,
            HarnessStage.LLM_JUDGEMENT: 0.7,
            HarnessStage.FUZZ_LONG: 0.85,
            HarnessStage.DOCKER_CLEANUP: 0.9,
            HarnessStage.COMPLETED: 1.0,
            HarnessStage.FAILED: 0.0,
            HarnessStage.FIX: 0.05
        }
        
        total_progress = 0.0
        for status in self.categories.values():
            stage_progress = stage_weights.get(status.current_stage, 0.0)
            total_progress += stage_progress
        
        return total_progress / len(self.categories) if self.categories else 0.0
    
    def get_stage_color(self, stage: HarnessStage) -> str:
        """Return color for each stage"""
        colors = {
            HarnessStage.WAITING: "grey62",
            HarnessStage.FLOW: "bright_cyan",
            HarnessStage.GENERATING: "dodger_blue1",
            HarnessStage.BUILDING: "gold1",
            HarnessStage.FUZZ_SHORT: "medium_orchid1",
            HarnessStage.MEASURING: "cyan2",
            HarnessStage.LLM_JUDGEMENT: "light_slate_blue",
            HarnessStage.FUZZ_LONG: "dark_orange",
            HarnessStage.DOCKER_CLEANUP: "spring_green2",
            HarnessStage.COMPLETED: "green1",
            HarnessStage.FAILED: "red1",
            HarnessStage.FIX: "purple3",
        }
        return colors.get(stage, "white")
    
    def create_summary_panel(self) -> Panel:
        """Create summary panel"""
        elapsed_time = time.time() - self.start_time

        summary_text = f"""
📚 Library: {self.current_library}
⏱️  Elapsed Time: {elapsed_time:.0f}s
📁 Total Categories: {self.total_categories}
🧪 Current Run Index: {self.current_run_index}
📈 Current Total Coverage: {self.current_total_coverage:.2f}% (Target {self.coverage_threshold:.2f}%)
🧠 Token Usage:  IN={self.total_input_tokens:,}  OUT={self.total_output_tokens:,}  (calls={self.token_call_count})
"""
        
        return Panel(
            summary_text.strip(),
            title="[bold bright_blue]💤 AFK 💤[/bold bright_blue]",
            border_style="bright_blue",
            box=box.ROUNDED
        )
    
    def truncate_category_name(self, category: str, max_width: int = 25) -> str:
        """Truncate category name to appropriate length"""
        if len(category) <= max_width:
            return category
        
        # Apply common abbreviation rules
        abbreviations = {
            "Management": "Mgmt",
            "Processing": "Proc",
            "Handling": "Handle",
            "Functions": "Funcs",
            "Operations": "Ops",
            "Information": "Info",
            "Initialization": "Init",
            "Configuration": "Config",
            "Validation": "Valid",
            "Transformation": "Transform",
            "and ": "& ",
            " and ": " & "
        }
        
        shortened = category
        for full, abbr in abbreviations.items():
            shortened = shortened.replace(full, abbr)
        
        # If still long, truncate
        if len(shortened) > max_width:
            shortened = shortened[:max_width-3] + "..."
        
        return shortened
    
    def get_optimal_category_width(self) -> int:
        """Calculate optimal column width based on category names"""
        if not self.categories:
            return 20
        
        max_width = 25  # Maximum width limit
        min_width = 15  # Minimum width guarantee
        
        # Calculate max length of shortened names
        max_length = max(
            len(self.truncate_category_name(cat)) 
            for cat in self.categories.keys()
        )
        
        # Adjust within min/max range
        optimal_width = max(min_width, min(max_length + 2, max_width))
        return optimal_width
    
    def create_categories_table(self) -> Table:
        """Create category status table"""
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED, title="📊 Category Progress")
        
        # Dynamically set category column width
        category_width = self.get_optimal_category_width()
        table.add_column("Category", style="cyan", width=category_width)
        table.add_column("APIs", justify="center", width=8)
        table.add_column("Stage", width=12)
        table.add_column("Elapsed", width=10)
        table.add_column("Coverage", width=10)
        table.add_column("Fix Retry", justify="center", width=10)
        table.add_column("Gen Retry", justify="center", width=10)
        table.add_column("Status", width=20)
        
        for status in sorted(self.categories.values(), key=lambda x: x.category):
            stage_color = self.get_stage_color(status.current_stage)
            elapsed = status.get_elapsed_time()
            
            # Truncate category name
            display_category = self.truncate_category_name(status.category)
            
            # Coverage display
            coverage_text = f"{status.coverage:.1f}%" if status.coverage > 0 else "-"
            
            # Status message - show based on current stage
            status_msg = ""
            if status.error_message:
                status_msg = f"[red]{status.error_message[:30]}[/red]"
            elif status.current_stage == HarnessStage.FUZZ_SHORT:
                status_msg = f"[magenta]Short Fuzzing ({status.fuzz_duration}s)[/magenta]"
            elif status.current_stage == HarnessStage.FUZZ_LONG:
                status_msg = f"[dark_orange]Long Fuzzing ({status.fuzz_duration}s)[/dark_orange]"
            elif status.current_stage == HarnessStage.FAILED:
                status_msg = f"[red1]Failed[/red1]"
            elif status.current_stage == HarnessStage.COMPLETED:
                status_msg = f"[green]Completed[/green]"
            else:
                status_msg = f"[{stage_color}]Running[/{stage_color}]"
            
            table.add_row(
                display_category,
                str(status.api_count),
                f"[{stage_color}]{status.current_stage.value}[/{stage_color}]",
                f"{elapsed:.0f}s",
                coverage_text,
                str(status.fix_retry_count) if status.fix_retry_count > 0 else "0",
                str(status.generate_retry_count) if status.generate_retry_count > 0 else "0",
                status_msg
            )
        
        return table
    
    def create_fuzz_stats_table(self) -> Table:
        """Create fuzzing statistics table"""
        # Check categories currently fuzzing
        fuzzing_categories = [
            status for status in self.categories.values()
            if status.current_stage in [HarnessStage.FUZZ_SHORT, HarnessStage.FUZZ_LONG]
        ]
        
        if not fuzzing_categories:
            # Return empty table if no categories are fuzzing
            table = Table(show_header=True, header_style="bold green", box=box.ROUNDED, title="🔍 Fuzzing Stats (No categories currently fuzzing)")
            table.add_column("Status", style="dim", width=50)
            table.add_row("[dim]No categories are currently fuzzing.[/dim]")
            return table
        
        table = Table(show_header=True, header_style="bold green", box=box.ROUNDED, title="🔍 Fuzzing Stats")
        
        # Dynamically set category column width
        category_width = self.get_optimal_category_width()
        table.add_column("Category", style="cyan", width=category_width)
        table.add_column("Queue", justify="right", width=8)
        table.add_column("Hangs", justify="right", width=8)
        table.add_column("Crashes", justify="right", width=10)
        table.add_column("Stability", justify="right", width=10)
        table.add_column("Bitmap Cov", justify="right", width=12)
        
        for status in sorted(fuzzing_categories, key=lambda x: x.category):
            # Truncate category name
            display_category = self.truncate_category_name(status.category)
            
            # Extract fuzzing statistics info
            stats = status.fuzz_stats
            if stats:
                stability = stats.get("stability", 0.0)
                bitmap_cvg = stats.get("bitmap_cvg", 0.0)
                corpus_count = stats.get("corpus_count", 0)
                saved_crashes = stats.get("saved_crashes", 0)
                saved_hangs = stats.get("saved_hangs", 0)
                
                # Determine color (only active during fuzzing stages)
                if status.current_stage in [HarnessStage.FUZZ_SHORT, HarnessStage.FUZZ_LONG]:
                    row_style = "bright_white"
                else:
                    row_style = "dim"
                
                table.add_row(
                    f"[{row_style}]{display_category}[/{row_style}]",
                    f"[{row_style}]{corpus_count}[/{row_style}]",
                    f"[{row_style}]{saved_hangs}[/{row_style}]",
                    f"[{row_style}]{saved_crashes}[/{row_style}]",
                    f"[{row_style}]{stability:.1f}%[/{row_style}]",
                    f"[{row_style}]{bitmap_cvg:.1f}%[/{row_style}]"
                )
            else:
                # Fuzzing but no stats yet
                table.add_row(
                    f"[yellow]{display_category}[/yellow]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]"
                )
        
        return table
    
    def create_layout(self) -> Layout:
        """Create layout"""
        layout = Layout()
        layout.split_column(
            Layout(self.create_summary_panel(), size=8, name="summary"),
            Layout(self.create_categories_table(), name="categories"),
            Layout(self.create_fuzz_stats_table(), name="fuzz_stats")
        )
        return layout
    
    def _disable_console_logging(self):
        """Disable console output to prevent conflict with dashboard"""
        self._disabled_console_handlers = disable_all_console_logging()
    
    def _enable_console_logging(self):
        """Re-enable console output"""
        if hasattr(self, '_disabled_console_handlers'):
            enable_console_logging(self._disabled_console_handlers)
            self._disabled_console_handlers = []
    
    async def run_dashboard_updater(self):
        """Dashboard update loop (manage Live object in one place)"""
        self._disable_console_logging()
        self._running = True
        
        with Live(console=self.console, screen=True, auto_refresh=False) as live:
            self.live = live
            try:
                while self._running:
                    await self.update_fuzzing_stats()
                    self.live.update(self.create_layout(), refresh=True)
                    await asyncio.sleep(2)  # Update stats every 2 seconds
            except asyncio.CancelledError:
                pass
            finally:
                self._running = False
    
    async def update_fuzzing_stats(self):
        """Read fuzzer_stats file for fuzzing categories and update"""
        for category, status in self.categories.items():
            if status.current_stage in [HarnessStage.FUZZ_SHORT, HarnessStage.FUZZ_LONG]:
                try:
                    # Construct fuzzer_stats file path
                    safe_category = category.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
                    # Use gen{N}_fix{M} directory for current attempt
                    gen_dir = f"gen{status.generate_retry_count}_fix{status.fix_retry_count}"
                    stats_file_path = os.path.join(
                        "library",
                        self.current_library,
                        f"harness_{self.current_run_index:03d}",
                        safe_category,
                        gen_dir,
                        "out",
                        "default",
                        "fuzzer_stats",
                    )
                    
                    if stats_file_path and os.path.exists(stats_file_path):
                        # Read stats file
                        stats = parse_fuzzer_stats(stats_file_path)
                        if stats:
                            # Directly update CategoryStatus fuzz_stats
                            status.fuzz_stats = stats
                except Exception as e:
                    # Output log at debug level (no impact on dashboard)
                    pass
    
    async def stop_dashboard(self):
        """Stop dashboard"""
        self._running = False  # Stop loop signal
        
        if self.live:
            self.live.stop()
        
        # Re-enable console logging
        self._enable_console_logging()
    
    def save_final_report(self, library: str = "library"):
        """Save final report"""
        # Create log/harness_report directory
        report_dir = os.path.join("log", "harness_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate filename: {library}_harness_report_{date}_{time}.json
        timestamp = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
        date_str = timestamp.strftime('%Y%m%d')
        time_str = timestamp.strftime('%H%M%S')
        filename = f"{library}_harness_report_{date_str}_{time_str}.json"
        output_path = os.path.join(report_dir, filename)
        
        report = {
            "timestamp": time.time(),
            "library": library,
            "generation_date": timestamp.strftime('%Y-%m-%d'),
            "generation_time": timestamp.strftime('%H:%M:%S'),
            "total_categories": self.total_categories,
            "total_elapsed_time": time.time() - self.start_time,
            "overall_progress": self.get_overall_progress(),
            "categories": {}
        }
        
        for category, status in self.categories.items():
            report["categories"][category] = {
                "api_count": status.api_count,
                "current_stage": status.current_stage.value,
                "elapsed_time": status.get_elapsed_time(),
                "coverage": status.coverage,
                "error_message": status.error_message,
                "is_long_fuzz": status.is_long_fuzz,
                "fuzz_duration": status.fuzz_duration
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.console.print(f"\n📄 Final report saved to {output_path}")


# Global dashboard manager instance
dashboard_manager = DashboardManager()