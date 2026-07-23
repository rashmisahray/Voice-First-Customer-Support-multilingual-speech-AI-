import threading
import time
from typing import Dict, Any

class SystemMetricsTracker:
    """Thread-safe metrics recorder for tracking voice pipeline performance."""
    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.total_asr_latency = 0.0
        self.total_dialogue_latency = 0.0
        self.successful_tools = 0
        self.failed_requests = 0

    def record_turn(self, asr_ms: float, dialogue_ms: float, tool_success: bool = None):
        with self._lock:
            self.total_requests += 1
            self.total_asr_latency += asr_ms
            self.total_dialogue_latency += dialogue_ms
            if tool_success is True:
                self.successful_tools += 1
            elif tool_success is False:
                self.failed_requests += 1

    def get_report(self) -> Dict[str, Any]:
        with self._lock:
            avg_asr = (self.total_asr_latency / self.total_requests) if self.total_requests > 0 else 0.0
            avg_dialogue = (self.total_dialogue_latency / self.total_requests) if self.total_requests > 0 else 0.0
            avg_total = avg_asr + avg_dialogue
            success_rate = (self.successful_tools / self.total_requests) * 100 if self.total_requests > 0 else 100.0
            
            return {
                "total_turns": self.total_requests,
                "avg_asr_latency_ms": round(avg_asr, 2),
                "avg_dialogue_latency_ms": round(avg_dialogue, 2),
                "avg_total_latency_ms": round(avg_total, 2),
                "successful_transactions": self.successful_tools,
                "failed_requests": self.failed_requests,
                "task_success_rate_percent": round(success_rate, 2)
            }

# Singleton instance of system metrics
metrics_tracker = SystemMetricsTracker()
