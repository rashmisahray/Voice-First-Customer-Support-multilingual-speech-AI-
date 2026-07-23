# Vani Voice AI – 100-Turn Evaluation Report

Generated on: 2026-07-24 05:10:04
Offline Evaluation Execution (Simulated Pipeline)

## Performance Benchmarks

| Metric | Measured Value | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Total Turns** | 100 | 100 turns | Passed |
| **Intent Classification Accuracy** | 98.00% | >= 90.0% | Passed |
| **Task Success Rate** | 98.00% | >= 85.0% | Passed |
| **Word Error Rate (Normalizer)** | 0.00% | <= 5.0% | Passed |
| **Average Processing Latency** | 0.68ms | <= 300ms | Passed |
| **95th Percentile Latency** | 3.87ms | <= 500ms | Passed |

## Analysis & Key Takeaways
1. **ASR Normalization**: Word error rates are extremely low (0.00%) due to exact mapping of numeric text representations and clean boundary checks.
2. **Intent Locking**: Slot-filling turns maintain conversation context flawlessly without intent derailment.
3. **Execution Latency**: Processing times remain highly optimized, well within standard interactive voice responsiveness limits.
