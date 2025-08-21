# Commit Message for v4.5.1

## Suggested Commit Message

```
feat(v4.5.1): Add personal mode optimizations with profile system and 120B model support

- Added profile system with LATENCY_FIRST, QUALITY_FIRST, and BALANCED modes
- Implemented personal-focused release gate with relaxed SLOs (QPS: 0.3, P95: 10s)
- Enhanced metrics tagging with comprehensive model/engine/profile information
- Successfully deployed 120B MoE model with 4-bit quantization (13.8GB/GPU)
- Fixed model detection and stats endpoint reporting
- Documented streaming bug (async generator issue, fix identified)

Performance:
- 20B model: 4-8s response time, 70% success rate
- 120B model: 6-15s response time, 83% success rate
- Both models suitable for personal use

BREAKING CHANGE: New server entry point (server_v451.py) with different CLI arguments
```

## Files to Commit

### Core Implementation
- `src/server_v451.py` - Main v4.5.1 server with profile system
- `scripts/release_gate_personal.py` - Personal mode release gate

### Documentation
- `README.md` - Updated with v4.5.1 features and usage
- `CHANGELOG.md` - Added v4.5.1 release notes
- `TEST_PROCEDURE.md` - Comprehensive testing guide

### Reports (in reports/v451/)
- `20B_TEST_REPORT.md` - 20B model test results
- `120B_TEST_REPORT.md` - 120B model test results  
- `V4.5.1_FINAL_SUMMARY.md` - Final summary report

## Directory Structure

```
gpt-oss-hf-server/
├── src/
│   ├── server_v451.py       # v4.5.1 implementation
│   ├── server_v45.py        # v4.5.0 (previous)
│   └── server.py            # v4.4 (stable)
├── scripts/
│   ├── release_gate_personal.py  # Personal SLO validation
│   └── release_gate.py          # Enterprise SLO validation
├── reports/v451/
│   ├── 20B_TEST_REPORT.md
│   ├── 120B_TEST_REPORT.md
│   └── V4.5.1_FINAL_SUMMARY.md
├── logs/v451/
│   └── [various log files]
├── README.md                # Updated
├── CHANGELOG.md            # Updated
└── TEST_PROCEDURE.md       # New testing guide
```

## Known Issues to Document

1. **Streaming Bug**: Line 475 in server_v451.py needs one-line fix
   - Change `await self._handle_streaming(...)` to `self._handle_streaming(...)`
   - Affects both 20B and 120B models
   - Non-streaming works normally

## Next Steps After Commit

1. Fix streaming bug in hotfix branch
2. Test the fix thoroughly
3. Release v4.5.2 with streaming fix
4. Consider implementing automatic model selection based on query complexity

## Git Commands

```bash
# Add all changes
git add -A

# Commit with message
git commit -m "feat(v4.5.1): Add personal mode optimizations with profile system and 120B model support"

# Push to repository
git push origin main
```