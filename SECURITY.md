# Security Policy

## Supported Versions

This is an academic research repository and book project. The primary "product" is the manuscript and the NumPy scratch implementations. 

Security here means:
1. **Code Integrity:** Ensuring the NumPy scratch code performs the exact mathematical operations described in the original papers.
2. **Data Safety:** There is no proprietary data in this repo.

| Version | Supported |
| :--- | :--- |
| v0.2.x (Current Infrastructure) | ✅ Security patches for code logic are accepted. |
| < v0.2.0 | ❌ Not supported. |

## Reporting a Vulnerability

If you discover a mathematical error in the equations, a citation mismatch, or a logic flaw in the NumPy scratch code, please do **not** open a public issue immediately.

Instead, email the project maintainer at: **[anshumansingh3697@gmail.com]**

We will respond within 72 hours and coordinate a fix via a private fork or a dedicated hotfix branch.

## Threat Model
- **Typos in equations:** High priority. Will be fixed immediately in the next patch release.
- **Citation fakes:** Critical priority. We run `scripts/verify_citations.py` before every merge.
- **NumPy overflow vulnerabilities:** Medium priority. All computations use standard `np.clip()` where necessary to prevent runaway exponents.