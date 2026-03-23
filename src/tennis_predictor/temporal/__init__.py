"""Temporal isolation and validation framework.

This is the most critical module in the system. It ensures that no future
information leaks into training or feature computation. The phosphenq model's
critical flaw was ELO data leakage — we prevent this architecturally.
"""
