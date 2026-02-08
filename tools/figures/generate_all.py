#!/usr/bin/env python3
"""Regenerate all documentation gallery figures."""

from figgen import generate_all, laser_tracking_report


if __name__ == "__main__":
    generate_all()
    rows_med, cols_med = laser_tracking_report()
    print(f"laser median center error: rows={rows_med:.3f}px cols={cols_med:.3f}px")
