"""
Probabilistic Dynamic Mode Primitives for Robotics
"""

from demos.demo_0_dmd_simple_data import demo as demo1
from demos.demo_1_dmd_min_jerk import demo as demo2


def main():
    demo1()  # minimum_jerk_trajectory.py
    # demo2()  # Exact_DMD


if __name__ == '__main__':
    main()
