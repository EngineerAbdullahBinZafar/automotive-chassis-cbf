# Contributing to Automotive Chassis CBF

Contributions extending road synthesis profiles, DARE/MPC solvers, or HIL CAN telemetry bridges are welcome.

## Development Workflow
1. Create a virtual environment and install dependencies (`numpy`, `scipy`, `osqp`, `cvxpy`, `pytest`).
2. Run the complete 88-test verification suite:
   ```bash
   pytest tests/ -v
   ```
3. Ensure all changes preserve strict stroke invariance ($|z_s - z_u| \le 50\text{ mm}$) and zero tire liftoff ($F_{z,\text{total}} \ge 0$).
