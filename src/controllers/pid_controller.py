"""
Discrete-time PID controller with anti-windup.
Used by B1 Seg+PID baseline for each control axis.
"""
from __future__ import annotations


class PIDController:
    """
    Discrete-time PID controller.

    u(t) = Kp*e(t) + Ki*integral(e) + Kd*(e(t)-e(t-1))/dt

    Anti-windup: integral clamped to [-integral_limit, +integral_limit].
    Output clamped to output_limits.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limits: tuple[float, float] = (-1.0, 1.0),
        integral_limit: float = 2.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limit = integral_limit

        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._initialized: bool = False  # prevents cold-start derivative spike

    def compute(self, error: float, dt: float = 1.0) -> float:
        """
        Compute control output for given error and timestep.

        Args:
            error: current error (setpoint - measurement)
            dt:    control period in seconds

        Returns:
            Control output clipped to output_limits.
        """
        # Integral with anti-windup
        self._integral += error * dt
        self._integral = float(max(
            -self.integral_limit,
            min(self.integral_limit, self._integral)
        ))

        # Derivative (skip on first call to avoid cold-start spike)
        if self._initialized:
            derivative = (error - self._prev_error) / max(dt, 1e-6)
        else:
            derivative = 0.0
            self._initialized = True

        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(max(self.output_limits[0], min(self.output_limits[1], output)))

    def reset(self) -> None:
        """Clear integral and derivative history (call on phase transitions)."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._initialized = False

    @property
    def integral(self) -> float:
        return self._integral
