import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class DynamicBicycleModel:
    def __init__(self):
        ##### param #####
        # Pacejka parameters (f for front, r for rear)
        self.F_z0 = 3  # Nominal z force on tyre N

        # Front tyre parameters
        self.Bf = 9.62
        self.Cf = 2.59
        self.Df = 4.120
        self.Ef = 1
        self.epsf = -0.0813
        # Rear tyre parameters
        self.Br = 8.62
        self.Cr = 2.65
        self.Dr = 4.617
        self.Er = 1
        self.epsr = -0.1263

        self.mass = 1.160  # Mass kg
        self.Iz = 1.260  # Vehicle intertia kg m^2
        self.g = 9.81  # Gravity m/s^2
        self.h = 0.35  # Height of CG. (m)
        self.lf = 1.51  # Dist from CG to front (m).
        self.lr = 1.388  # Dist from CG to rear (m)
        self.length = self.lf + self.lr
        self.u_fr = 1.0489  # Friction cooeficient.
        self.brake_bias = 0.7  # Front to rear

        self.F_zf = self.mass * self.g * self.lr / (self.lr + self.lf)
        self.F_zr = self.mass * self.g * self.lf / (self.lr + self.lf)

        self.acceleration_data = np.array(
            [
                [0.0, 1.0, 6612],
                [27.78, 1.0, 5684],
                [55.56, 1.0, 1160],
                [55.56, 0.0, -2436],
                [27.78, 0.0, -812],
                [11.11, 0.0, -238],
            ]
        ).T
        self.braking_data = np.array(
            [
                [55.56, 0.0, -2436],
                [27.78, 0.0, -812],
                [11.11, 0.0, -238],
                [55.56, -1.0, -18908],
                [27.78, -1.0, -17748],
                [11.11, -1.0, -17168],
            ]
        ).T
        self.acceleration_params, _ = curve_fit(
            self.long_force, self.acceleration_data[:2], self.acceleration_data[2]
        )
        self.braking_params, _ = curve_fit(
            self.long_force, self.braking_data[:2], self.braking_data[2]
        )

        # Motor coefficients
        self.Cm1 = self.acceleration_params[0]  # Motor parameter 1
        self.Cm2 = self.acceleration_params[1]  # Motor parameter 2
        self.Cm3 = self.acceleration_params[2]  # Motor parameter 3

        # Braking coefficients
        self.Cb1 = self.braking_params[0]  # Braking parameter 1
        self.Cb2 = self.braking_params[1]  # Braking parameter 2
        self.Cb3 = self.braking_params[2]  # Braking parameter 3

        # Friction coefficients (rolling, aero) exerimentally gathered
        self.Cfric1 = self.braking_params[3]
        self.Cfric2 = self.braking_params[4]
        self.Cfric3 = self.braking_params[5]

    @staticmethod
    def long_force(data, cm1, cm2, cm3, cphys1, cphys2, cphys3):
        return (
            (cm1 - cm2 * data[0] - cm3 * data[0] ** 2) * data[1]
            - cphys1
            - cphys2 * data[0]
            - cphys3 * data[0] ** 2
        )

    def predict_next_state(self, initial_state, u, dt=0.05):
        """
        u is (steering, acceleration)
        -0.3 < steering 0.3
        -16 < acceleration < 6
        """
        delta, acc = u
        pos_x, pos_y, yaw, vel_x, vel_y, yaw_rate = initial_state

        alpha_f = -np.arctan((yaw_rate * self.lf + vel_y) / (vel_x + 1e-3)) + delta
        alpha_r = np.arctan((yaw_rate * self.lr - vel_y) / (vel_x + 1e-3))

        # F_zf = self.mass * self.g * self.lr / (self.lr + self.lf)
        # F_zr = self.mass * self.g * self.lf / (self.lr + self.lf)

        F_fy = (
            self.Df
            * (1 + self.epsf * self.F_zf / self.F_z0)
            * self.F_zf
            / self.F_z0
            * np.sin(
                self.Cf
                * np.arctan2(
                    self.Bf * alpha_f
                    - self.Ef * (self.Bf * alpha_f - np.arctan2(self.Bf * alpha_f, 1)),
                    1,
                )
            )
        )
        F_ry = (
            self.Dr
            * (1 + self.epsr * self.F_zr / self.F_z0)
            * self.F_zr
            / self.F_z0
            * np.sin(
                self.Cr
                * np.arctan2(
                    self.Br * alpha_r
                    - self.Er * (self.Br * alpha_r - np.arctan2(self.Br * alpha_r, 1)),
                    1,
                )
            )
        )

        # F_fy = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f))
        # F_ry = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r))
        F_fric = -self.Cfric1 - self.Cfric2 * vel_x - self.Cfric3 * vel_x**2
        F_rx_braking = (
            (self.Cb1 - self.Cb2 * vel_x - self.Cb3 * vel_x**2) * (1 - self.brake_bias)
        ) * min(0.0, acc)
        F_rx_acceleration = (self.Cm1 - self.Cm2 * vel_x - self.Cm3 * vel_x**2) * max(
            0.0, acc
        )
        F_rx = F_rx_braking + F_rx_acceleration
        F_fx = (
            (self.Cb1 - self.Cb2 * vel_x - self.Cb3 * vel_x**2)
            * self.brake_bias
            * min(0.0, acc)
        )

        x_dot = np.zeros(6)
        x_dot[0] = vel_x * np.cos(yaw) - vel_y * np.sin(yaw)
        x_dot[1] = vel_x * np.sin(yaw) + vel_y * np.cos(yaw)
        x_dot[2] = yaw_rate
        x_dot[3] = (1 / self.mass) * (
            F_rx + F_fx + F_fric - F_fy * np.sin(delta) + self.mass * vel_y * yaw_rate
        )
        x_dot[4] = (1 / self.mass) * (
            F_ry + F_fy * np.cos(delta) - self.mass * vel_x * yaw_rate
        )
        x_dot[5] = (1 / self.Iz) * (F_fy * self.lf * np.cos(delta) - F_ry * self.lr)

        return (initial_state + x_dot * dt, x_dot, [F_fy, F_ry, F_fx, F_rx])


if __name__ == "__main__":
    dynamics_model = DynamicBicycleModel()

    dt = 0.05
    time = 2  # seconds
    start_vel = 5
    state = np.array([0, 0, 0, start_vel, 0, 0])
    control = [0.01, 0.0]

    states = []
    tyre_forces = []
    x_dots = []
    times = []
    total_time = 0
    for i in range(int(time / dt)):
        total_time += dt
        state, x_dot, tyre_force = dynamics_model.predict_next_state(state, control, dt)
        state[3] = np.clip(state[3], 0, np.inf)
        states.append(state)
        tyre_forces.append(tyre_force)
        x_dots.append(x_dot)
        times.append(total_time)

    states = np.array(states).T
    x_dots = np.array(x_dots).T
    tyre_forces = np.array(tyre_forces).T

    fig1, ax1 = plt.subplots()
    ax1.plot(states[0], states[1])
    ax1.set_title("Movement")
    # ax1.set_aspect(1)

    fig2, ax2 = plt.subplots()
    ax2.plot(times, states[3], label="x")
    ax2.set_title("Velocity X")
    ax2.legend()

    fig3, ax3 = plt.subplots()
    ax3.plot(times, x_dots[3])
    ax3.set_title("Acceleration X")

    fig4, ax4 = plt.subplots()
    ax4.plot(times, x_dots[2])
    ax4.set_title("Yaw")

    fig5, ax5 = plt.subplots()
    ax5.plot(times, tyre_forces[0], label="front")
    ax5.plot(times, tyre_forces[1], label="rear")
    ax5.set_title("Lateral tyre forces")
    ax5.legend()

    plt.show()
