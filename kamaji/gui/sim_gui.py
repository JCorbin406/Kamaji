# kamaji/gui/sim_gui.py

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QSlider, QProgressBar,
    QHBoxLayout, QComboBox, QTextEdit
)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class KamajiSimGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.sim = None
        self.manual_control_fn = None
        self.timestep = 0
        self.total_steps = 1
        self.running = False
        self._on_complete = None

        self.layout = QVBoxLayout(self)

        # Status
        self.status_label = QLabel("Simulation Not Initialized")
        self.layout.addWidget(self.status_label)

        # Plot setup
        self.figure = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)
        self.layout.addWidget(self.canvas)

        # Diagnostics
        self.agent_selector = QComboBox()
        self.agent_selector.currentTextChanged.connect(self.update_diagnostics)
        self.diagnostics = QTextEdit()
        self.diagnostics.setReadOnly(True)

        diag_layout = QVBoxLayout()
        diag_layout.addWidget(QLabel("Selected Agent:"))
        diag_layout.addWidget(self.agent_selector)
        diag_layout.addWidget(QLabel("Agent Diagnostics"))
        diag_layout.addWidget(self.diagnostics)
        self.layout.addLayout(diag_layout)

        # Controls
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("▶ Start")
        self.restart_button = QPushButton("⟳ Restart")
        self.start_button.clicked.connect(self.toggle_simulation)
        self.restart_button.clicked.connect(self.restart_simulation)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.restart_button)
        self.layout.addLayout(control_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setFixedHeight(24)
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Simulation Progress: %p%")
        self.layout.addWidget(self.progress)


        # Time slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_label = QLabel("Time: 0.00s")
        self.time_slider.setMinimum(0)
        self.time_slider.sliderReleased.connect(self.slider_changed)
        time_slider_layout = QHBoxLayout()
        time_slider_layout.addWidget(QLabel("Time Slider"))
        time_slider_layout.addWidget(self.time_label)
        self.layout.addLayout(time_slider_layout)
        self.layout.addWidget(self.time_slider)


        # Timer
        self._timer = QTimer()
        self._timer.timeout.connect(self.advance_timestep)

    def set_simulator(self, simulator, manual_control_fn=None, on_complete=None):
        self.sim = simulator
        self.manual_control_fn = manual_control_fn
        self._on_complete = on_complete
        self.total_steps = simulator.num_timesteps
        self.status_label.setText("Simulation Loaded")
        self.running = False
        self.timestep = 0
        self.start_button.setText("▶ Start")

        self.ax.clear()
        self.ax.grid(True)
        self.lines = []
        self.markers = []

        agents = self.sim.active_agents
        if not agents:
            self.status_label.setText("No agents loaded.")
            return

        self.agent_selector.clear()
        for agent in agents:
            self.agent_selector.addItem(agent._id)

        color_cycle = plt.rcParams['axes.prop_cycle']()
        self.colors = []

        for agent in agents:
            color = next(color_cycle)['color']
            self.colors.append(color)
            x_vals = agent.state_log["position_x"]
            y_vals = agent.state_log["position_y"]
            line, = self.ax.plot(x_vals, y_vals, '-', color=color, marker=None)
            marker, = self.ax.plot([], [], 'o', color=color)
            self.lines.append(line)
            self.markers.append(marker)

        all_x = [agent.state_log["position_x"].iloc[0] for agent in agents]
        all_y = [agent.state_log["position_y"].iloc[0] for agent in agents]
        self.ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        self.ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
        self.canvas.draw()

        self.time_slider.setMaximum(self.total_steps - 1)
        self.progress.setMaximum(self.total_steps - 1)

        self.update_diagnostics()
        self._update_plot()

    def toggle_simulation(self):
        if not self.sim:
            self.status_label.setText("No simulator loaded.")
            return

        if self.running:
            self._timer.stop()
            self.start_button.setText("▶ Start")
            self.status_label.setText("Paused")
        else:
            self._timer.start(50)
            self.start_button.setText("⏸ Pause")
            self.status_label.setText("Running")
        self.running = not self.running

    def restart_simulation(self):
        if not self.sim:
            return
        self.timestep = 0
        self.time_slider.setValue(0)
        self.progress.setValue(0)
        self.status_label.setText("Restarted")
        self.running = False
        self.start_button.setText("▶ Start")
        self._update_plot()
        self.update_diagnostics()

    def slider_changed(self):
        if not self.sim:
            return
        self.timestep = self.time_slider.value()
        self.time_label.setText(f"Time: {self.timestep * self.sim.dt:.2f}s")
        self._update_plot()
        self.update_diagnostics()

    def advance_timestep(self):
        if not self.sim:
            return
        if self.timestep >= self.total_steps - 1:
            self._timer.stop()
            self.status_label.setText("Finished")
            self.start_button.setText("▶ Restart")
            self.running = False
            if self._on_complete:
                self._on_complete(self.sim)
            return

        if self.manual_control_fn:
            self.manual_control_fn(self.sim, self.sim.sim_time)

        self.sim.step()
        self.sim.sim_time += self.sim.dt
        self.timestep += 1
        self.time_label.setText(f"Time: {self.timestep * self.sim.dt:.2f}s")

        self._update_plot()
        self.update_diagnostics()
        self.progress.setValue(self.timestep)
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(self.timestep)
        self.time_slider.blockSignals(False)

    def _update_plot(self):
        for i, agent in enumerate(self.sim.active_agents):
            x_vals = agent.state_log["position_x"]
            y_vals = agent.state_log["position_y"]

            self.lines[i].set_data(x_vals[:self.timestep + 1], y_vals[:self.timestep + 1])
            if self.timestep < len(x_vals):
                self.markers[i].set_data([x_vals.iloc[self.timestep]], [y_vals.iloc[self.timestep]])

        self.canvas.draw()

    def update_diagnostics(self):
        aid = self.agent_selector.currentText()
        agent = next((a for a in self.sim.active_agents if a._id == aid), None)
        if not agent:
            self.diagnostics.setText("Agent not found.")
            return

        s_log = agent.state_log
        c_log = agent.control_log

        try:
            state = s_log.iloc[self.timestep].to_dict()
            control = c_log.iloc[self.timestep].to_dict()
        except IndexError:
            self.diagnostics.setText("No data at current timestep.")
            return

        lines = ["📊 Current State:"]
        for k, v in state.items():
            lines.append(f"{k}: {v:.3f}")
        lines.append("🎮 Control Input:")
        for k, v in control.items():
            if k != "time":
                lines.append(f"{k}: {v:.3f}")

        self.diagnostics.setText("\n".join(lines))