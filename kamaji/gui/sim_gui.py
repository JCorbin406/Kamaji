# (existing imports)
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QTextEdit, QMessageBox
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

class KamajiSimGUI(QWidget):
    def __init__(self, simulator=None, manual_control_fn=None, update_interval_ms=50, on_complete=None):
        super().__init__()
        self.sim = simulator
        self.manual_control_fn = manual_control_fn or (lambda sim, t: None)
        self.on_complete = on_complete
        self.update_interval_ms = update_interval_ms
        self.running = False

        self._init_ui()
        if self.sim:
            self._init_plot()

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_simulation)

    def _init_ui(self):
        self.layout = QHBoxLayout(self)

        # Left side: plot + controls
        left_panel = QVBoxLayout()
        btn_row = QHBoxLayout()

        self.start_button = QPushButton("▶ Start")
        self.start_button.clicked.connect(self.toggle_simulation)

        self.save_button = QPushButton("💾 Save Now")
        self.save_button.clicked.connect(self.save_simulation)

        btn_row.addWidget(self.start_button)
        btn_row.addWidget(self.save_button)

        self.status_label = QLabel("Status: Paused")

        self.figure = Figure(figsize=(6, 6))
        self.canvas = Canvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Agent Trajectories")
        self.ax.grid(True)

        left_panel.addLayout(btn_row)
        left_panel.addWidget(self.status_label)
        left_panel.addWidget(self.canvas)

        # Right side: agent diagnostics
        right_panel = QVBoxLayout()
        self.agent_selector = QComboBox()
        self.agent_selector.currentTextChanged.connect(self._refresh_agent_display)

        self.agent_display = QTextEdit()
        self.agent_display.setReadOnly(True)

        right_panel.addWidget(QLabel("🧠 Agent Diagnostics"))
        right_panel.addWidget(self.agent_selector)
        right_panel.addWidget(self.agent_display)

        self.layout.addLayout(left_panel, 3)
        self.layout.addLayout(right_panel, 2)

    def set_simulator(self, simulator, manual_control_fn=None, on_complete=None):
        self.sim = simulator
        self.manual_control_fn = manual_control_fn or (lambda sim, t: None)
        self.on_complete = on_complete
        self.sim.sim_time = 0.0
        self._init_plot()
        self.status_label.setText("Status: Ready")
        self.running = False
        self.start_button.setText("▶ Start")

    def _init_plot(self):
        self.ax.clear()
        self.ax.grid(True)

        self.lines = []
        self.markers = []
        self.agent_selector.clear()
        all_positions = []

        for agent in self.sim.active_agents:
            x = agent.state.get("position_x", 0.0)
            y = agent.state.get("position_y", 0.0)
            all_positions.append((x, y))

            self.agent_selector.addItem(agent._id)
            line, = self.ax.plot([], [], label=agent._id)
            marker, = self.ax.plot([], [], 'o', color=line.get_color())
            self.lines.append(line)
            self.markers.append(marker)

        self.ax.legend()
        if all_positions:
            xs, ys = zip(*all_positions)
            buffer = 2.0
            self.ax.set_xlim(min(xs) - buffer, max(xs) + buffer)
            self.ax.set_ylim(min(ys) - buffer, max(ys) + buffer)
        self.canvas.draw()

    def toggle_simulation(self):
        if not self.sim:
            QMessageBox.warning(self, "Simulation Not Initialized", "Please initialize the simulation first.")
            return

        if self.running:
            self._timer.stop()
            self.status_label.setText("Status: Paused")
            self.start_button.setText("▶ Start")
        else:
            self._timer.start(self.update_interval_ms)
            self.status_label.setText("Status: Running")
            self.start_button.setText("⏸ Pause")
        self.running = not self.running

    def _update_simulation(self):
        if not self.sim:
            return

        if self.sim.sim_time >= self.sim.duration:
            self._timer.stop()
            self.status_label.setText("Status: Complete")
            self.start_button.setText("▶ Restart")
            self.running = False

            if self.sim.logging_params.get("autosave", False):
                self.save_simulation()

            if self.on_complete:
                self.on_complete(self.sim)
            return

        self.manual_control_fn(self.sim, self.sim.sim_time)
        self.sim.step()
        self.sim.sim_time += self.sim.dt
        self._update_plot()
        self._refresh_agent_display()

    def _update_plot(self):
        for i, agent in enumerate(self.sim.active_agents):
            x_vals = agent.state_log.get("position_x", [])
            y_vals = agent.state_log.get("position_y", [])
            if len(x_vals) > 0 and len(y_vals) > 0:
                x_np = x_vals.to_numpy()
                y_np = y_vals.to_numpy()
                self.lines[i].set_data(x_np, y_np)
                self.markers[i].set_data([x_np[-1]], [y_np[-1]])

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def _refresh_agent_display(self):
        aid = self.agent_selector.currentText()
        agent = next((a for a in self.sim.active_agents if a._id == aid), None)
        if not agent:
            self.agent_display.setText("No agent selected.")
            return

        state_lines = [f"{k}: {v:.2f}" for k, v in agent.state.items()]
        control_row = agent.control_log.iloc[-1] if not agent.control_log.empty else None
        control_lines = []
        if control_row is not None:
            for k in control_row.keys():
                if k != "time":
                    control_lines.append(f"{k}: {control_row[k]:.2f}")

        self.agent_display.setText("\n".join([
            f"⏱ Time: {self.sim.sim_time:.2f}",
            "--- State ---",
            *state_lines,
            "--- Control ---",
            *control_lines
        ]))

    def save_simulation(self):
        if not hasattr(self.sim, "logger"):
            QMessageBox.warning(self, "Save Failed", "Simulator does not have a logger.")
            return
        try:
            self.sim.logger.log_to_hdf5()
            self.status_label.setText("Data saved ✅")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save: {e}")
