from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
import numpy as np


class DiagnosticsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.sim = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.plot_type_selector = QComboBox()
        self.plot_type_selector.addItems([
            "Control Inputs",
            "State Variables",
            "Agent Count Over Time"
        ])

        self.agent_selector = QComboBox()
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.update_plot)

        self.layout.addWidget(QLabel("Select Plot Type:"))
        self.layout.addWidget(self.plot_type_selector)

        self.layout.addWidget(QLabel("Select Agent (if applicable):"))
        self.layout.addWidget(self.agent_selector)
        self.layout.addWidget(self.update_button)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = Canvas(self.figure)
        self.layout.addWidget(self.canvas)

    def set_simulator(self, simulator):
        self.sim = simulator
        self.agent_selector.clear()
        if self.sim:
            self.agent_selector.addItems([a._id for a in self.sim.active_agents])
        self.figure.clear()
        self.canvas.draw()

    def update_plot(self):
        if not self.sim:
            return

        plot_type = self.plot_type_selector.currentText()
        self.figure.clear()

        if plot_type == "Control Inputs":
            self.plot_control_inputs()
        elif plot_type == "State Variables":
            self.plot_state_variables()
        elif plot_type == "Agent Count Over Time":
            self.plot_agent_count()

        self.canvas.draw()

    def plot_control_inputs(self):
        selected_id = self.agent_selector.currentText()
        agent = next((a for a in self.sim.active_agents if a._id == selected_id), None)
        if not agent or agent.control_log.empty:
            ax = self.figure.add_subplot(111)
            ax.set_title(f"No control data for {selected_id}")
            return

        df = agent.control_log
        control_cols = [col for col in df.columns if col != "time"]
        num_plots = len(control_cols)

        for i, col in enumerate(control_cols, 1):
            ax = self.figure.add_subplot(num_plots, 1, i)
            ax.plot(df["time"], df[col], label=col)
            ax.set_ylabel(col)
            if i == 1:
                ax.set_title(f"Control Inputs for {selected_id}")
            if i == num_plots:
                ax.set_xlabel("Time (s)")

    def plot_state_variables(self):
        selected_id = self.agent_selector.currentText()
        agent = next((a for a in self.sim.active_agents if a._id == selected_id), None)
        if not agent or agent.state_log.empty:
            ax = self.figure.add_subplot(111)
            ax.set_title(f"No state data for {selected_id}")
            return

        df = agent.state_log
        state_cols = [col for col in df.columns if col != "time"]
        num_plots = len(state_cols)

        for i, col in enumerate(state_cols, 1):
            ax = self.figure.add_subplot(num_plots, 1, i)
            ax.plot(df["time"], df[col], label=col)
            ax.set_ylabel(col)
            if i == 1:
                ax.set_title(f"State Variables for {selected_id}")
            if i == num_plots:
                ax.set_xlabel("Time (s)")

    def plot_agent_count(self):
        if not self.sim:
            return

        agent_logs = [a.state_log["time"].values for a in self.sim.active_agents if not a.state_log.empty]
        if not agent_logs:
            ax = self.figure.add_subplot(111)
            ax.set_title("No agents with log data")
            return

        all_times = sorted(set(np.concatenate(agent_logs)))
        counts = [sum(t in log for log in agent_logs) for t in all_times]

        ax = self.figure.add_subplot(111)
        ax.plot(all_times, counts, label="Active Agents")
        ax.set_title("Number of Active Agents Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count")
        ax.legend()
