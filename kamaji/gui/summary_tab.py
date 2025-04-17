import numpy as np
import h5py
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure


class SummaryTab(QWidget):
    def __init__(self):
        super().__init__()
        self.sim = None
        self.agent_data = {}

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.load_button = QPushButton("📂 Load External HDF5 Log")
        self.load_button.clicked.connect(self.load_hdf5_file)

        self.text_summary = QTextEdit()
        self.text_summary.setReadOnly(True)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = Canvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.layout.addWidget(QLabel("Summary Analysis"))
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.text_summary)

    def set_simulator(self, simulator):
        self.sim = simulator
        self.agent_data = self._extract_from_sim(simulator)
        self.update_summary()

    def load_hdf5_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select HDF5 File", ".", "HDF5 files (*.h5 *.hdf5)")
        if not fname:
            return
        try:
            self.agent_data = self._extract_from_hdf5(fname)
            self.sim = None
            self.update_summary()
        except Exception as e:
            self.text_summary.setText(f"❌ Failed to load: {e}")

    def update_summary(self):
        if not self.agent_data:
            self.text_summary.setText("No data loaded.")
            return

        self.ax.clear()
        summary_lines = []
        efforts = []
        names = []

        for aid, data in self.agent_data.items():
            control_log = data["control"]
            effort = np.sum(np.linalg.norm(control_log[:, 1:], axis=1))
            min_dist = self._min_distance_to_others(aid, self.agent_data)

            summary_lines.append(f"🛸 {aid} | Control Effort: {effort:.2f} | Min Distance: {min_dist:.2f}")
            efforts.append(effort)
            names.append(aid)

        self.ax.bar(names, efforts)
        self.ax.set_ylabel("Total Control Effort")
        self.ax.set_title("Control Effort per Agent")
        self.canvas.draw()
        self.text_summary.setText("\n".join(summary_lines))

    def _min_distance_to_others(self, agent_id, all_data):
        a_pos = all_data[agent_id]["state"][:, 1:3]  # X, Y
        min_dist = float('inf')
        for other_id, other_data in all_data.items():
            if other_id == agent_id:
                continue
            b_pos = other_data["state"][:, 1:3]
            dists = np.linalg.norm(a_pos - b_pos, axis=1)
            min_dist = min(min_dist, np.min(dists))
        return min_dist

    def _extract_from_sim(self, sim):
        out = {}
        for agent in sim.active_agents + sim.inactive_agents:
            state = agent.state_log.to_numpy()
            control = agent.control_log.to_numpy()
            out[agent._id] = {"state": state, "control": control}
        return out

    def _extract_from_hdf5(self, path):
        out = {}
        with h5py.File(path, "r") as f:
            for aid in f:
                if "state" in f[aid] and "control" in f[aid]:
                    state_data = np.vstack([f[aid]["state"][col][...] for col in f[aid]["state"]]).T
                    control_data = np.vstack([f[aid]["control"][col][...] for col in f[aid]["control"]]).T
                    out[aid] = {"state": state_data, "control": control_data}
        return out
