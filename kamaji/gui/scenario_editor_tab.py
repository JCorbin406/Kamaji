import yaml
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog, QMessageBox, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
import matplotlib.patches as patches


class ScenarioEditorTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.agents = []
        self.obstacles = []
        self.mode = "agent"

        self.fig = Figure()
        self.canvas = Canvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Scenario Editor")
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.layout = QVBoxLayout()
        self.button_row = QHBoxLayout()

        self.add_agent_btn = QPushButton("🧍 Add Agent")
        self.add_agent_btn.clicked.connect(lambda: self.set_mode("agent"))

        self.add_obstacle_btn = QPushButton("🪨 Add Obstacle")
        self.add_obstacle_btn.clicked.connect(lambda: self.set_mode("obstacle"))

        self.save_btn = QPushButton("💾 Export YAML")
        self.save_btn.clicked.connect(self.export_yaml)

        self.button_row.addWidget(self.add_agent_btn)
        self.button_row.addWidget(self.add_obstacle_btn)
        self.button_row.addWidget(self.save_btn)

        self.layout.addLayout(self.button_row)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def set_mode(self, mode):
        self.mode = mode

    def on_click(self, event):
        if not event.inaxes:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:  # Left-click
            if self.mode == "agent":
                self.add_agent(x, y)
            elif self.mode == "obstacle":
                self.add_obstacle(x, y)
        elif event.button == 3:  # Right-click
            self.clear_canvas()

    def add_agent(self, x, y):
        agent_id = f"agent_{len(self.agents) + 1}"
        self.agents.append({
            "id": agent_id,
            "position": (x, y)
        })
        self.ax.plot(x, y, marker="o", color="blue", label=agent_id)
        self.canvas.draw()

    def add_obstacle(self, x, y):
        radius = 1.0
        self.obstacles.append({
            "type": "circle",
            "position": [x, y],
            "radius": radius
        })
        circle = patches.Circle((x, y), radius=radius, color="red", alpha=0.5)
        self.ax.add_patch(circle)
        self.canvas.draw()

    def clear_canvas(self):
        self.agents.clear()
        self.obstacles.clear()
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.canvas.draw()

    def export_yaml(self):
        output = {
            "agents": {},
            "environment": {
                "obstacles": self.obstacles
            }
        }
        for agent in self.agents:
            output["agents"][agent["id"]] = {
                "type": "agent",
                "dynamics_model": "SingleIntegrator2DOF",
                "initial_state": {
                    "position_x": agent["position"][0],
                    "position_y": agent["position"][1],
                    "velocity_x": 0.0,
                    "velocity_y": 0.0
                },
                "controller": {
                    "type": "PID",
                    "specs": [
                        {
                            "state": "position_x",
                            "goal": 0.0,
                            "kp": 1.0,
                            "ki": 0.0,
                            "kd": 0.0
                        },
                        {
                            "state": "position_y",
                            "goal": 0.0,
                            "kp": 1.0,
                            "ki": 0.0,
                            "kd": 0.0
                        }
                    ]
                }
            }

        fname, _ = QFileDialog.getSaveFileName(self, "Export Scenario", "scenario.yaml", "YAML Files (*.yaml *.yml)")
        if fname:
            with open(fname, "w") as f:
                yaml.dump(output, f)
            QMessageBox.information(self, "Saved", f"Scenario exported to {fname}")
