from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog, QMessageBox,
    QCheckBox, QComboBox, QStackedWidget
)
from kamaji.controllers.manual_behaviors import GoToOrigin, ConstantControl
import traceback
import numpy as np

DEFAULT_TEMPLATE = """\
def manual_control(sim, t):
    for agent in sim.active_agents:
        x = agent.state.get("position_x", 0.0)
        y = agent.state.get("position_y", 0.0)
        dx = -x
        dy = -y
        u = [dx, dy]
        sim.set_manual_control(agent._id, u)
"""

BEHAVIOR_MAP = {
    "Go to Origin": GoToOrigin,
    "Constant Control (1.0, 0.0)": lambda: ConstantControl([1.0, 0.0])
}


class ManualControlTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.use_custom_code = QCheckBox("✏️ Use Custom Python Code")
        self.use_custom_code.setChecked(False)
        self.use_custom_code.toggled.connect(self.toggle_mode)

        self.mode_stack = QStackedWidget()
        self.layout.addWidget(self.use_custom_code)
        self.layout.addWidget(self.mode_stack)

        # === Predefined Mode ===
        self.behavior_mode = QWidget()
        behavior_layout = QVBoxLayout()
        self.behavior_dropdown = QComboBox()
        self.behavior_dropdown.addItems(BEHAVIOR_MAP.keys())
        behavior_layout.addWidget(QLabel("Choose a built-in behavior:"))
        behavior_layout.addWidget(self.behavior_dropdown)
        self.behavior_mode.setLayout(behavior_layout)

        # === Custom Code Mode ===
        self.code_mode = QWidget()
        code_layout = QVBoxLayout()
        self.editor = QTextEdit()
        self.editor.setPlainText(DEFAULT_TEMPLATE)
        self.load_button = QPushButton("📂 Load from .py")
        self.load_button.clicked.connect(self.load_from_file)
        self.validate_button = QPushButton("✅ Validate")
        self.validate_button.clicked.connect(self.validate_function)
        code_layout.addWidget(QLabel("Define `manual_control(sim, t)` below:"))
        code_layout.addWidget(self.editor)
        code_layout.addWidget(self.load_button)
        code_layout.addWidget(self.validate_button)
        self.code_mode.setLayout(code_layout)

        self.mode_stack.addWidget(self.behavior_mode)
        self.mode_stack.addWidget(self.code_mode)
        self.toggle_mode(False)

    def toggle_mode(self, checked):
        self.mode_stack.setCurrentIndex(1 if checked else 0)

    def load_from_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Python File", ".", "Python Files (*.py)")
        if fname:
            try:
                with open(fname, "r") as f:
                    self.editor.setPlainText(f.read())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load file:\n{e}")

    def validate_function(self):
        func = self.get_function()
        if callable(func):
            QMessageBox.information(self, "Success", "Function is valid.")
        else:
            QMessageBox.critical(self, "Error", "No valid `manual_control(sim, t)` function found.")

    def get_function(self):
        if not self.use_custom_code.isChecked():
            behavior_class = BEHAVIOR_MAP[self.behavior_dropdown.currentText()]
            strategy = behavior_class()
            return strategy.update

        code = self.editor.toPlainText()
        try:
            local_scope = {}
            exec(code, {"np": np}, local_scope)
            func = local_scope.get("manual_control", None)
            if callable(func):
                func.__globals__["np"] = np
                return func
        except Exception:
            return None
