import sys
import os
import yaml
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QLineEdit, QFormLayout, QHBoxLayout, QMessageBox, QTabWidget, QComboBox,
    QListWidget, QTextEdit, QCheckBox, QStackedWidget, QSplitter, QListWidgetItem
)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox, QFileDialog,
    QFormLayout, QMessageBox, QScrollArea, QGroupBox
)
from PyQt5.QtCore import Qt
from kamaji.simulation.simulator import Simulator
from kamaji.gui.sim_gui import KamajiSimGUI
from kamaji.gui.diag_gui import DiagnosticsTab
from kamaji.tools.helpers import inject_manual_control
from kamaji.tools.stdout_redirector import EmittingStream
from kamaji.dynamics import dynamics as dyn_models  # import your dynamics module
from kamaji.gui.manual_control_tab import ManualControlTab
from kamaji.gui.summary_tab import SummaryTab
from kamaji.gui.scenario_editor_tab import ScenarioEditorTab
from typing import Optional


from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QVBoxLayout, QComboBox, QPushButton, QLineEdit,
    QLabel, QGroupBox
)
from typing import Optional
from kamaji.dynamics import dynamics as dyn_models


class PIDBlock(QWidget):
    def __init__(self, spec=None, state_options=None):
        super().__init__()
        self.layout = QFormLayout()

        self.state_input = QComboBox()
        if state_options:
            self.state_input.addItems(state_options)
        else:
            self.state_input.addItems(["position_x", "position_y", "velocity_x", "velocity_y"])  # fallback

        self.goal_input = QLineEdit("0.0")
        self.kp_input = QLineEdit("1.0")
        self.ki_input = QLineEdit("0.0")
        self.kd_input = QLineEdit("0.0")

        # Safe loading without popups
        if spec:
            self.state_input.setCurrentText(str(spec.get("state", state_options[0] if state_options else "position_x")))
            self.goal_input.setText(str(spec.get("goal", 0.0)))
            # Handle both flat or nested gain formats
            if "kp" in spec:
                self.kp_input.setText(str(spec.get("kp", 1.0)))
                self.ki_input.setText(str(spec.get("ki", 0.0)))
                self.kd_input.setText(str(spec.get("kd", 0.0)))
            elif "gains" in spec:
                gains = spec.get("gains", {})
                self.kp_input.setText(str(gains.get("Kp", 1.0)))
                self.ki_input.setText(str(gains.get("Ki", 0.0)))
                self.kd_input.setText(str(gains.get("Kd", 0.0)))

        self.layout.addRow("State", self.state_input)
        self.layout.addRow("Goal", self.goal_input)
        self.layout.addRow("Kp", self.kp_input)
        self.layout.addRow("Ki", self.ki_input)
        self.layout.addRow("Kd", self.kd_input)
        self.setLayout(self.layout)

    def to_dict(self):
        return {
            "state": self.state_input.currentText(),
            "goal": float(self.goal_input.text()),
            "kp": float(self.kp_input.text()),
            "ki": float(self.ki_input.text()),
            "kd": float(self.kd_input.text())
        }


class AgentEditor(QWidget):
    def __init__(self, agent_id: str, config: Optional[dict] = None):
        super().__init__()
        self.agent_id = agent_id
        self.state_inputs = {}
        self.control_editors = {}
        self.loading = False  # ✅ Prevent popups

        self.layout = QVBoxLayout()
        self.form = QFormLayout()

        self.dynamics_map = {
            "CruiseControl": dyn_models.CruiseControl,
            "Unicycle": dyn_models.Unicycle,
            "SingleIntegrator1DOF": dyn_models.SingleIntegrator1DOF,
            "SingleIntegrator2DOF": dyn_models.SingleIntegrator2DOF,
            "SingleIntegrator3DOF": dyn_models.SingleIntegrator3DOF,
            "DoubleIntegrator1DOF": dyn_models.DoubleIntegrator1DOF,
            "DoubleIntegrator2DOF": dyn_models.DoubleIntegrator2DOF,
            "DoubleIntegrator3DOF": dyn_models.DoubleIntegrator3DOF
        }

        self.dynamics_box = QComboBox()
        self.dynamics_box.addItems(self.dynamics_map.keys())
        self.dynamics_box.currentTextChanged.connect(self.rebuild_fields)
        self.form.addRow("Dynamics Model:", self.dynamics_box)

        self.state_fields_layout = QFormLayout()
        self.form.addRow(QLabel("Initial State:"), QWidget())
        self.form.addRow(self.state_fields_layout)
        self.layout.addLayout(self.form)

        # ✅ Wrap control layout in scroll area
        self.control_group = QGroupBox("Controllers per Control Channel")
        self.control_layout = QFormLayout()

        control_container = QWidget()
        control_container.setLayout(self.control_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(control_container)

        group_layout = QVBoxLayout()
        group_layout.addWidget(self.scroll_area)
        self.control_group.setLayout(group_layout)

        self.layout.addWidget(self.control_group)
        self.setLayout(self.layout)

        if config:
            self.load_from_spec(config)
        else:
            self.rebuild_fields()

    def rebuild_fields(self):
        self.rebuild_state_fields()
        self.rebuild_control_fields()

    def rebuild_state_fields(self):
        for i in reversed(range(self.state_fields_layout.count())):
            widget = self.state_fields_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.state_inputs.clear()
        model_name = self.dynamics_box.currentText()
        state_vars = self.dynamics_map[model_name].state_variables()
        for var in state_vars:
            box = QLineEdit("0.0")
            self.state_inputs[var] = box
            self.state_fields_layout.addRow(var, box)

    def rebuild_control_fields(self):
        for i in reversed(range(self.control_layout.count())):
            widget = self.control_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.control_editors.clear()

        model_name = self.dynamics_box.currentText()
        control_vars = self.dynamics_map[model_name].control_variables()
        state_vars = self.dynamics_map[model_name].state_variables()

        for control in control_vars:
            ctrl_type = QComboBox()
            ctrl_type.addItems(["Constant", "PID"])

            const_val = QLineEdit("0.0")
            pid_widget = PIDBlock(state_options=state_vars)
            pid_widget.hide()

            ctrl_type.currentTextChanged.connect(
                lambda mode, c=control: self.toggle_ctrl_widget(c)
            )

            layout = QVBoxLayout()
            layout.addWidget(ctrl_type)
            layout.addWidget(const_val)
            layout.addWidget(pid_widget)

            container = QWidget()
            container.setLayout(layout)

            self.control_layout.addRow(QLabel(control), container)

            self.control_editors[control] = {
                "dropdown": ctrl_type,
                "constant_input": const_val,
                "pid_block": pid_widget,
                "container": container
            }

    def toggle_ctrl_widget(self, control_name):
        editor = self.control_editors[control_name]
        mode = editor["dropdown"].currentText()
        editor["constant_input"].setVisible(mode == "Constant")
        editor["pid_block"].setVisible(mode == "PID")

    def to_dict(self):
        state_dict = {key: float(box.text()) for key, box in self.state_inputs.items()}
        controller_dict = {}
        for ctrl_name, editor in self.control_editors.items():
            mode = editor["dropdown"].currentText()
            if mode == "Constant":
                val = float(editor["constant_input"].text())
                controller_dict[ctrl_name] = {"type": "Constant", "value": val}
            elif mode == "PID":
                controller_dict[ctrl_name] = {
                    "type": "PID",
                    "specs": [editor["pid_block"].to_dict()]
                }

        return {
            "type": "custom",
            "dynamics_model": self.dynamics_box.currentText(),
            "initial_state": state_dict,
            "controller": controller_dict
        }

    def load_from_spec(self, spec):
        self.loading = True
        self.dynamics_box.setCurrentText(spec.get("dynamics_model", "SingleIntegrator2DOF"))
        self.rebuild_fields()

        init = spec.get("initial_state", {})
        for key, box in self.state_inputs.items():
            box.setText(str(init.get(key, 0.0)))

        controller = spec.get("controller", {})
        for ctrl_name, settings in controller.items():
            editor = self.control_editors.get(ctrl_name)
            if not editor:
                continue
            editor["dropdown"].setCurrentText(settings["type"])
            if settings["type"] == "Constant":
                editor["constant_input"].setText(str(settings["value"]))
            elif settings["type"] == "PID" and settings.get("specs"):
                editor["pid_block"].state_input.setCurrentText(settings["specs"][0].get("state", ""))
                editor["pid_block"].goal_input.setText(str(settings["specs"][0].get("goal", 0.0)))
                editor["pid_block"].kp_input.setText(str(settings["specs"][0].get("kp", 1.0)))
                editor["pid_block"].ki_input.setText(str(settings["specs"][0].get("ki", 0.0)))
                editor["pid_block"].kd_input.setText(str(settings["specs"][0].get("kd", 0.0)))
                self.toggle_ctrl_widget(ctrl_name)
        self.loading = False


class AgentTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.agent_forms = {}
        self.list_widget = QListWidget()
        self.editor_stack = QStackedWidget()

        self.list_widget.currentItemChanged.connect(self.switch_agent)

        add_btn = QPushButton("Add Agent")
        add_btn.clicked.connect(lambda: self.add_agent())

        remove_btn = QPushButton("Remove Agent")
        remove_btn.clicked.connect(lambda: self.remove_selected_agent())

        sidebar = QVBoxLayout()
        sidebar.addWidget(QLabel("Agents"))
        sidebar.addWidget(self.list_widget)
        sidebar.addWidget(add_btn)
        sidebar.addWidget(remove_btn)

        layout = QHBoxLayout()
        layout.addLayout(sidebar)
        layout.addWidget(self.editor_stack)
        self.setLayout(layout)

    def add_agent(self, aid: Optional[str] = None, config: Optional[dict] = None):
        if aid is None:
            base = "agent"
            i = 1
            while f"{base}_{i}" in self.agent_forms:
                i += 1
            aid = f"{base}_{i}"

        if not isinstance(aid, str):
            raise TypeError(f"[AgentTab] Generated aid is not a string: {aid} ({type(aid)})")

        from kamaji.gui.gui_main import AgentEditor  # ensure dynamic import avoids circular issues
        form = AgentEditor(aid, config)
        self.agent_forms[aid] = form
        self.editor_stack.addWidget(form)

        item = QListWidgetItem(aid)
        self.list_widget.addItem(item)
        self.list_widget.setCurrentItem(item)

    def switch_agent(self, current, _):
        if current:
            aid = current.text()
            self.editor_stack.setCurrentWidget(self.agent_forms[aid])

    def remove_selected_agent(self):
        current_item = self.list_widget.currentItem()
        if not current_item:
            return
        aid = current_item.text()

        row = self.list_widget.row(current_item)
        self.list_widget.takeItem(row)

        form = self.agent_forms.pop(aid)
        index = self.editor_stack.indexOf(form)
        if index != -1:
            self.editor_stack.removeWidget(form)
        form.setParent(None)

    def get_agent_config(self):
        return {aid: form.to_dict() for aid, form in self.agent_forms.items()}

    def load_agents_from_config(self, config):
        self.agent_forms.clear()
        self.list_widget.clear()

        while self.editor_stack.count():
            widget = self.editor_stack.widget(0)
            self.editor_stack.removeWidget(widget)
            widget.deleteLater()

        for aid, conf in config.get("agents", {}).items():
            if "dynamics_model" not in conf:
                print(f"[Warning] Skipping agent '{aid}' due to missing dynamics_model.")
                continue
            try:
                self.add_agent(aid, conf)
            except Exception as e:
                print(f"[Warning] Failed to load agent '{aid}': {e}")



class ConfigTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.config_data = None

        main_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)

        # Load YAML button
        load_btn = QPushButton("📂 Load YAML Config")
        load_btn.clicked.connect(self.load_yaml)
        layout.addWidget(load_btn)

        # --- Simulation Settings ---
        sim_box = QGroupBox("⚙️ Simulation Settings")
        sim_form = QFormLayout()

        self.duration_input = QLineEdit("30.0")
        self.time_step_input = QLineEdit("0.1")
        self.integrator_input = QComboBox()
        self.integrator_input.addItems(["Euler", "RK2", "RK4", "RK45"])
        self.seed_input = QLineEdit("42")
        self.verbose_box = QCheckBox()
        self.verbose_box.setChecked(True)

        sim_form.addRow("Duration", self.duration_input)
        sim_form.addRow("Time Step", self.time_step_input)
        sim_form.addRow("Integrator", self.integrator_input)
        sim_form.addRow("Seed", self.seed_input)
        sim_form.addRow("Verbose Output", self.verbose_box)
        sim_box.setLayout(sim_form)

        # --- Logging Settings ---
        log_box = QGroupBox("📝 Logging Settings")
        log_form = QFormLayout()

        self.logging_enabled = QCheckBox()
        self.logging_enabled.setChecked(True)

        self.logging_format = QComboBox()
        self.logging_format.addItems(["hdf5"])
        self.logging_format.currentTextChanged.connect(self.update_filename_extension)

        self.logging_output_dir = QLineEdit("logs/")
        self.logging_filename = QLineEdit("simulation")

        self.logging_compress = QCheckBox()
        self.logging_verbose = QCheckBox()
        self.logging_verbose.setChecked(True)
        self.logging_autosave = QCheckBox()
        self.logging_autosave.setChecked(True)

        log_form.addRow("Enable Logging", self.logging_enabled)
        log_form.addRow("Format", self.logging_format)
        log_form.addRow("Output Directory", self.logging_output_dir)
        log_form.addRow("Filename", self.logging_filename)
        log_form.addRow("Compress (future)", self.logging_compress)
        log_form.addRow("Verbose Logging", self.logging_verbose)
        log_form.addRow("Auto-Save at End", self.logging_autosave)
        log_box.setLayout(log_form)

        # Final assembly
        layout.addWidget(sim_box)
        layout.addWidget(log_box)
        layout.addStretch(1)
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def update_filename_extension(self):
        base = os.path.splitext(self.logging_filename.text())[0]
        if self.logging_format.currentText() == "hdf5":
            ext = "h5"
        self.logging_filename.setText(f"{base}.{ext}")

    def load_yaml(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open config file', '.', 'YAML Files (*.yml *.yaml)')
        if fname:
            with open(fname, 'r') as f:
                self.config_data = yaml.safe_load(f)

            sim = self.config_data.get("simulation", {})
            self.duration_input.setText(str(sim.get("duration", 30.0)))
            self.time_step_input.setText(str(sim.get("time_step", 0.1)))
            self.integrator_input.setCurrentText(sim.get("integrator", "RK4"))
            self.seed_input.setText(str(sim.get("seed", 42)))
            self.verbose_box.setChecked(sim.get("verbose", True))

            logging = self.config_data.get("logging", {})
            self.logging_enabled.setChecked(logging.get("enabled", True))
            self.logging_format.setCurrentText(logging.get("format", "hdf5"))
            self.logging_output_dir.setText(logging.get("output_directory", "logs/"))
            self.logging_filename.setText(logging.get("filename", "simulation"))
            self.logging_compress.setChecked(logging.get("compress", False))
            self.logging_verbose.setChecked(logging.get("verbose", True))
            self.logging_autosave.setChecked(logging.get("autosave", True))

            self.main_window.agent_tab.load_agents_from_config(self.config_data)
            QMessageBox.information(self, "Loaded", f"Loaded config from {fname}")

    def get_config(self):
        filename = self.logging_filename.text()
        if self.logging_format.currentText() == "hdf5":
            ext = "h5"
        if not filename.endswith(f".{ext}"):
            filename += f".{ext}"

        return {
            "simulation": {
                "duration": float(self.duration_input.text()),
                "time_step": float(self.time_step_input.text()),
                "integrator": self.integrator_input.currentText(),
                "seed": int(self.seed_input.text()),
                "verbose": self.verbose_box.isChecked()
            },
            "agents": self.main_window.agent_tab.get_agent_config(),
            "logging": {
                "enabled": self.logging_enabled.isChecked(),
                "format": ext,
                "output_directory": self.logging_output_dir.text(),
                "filename": filename,
                "compress": self.logging_compress.isChecked(),
                "verbose": self.logging_verbose.isChecked(),
                "autosave": self.logging_autosave.isChecked()
            },
            "environment": {}
        }


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kamaji Simulation Launcher")

        self.tabs = QTabWidget()

        self.agent_tab = AgentTab(self)
        self.config_tab = ConfigTab(self)
        self.sim_tab = KamajiSimGUI()  # initialized but empty
        self.diagnostics_tab = DiagnosticsTab()
        self.manual_control_tab = ManualControlTab()
        self.summary_tab = SummaryTab()
        self.scenario_tab = ScenarioEditorTab()

        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)

        self.tabs.addTab(self.config_tab, "Configuration")
        self.tabs.addTab(self.scenario_tab, "Scenario Editor")
        self.tabs.addTab(self.agent_tab, "Agents")
        self.tabs.addTab(self.manual_control_tab, "Manual Control")
        self.tabs.addTab(self.sim_tab, "Simulation")
        self.tabs.addTab(self.summary_tab, "Summary")
        self.tabs.addTab(self.diagnostics_tab, "Diagnostics")
        self.tabs.addTab(self.console_log, "Console")

        self.reset_btn = QPushButton("⟳ Reset Simulation")
        self.reset_btn.clicked.connect(self.reset_simulation)

        self.write_btn = QPushButton("📤 Write Parameters")
        self.write_btn.clicked.connect(self.write_parameters)

        self.init_btn = QPushButton("🛠 Initialize Simulation")
        self.init_btn.clicked.connect(self.initialize_simulation)

        self.manual_control_checkbox = QCheckBox("Inject Manual Control")
        self.manual_control_checkbox.setChecked(True)  # or False by default

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.manual_control_checkbox)
        btn_layout.addWidget(self.init_btn)
        btn_layout.addWidget(self.write_btn)
        btn_layout.addWidget(self.reset_btn)


        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.redirect_stdout()

    def initialize_simulation(self):
        config_data = self.config_tab.get_config()
        print("[Launcher] Initializing simulator with config:", config_data)

        try:
            self.simulator = Simulator(config_data)
            if self.manual_control_checkbox.isChecked():
                control_fn = self.manual_control_tab.get_function()
                if control_fn is None:
                    QMessageBox.warning(self, "Manual Control Error", "Manual control is enabled but your function is invalid.")
                    return
            else:
                control_fn = None
            self.sim_tab.set_simulator(
                self.simulator,
                manual_control_fn=control_fn,
                on_complete=self.summary_tab.set_simulator
            )
            self.tabs.setCurrentWidget(self.sim_tab)
            self.diagnostics_tab.set_simulator(self.simulator)
            self.summary_tab.set_simulator(self.simulator)
            QMessageBox.information(self, "Simulator Ready", "Simulation has been initialized.")
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize Simulator:\n{e}")


    def write_parameters(self):
        config_data = self.config_tab.get_config()
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Config As", "config_out.yaml", "YAML Files (*.yaml *.yml)")
        if out_path:
            with open(out_path, "w") as f:
                yaml.dump(config_data, f)
            QMessageBox.information(self, "Saved", f"Parameters written to {out_path}")


    def redirect_stdout(self):
        self.output_stream = EmittingStream()
        self.output_stream.text_written.connect(self.console_log.append)
        # sys.stdout = self.output_stream
        # sys.stderr = self.output_stream

    def launch_simulation(self):
        config_data = self.config_tab.get_config()
        print("[Launcher] Starting simulation with config:", config_data)

        try:
            self.simulator = Simulator(config_data)
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", f"Failed to initialize Simulator:\n{e}")
            return

        if self.manual_control_checkbox.isChecked():
            control_fn = self.manual_control_tab.get_function()
            if control_fn is None:
                QMessageBox.warning(self, "Manual Control Error", "Manual control is enabled but your function is invalid.")
                return
        else:
            control_fn = None

        self.sim_tab.set_simulator(
            self.simulator,
            manual_control_fn=control_fn,
            on_complete=self.summary_tab.set_simulator
        )

        self.diagnostics_tab.set_simulator(self.simulator)
        self.summary_tab.set_simulator(self.simulator)
        self.tabs.setCurrentWidget(self.sim_tab)

    def reset_simulation(self):
        print("[Launcher] Resetting simulation...")
        self.launch_simulation()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 700)
    win.show()
    sys.exit(app.exec_())
