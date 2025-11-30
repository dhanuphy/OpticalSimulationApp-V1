import sys
import os
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QLineEdit, QLabel, QHBoxLayout, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter


class OpticalEnvelopeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Envelope Fitting Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Data
        self.data = None
        self.wavelength = None
        self.transmittance = None  # normalized 0-1

        # Manual Tmax/Tmin points & envelopes
        self.tmax = []               # list of (x, y)
        self.tmin = []               # list of (x, y)
        self.envelope_upper = None   # (xx, yy)
        self.envelope_lower = None

        # Auto-selected Tmax/Tmin for thickness estimation (from raw spectrum)
        self.auto_tmax = []          # list of (x, y)
        self.auto_tmin = []          # list of (x, y)

        # Optical parameters
        self.thickness_nm = 100.0
        self.substrate_refractive_index = 1.4585
        self.save_folder = os.getcwd()
        self.min_points_required = 5

        self.xx = None
        self.TM = None
        self.Tm = None
        self.n2 = None
        self.alpha = None           # from envelope-based simulation
        self.T_simulated = None
        self.T_exp_xx = None        # experimental T interpolated on xx

        # Manual selection/editing
        self.selection_mode = 'Tmax'   # 'Tmax' or 'Tmin'
        self.dragging = False
        self.drag_point_type = None    # 'Tmax' or 'Tmin'
        self.drag_point_index = None

        self.init_ui()

    # ----------------------------------------------------------------------
    # UI SETUP
    # ----------------------------------------------------------------------
    def init_ui(self):
        container = QWidget()
        layout = QHBoxLayout(container)

        # Styles
        button_base = """
            QPushButton {
                min-height: 40px;
                font-size: 15px;
                font-weight: bold;
                padding: 6px 10px;
                border-radius: 10px;
                border: 1px solid #2c3e50;
            }
            QPushButton:checked {
                border: 2px solid white;
            }
        """

        self.blue_btn = button_base + "QPushButton {background-color:#4A90E2;color:white;}"
        self.green_btn = button_base + "QPushButton {background-color:#27AE60;color:white;}"
        self.red_btn = button_base + "QPushButton {background-color:#E74C3C;color:white;}"
        self.gray_btn = button_base + "QPushButton {background-color:#7F8C8D;color:white;}"

        label_style = "QLabel { font-size: 14px; }"
        line_style = "QLineEdit { min-height: 28px; font-size: 14px; padding: 3px; }"

        # Sidebar
        sidebar = QVBoxLayout()
        sidebar.setSpacing(15)
        sidebar.setContentsMargins(10, 10, 10, 10)

        # 1. Load data
        self.load_btn = QPushButton("Load Data (CSV/TXT/Excel)")
        self.load_btn.setStyleSheet(self.blue_btn)
        self.load_btn.clicked.connect(self.load_data)
        sidebar.addWidget(self.load_btn)

        # 1) Manual envelope selection
        lbl_mode = QLabel("1) Manual Envelope Selection")
        lbl_mode.setStyleSheet(label_style)
        sidebar.addWidget(lbl_mode)

        mode_layout = QHBoxLayout()
        self.tmax_btn = QPushButton("Add Tmax")
        self.tmax_btn.setCheckable(True)
        self.tmax_btn.setChecked(True)
        self.tmax_btn.setStyleSheet(self.gray_btn)
        self.tmax_btn.clicked.connect(lambda: self.set_selection_mode('Tmax'))

        self.tmin_btn = QPushButton("Add Tmin")
        self.tmin_btn.setCheckable(True)
        self.tmin_btn.setStyleSheet(self.gray_btn)
        self.tmin_btn.clicked.connect(lambda: self.set_selection_mode('Tmin'))

        mode_layout.addWidget(self.tmax_btn)
        mode_layout.addWidget(self.tmin_btn)
        sidebar.addLayout(mode_layout)

        self.interp_btn = QPushButton("Interpolate Envelopes")
        self.interp_btn.setStyleSheet(self.blue_btn)
        self.interp_btn.clicked.connect(self.update_envelopes)
        self.interp_btn.setEnabled(False)
        sidebar.addWidget(self.interp_btn)

        # 2) Auto thickness from RAW Tmax/Tmin
        lbl_auto = QLabel("2) Auto Thickness (Raw Spectrum Peaks)")
        lbl_auto.setStyleSheet(label_style)
        sidebar.addWidget(lbl_auto)

        auto_layout = QHBoxLayout()
        self.auto_tmax_btn = QPushButton("Auto Select Tmax")
        self.auto_tmax_btn.setStyleSheet(self.gray_btn)
        self.auto_tmax_btn.clicked.connect(self.auto_select_tmax)

        self.auto_tmin_btn = QPushButton("Auto Select Tmin")
        self.auto_tmin_btn.setStyleSheet(self.gray_btn)
        self.auto_tmin_btn.clicked.connect(self.auto_select_tmin)

        auto_layout.addWidget(self.auto_tmax_btn)
        auto_layout.addWidget(self.auto_tmin_btn)
        sidebar.addLayout(auto_layout)

        self.sim_estimate_btn = QPushButton("Simulate (Estimate Thickness)")
        self.sim_estimate_btn.setStyleSheet(self.blue_btn)
        self.sim_estimate_btn.clicked.connect(self.simulate_with_auto_thickness)
        self.sim_estimate_btn.setEnabled(False)
        sidebar.addWidget(self.sim_estimate_btn)

        # Thickness box (auto-updated after simulate)
        lbl_t = QLabel("Film Thickness (nm):")
        lbl_t.setStyleSheet(label_style)
        sidebar.addWidget(lbl_t)

        self.thickness_input = QLineEdit("100")
        self.thickness_input.setStyleSheet(line_style)
        self.thickness_input.textChanged.connect(self.update_thickness)
        sidebar.addWidget(self.thickness_input)

        # 3) Fine auto fit
        lbl_auto_fit = QLabel("3) Auto Simulate (Fine Fit)")
        lbl_auto_fit.setStyleSheet(label_style)
        sidebar.addWidget(lbl_auto_fit)

        self.auto_btn = QPushButton("Auto Simulate (Fine Fit)")
        self.auto_btn.setStyleSheet(self.blue_btn)
        self.auto_btn.clicked.connect(self.auto_simulate)
        self.auto_btn.setEnabled(False)
        sidebar.addWidget(self.auto_btn)

        # Save folder
        self.save_btn = QPushButton("Select Save Folder")
        self.save_btn.setStyleSheet(self.gray_btn)
        self.save_btn.clicked.connect(self.select_save_folder)
        sidebar.addWidget(self.save_btn)

        # Export CSV
        self.export_btn = QPushButton("Extract & Save Results (CSV)")
        self.export_btn.setStyleSheet(self.green_btn)
        self.export_btn.clicked.connect(self.extract_optical_properties)
        self.export_btn.setEnabled(False)
        sidebar.addWidget(self.export_btn)

        # Clear
        self.clear_btn = QPushButton("Clear Points & Envelopes")
        self.clear_btn.setStyleSheet(self.red_btn)
        self.clear_btn.clicked.connect(self.clear_points)
        sidebar.addWidget(self.clear_btn)

        # Point counter
        self.point_counter = QLabel("Points: Tmax=0, Tmin=0")
        self.point_counter.setStyleSheet(label_style)
        sidebar.addWidget(self.point_counter)

        # Cursor label
        self.cursor_label = QLabel("Cursor: λ = -, T = -")
        self.cursor_label.setStyleSheet(label_style)
        sidebar.addWidget(self.cursor_label)

        sidebar.addStretch()

        # Plot
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # Matplotlib events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        layout.addLayout(sidebar, 1)
        layout.addWidget(self.canvas, 4)

        self.setCentralWidget(container)
        self.showMaximized()

    # ----------------------------------------------------------------------
    # STATE / BUTTON HELPERS
    # ----------------------------------------------------------------------
    def set_selection_mode(self, mode):
        self.selection_mode = mode
        if mode == 'Tmax':
            self.tmax_btn.setChecked(True)
            self.tmin_btn.setChecked(False)
        else:
            self.tmax_btn.setChecked(False)
            self.tmin_btn.setChecked(True)

    def update_button_states(self):
        has_data = self.data is not None
        has_pts = len(self.tmax) >= self.min_points_required and len(self.tmin) >= self.min_points_required
        has_env = self.envelope_upper is not None and self.envelope_lower is not None
        has_auto_points = (len(self.auto_tmax) + len(self.auto_tmin)) >= 2
        has_Texp = self.T_exp_xx is not None
        has_sim = self.T_simulated is not None

        self.interp_btn.setEnabled(has_data and has_pts)
        self.auto_tmax_btn.setEnabled(has_data)
        self.auto_tmin_btn.setEnabled(has_data)
        self.sim_estimate_btn.setEnabled(has_env and has_auto_points)
        self.auto_btn.setEnabled(has_env and has_Texp)
        self.export_btn.setEnabled(has_sim)

    def select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.save_folder = folder

    # ----------------------------------------------------------------------
    # DATA LOADING
    # ----------------------------------------------------------------------
    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "",
            "CSV/TXT/Excel (*.csv *.txt *.xlsx *.xls)"
        )
        if not path:
            return

        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, header=None)
            elif path.endswith(".txt"):
                df = pd.read_csv(path, sep=r"\s+", header=None)
            else:
                df = pd.read_excel(path, header=None)

            if df.shape[1] < 2:
                QMessageBox.warning(self, "Error", "File must contain at least 2 columns.")
                return

            wl = df.iloc[:, 0].astype(float).to_numpy()
            T = df.iloc[:, 1].astype(float).to_numpy()

            if np.any(T > 1.0):
                T = T / 100.0

            max_val = np.nanmax(T)
            if max_val <= 0:
                QMessageBox.warning(self, "Error", "Transmittance values are zero or negative.")
                return

            T = T / max_val

            if np.any(T < 0) or np.any(T > 1.01):
                QMessageBox.warning(self, "Error", "Transmittance must be between 0 and 1 (or 0–100).")
                return

            self.data = df.values
            self.wavelength = wl
            self.transmittance = T

            self.clear_points()
            self.draw_base()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def draw_base(self):
        self.ax.clear()
        if self.wavelength is not None and self.transmittance is not None:
            self.ax.plot(self.wavelength, self.transmittance, 'k-', label="Transmittance")
            self.ax.set_xlabel("Wavelength (nm)")
            self.ax.set_ylabel("T (normalized)")
            self.ax.set_title("1) Click to add Tmax (red) / Tmin (blue) for envelopes")
            self.ax.legend()
        self.canvas.draw()
        self.update_button_states()

    # ----------------------------------------------------------------------
    # MOUSE EVENTS: MANUAL ADD + DRAG EDIT
    # ----------------------------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.xdata is not None and event.ydata is not None:
            self.cursor_label.setText(f"Cursor: λ = {event.xdata:.1f} nm, T = {event.ydata:.3f}")

        if self.wavelength is None or self.transmittance is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Check for dragging existing points
        tolx = (self.wavelength.max() - self.wavelength.min()) * 0.01
        toly = 0.02

        for i, (px, py) in enumerate(self.tmax):
            if abs(px - x) < tolx and abs(py - y) < toly:
                self.dragging = True
                self.drag_point_type = 'Tmax'
                self.drag_point_index = i
                return

        for i, (px, py) in enumerate(self.tmin):
            if abs(px - x) < tolx and abs(py - y) < toly:
                self.dragging = True
                self.drag_point_type = 'Tmin'
                self.drag_point_index = i
                return

        # Otherwise, add new point according to selection_mode
        if self.selection_mode == 'Tmax':
            self.tmax.append((x, y))
        else:
            self.tmin.append((x, y))

        self.point_counter.setText(f"Points: Tmax={len(self.tmax)}, Tmin={len(self.tmin)}")
        self.redraw()
        self.update_button_states()

    def on_motion(self, event):
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.cursor_label.setText(f"Cursor: λ = {event.xdata:.1f} nm, T = {event.ydata:.3f}")

        if not self.dragging:
            return
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        if self.drag_point_type == 'Tmax' and self.drag_point_index is not None:
            self.tmax[self.drag_point_index] = (x, y)
        elif self.drag_point_type == 'Tmin' and self.drag_point_index is not None:
            self.tmin[self.drag_point_index] = (x, y)

        self.redraw()

    def on_release(self, event):
        self.dragging = False
        self.drag_point_type = None
        self.drag_point_index = None

    # ----------------------------------------------------------------------
    # PLOT REDRAW
    # ----------------------------------------------------------------------
    def redraw(self):
        self.ax.clear()
        if self.wavelength is not None and self.transmittance is not None:
            self.ax.plot(self.wavelength, self.transmittance, 'k-', label="Transmittance")

        # Manual Tmax/Tmin
        if self.tmax:
            x_tmax, y_tmax = zip(*self.tmax)
            self.ax.plot(x_tmax, y_tmax, 'ro-', label="Tmax (manual)")
        if self.tmin:
            x_tmin, y_tmin = zip(*self.tmin)
            self.ax.plot(x_tmin, y_tmin, 'bo-', label="Tmin (manual)")

        # Envelopes
        if self.envelope_upper is not None:
            self.ax.plot(*self.envelope_upper, 'r--', label="Upper Envelope")
        if self.envelope_lower is not None:
            self.ax.plot(*self.envelope_lower, 'b--', label="Lower Envelope")

        # Auto-selected raw Tmax/Tmin for thickness
        if self.auto_tmax:
            x_auto_tmax, y_auto_tmax = zip(*self.auto_tmax)
            self.ax.plot(x_auto_tmax, y_auto_tmax, 'm*', markersize=12, label="Auto Tmax (raw)")
        if self.auto_tmin:
            x_auto_tmin, y_auto_tmin = zip(*self.auto_tmin)
            self.ax.plot(x_auto_tmin, y_auto_tmin, 'c*', markersize=12, label="Auto Tmin (raw)")

        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("T (normalized)")

        handles, labels = self.ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        self.ax.legend(uniq.values(), uniq.keys())

        self.canvas.draw()

    # ----------------------------------------------------------------------
    # ENVELOPE INTERPOLATION
    # ----------------------------------------------------------------------
    def update_envelopes(self):
        if len(self.tmax) < self.min_points_required or len(self.tmin) < self.min_points_required:
            QMessageBox.warning(
                self, "Error",
                f"Need at least {self.min_points_required} points each for Tmax and Tmin."
            )
            return

        try:
            self.envelope_upper = self.interpolate_env(self.tmax)
            self.envelope_lower = self.interpolate_env(self.tmin)
            self.redraw()
            self.update_button_states()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Envelope interpolation failed:\n{e}")

    def interpolate_env(self, pts):
        x, y = zip(*sorted(pts))
        x = np.array(x)
        y = np.array(y)

        ux, idx = np.unique(x, return_index=True)
        uy = y[idx]

        kind = 'cubic'
        if len(ux) < 4:
            kind = 'linear'

        f = interp1d(ux, uy, kind=kind, fill_value="extrapolate")
        xx = np.linspace(ux.min(), ux.max(), 200)
        yy = f(xx)
        return xx, yy

    def update_thickness(self):
        try:
            t = float(self.thickness_input.text())
            if t <= 0:
                raise ValueError
            self.thickness_nm = t
        except ValueError:
            self.thickness_input.setText("100")
            self.thickness_nm = 100.0

    def clear_points(self):
        self.tmax = []
        self.tmin = []
        self.auto_tmax = []
        self.auto_tmin = []
        self.envelope_upper = None
        self.envelope_lower = None
        self.xx = None
        self.TM = None
        self.Tm = None
        self.n2 = None
        self.alpha = None
        self.T_simulated = None
        self.T_exp_xx = None
        self.dragging = False
        self.drag_point_type = None
        self.drag_point_index = None
        self.point_counter.setText("Points: Tmax=0, Tmin=0")
        self.draw_base()

    # ----------------------------------------------------------------------
    # AUTO SELECT Tmax/Tmin FROM RAW SPECTRUM
    # ----------------------------------------------------------------------
    def auto_select_tmax(self):
        if self.wavelength is None or self.transmittance is None:
            QMessageBox.warning(self, "Error", "Load data first.")
            return

        try:
            T = self.transmittance
            wl = self.wavelength

            win = min(51, (len(T) // 2) * 2 + 1)
            if win < 5:
                win = 5
            T_smooth = savgol_filter(T, win, 3)

            peaks, _ = find_peaks(T_smooth, distance=20, prominence=0.005)
            if len(peaks) == 0:
                QMessageBox.warning(self, "Auto Tmax", "No maxima found in spectrum.")
                return

            # Choose 1–2 central peaks
            if len(peaks) == 1:
                selected = [peaks[0]]
            else:
                mid = len(peaks) // 2
                selected = [peaks[mid - 1], peaks[mid]] if mid > 0 else [peaks[mid]]

            self.auto_tmax = [(wl[i], T[i]) for i in selected]
            self.redraw()
            self.update_button_states()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto Tmax failed:\n{e}")

    def auto_select_tmin(self):
        if self.wavelength is None or self.transmittance is None:
            QMessageBox.warning(self, "Error", "Load data first.")
            return

        try:
            T = self.transmittance
            wl = self.wavelength

            win = min(51, (len(T) // 2) * 2 + 1)
            if win < 5:
                win = 5
            T_smooth = savgol_filter(T, win, 3)

            valleys, _ = find_peaks(-T_smooth, distance=20, prominence=0.005)
            if len(valleys) == 0:
                QMessageBox.warning(self, "Auto Tmin", "No minima found in spectrum.")
                return

            # Choose 1–2 central valleys
            if len(valleys) == 1:
                selected = [valleys[0]]
            else:
                mid = len(valleys) // 2
                selected = [valleys[mid - 1], valleys[mid]] if mid > 0 else [valleys[mid]]

            self.auto_tmin = [(wl[i], T[i]) for i in selected]
            self.redraw()
            self.update_button_states()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto Tmin failed:\n{e}")

    # ----------------------------------------------------------------------
    # CORE OPTICAL CALC FUNCTIONS
    # ----------------------------------------------------------------------
    def prepare_envelope_TM_Tm(self):
        TM_interp = interp1d(*self.envelope_upper, fill_value="extrapolate")
        Tm_interp = interp1d(*self.envelope_lower, fill_value="extrapolate")
        self.TM = TM_interp(self.xx)
        self.Tm = Tm_interp(self.xx)

    def run_simulation_for_thickness(self, d):
        """
        Given thickness d (nm), compute n2, alpha, and T_simulated on self.xx
        using existing TM, Tm, and substrate index.
        """
        s = self.substrate_refractive_index
        TM = self.TM
        Tm = self.Tm
        xx = self.xx

        N1 = (2 * s * (TM - Tm) / (TM * Tm)) + (s**2 + 1) / 2
        inner = np.maximum(N1**2 - s**2, 0)
        outer = N1 + np.sqrt(inner)
        n2 = np.sqrt(np.maximum(outer, 0))

        Ti = np.clip((2 * TM * Tm) / (TM + Tm), 1e-10, 1)
        alpha = (1 / (d * 1e-7)) * np.log(1 / Ti)

        phi = 4 * np.pi * n2 * d / xx
        A = 16 * n2**2 * s
        B = (n2 + 1)**3 * (n2 + s**2)
        C = 2 * (n2**2 - 1) * (n2**2 - s**2)
        D = (n2 - 1)**3 * (n2 - s**2)
        F = (8 * n2**2 * s) / Ti

        disc = np.maximum(F**2 - (n2**2 - 1)**3 * (n2**2 - s**4), 0)
        x1 = (F - np.sqrt(disc)) / D
        Tsim = (A * x1) / (B - C * x1 * np.cos(phi) + D * x1**2)

        return n2, alpha, Tsim

    # ----------------------------------------------------------------------
    # ROUGH SIM: ESTIMATE THICKNESS FROM AUTO Tmax/Tmin + SIMULATE
    # ----------------------------------------------------------------------
    def estimate_thickness_from_auto_peaks(self):
        """
        Use auto_tmin (first) and auto_tmax points (2 or 4 total) to estimate thickness:
        Prefer alternating Tmin–Tmax or Tmax–Tmin pairs, with decent spacing.
        d ≈ (λ1 * λ2) / (2 |λ1 - λ2|).
        """
        combined = []
        for x, _ in self.auto_tmin:
            combined.append((x, 'Tmin'))
        for x, _ in self.auto_tmax:
            combined.append((x, 'Tmax'))

        if len(combined) < 2:
            return self.thickness_nm

        combined.sort(key=lambda p: p[0])

        # Find alternating pairs Tmin<->Tmax with spacing >= 25 nm
        alt_pairs = []
        for i in range(len(combined) - 1):
            x1, t1 = combined[i]
            x2, t2 = combined[i + 1]
            if t1 != t2 and abs(x2 - x1) >= 25.0:
                alt_pairs.append((x1, x2))

        ds = []
        if alt_pairs:
            # Use up to 2 central alternating pairs
            if len(alt_pairs) == 1:
                pairs_to_use = [alt_pairs[0]]
            else:
                mid = len(alt_pairs) // 2
                start = max(0, mid - 1)
                end = min(len(alt_pairs), start + 2)
                pairs_to_use = alt_pairs[start:end]

            for lam1, lam2 in pairs_to_use:
                if lam1 == lam2:
                    continue
                d_est = abs((lam1 * lam2) / (2 * (lam1 - lam2)))
                if 10 <= d_est <= 5000:
                    ds.append(d_est)

        # Fallback: use all auto points if alt_pairs empty or ds empty
        if not ds:
            xs = sorted([p[0] for p in self.auto_tmax + self.auto_tmin])
            for i in range(len(xs) - 1):
                lam1, lam2 = xs[i], xs[i + 1]
                if lam1 == lam2:
                    continue
                if abs(lam2 - lam1) < 25.0:
                    continue
                d_est = abs((lam1 * lam2) / (2 * (lam1 - lam2)))
                if 10 <= d_est <= 5000:
                    ds.append(d_est)

        if not ds:
            return self.thickness_nm

        return float(np.median(ds))

    def simulate_with_auto_thickness(self):
        """
        Use auto-selected raw Tmax/Tmin (2 or 4 points) to estimate thickness,
        then run a rough simulation and update thickness box.
        """
        if self.envelope_upper is None or self.envelope_lower is None:
            QMessageBox.warning(self, "Error", "Interpolate envelopes first.")
            return

        if len(self.auto_tmax) + len(self.auto_tmin) < 2:
            QMessageBox.warning(self, "Error", "Use Auto Select Tmax/Tmin first.")
            return

        try:
            # Common wavelength grid
            self.xx = np.linspace(self.wavelength.min(), self.wavelength.max(), 200)
            self.prepare_envelope_TM_Tm()

            # Rough thickness estimate from auto peaks
            d_est = self.estimate_thickness_from_auto_peaks()
            self.thickness_nm = d_est
            self.thickness_input.setText(f"{d_est:.2f}")

            # Simulate for rough thickness
            self.n2, self.alpha, self.T_simulated = self.run_simulation_for_thickness(d_est)

            # Prepare experimental curve on xx for later fine fit
            exp_interp = interp1d(self.wavelength, self.transmittance, fill_value="extrapolate")
            self.T_exp_xx = exp_interp(self.xx)

            self.redraw()
            self.ax.plot(self.xx, self.T_simulated, 'm-', label="Rough Simulated (Auto d)")
            handles, labels = self.ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            self.ax.legend(uniq.values(), uniq.keys())
            self.canvas.draw()

            self.update_button_states()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Simulate (Estimate Thickness) failed:\n{e}")

    # ----------------------------------------------------------------------
    # STEP 3: FINE AUTO FIT AROUND CURRENT THICKNESS
    # ----------------------------------------------------------------------
    def auto_simulate(self):
        """
        Fine automatic fit (full-curve RMSE) around current thickness.
        Requires:
        - envelopes
        - previous rough simulation (so T_exp_xx, TM, Tm, xx exist)
        """
        if self.envelope_upper is None or self.envelope_lower is None or self.T_exp_xx is None:
            QMessageBox.warning(self, "Error", "Run 'Simulate (Estimate Thickness)' first.")
            return

        try:
            center = self.thickness_nm
            best_t = self.optimize_full_curve(center)

            self.thickness_nm = best_t
            self.thickness_input.setText(f"{best_t:.2f}")

            # simulate with best thickness
            self.n2, self.alpha, self.T_simulated = self.run_simulation_for_thickness(best_t)

            self.redraw()
            self.ax.plot(self.xx, self.T_simulated, 'g-', label="Fine Fit (Auto d)")
            handles, labels = self.ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            self.ax.legend(uniq.values(), uniq.keys())
            self.canvas.draw()

            self.update_button_states()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Auto simulate failed:\n{e}")

    def optimize_full_curve(self, center_thickness):
        """
        Multi-stage coarse-to-fine search around center_thickness,
        minimizing RMSE between experimental T_exp_xx and simulated T.
        """
        def clip_range(a, b):
            return max(a, 1.0), min(b, 5000.0)

        best = center_thickness

        # Stage 1: ±40%, then ±15%, ±5%, ±1%
        for pct, n in [(0.40, 120), (0.15, 200), (0.05, 300), (0.01, 400)]:
            lo, hi = clip_range(best * (1 - pct), best * (1 + pct))
            arr = np.linspace(lo, hi, n)
            best = self.evaluate_thickness_list(arr)

        return best

    def evaluate_thickness_list(self, thickness_array):
        """
        Evaluate RMSE for each thickness in thickness_array.
        """
        best_d = thickness_array[0]
        best_err = 1e99

        for d in thickness_array:
            try:
                n2, alpha, Tsim = self.run_simulation_for_thickness(d)
                diff = self.T_exp_xx - Tsim
                rmse = np.sqrt(np.nanmean(diff**2))
                if rmse < best_err:
                    best_err = rmse
                    best_d = d
            except Exception:
                continue

        return best_d

    # ----------------------------------------------------------------------
    # EXPORT RESULTS (with interpolated envelopes & α from raw T)
    # ----------------------------------------------------------------------
    def extract_optical_properties(self):
        if self.n2 is None or self.alpha is None or self.T_simulated is None:
            QMessageBox.warning(self, "Error", "Run simulation first.")
            return

        wl = self.wavelength  # nm
        T_exp = self.transmittance  # 0–1
        xx = self.xx

        # Interpolate n, Tsim, envelopes onto experimental wavelength grid
        n = interp1d(xx, self.n2, fill_value="extrapolate")(wl)
        Tsim = interp1d(xx, self.T_simulated, fill_value="extrapolate")(wl)

        if self.envelope_upper is not None:
            T_upper = interp1d(
                self.envelope_upper[0], self.envelope_upper[1],
                bounds_error=False, fill_value=np.nan
            )(wl)
        else:
            T_upper = np.full_like(wl, np.nan, dtype=float)

        if self.envelope_lower is not None:
            T_lower = interp1d(
                self.envelope_lower[0], self.envelope_lower[1],
                bounds_error=False, fill_value=np.nan
            )(wl)
        else:
            T_lower = np.full_like(wl, np.nan, dtype=float)

        # Energy
        E = 1240.0 / wl  # eV

        # alpha_raw from experimental T only (use T_clean = max(T, 1e-6))
        d_cm = self.thickness_nm * 1e-7  # nm -> cm
        T_clean = np.clip(T_exp, 1e-6, 1.0)
        alpha_raw = -(1.0 / d_cm) * np.log(T_clean)  # cm^-1

        # k, dielectric, etc., based on alpha_raw and n
        k = (alpha_raw * wl) / (4.0 * np.pi * 1e7)  # extinction coefficient
        e1 = n**2 - k**2
        e2 = 2.0 * n * k

        with np.errstate(divide='ignore', invalid='ignore'):
            tan_delta = e2 / e1

        # Skin depth (nm): δ(cm) = 1/α, convert to nm: δ(nm) = 1e7 / α
        skin_depth_nm = 1e7 / alpha_raw

        # Optical conductivity
        c_cm_s = 3e10
        sigma = (alpha_raw * n * c_cm_s) / (4.0 * np.pi)

        # Optical density
        OD = alpha_raw * d_cm

        # α·E and related
        alphaE = alpha_raw * E
        alphaE_sqrt = np.sqrt(alphaE)
        alphaE_sq = alphaE**2

        # dα/dE
        d_alpha_dE = np.gradient(alpha_raw, E)

        # ln(alpha)
        ln_alpha = np.log(alpha_raw)

        df = pd.DataFrame({
            "Wavelength (nm)": wl,
            "Transmittance (0-1)": T_exp,
            "Transmittance (%)": T_exp * 100.0,
            "Upper Envelope T": T_upper,
            "Lower Envelope T": T_lower,
            "Simulated T": Tsim,
            "n": n,
            "k": k,
            "e1": e1,
            "e2": e2,
            "tan_delta": tan_delta,
            "skin_depth (nm)": skin_depth_nm,
            "sigma": sigma,
            "optical_density": OD,
            "Energy (eV)": E,
            "alpha_raw (cm^-1)": alpha_raw,
            "alphaE": alphaE,
            "(alphaE)^0.5": alphaE_sqrt,
            "(alphaE)^2": alphaE_sq,
            "dAlpha/dE": d_alpha_dE,
            "ln(alpha)": ln_alpha
        })

        out = os.path.join(self.save_folder, "Optical_Properties.csv")
        try:
            df.to_csv(out, index=False)
            QMessageBox.information(self, "Saved", f"Results saved to:\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OpticalEnvelopeApp()
    sys.exit(app.exec_())
