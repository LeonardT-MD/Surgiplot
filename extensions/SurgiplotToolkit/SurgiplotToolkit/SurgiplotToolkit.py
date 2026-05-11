from __future__ import annotations

import csv
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

from SurgiplotToolkitLib.surgiplot_bridge import SurgiplotMetricBridge


class SurgiplotToolkit(ScriptedLoadableModule):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        parent.title = "SurgiplotToolkit"
        parent.categories = ["Neurosurgery"]
        parent.dependencies = ["Markups"]
        parent.contributors = ["Leonardo Tariciotti"]
        parent.helpText = (
            "SurgiplotToolkit integrates Surgiplot metric families into 3D Slicer. "
            "This first prototype reads Slicer markups, calls the shared Surgiplot "
            "backend, and renders selected debug geometry back into the Slicer scene."
        )
        parent.acknowledgementText = (
            "Prototype extension scaffold for Surgiplot integration into 3D Slicer."
        )


class SurgiplotToolkitWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        super().setup()
        self.logic = SurgiplotToolkitLogic()
        self._session_mode: Optional[str] = None
        self._session_observed_nodes: list[Any] = []
        self._landmark_node = None
        self._landmark_node_observer = None
        self._landmark_last_count = 0
        self._scale_node = None
        self._scale_node_observer = None
        self._scale_last_count = 0
        self._pending_collect_spec: Optional[dict[str, str]] = None
        self._scene_scale_to_mm = 1.0
        self._updating_landmark_table = False
        self._metric_node_observed_nodes: list[Any] = []
        self._pending_scale_factor_to_mm: Optional[float] = None
        self._active_multicollect_spec: Optional[dict[str, str]] = None
        self._place_click_observers: list[tuple[Any, Any]] = []
        self._pending_place_click = False
        self._last_place_click_time = 0.0
        self._last_metric_payload: Optional[dict[str, Any]] = None
        self._build_ui()
        self._attach_status_observers()
        self._refresh_all_status()

    def _build_ui(self) -> None:
        self.topBarWidget = self._build_top_toolbar()
        self.datasetSection = self._build_dataset_section()
        self.analysisSection = self._build_morphometric_analysis_section()
        self.resultsSection = self._build_results_section()
        right_panel = qt.QWidget()
        right_layout = qt.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.addWidget(self.analysisSection, 1)
        right_layout.addWidget(self.resultsSection, 0)

        self.mainSplitter = qt.QSplitter(qt.Qt.Vertical)
        self.mainSplitter.addWidget(self.datasetSection)
        self.mainSplitter.addWidget(right_panel)
        self.mainSplitter.setChildrenCollapsible(False)
        self.mainSplitter.setStretchFactor(0, 4)
        self.mainSplitter.setStretchFactor(1, 6)
        self.mainSplitter.setSizes([440, 760])

        self.layout.addWidget(self.topBarWidget)
        self.layout.addWidget(self.mainSplitter, 1)
        self.layout.addStretch(1)

    def _build_top_toolbar(self):
        widget = qt.QWidget()
        layout = qt.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title = qt.QLabel("SurgiplotToolkit")
        title_font = title.font
        if callable(title_font):
            title_font = title.font()
        title_font.setPointSize(max(11, int(title_font.pointSize())))
        title_font.setBold(True)
        title.setFont(title_font)

        self.reloadModuleButton = qt.QPushButton("Reload")
        self.restartSlicerButton = qt.QPushButton("Restart")
        self.reloadModuleButton.setMaximumWidth(80)
        self.restartSlicerButton.setMaximumWidth(80)
        self.reloadModuleButton.clicked.connect(self.onReloadModule)
        self.restartSlicerButton.clicked.connect(self.onRestartSlicer)

        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.reloadModuleButton)
        layout.addWidget(self.restartSlicerButton)
        return widget

    def _build_dataset_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Scene Context & Landmark Dataset"
        box.collapsed = False
        layout = qt.QVBoxLayout(box)

        info = qt.QLabel(
            "Choose the active anatomical context, collect reusable landmarks into the shared dataset, "
            "and reuse those points across metrics."
        )
        info.wordWrap = True
        layout.addWidget(info)

        context_form = qt.QFormLayout()
        layout.addLayout(context_form)
        self.referenceSelector = self._new_reference_selector()
        self.imageVolumeSelector = self._new_volume_selector()
        self.setBackgroundButton = qt.QPushButton("Set Background Volume")
        self.setBackgroundButton.clicked.connect(self.onSetBackgroundVolume)
        context_form.addRow("Analysis context (3D model / CT / MRI / segmentation):", self.referenceSelector)
        context_form.addRow("CT / MRI slice background (optional):", self.imageVolumeSelector)
        context_form.addRow(self.setBackgroundButton)

        return self._populate_landmark_dataset_content(box, layout)

    def _populate_landmark_dataset_content(self, box, layout):
        top_row = qt.QHBoxLayout()
        self.createDatasetNodeButton = qt.QPushButton("Open / Use Dataset")
        self.createScaleNodeButton = qt.QPushButton("Open / Use Scale Node")
        self.scaleDistanceSpin = qt.QDoubleSpinBox()
        self.scaleDistanceSpin.setRange(0.01, 500.0)
        self.scaleDistanceSpin.setDecimals(2)
        self.scaleDistanceSpin.setValue(1.0)
        self.scaleDistanceSpin.setSuffix(" cm")
        self.pickScaleButton = qt.QPushButton("Pick 2 Scale Points")
        self.applyScaleButton = qt.QPushButton("Apply Rescaling")
        self.applyScaleButton.setEnabled(False)
        top_row.addWidget(self.createDatasetNodeButton)
        top_row.addWidget(self.createScaleNodeButton)
        top_row.addWidget(qt.QLabel("Known distance:"))
        top_row.addWidget(self.scaleDistanceSpin)
        top_row.addWidget(self.pickScaleButton)
        top_row.addWidget(self.applyScaleButton)
        layout.addLayout(top_row)

        pick_row = qt.QHBoxLayout()
        self.customLabelEdit = qt.QLineEdit()
        self.customLabelEdit.setPlaceholderText("Custom landmark label")
        self.pickCustomLabelButton = qt.QPushButton("Pick Custom Label")
        pick_row.addWidget(self.customLabelEdit, 1)
        pick_row.addWidget(self.pickCustomLabelButton)
        layout.addLayout(pick_row)

        self.datasetStatus = qt.QPlainTextEdit()
        self.datasetStatus.readOnly = True
        self.datasetStatus.setMaximumHeight(95)
        self.datasetStatus.setPlaceholderText("Dataset status will appear here.")
        layout.addWidget(self.datasetStatus)

        self.landmarkTable = qt.QTableWidget()
        self.landmarkTable.setColumnCount(4)
        self.landmarkTable.setHorizontalHeaderLabels(["Label", "X", "Y", "Z"])
        header = self.landmarkTable.horizontalHeader()
        if hasattr(header, "setSectionResizeMode"):
            header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
            for col in (1, 2, 3):
                header.setSectionResizeMode(col, qt.QHeaderView.ResizeToContents)
        self.landmarkTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.landmarkTable.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self.landmarkTable.setMinimumHeight(220)
        layout.addWidget(self.landmarkTable)

        edit_row = qt.QHBoxLayout()
        self.renameSelectedLandmarkButton = qt.QPushButton("Rename Selected")
        self.saveLandmarkLabelsButton = qt.QPushButton("Save Label Edits")
        self.deleteSelectedLandmarksButton = qt.QPushButton("Delete Selected")
        self.moveLandmarkUpButton = qt.QPushButton("Move Up")
        self.moveLandmarkDownButton = qt.QPushButton("Move Down")
        self.showDatasetPointsButton = qt.QPushButton("Show Dataset Points")
        self.hideDatasetPointsButton = qt.QPushButton("Hide Dataset Points")
        self.clearLandmarksButton = qt.QPushButton("Clear Dataset")
        for button in (
            self.renameSelectedLandmarkButton,
            self.saveLandmarkLabelsButton,
            self.deleteSelectedLandmarksButton,
            self.moveLandmarkUpButton,
            self.moveLandmarkDownButton,
            self.showDatasetPointsButton,
            self.hideDatasetPointsButton,
            self.clearLandmarksButton,
        ):
            edit_row.addWidget(button)
        edit_row.addStretch(1)
        layout.addLayout(edit_row)

        self.createDatasetNodeButton.clicked.connect(self.onCreateLandmarkDatasetNode)
        self.createScaleNodeButton.clicked.connect(self.onCreateScaleNode)
        self.pickScaleButton.clicked.connect(self.onPickScalePoints)
        self.applyScaleButton.clicked.connect(self.onApplyScaleCalibration)
        self.pickCustomLabelButton.clicked.connect(self.onPickCustomLabel)
        self.renameSelectedLandmarkButton.clicked.connect(self.onRenameSelectedLandmark)
        self.saveLandmarkLabelsButton.clicked.connect(self.onSaveLandmarkLabelEdits)
        self.deleteSelectedLandmarksButton.clicked.connect(self.onDeleteSelectedLandmarks)
        self.moveLandmarkUpButton.clicked.connect(lambda: self.onMoveSelectedLandmark(-1))
        self.moveLandmarkDownButton.clicked.connect(lambda: self.onMoveSelectedLandmark(1))
        self.showDatasetPointsButton.clicked.connect(lambda: self.onSetDatasetVisibility(True))
        self.hideDatasetPointsButton.clicked.connect(lambda: self.onSetDatasetVisibility(False))
        self.clearLandmarksButton.clicked.connect(self.onClearLandmarkDataset)
        self.landmarkTable.itemSelectionChanged.connect(self.onLandmarkSelectionChanged)
        return box

    def _build_morphometric_analysis_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Morphometric Analysis"
        box.collapsed = False
        layout = qt.QVBoxLayout(box)

        button_row = qt.QHBoxLayout()
        self.btnOpenDistance = qt.QPushButton("Distance")
        self.btnOpenArea = qt.QPushButton("Area of Exposure")
        self.btnOpenVolume = qt.QPushButton("Volume")
        self.btnOpenAoa = qt.QPushButton("AoA / SF")
        self.btnOpenAoe = qt.QPushButton("AoE")
        self.btnOpenVom = qt.QPushButton("VOM / sVOM / VoA")
        for button in (
            self.btnOpenDistance,
            self.btnOpenArea,
            self.btnOpenVolume,
            self.btnOpenAoa,
            self.btnOpenAoe,
            self.btnOpenVom,
        ):
            button_row.addWidget(button)
        layout.addLayout(button_row)

        self.workspaceStatus = qt.QPlainTextEdit()
        self.workspaceStatus.readOnly = True
        self.workspaceStatus.setMaximumHeight(90)
        self.workspaceStatus.setPlaceholderText("Select a metric to open its panel.")
        layout.addWidget(self.workspaceStatus)

        self.entrySelector = self._new_markups_selector()
        self.targetSelector = self._new_markups_selector()
        self.pivotSelector = self._new_markups_selector()
        self.aSelector = self._new_markups_selector()
        self.bSelector = self._new_markups_selector()
        self.cSelector = self._new_markups_selector()
        self.polygonSelector = self._new_markups_selector()
        self.volumeSelector = self._new_markups_selector()

        self.standDistSpin = qt.QDoubleSpinBox()
        self.standDistSpin.setRange(0.01, 1000.0)
        self.standDistSpin.setDecimals(2)
        self.standDistSpin.setValue(10.0)
        self.standDistSpin.setSuffix(" mm")

        self.sfRadiusSpin = qt.QDoubleSpinBox()
        self.sfRadiusSpin.setRange(0.0, 1000.0)
        self.sfRadiusSpin.setDecimals(2)
        self.sfRadiusSpin.setValue(10.0)
        self.sfRadiusSpin.setSuffix(" mm")
        self.sfStandardizedCheckBox = qt.QCheckBox("Enable standardized SF")
        self.sfStandardizedCheckBox.setChecked(False)
        self.sfRadiusSpin.setEnabled(False)
        self.sfStandardizedCheckBox.toggled.connect(self.sfRadiusSpin.setEnabled)

        self.metricPanelDistance = self._build_distance_panel()
        self.metricPanelArea = self._build_area_panel()
        self.metricPanelVolume = self._build_volume_panel()
        self.metricPanelAoa = self._build_aoa_panel()
        self.metricPanelAoe = self._build_aoe_panel()
        self.metricPanelVom = self._build_vom_panel()

        for panel in (
            self.metricPanelDistance,
            self.metricPanelArea,
            self.metricPanelVolume,
            self.metricPanelAoa,
            self.metricPanelAoe,
            self.metricPanelVom,
        ):
            panel.collapsed = True
            layout.addWidget(panel)

        self.btnOpenDistance.clicked.connect(lambda: self._open_metric_panel("distance"))
        self.btnOpenArea.clicked.connect(lambda: self._open_metric_panel("area"))
        self.btnOpenVolume.clicked.connect(lambda: self._open_metric_panel("volume"))
        self.btnOpenAoa.clicked.connect(lambda: self._open_metric_panel("aoa"))
        self.btnOpenAoe.clicked.connect(lambda: self._open_metric_panel("aoe"))
        self.btnOpenVom.clicked.connect(lambda: self._open_metric_panel("vom"))
        return box

    def _build_landmark_dataset_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Landmark Dataset Workflow"
        layout = qt.QVBoxLayout(box)

        info = qt.QLabel(
            "This mirrors the Surgiplot 3D-scene workflow: collect labeled landmarks progressively into a shared dataset, "
            "then export the selected labels into Entry, Target, or Pivot markups for the metric you want to compute."
        )
        info.wordWrap = True
        layout.addWidget(info)

        top_row = qt.QHBoxLayout()
        self.createDatasetNodeButton = qt.QPushButton("Create / Use Landmark Dataset")
        self.createScaleNodeButton = qt.QPushButton("Create / Use Scale Node")
        self.scaleDistanceSpin = qt.QDoubleSpinBox()
        self.scaleDistanceSpin.setRange(0.01, 500.0)
        self.scaleDistanceSpin.setDecimals(2)
        self.scaleDistanceSpin.setValue(1.0)
        self.scaleDistanceSpin.setSuffix(" cm")
        self.pickScaleButton = qt.QPushButton("Pick 2 Scale Points")
        top_row.addWidget(self.createDatasetNodeButton)
        top_row.addWidget(self.createScaleNodeButton)
        top_row.addWidget(qt.QLabel("Known distance:"))
        top_row.addWidget(self.scaleDistanceSpin)
        top_row.addWidget(self.pickScaleButton)
        layout.addLayout(top_row)

        pick_row_1 = qt.QHBoxLayout()
        self.pickCranialButton = qt.QPushButton("Pick Cranial")
        self.pickCaudalButton = qt.QPushButton("Pick Caudal")
        self.pickMedialButton = qt.QPushButton("Pick Medial")
        self.pickLateralButton = qt.QPushButton("Pick Lateral")
        self.pickPivotDatasetButton = qt.QPushButton("Pick Pivot")
        for button in (
            self.pickCranialButton,
            self.pickCaudalButton,
            self.pickMedialButton,
            self.pickLateralButton,
            self.pickPivotDatasetButton,
        ):
            pick_row_1.addWidget(button)
        layout.addLayout(pick_row_1)

        pick_row_2 = qt.QHBoxLayout()
        self.pickEntryPointButton = qt.QPushButton("Pick Entry Point")
        self.pickTargetPointButton = qt.QPushButton("Pick Target Point")
        self.customLabelEdit = qt.QLineEdit()
        self.customLabelEdit.setPlaceholderText("Custom landmark label")
        self.pickCustomLabelButton = qt.QPushButton("Pick Custom Label")
        pick_row_2.addWidget(self.pickEntryPointButton)
        pick_row_2.addWidget(self.pickTargetPointButton)
        pick_row_2.addWidget(self.customLabelEdit, 1)
        pick_row_2.addWidget(self.pickCustomLabelButton)
        layout.addLayout(pick_row_2)

        self.datasetStatus = qt.QPlainTextEdit()
        self.datasetStatus.readOnly = True
        self.datasetStatus.setMaximumHeight(110)
        self.datasetStatus.setPlaceholderText("Landmark dataset status will appear here.")
        layout.addWidget(self.datasetStatus)

        self.landmarkTable = qt.QTableWidget()
        self.landmarkTable.setColumnCount(4)
        self.landmarkTable.setHorizontalHeaderLabels(["Label", "X", "Y", "Z"])
        self.landmarkTable.horizontalHeader().setStretchLastSection(True)
        self.landmarkTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.landmarkTable.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.landmarkTable)

        export_row = qt.QHBoxLayout()
        self.useSelectedEntryButton = qt.QPushButton("Use Selected for Entry")
        self.useSelectedTargetButton = qt.QPushButton("Use Selected for Target")
        self.useSelectedPivotButton = qt.QPushButton("Use Selected for Pivot")
        self.autoFillAoaButton = qt.QPushButton("Auto-fill AoA From Labels")
        for button in (
            self.useSelectedEntryButton,
            self.useSelectedTargetButton,
            self.useSelectedPivotButton,
            self.autoFillAoaButton,
        ):
            export_row.addWidget(button)
        layout.addLayout(export_row)

        clear_row = qt.QHBoxLayout()
        self.renameSelectedLandmarkButton = qt.QPushButton("Rename Selected")
        self.deleteSelectedLandmarksButton = qt.QPushButton("Delete Selected")
        self.moveLandmarkUpButton = qt.QPushButton("Move Up")
        self.moveLandmarkDownButton = qt.QPushButton("Move Down")
        self.clearLandmarksButton = qt.QPushButton("Clear Landmark Dataset")
        clear_row.addWidget(self.renameSelectedLandmarkButton)
        clear_row.addWidget(self.deleteSelectedLandmarksButton)
        clear_row.addWidget(self.moveLandmarkUpButton)
        clear_row.addWidget(self.moveLandmarkDownButton)
        clear_row.addWidget(self.clearLandmarksButton)
        clear_row.addStretch(1)
        layout.addLayout(clear_row)

        metric_assign_row = qt.QHBoxLayout()
        self.useSelectedAButton = qt.QPushButton("Use Selected for A")
        self.useSelectedBButton = qt.QPushButton("Use Selected for B")
        self.useSelectedCButton = qt.QPushButton("Use Selected for C")
        self.useSelectedPolygonButton = qt.QPushButton("Use Selected for Polygon")
        self.useSelectedVolumeButton = qt.QPushButton("Use Selected for Volume")
        for button in (
            self.useSelectedAButton,
            self.useSelectedBButton,
            self.useSelectedCButton,
            self.useSelectedPolygonButton,
            self.useSelectedVolumeButton,
        ):
            metric_assign_row.addWidget(button)
        layout.addLayout(metric_assign_row)

        self.createDatasetNodeButton.clicked.connect(self.onCreateLandmarkDatasetNode)
        self.createScaleNodeButton.clicked.connect(self.onCreateScaleNode)
        self.pickScaleButton.clicked.connect(self.onPickScalePoints)
        self.pickCranialButton.clicked.connect(lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "cranial"}))
        self.pickCaudalButton.clicked.connect(lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "caudal"}))
        self.pickMedialButton.clicked.connect(lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "medial"}))
        self.pickLateralButton.clicked.connect(lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "lateral"}))
        self.pickPivotDatasetButton.clicked.connect(lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "pivot"}))
        self.pickEntryPointButton.clicked.connect(lambda: self.onCollectLabeledLandmark({"mode": "prefix", "prefix": "entry"}))
        self.pickTargetPointButton.clicked.connect(lambda: self.onCollectLabeledLandmark({"mode": "prefix", "prefix": "target"}))
        self.pickCustomLabelButton.clicked.connect(self.onPickCustomLabel)
        self.useSelectedEntryButton.clicked.connect(lambda: self.onExportSelectedLandmarks("entry"))
        self.useSelectedTargetButton.clicked.connect(lambda: self.onExportSelectedLandmarks("target"))
        self.useSelectedPivotButton.clicked.connect(lambda: self.onExportSelectedLandmarks("pivot"))
        self.useSelectedAButton.clicked.connect(lambda: self.onExportSelectedLandmarks("a"))
        self.useSelectedBButton.clicked.connect(lambda: self.onExportSelectedLandmarks("b"))
        self.useSelectedCButton.clicked.connect(lambda: self.onExportSelectedLandmarks("c"))
        self.useSelectedPolygonButton.clicked.connect(lambda: self.onExportSelectedLandmarks("polygon"))
        self.useSelectedVolumeButton.clicked.connect(lambda: self.onExportSelectedLandmarks("volume"))
        self.autoFillAoaButton.clicked.connect(self.onAutoFillAoaFromDataset)
        self.renameSelectedLandmarkButton.clicked.connect(self.onRenameSelectedLandmark)
        self.deleteSelectedLandmarksButton.clicked.connect(self.onDeleteSelectedLandmarks)
        self.moveLandmarkUpButton.clicked.connect(lambda: self.onMoveSelectedLandmark(-1))
        self.moveLandmarkDownButton.clicked.connect(lambda: self.onMoveSelectedLandmark(1))
        self.clearLandmarksButton.clicked.connect(self.onClearLandmarkDataset)
        return box

    def _build_workflow_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Workflow Overview"
        layout = qt.QVBoxLayout(box)

        info = qt.QLabel(
            "Recommended order: 1) choose an image/model context, 2) create or select Entry/Target/Pivot markups, "
            "3) use guided placement to acquire points, 4) validate inputs, 5) compute metrics and inspect the generated scene nodes."
        )
        info.wordWrap = True
        layout.addWidget(info)

        self.validationStatus = qt.QPlainTextEdit()
        self.validationStatus.readOnly = True
        self.validationStatus.setMaximumHeight(180)
        self.validationStatus.setPlaceholderText("Current workflow validation will appear here.")
        layout.addWidget(self.validationStatus)

        row = qt.QHBoxLayout()
        self.validateInputsButton = qt.QPushButton("Validate Current Inputs")
        self.clearResultsButton = qt.QPushButton("Clear Result Nodes")
        self.validateInputsButton.clicked.connect(self.onValidateInputs)
        self.clearResultsButton.clicked.connect(self.onClearResults)
        row.addWidget(self.validateInputsButton)
        row.addWidget(self.clearResultsButton)
        layout.addLayout(row)

        workspace_row_1 = qt.QHBoxLayout()
        self.openAoaWorkspaceButton = qt.QPushButton("Open AoA / SF Workspace")
        self.openVomWorkspaceButton = qt.QPushButton("Open VOM Workspace")
        self.openAoeWorkspaceButton = qt.QPushButton("Open AoE Workspace")
        for button in (
            self.openAoaWorkspaceButton,
            self.openVomWorkspaceButton,
            self.openAoeWorkspaceButton,
        ):
            workspace_row_1.addWidget(button)
        layout.addLayout(workspace_row_1)

        workspace_row_2 = qt.QHBoxLayout()
        self.openDistanceWorkspaceButton = qt.QPushButton("Open Distance Workspace")
        self.openAreaWorkspaceButton = qt.QPushButton("Open Area Workspace")
        self.openVolumeWorkspaceButton = qt.QPushButton("Open Volume Workspace")
        for button in (
            self.openDistanceWorkspaceButton,
            self.openAreaWorkspaceButton,
            self.openVolumeWorkspaceButton,
        ):
            workspace_row_2.addWidget(button)
        layout.addLayout(workspace_row_2)

        self.workspaceStatus = qt.QPlainTextEdit()
        self.workspaceStatus.readOnly = True
        self.workspaceStatus.setMaximumHeight(110)
        self.workspaceStatus.setPlaceholderText("Metric-specific workflow guidance will appear here.")
        layout.addWidget(self.workspaceStatus)

        self.openAoaWorkspaceButton.clicked.connect(lambda: self._open_metric_workspace("aoa"))
        self.openVomWorkspaceButton.clicked.connect(lambda: self._open_metric_workspace("vom"))
        self.openAoeWorkspaceButton.clicked.connect(lambda: self._open_metric_workspace("aoe"))
        self.openDistanceWorkspaceButton.clicked.connect(lambda: self._open_metric_workspace("distance"))
        self.openAreaWorkspaceButton.clicked.connect(lambda: self._open_metric_workspace("area"))
        self.openVolumeWorkspaceButton.clicked.connect(lambda: self._open_metric_workspace("volume"))
        return box

    def _metric_action_row(self, actions: list[tuple[str, Any]]) -> qt.QHBoxLayout:
        row = qt.QHBoxLayout()
        for label, callback in actions:
            btn = qt.QPushButton(label)
            btn.clicked.connect(callback)
            row.addWidget(btn)
        row.addStretch(1)
        return row

    def _metric_scene_controls(self, metric: str, input_roles: list[str]) -> qt.QHBoxLayout:
        row = qt.QHBoxLayout()
        show_overlay = qt.QPushButton("Show 3D Overlay")
        hide_overlay = qt.QPushButton("Hide 3D Overlay")
        show_points = qt.QPushButton("Show Points")
        hide_points = qt.QPushButton("Hide Points")
        export_scene = qt.QPushButton("Save 3D Scene")
        show_overlay.clicked.connect(lambda: self.onSetMetricOverlayVisibility(metric, True))
        hide_overlay.clicked.connect(lambda: self.onSetMetricOverlayVisibility(metric, False))
        show_points.clicked.connect(lambda: self.onSetMetricInputVisibility(input_roles, True))
        hide_points.clicked.connect(lambda: self.onSetMetricInputVisibility(input_roles, False))
        export_scene.clicked.connect(lambda: self.onExportMetricScene(metric))
        for button in (show_overlay, hide_overlay, show_points, hide_points, export_scene):
            row.addWidget(button)
        row.addStretch(1)
        return row

    def _metric_results_box(self, placeholder: str) -> qt.QPlainTextEdit:
        box = qt.QPlainTextEdit()
        box.readOnly = True
        box.setMinimumHeight(180)
        box.setMaximumHeight(260)
        box.setPlaceholderText(placeholder)
        return box

    def _build_distance_panel(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Distance"
        layout = qt.QVBoxLayout(box)
        form = qt.QFormLayout()
        form.addRow("Point A:", self.aSelector)
        form.addRow("Point B:", self.bSelector)
        layout.addLayout(form)
        layout.addLayout(self._metric_action_row([
            ("Use Selected for A", lambda: self.onExportSelectedLandmarks("a")),
            ("Use Selected for B", lambda: self.onExportSelectedLandmarks("b")),
            ("Pick A", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "A", "mirror_role": "a"})),
            ("Pick B", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "B", "mirror_role": "b"})),
        ]))
        layout.addLayout(self._metric_scene_controls("distance", ["a", "b"]))
        self.computeDistanceButton = qt.QPushButton("Compute Distance")
        self.computeDistanceButton.clicked.connect(self.onComputeDistance)
        layout.addWidget(self.computeDistanceButton)
        self.distanceResultsText = self._metric_results_box("Distance results will appear here.")
        layout.addWidget(self.distanceResultsText)
        return box

    def _build_area_panel(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Area of Exposure"
        layout = qt.QVBoxLayout(box)
        form = qt.QFormLayout()
        form.addRow("Polygon points:", self.polygonSelector)
        layout.addLayout(form)
        hint = qt.QLabel("Select landmarks in the desired polygon order, then use them for Polygon.")
        hint.wordWrap = True
        layout.addWidget(hint)
        layout.addLayout(self._metric_action_row([
            ("Use Selected for Polygon", lambda: self.onExportSelectedLandmarks("polygon")),
            ("Start Polygon Picking", lambda: self.onStartSequentialCollection({"mode": "prefix", "prefix": "polygon", "mirror_role": "polygon", "repeat": "true"})),
            ("Stop Picking", self.onStopSequentialCollection),
        ]))
        layout.addLayout(self._metric_scene_controls("area", ["polygon"]))
        self.computeAreaButton = qt.QPushButton("Compute Area")
        self.computeAreaButton.clicked.connect(self.onComputeArea)
        layout.addWidget(self.computeAreaButton)
        self.areaResultsText = self._metric_results_box("Area of exposure results will appear here.")
        layout.addWidget(self.areaResultsText)
        return box

    def _build_volume_panel(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Volume"
        layout = qt.QVBoxLayout(box)
        form = qt.QFormLayout()
        form.addRow("Volume points:", self.volumeSelector)
        layout.addLayout(form)
        layout.addLayout(self._metric_action_row([
            ("Use Selected for Volume", lambda: self.onExportSelectedLandmarks("volume")),
            ("Start Volume Picking", lambda: self.onStartSequentialCollection({"mode": "prefix", "prefix": "volume", "mirror_role": "volume", "repeat": "true"})),
            ("Stop Picking", self.onStopSequentialCollection),
        ]))
        layout.addLayout(self._metric_scene_controls("volume", ["volume"]))
        self.computeVolumeButton = qt.QPushButton("Compute Volume")
        self.computeVolumeButton.clicked.connect(self.onComputeVolume)
        layout.addWidget(self.computeVolumeButton)
        self.volumeResultsText = self._metric_results_box("Volume results will appear here.")
        layout.addWidget(self.volumeResultsText)
        return box

    def _build_aoa_panel(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "AoA / SF"
        layout = qt.QVBoxLayout(box)
        form = qt.QFormLayout()
        form.addRow("Entry points:", self.entrySelector)
        form.addRow("Pivot point:", self.pivotSelector)
        form.addRow(self.sfStandardizedCheckBox)
        form.addRow("Standardized SF radius:", self.sfRadiusSpin)
        layout.addLayout(form)
        layout.addLayout(self._metric_action_row([
            ("Use Selected for Entry", lambda: self.onExportSelectedLandmarks("entry")),
            ("Use Selected for Pivot", lambda: self.onExportSelectedLandmarks("pivot")),
            ("Auto-fill AoA From Labels", self.onAutoFillAoaFromDataset),
            ("Pick Cranial", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "cranial", "mirror_role": "entry"})),
            ("Pick Caudal", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "caudal", "mirror_role": "entry"})),
            ("Pick Medial", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "medial", "mirror_role": "entry"})),
            ("Pick Lateral", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "lateral", "mirror_role": "entry"})),
            ("Pick Pivot", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "pivot", "mirror_role": "pivot"})),
        ]))
        layout.addLayout(self._metric_scene_controls("aoa", ["entry", "pivot"]))
        self.aoaRoleStatus = qt.QPlainTextEdit()
        self.aoaRoleStatus.readOnly = True
        self.aoaRoleStatus.setMaximumHeight(120)
        layout.addWidget(self.aoaRoleStatus)
        buttons_row = qt.QHBoxLayout()
        self.computeAoaButton = qt.QPushButton("Compute AoA / SF")
        self.computeStandardizedSfButton = qt.QPushButton("Compute Standardized SF")
        self.computeAoaButton.clicked.connect(self.onComputeAoa)
        self.computeStandardizedSfButton.clicked.connect(self.onComputeAoaStandardized)
        buttons_row.addWidget(self.computeAoaButton)
        buttons_row.addWidget(self.computeStandardizedSfButton)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)
        self.aoaResultsText = self._metric_results_box("AoA / SF results will appear here.")
        layout.addWidget(self.aoaResultsText)
        return box

    def _build_aoe_panel(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "AoE"
        layout = qt.QVBoxLayout(box)
        form = qt.QFormLayout()
        form.addRow("Point A:", self.aSelector)
        form.addRow("Point B:", self.bSelector)
        form.addRow("Point C:", self.cSelector)
        layout.addLayout(form)
        layout.addLayout(self._metric_action_row([
            ("Use Selected for A", lambda: self.onExportSelectedLandmarks("a")),
            ("Use Selected for B", lambda: self.onExportSelectedLandmarks("b")),
            ("Use Selected for C", lambda: self.onExportSelectedLandmarks("c")),
            ("Pick A", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "A", "mirror_role": "a"})),
            ("Pick B", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "B", "mirror_role": "b"})),
            ("Pick C", lambda: self.onCollectLabeledLandmark({"mode": "fixed", "label": "C", "mirror_role": "c"})),
        ]))
        layout.addLayout(self._metric_scene_controls("aoe", ["a", "b", "c"]))
        self.computeAoeButton = qt.QPushButton("Compute AoE")
        self.computeAoeButton.clicked.connect(self.onComputeAoe)
        layout.addWidget(self.computeAoeButton)
        self.aoeResultsText = self._metric_results_box("AoE results will appear here.")
        layout.addWidget(self.aoeResultsText)
        return box

    def _build_vom_panel(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "VOM / sVOM / VoA"
        layout = qt.QVBoxLayout(box)
        form = qt.QFormLayout()
        form.addRow("Entry polygon:", self.entrySelector)
        form.addRow("Target polygon:", self.targetSelector)
        form.addRow("Standard distance:", self.standDistSpin)
        layout.addLayout(form)
        layout.addLayout(self._metric_action_row([
            ("Use Selected for Entry", lambda: self.onExportSelectedLandmarks("entry")),
            ("Use Selected for Target", lambda: self.onExportSelectedLandmarks("target")),
            ("Start Entry Picking", lambda: self.onStartSequentialCollection({"mode": "prefix", "prefix": "entry", "mirror_role": "entry", "repeat": "true"})),
            ("Start Target Picking", lambda: self.onStartSequentialCollection({"mode": "prefix", "prefix": "target", "mirror_role": "target", "repeat": "true"})),
            ("Stop Picking", self.onStopSequentialCollection),
        ]))
        layout.addLayout(self._metric_scene_controls("vom", ["entry", "target"]))
        self.computeVomButton = qt.QPushButton("Compute VOM / sVOM / VoA")
        self.computeVomButton.clicked.connect(self.onComputeVom)
        layout.addWidget(self.computeVomButton)
        self.vomResultsText = self._metric_results_box("VOM / sVOM / VoA results will appear here.")
        layout.addWidget(self.vomResultsText)
        return box

    def _new_markups_selector(self, none_enabled: bool = False):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        selector.selectNodeUponCreation = True
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = none_enabled
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def _new_reference_selector(self):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = [
            "vtkMRMLModelNode",
            "vtkMRMLScalarVolumeNode",
            "vtkMRMLSegmentationNode",
        ]
        selector.selectNodeUponCreation = False
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def _new_volume_selector(self):
        selector = slicer.qMRMLNodeComboBox()
        selector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        selector.selectNodeUponCreation = False
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.showHidden = False
        selector.showChildNodeTypes = False
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def _build_corridor_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Corridor Metrics"
        layout = qt.QVBoxLayout(box)
        info = qt.QLabel(
            "Use markups fiducial lists to define corridor inputs. "
            "For VOM, entry and target should each contain at least 3 control points. "
            "For AoA / SF, the entry node should contain exactly 4 control points corresponding to "
            "cranial, caudal, medial, and lateral. If those labels are present on the control points, "
            "SurgiplotToolkit will auto-order them; otherwise it will use the current control-point order. "
            "Computed surfaces and guides are added back into the Slicer scene."
        )
        info.wordWrap = True
        layout.addWidget(info)
        form = qt.QFormLayout()
        layout.addLayout(form)

        self.entrySelector = self._new_markups_selector()
        self.targetSelector = self._new_markups_selector()
        self.pivotSelector = self._new_markups_selector()
        self.referenceSelector = self._new_reference_selector()

        self.standDistSpin = qt.QDoubleSpinBox()
        self.standDistSpin.setRange(0.01, 1000.0)
        self.standDistSpin.setDecimals(2)
        self.standDistSpin.setValue(10.0)
        self.standDistSpin.setSuffix(" mm")

        self.sfRadiusSpin = qt.QDoubleSpinBox()
        self.sfRadiusSpin.setRange(0.0, 1000.0)
        self.sfRadiusSpin.setDecimals(2)
        self.sfRadiusSpin.setValue(10.0)
        self.sfRadiusSpin.setSuffix(" mm")

        self.computeVomButton = qt.QPushButton("Compute VOM / sVOM / VoA")
        self.computeAoaButton = qt.QPushButton("Compute AoA / SF")
        self.computeVomButton.clicked.connect(self.onComputeVom)
        self.computeAoaButton.clicked.connect(self.onComputeAoa)

        form.addRow("Entry markups:", self.entrySelector)
        form.addRow("Target markups:", self.targetSelector)
        form.addRow("Pivot markups:", self.pivotSelector)
        form.addRow("Reference anatomy:", self.referenceSelector)
        form.addRow("Standard distance:", self.standDistSpin)
        form.addRow("SF rescale radius:", self.sfRadiusSpin)
        form.addRow(self.computeVomButton)
        form.addRow(self.computeAoaButton)

        role_box = ctk.ctkCollapsibleButton()
        role_box.text = "AoA / SF Role Assistant"
        role_layout = qt.QVBoxLayout(role_box)
        role_help = qt.QLabel(
            "This assistant helps define the anatomical role order expected by AoA / SF. "
            "Preferred labels are: cranial, caudal, medial, lateral. "
            "If those labels are missing, the metric falls back to raw control-point order."
        )
        role_help.wordWrap = True
        role_layout.addWidget(role_help)

        self.aoaRoleStatus = qt.QPlainTextEdit()
        self.aoaRoleStatus.readOnly = True
        self.aoaRoleStatus.setMaximumHeight(130)
        role_layout.addWidget(self.aoaRoleStatus)

        role_buttons = qt.QHBoxLayout()
        self.refreshAoaRolesButton = qt.QPushButton("Refresh Role Status")
        self.assignAoaRolesButton = qt.QPushButton("Assign Standard Roles By Current Order")
        self.refreshAoaRolesButton.clicked.connect(self._refresh_aoa_role_status)
        self.assignAoaRolesButton.clicked.connect(self.onAssignAoaRoles)
        role_buttons.addWidget(self.refreshAoaRolesButton)
        role_buttons.addWidget(self.assignAoaRolesButton)
        role_layout.addLayout(role_buttons)
        layout.addWidget(role_box)
        return box

    def _build_imaging_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Imaging & Markups Workflow"
        layout = qt.QVBoxLayout(box)
        info = qt.QLabel(
            "Use this section to prepare a Slicer-native CT/MRI workflow: choose a background volume, "
            "create standard markup nodes for Surgiplot, and jump all slice views to entry, target, or pivot locations."
        )
        info.wordWrap = True
        layout.addWidget(info)

        form = qt.QFormLayout()
        layout.addLayout(form)

        self.imageVolumeSelector = self._new_volume_selector()
        form.addRow("Slice background volume:", self.imageVolumeSelector)

        row1 = qt.QHBoxLayout()
        self.setBackgroundButton = qt.QPushButton("Set Background Volume")
        self.createEntryNodeButton = qt.QPushButton("Create Entry Node")
        self.createTargetNodeButton = qt.QPushButton("Create Target Node")
        row1.addWidget(self.setBackgroundButton)
        row1.addWidget(self.createEntryNodeButton)
        row1.addWidget(self.createTargetNodeButton)
        layout.addLayout(row1)

        row2 = qt.QHBoxLayout()
        self.createPivotNodeButton = qt.QPushButton("Create Pivot Node")
        self.jumpToEntryButton = qt.QPushButton("Jump Slices To Entry")
        self.jumpToTargetButton = qt.QPushButton("Jump Slices To Target")
        self.jumpToPivotButton = qt.QPushButton("Jump Slices To Pivot")
        row2.addWidget(self.createPivotNodeButton)
        row2.addWidget(self.jumpToEntryButton)
        row2.addWidget(self.jumpToTargetButton)
        row2.addWidget(self.jumpToPivotButton)
        layout.addLayout(row2)

        row3 = qt.QHBoxLayout()
        self.placeEntryButton = qt.QPushButton("Place Into Entry")
        self.placeTargetButton = qt.QPushButton("Place Into Target")
        self.placePivotButton = qt.QPushButton("Place Into Pivot")
        self.placeNextGuidedButton = qt.QPushButton("Place Next Guided Point")
        row3.addWidget(self.placeEntryButton)
        row3.addWidget(self.placeTargetButton)
        row3.addWidget(self.placePivotButton)
        row3.addWidget(self.placeNextGuidedButton)
        layout.addLayout(row3)

        self.imagingStatus = qt.QPlainTextEdit()
        self.imagingStatus.readOnly = True
        self.imagingStatus.setMaximumHeight(120)
        self.imagingStatus.setPlaceholderText("Imaging workflow status and guidance will appear here.")
        layout.addWidget(self.imagingStatus)

        self.setBackgroundButton.clicked.connect(self.onSetBackgroundVolume)
        self.createEntryNodeButton.clicked.connect(self.onCreateEntryNode)
        self.createTargetNodeButton.clicked.connect(self.onCreateTargetNode)
        self.createPivotNodeButton.clicked.connect(self.onCreatePivotNode)
        self.jumpToEntryButton.clicked.connect(self.onJumpToEntry)
        self.jumpToTargetButton.clicked.connect(self.onJumpToTarget)
        self.jumpToPivotButton.clicked.connect(self.onJumpToPivot)
        self.placeEntryButton.clicked.connect(self.onPlaceEntry)
        self.placeTargetButton.clicked.connect(self.onPlaceTarget)
        self.placePivotButton.clicked.connect(self.onPlacePivot)
        self.placeNextGuidedButton.clicked.connect(self.onPlaceNextGuidedPoint)
        return box

    def _build_guided_session_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Guided Acquisition"
        layout = qt.QVBoxLayout(box)

        info = qt.QLabel(
            "Use guided sessions for slice-linked acquisition. The toolkit will prepare the required markup nodes, "
            "track how many points have been collected, and indicate what point set is expected next."
        )
        info.wordWrap = True
        layout.addWidget(info)

        button_row = qt.QHBoxLayout()
        self.startAoaSessionButton = qt.QPushButton("Start AoA / SF Session")
        self.startVomSessionButton = qt.QPushButton("Start VOM Session")
        self.stopSessionButton = qt.QPushButton("Stop Session")
        self.startAoaSessionButton.clicked.connect(self.onStartAoaSession)
        self.startVomSessionButton.clicked.connect(self.onStartVomSession)
        self.stopSessionButton.clicked.connect(self.onStopSession)
        button_row.addWidget(self.startAoaSessionButton)
        button_row.addWidget(self.startVomSessionButton)
        button_row.addWidget(self.stopSessionButton)
        layout.addLayout(button_row)

        self.sessionStatus = qt.QPlainTextEdit()
        self.sessionStatus.readOnly = True
        self.sessionStatus.setMaximumHeight(170)
        self.sessionStatus.setPlaceholderText("Guided session status will appear here.")
        layout.addWidget(self.sessionStatus)

        helper_row = qt.QHBoxLayout()
        self.jumpToGuidedButton = qt.QPushButton("Jump To Guided Target")
        self.refreshSessionButton = qt.QPushButton("Refresh Session Status")
        self.jumpToGuidedButton.clicked.connect(self.onJumpToGuidedTarget)
        self.refreshSessionButton.clicked.connect(self._refresh_session_status)
        helper_row.addWidget(self.jumpToGuidedButton)
        helper_row.addWidget(self.refreshSessionButton)
        layout.addLayout(helper_row)
        return box

    def _attach_status_observers(self) -> None:
        selectors = [
            self.entrySelector,
            self.targetSelector,
            self.pivotSelector,
            self.referenceSelector,
            self.imageVolumeSelector,
            self.aSelector,
            self.bSelector,
            self.cSelector,
            self.polygonSelector,
            self.volumeSelector,
        ]
        for selector in selectors:
            selector.connect("currentNodeChanged(vtkMRMLNode*)", self._refresh_all_status)
            selector.connect("currentNodeChanged(vtkMRMLNode*)", self._refresh_metric_node_observers)
        self._refresh_metric_node_observers()

    def _clear_metric_node_observers(self) -> None:
        for node in self._metric_node_observed_nodes:
            try:
                node.RemoveObservers(vtk.vtkCommand.ModifiedEvent)
            except Exception:
                pass
        self._metric_node_observed_nodes = []

    def _refresh_metric_node_observers(self, *args) -> None:
        self._clear_metric_node_observers()
        nodes = [
            self.entrySelector.currentNode(),
            self.targetSelector.currentNode(),
            self.pivotSelector.currentNode(),
            self.aSelector.currentNode(),
            self.bSelector.currentNode(),
            self.cSelector.currentNode(),
            self.polygonSelector.currentNode(),
            self.volumeSelector.currentNode(),
        ]
        seen = set()
        for node in nodes:
            if node is None:
                continue
            node_id = node.GetID()
            if node_id in seen:
                continue
            seen.add(node_id)
            try:
                node.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_metric_node_modified)
                self._metric_node_observed_nodes.append(node)
            except Exception:
                pass

    def _on_metric_node_modified(self, caller=None, event=None) -> None:
        self._sync_metric_nodes_to_dataset()
        self._refresh_all_status()

    def _refresh_all_status(self, *args) -> None:
        self._refresh_aoa_role_status()
        self._refresh_session_status()
        self._refresh_validation_status()
        self._refresh_landmark_dataset_status()
        if not getattr(self, "_active_workspace", None):
            self._set_workspace_status(
                "No metric workspace selected.",
                "Use the workspace buttons above to open the metric-specific workflow you want to run.",
            )

    def _set_workspace_status(self, title: str, body: str) -> None:
        if hasattr(self, "workspaceStatus"):
            self.workspaceStatus.setPlainText(f"{title}\n\n{body}")

    def _open_metric_panel(self, metric: str) -> None:
        panels = {
            "distance": self.metricPanelDistance,
            "area": self.metricPanelArea,
            "volume": self.metricPanelVolume,
            "aoa": self.metricPanelAoa,
            "aoe": self.metricPanelAoe,
            "vom": self.metricPanelVom,
        }
        for name, panel in panels.items():
            panel.collapsed = name != metric
        self._active_workspace = metric

        messages = {
            "distance": (
                "Distance panel open",
                "Select or collect two landmarks, assign them to A and B, then compute Distance.",
            ),
            "area": (
                "Area of Exposure panel open",
                "Collect polygon landmarks, keep them in the desired order, export them to Polygon, then compute Area.",
            ),
            "volume": (
                "Volume panel open",
                "Collect the sparse 3D points you want, export them to Volume, then compute Volume.",
            ),
            "aoa": (
                "AoA / SF panel open",
                "Collect or relabel cranial, caudal, medial, lateral, and pivot, then auto-fill or export them before computing AoA / SF.",
            ),
            "aoe": (
                "AoE panel open",
                "Assign one point each to A, B, and C, then compute AoE.",
            ),
            "vom": (
                "VOM / sVOM / VoA panel open",
                "Collect entry_* and target_* landmarks, export them to Entry and Target, then compute the VOM family.",
            ),
        }
        title, body = messages.get(metric, ("Metric panel open", ""))
        self._set_workspace_status(title, body)
        self.analysisSection.collapsed = False
        self.resultsSection.collapsed = True
        if hasattr(self, "mainSplitter"):
            self.mainSplitter.setSizes([380, 780])

    def _open_metric_workspace(self, metric: str) -> None:
        self._active_workspace = metric
        for section in (self.landmarkSection, self.corridorSection, self.imagingSection, self.guidedSection, self.geometrySection, self.resultsSection):
            section.collapsed = False
        self.workflowSection.collapsed = False

        if metric == "aoa":
            self.corridorSection.collapsed = False
            self.guidedSection.collapsed = False
            self.geometrySection.collapsed = True
            self._set_workspace_status(
                "AoA / SF workspace",
                "Collect cranial, caudal, medial, lateral, and pivot in the landmark dataset, then auto-fill or export them into Entry and Pivot before computing AoA / SF.",
            )
        elif metric == "vom":
            self.corridorSection.collapsed = False
            self.guidedSection.collapsed = False
            self.geometrySection.collapsed = True
            self._set_workspace_status(
                "VOM / sVOM / VoA workspace",
                "Collect progressive entry_* and target_* landmarks, select the groups you want, export them into Entry and Target, then compute VOM / sVOM / VoA.",
            )
        elif metric == "aoe":
            self.corridorSection.collapsed = True
            self.guidedSection.collapsed = True
            self.geometrySection.collapsed = False
            self._set_workspace_status(
                "AoE workspace",
                "Collect or select exactly one landmark for A, one for B, and one for C, export them into A/B/C, then compute AoE.",
            )
        elif metric == "distance":
            self.corridorSection.collapsed = True
            self.guidedSection.collapsed = True
            self.geometrySection.collapsed = False
            self._set_workspace_status(
                "Distance workspace",
                "Collect or select exactly one landmark for A and one for B, export them into A and B, then compute Distance.",
            )
        elif metric == "area":
            self.corridorSection.collapsed = True
            self.guidedSection.collapsed = True
            self.geometrySection.collapsed = False
            self._set_workspace_status(
                "Area workspace",
                "Collect the polygon landmarks you want, select them in order, export them into Polygon, then compute Area.",
            )
        elif metric == "volume":
            self.corridorSection.collapsed = True
            self.guidedSection.collapsed = True
            self.geometrySection.collapsed = False
            self._set_workspace_status(
                "Volume workspace",
                "Collect the 3D points defining the sparse volume cloud, select them, export them into Volume, then compute Volume.",
            )

    def _ensure_landmark_node(self):
        node, _payload = self.logic.create_landmark_dataset_node(reuse_existing=True)
        self._landmark_node = node
        if self._landmark_node_observer is None and node is not None:
            self._landmark_node_observer = node.AddObserver(node.PointPositionDefinedEvent, self._on_landmark_node_modified)
        self._landmark_last_count = self._defined_markups_count(node)
        return node

    def _ensure_scale_node(self):
        node, _payload = self.logic.create_scale_markups_node(reuse_existing=True)
        self._scale_node = node
        if self._scale_node_observer is None and node is not None:
            self._scale_node_observer = node.AddObserver(node.PointPositionDefinedEvent, self._on_scale_node_modified)
        self._scale_last_count = self._defined_markups_count(node)
        return node

    def _defined_control_point_indices(self, node) -> list[int]:
        if node is None:
            return []
        indices: list[int] = []
        total = self.logic.bridge.markups_count(node)
        defined_status = getattr(node, "PositionDefined", None)
        for index in range(total):
            try:
                status = node.GetNthControlPointPositionStatus(index)
            except Exception:
                status = defined_status
            if defined_status is None or status == defined_status:
                indices.append(index)
        return indices

    def _defined_markups_count(self, node) -> int:
        return len(self._defined_control_point_indices(node))

    def _rearm_place_mode_later(self, node, place_label: str) -> None:
        if node is None:
            return
        def _rearm():
            try:
                self._arm_place_click_gate()
                self.logic.activate_place_mode(node, place_label=place_label, persistent=False)
            except Exception:
                pass
        qt.QTimer.singleShot(0, _rearm)

    def _on_place_left_click(self, caller=None, event=None) -> None:
        self._pending_place_click = True
        self._last_place_click_time = time.monotonic()

    def _clear_place_click_observers(self) -> None:
        for interactor, tag in self._place_click_observers:
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._place_click_observers = []

    def _arm_place_click_gate(self) -> None:
        self._pending_place_click = False
        self._last_place_click_time = 0.0
        self._clear_place_click_observers()
        layout_manager = getattr(slicer.app, "layoutManager", lambda: None)()
        if layout_manager is None:
            return
        try:
            for index in range(int(layout_manager.threeDViewCount)):
                widget = layout_manager.threeDWidget(index)
                view = widget.threeDView() if widget is not None else None
                interactor = view.interactor() if view is not None else None
                if interactor is not None:
                    tag = interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self._on_place_left_click)
                    self._place_click_observers.append((interactor, tag))
        except Exception:
            pass
        try:
            slice_names = list(layout_manager.sliceViewNames())
        except Exception:
            slice_names = []
        for name in slice_names:
            try:
                widget = layout_manager.sliceWidget(name)
                view = widget.sliceView() if widget is not None else None
                interactor = view.interactor() if view is not None else None
                if interactor is not None:
                    tag = interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self._on_place_left_click)
                    self._place_click_observers.append((interactor, tag))
            except Exception:
                pass

    def _consume_place_click(self) -> bool:
        now = time.monotonic()
        recent_click = bool(self._pending_place_click) and (now - float(self._last_place_click_time or 0.0) <= 1.0)
        self._pending_place_click = False
        self._last_place_click_time = 0.0
        return recent_click

    def _scaled_point(self, point: np.ndarray) -> np.ndarray:
        return np.asarray(point, dtype=float).reshape(3,) * float(self._scene_scale_to_mm)

    def _spinbox_value(self, spinbox) -> float:
        value_attr = getattr(spinbox, "value", None)
        if callable(value_attr):
            return float(value_attr())
        return float(value_attr)

    def _lineedit_text(self, widget) -> str:
        text_attr = getattr(widget, "text", "")
        if callable(text_attr):
            return str(text_attr())
        return str(text_attr)

    def _slider_value(self, widget) -> int:
        value_attr = getattr(widget, "value", 0)
        if callable(value_attr):
            return int(value_attr())
        return int(value_attr)

    def _next_prefixed_label(self, prefix: str) -> str:
        node = self._landmark_node
        labels = self.logic.bridge.markups_labels(node) if node is not None else []
        used = set()
        for label in labels:
            lower = str(label or "").strip().lower()
            if lower.startswith(prefix.lower() + "_"):
                suffix = lower.split("_")[-1]
                if suffix.isdigit():
                    used.add(int(suffix))
        next_index = 1
        while next_index in used:
            next_index += 1
        return f"{prefix}_{next_index}"

    def _refresh_landmark_table(self) -> None:
        node = self._landmark_node
        count = self.logic.bridge.markups_count(node)
        self.landmarkTable.setRowCount(count)
        unit = "mm" if abs(float(self._scene_scale_to_mm) - 1.0) > 1e-12 else "scene"
        self.landmarkTable.setHorizontalHeaderLabels(["Label", f"X ({unit})", f"Y ({unit})", f"Z ({unit})"])
        if node is None:
            return
        self._updating_landmark_table = True
        points = self.logic.bridge.markups_to_points(node, minimum=1) if count > 0 else np.empty((0, 3), dtype=float)
        labels = self.logic.bridge.markups_labels(node)
        for row in range(count):
            point = self._scaled_point(points[row])
            values = [labels[row] or f"point_{row + 1}", f"{point[0]:.4f}", f"{point[1]:.4f}", f"{point[2]:.4f}"]
            for col, value in enumerate(values):
                item = qt.QTableWidgetItem(value)
                if col > 0:
                    item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
                self.landmarkTable.setItem(row, col, item)
        self._updating_landmark_table = False

    def _refresh_landmark_dataset_status(self) -> None:
        node = self._landmark_node
        count = self.logic.bridge.markups_count(node)
        labels = self.logic.bridge.markups_labels(node) if node is not None and count > 0 else []
        lines = [
            f"Landmark dataset node: {node.GetName() if node is not None else 'Not created'}",
            f"Collected landmarks: {count}",
            f"Scale factor to mm: {self._scene_scale_to_mm:.6f}",
            f"Pending scale factor: {self._pending_scale_factor_to_mm:.6f}" if self._pending_scale_factor_to_mm is not None else "Pending scale factor: none",
            f"Pending label action: {self._pending_collect_spec if self._pending_collect_spec else 'none'}",
            f"Sequential picking: {self._active_multicollect_spec if self._active_multicollect_spec else 'inactive'}",
        ]
        if labels:
            lines.append("Collected labels: " + ", ".join(labels))
        if self.referenceSelector.currentNode() is not None and self.referenceSelector.currentNode().IsA("vtkMRMLModelNode"):
            lines.append("Model reference detected: scale calibration is available.")
        else:
            lines.append("Volume/segmentation or no reference: use native Slicer mm coordinates; scale calibration is usually not needed.")
        self.datasetStatus.setPlainText("\n".join(lines))
        self._refresh_landmark_table()

    def _on_landmark_node_modified(self, caller=None, event=None) -> None:
        node = self._landmark_node
        if node is None:
            return
        defined_indices = self._defined_control_point_indices(node)
        count = len(defined_indices)
        if self._pending_collect_spec and count > self._landmark_last_count:
            new_indices = list(defined_indices[self._landmark_last_count:])
            if not self._consume_place_click():
                spec = dict(self._pending_collect_spec)
                for index in reversed(new_indices):
                    try:
                        node.RemoveNthControlPoint(index)
                    except Exception:
                        pass
                try:
                    self.logic.deactivate_place_mode()
                except Exception:
                    pass
                self._clear_place_click_observers()
                self._rearm_place_mode_later(node, json.dumps(spec, sort_keys=True))
                self._landmark_last_count = self._defined_markups_count(node)
                self._refresh_all_status()
                return
            reference_node = self.referenceSelector.currentNode() if hasattr(self, "referenceSelector") else None
            for index in new_indices:
                spec = dict(self._pending_collect_spec)
                point = [0.0, 0.0, 0.0]
                node.GetNthControlPointPositionWorld(index, point)
                point_arr = np.asarray(point, dtype=float)
                snapped_point = self.logic.snap_point_to_reference_model(point_arr, reference_node=reference_node)
                if snapped_point is not None:
                    node.SetNthControlPointPositionWorld(
                        index,
                        float(snapped_point[0]),
                        float(snapped_point[1]),
                        float(snapped_point[2]),
                    )
                    point_arr = snapped_point
                if spec.get("mode") == "fixed":
                    label = str(spec.get("label", "")).strip()
                elif spec.get("mode") == "prefix":
                    label = self._next_prefixed_label(str(spec.get("prefix", "point")).strip())
                else:
                    label = str(spec.get("label", "")).strip() or f"point_{index + 1}"
                if label:
                    node.SetNthControlPointLabel(index, label)
                    mirror_role = str(spec.get("mirror_role", "")).strip().lower()
                    if mirror_role:
                        metric_node = self.logic.append_metric_point(mirror_role, label, self._scaled_point(point_arr))
                        if metric_node is not None:
                            self._bind_selector_for_role(mirror_role, metric_node)
            if str(spec.get("repeat", "")).strip().lower() != "true":
                self._pending_collect_spec = None
                try:
                    self.logic.deactivate_place_mode()
                except Exception:
                    pass
                self._clear_place_click_observers()
            else:
                self._rearm_place_mode_later(node, json.dumps(spec, sort_keys=True))
        self._landmark_last_count = count
        self._refresh_all_status()

    def _on_scale_node_modified(self, caller=None, event=None) -> None:
        node = self._scale_node
        if node is None:
            return
        count = self._defined_markups_count(node)
        if count > self._scale_last_count and not self._consume_place_click():
            defined_indices = self._defined_control_point_indices(node)
            for index in reversed(defined_indices[self._scale_last_count:]):
                try:
                    node.RemoveNthControlPoint(index)
                except Exception:
                    pass
            try:
                self.logic.deactivate_place_mode()
            except Exception:
                pass
            self._clear_place_click_observers()
            self._rearm_place_mode_later(node, "scale")
            self._scale_last_count = self._defined_markups_count(node)
            self._refresh_landmark_dataset_status()
            return
        self._scale_last_count = count
        reference_node = self.referenceSelector.currentNode() if hasattr(self, "referenceSelector") else None
        defined_indices = self._defined_control_point_indices(node)
        for index in defined_indices:
            point = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(index, point)
            snapped_point = self.logic.snap_point_to_reference_model(np.asarray(point, dtype=float), reference_node=reference_node)
            if snapped_point is not None:
                node.SetNthControlPointPositionWorld(
                    index,
                    float(snapped_point[0]),
                    float(snapped_point[1]),
                    float(snapped_point[2]),
                )
        if count < 2:
            self._pending_scale_factor_to_mm = None
            self.applyScaleButton.setEnabled(False)
            self._rearm_place_mode_later(node, "scale")
            self._refresh_landmark_dataset_status()
            return
        try:
            points = self.logic.bridge.markups_to_points(node, minimum=2)
        except Exception:
            self._refresh_landmark_dataset_status()
            return
        p1, p2 = points[0], points[1]
        measured = float(np.linalg.norm(p1 - p2))
        if measured <= 1e-12:
            self._pending_scale_factor_to_mm = None
            self.applyScaleButton.setEnabled(False)
            self.datasetStatus.setPlainText("Scale calibration failed: the selected scale points overlap.")
            return
        desired_mm = self._spinbox_value(self.scaleDistanceSpin) * 10.0
        self._pending_scale_factor_to_mm = desired_mm / measured
        self.applyScaleButton.setEnabled(True)
        try:
            self.logic.deactivate_place_mode()
        except Exception:
            pass
        self._clear_place_click_observers()
        self._refresh_all_status()

    def _sync_metric_nodes_to_dataset(self) -> None:
        dataset = self._ensure_landmark_node()
        existing_labels = self.logic.bridge.markups_labels(dataset)
        existing_points = self.logic.bridge.markups_to_points(dataset, minimum=1) if existing_labels else np.empty((0, 3), dtype=float)
        label_to_index = {str(label).strip().lower(): idx for idx, label in enumerate(existing_labels)}

        role_nodes = {
            "entry": self.entrySelector.currentNode(),
            "target": self.targetSelector.currentNode(),
            "pivot": self.pivotSelector.currentNode(),
            "a": self.aSelector.currentNode(),
            "b": self.bSelector.currentNode(),
            "c": self.cSelector.currentNode(),
            "polygon": self.polygonSelector.currentNode(),
            "volume": self.volumeSelector.currentNode(),
        }
        for role, node in role_nodes.items():
            if node is None:
                continue
            count = self.logic.bridge.markups_count(node)
            if count == 0:
                continue
            labels = self.logic.bridge.markups_labels(node)
            points = self.logic.bridge.markups_to_points(node, minimum=1)
            for idx, point in enumerate(points):
                label = str(labels[idx] or "").strip()
                if not label:
                    label = f"{role}_{idx + 1}" if role in {"entry", "target", "polygon", "volume"} else role.upper()
                key = label.lower()
                if key in label_to_index:
                    dataset.SetNthControlPointLabel(label_to_index[key], label)
                    dataset.SetNthControlPointPositionWorld(label_to_index[key], float(point[0]), float(point[1]), float(point[2]))
                else:
                    new_idx = dataset.AddControlPointWorld(vtk.vtkVector3d(float(point[0]), float(point[1]), float(point[2])))
                    dataset.SetNthControlPointLabel(new_idx, label)
                    label_to_index[key] = new_idx

    def _build_geometry_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Geometric Metrics"
        layout = qt.QVBoxLayout(box)
        info = qt.QLabel(
            "These utilities operate directly on Slicer markups nodes. "
            "Use one-point lists for AoE and distance, polygonal lists for area, and sparse 3D point clouds for volume."
        )
        info.wordWrap = True
        layout.addWidget(info)
        form = qt.QFormLayout()
        layout.addLayout(form)

        self.aSelector = self._new_markups_selector()
        self.bSelector = self._new_markups_selector()
        self.cSelector = self._new_markups_selector()
        self.polygonSelector = self._new_markups_selector()
        self.volumeSelector = self._new_markups_selector()

        self.computeAoeButton = qt.QPushButton("Compute AoE")
        self.computeDistanceButton = qt.QPushButton("Compute Distance")
        self.computeAreaButton = qt.QPushButton("Compute Area")
        self.computeVolumeButton = qt.QPushButton("Compute Volume")

        self.computeAoeButton.clicked.connect(self.onComputeAoe)
        self.computeDistanceButton.clicked.connect(self.onComputeDistance)
        self.computeAreaButton.clicked.connect(self.onComputeArea)
        self.computeVolumeButton.clicked.connect(self.onComputeVolume)

        form.addRow("Point A markup:", self.aSelector)
        form.addRow("Point B markup:", self.bSelector)
        form.addRow("Point C markup:", self.cSelector)
        form.addRow("Polygon markups:", self.polygonSelector)
        form.addRow("Volume markups:", self.volumeSelector)
        form.addRow(self.computeAoeButton)
        form.addRow(self.computeDistanceButton)
        form.addRow(self.computeAreaButton)
        form.addRow(self.computeVolumeButton)
        return box

    def _build_results_section(self):
        box = ctk.ctkCollapsibleButton()
        box.text = "Results"
        box.collapsed = False
        layout = qt.QVBoxLayout(box)

        controls_form = qt.QFormLayout()
        self.modelOpacitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.modelOpacitySlider.setRange(0, 100)
        self.modelOpacitySlider.setValue(100)
        self.metricOpacitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.metricOpacitySlider.setRange(0, 100)
        self.metricOpacitySlider.setValue(65)
        self.modelOpacityValue = qt.QLabel("1.00")
        self.metricOpacityValue = qt.QLabel("0.65")
        self.applyModelOpacityButton = qt.QPushButton("Apply Model Opacity")
        self.applyMetricOpacityButton = qt.QPushButton("Apply Metric Opacity")
        self.applyModelOpacityButton.clicked.connect(self.onApplyModelOpacity)
        self.applyMetricOpacityButton.clicked.connect(self.onApplyMetricOpacity)
        self.modelOpacitySlider.valueChanged.connect(lambda v: self.modelOpacityValue.setText(f"{v/100.0:.2f}"))
        self.metricOpacitySlider.valueChanged.connect(lambda v: self.metricOpacityValue.setText(f"{v/100.0:.2f}"))
        self.modelOpacitySlider.sliderReleased.connect(self.onApplyModelOpacity)
        self.metricOpacitySlider.sliderReleased.connect(self.onApplyMetricOpacity)
        model_widget = qt.QWidget()
        model_row = qt.QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(self.modelOpacitySlider, 1)
        model_row.addWidget(self.modelOpacityValue)
        model_widget.setLayout(model_row)
        metric_widget = qt.QWidget()
        metric_row = qt.QHBoxLayout()
        metric_row.setContentsMargins(0, 0, 0, 0)
        metric_row.addWidget(self.metricOpacitySlider, 1)
        metric_row.addWidget(self.metricOpacityValue)
        metric_widget.setLayout(metric_row)
        controls_form.addRow("Model opacity:", model_widget)
        controls_form.addRow(self.applyModelOpacityButton)
        controls_form.addRow("Metrics opacity:", metric_widget)
        controls_form.addRow(self.applyMetricOpacityButton)
        layout.addLayout(controls_form)

        export_box = ctk.ctkCollapsibleButton()
        export_box.text = "Case Output Datasets"
        export_box.collapsed = False
        export_layout = qt.QVBoxLayout(export_box)
        point_visibility_row = qt.QHBoxLayout()
        self.showAllPointsButton = qt.QPushButton("Show All Points")
        self.hideAllPointsButton = qt.QPushButton("Hide All Points")
        self.showAllPointsButton.clicked.connect(lambda: self.onSetAllPointsVisibility(True))
        self.hideAllPointsButton.clicked.connect(lambda: self.onSetAllPointsVisibility(False))
        point_visibility_row.addWidget(self.showAllPointsButton)
        point_visibility_row.addWidget(self.hideAllPointsButton)
        point_visibility_row.addStretch(1)
        export_layout.addLayout(point_visibility_row)
        export_form = qt.QFormLayout()
        self.subjectIdEdit = qt.QLineEdit()
        self.subjectIdEdit.setPlaceholderText("Patient 1")
        self.anatomicalTargetEdit = qt.QLineEdit()
        self.anatomicalTargetEdit.setPlaceholderText("IAC")
        export_form.addRow("Subject ID:", self.subjectIdEdit)
        export_form.addRow("Anatomical target:", self.anatomicalTargetEdit)
        export_layout.addLayout(export_form)
        export_buttons_1 = qt.QHBoxLayout()
        self.appendResultDatasetButton = qt.QPushButton("Append Current Result")
        self.appendPointsDatasetButton = qt.QPushButton("Append Current Points")
        export_buttons_1.addWidget(self.appendResultDatasetButton)
        export_buttons_1.addWidget(self.appendPointsDatasetButton)
        export_layout.addLayout(export_buttons_1)
        export_buttons_2 = qt.QHBoxLayout()
        self.exportResultsCsvButton = qt.QPushButton("Export Results CSV")
        self.exportResultsXlsxButton = qt.QPushButton("Export Results XLSX")
        self.exportPointsCsvButton = qt.QPushButton("Export Points CSV")
        self.exportPointsXlsxButton = qt.QPushButton("Export Points XLSX")
        for btn in (
            self.exportResultsCsvButton,
            self.exportResultsXlsxButton,
            self.exportPointsCsvButton,
            self.exportPointsXlsxButton,
        ):
            export_buttons_2.addWidget(btn)
        export_layout.addLayout(export_buttons_2)
        self.appendResultDatasetButton.clicked.connect(self.onAppendCurrentResultToDataset)
        self.appendPointsDatasetButton.clicked.connect(self.onAppendCurrentPointsToDataset)
        self.exportResultsCsvButton.clicked.connect(lambda: self.onExportDatasetFile("results", "csv"))
        self.exportResultsXlsxButton.clicked.connect(lambda: self.onExportDatasetFile("results", "xlsx"))
        self.exportPointsCsvButton.clicked.connect(lambda: self.onExportDatasetFile("points", "csv"))
        self.exportPointsXlsxButton.clicked.connect(lambda: self.onExportDatasetFile("points", "xlsx"))
        layout.addWidget(export_box)

        preview_box = ctk.ctkCollapsibleButton()
        preview_box.text = "Dataset Preview"
        preview_box.collapsed = False
        preview_layout = qt.QVBoxLayout(preview_box)
        preview_button_row = qt.QHBoxLayout()
        self.refreshDatasetPreviewButton = qt.QPushButton("Refresh Preview")
        self.saveDatasetPreviewEditsButton = qt.QPushButton("Save Preview Edits")
        preview_button_row.addWidget(self.refreshDatasetPreviewButton)
        preview_button_row.addWidget(self.saveDatasetPreviewEditsButton)
        preview_layout.addLayout(preview_button_row)
        self.exportPreviewTabs = qt.QTabWidget()
        self.resultsPreviewTable = qt.QTableWidget()
        self.pointsPreviewTable = qt.QTableWidget()
        for table in (self.resultsPreviewTable, self.pointsPreviewTable):
            table.setSelectionBehavior(qt.QAbstractItemView.SelectItems)
            table.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
            table.setEditTriggers(qt.QAbstractItemView.DoubleClicked | qt.QAbstractItemView.EditKeyPressed)
            table.setMinimumHeight(180)
            header = table.horizontalHeader()
            if hasattr(header, "setStretchLastSection"):
                header.setStretchLastSection(True)
        self.exportPreviewTabs.addTab(self.resultsPreviewTable, "Case Results")
        self.exportPreviewTabs.addTab(self.pointsPreviewTable, "Points Dataset")
        preview_layout.addWidget(self.exportPreviewTabs)
        self.refreshDatasetPreviewButton.clicked.connect(self.onRefreshExportPreview)
        self.saveDatasetPreviewEditsButton.clicked.connect(self.onSaveExportPreviewEdits)
        layout.addWidget(preview_box)

        self.resultsText = qt.QPlainTextEdit()
        self.resultsText.readOnly = True
        self.resultsText.setPlaceholderText("Computed results will appear here as structured JSON.")
        self.resultsText.setMinimumHeight(220)
        self.resultsText.setMaximumHeight(340)
        layout.addWidget(self.resultsText)
        return box

    def _show_results(self, payload: dict[str, Any]) -> None:
        if isinstance(payload, dict) and payload.get("metric_family") and not payload.get("action"):
            self._last_metric_payload = dict(payload)
        self.resultsText.setPlainText(json.dumps(payload, indent=2, sort_keys=True))

    def _show_metric_results(self, metric: str, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True)
        mapping = {
            "distance": getattr(self, "distanceResultsText", None),
            "area": getattr(self, "areaResultsText", None),
            "volume": getattr(self, "volumeResultsText", None),
            "aoa": getattr(self, "aoaResultsText", None),
            "aoe": getattr(self, "aoeResultsText", None),
            "vom": getattr(self, "vomResultsText", None),
        }
        box = mapping.get(metric)
        if box is not None:
            box.setPlainText(text)

    def _show_error(self, title: str, exc: Exception) -> None:
        slicer.util.errorDisplay(f"{title}\n\n{exc}")

    def onReloadModule(self) -> None:
        try:
            slicer.util.reloadScriptedModule("SurgiplotToolkit")
        except Exception as exc:
            self._show_error("Reload module failed", exc)

    def onRestartSlicer(self) -> None:
        try:
            if hasattr(slicer.app, "restart"):
                slicer.app.restart()
            else:
                raise RuntimeError("Slicer restart is not available in this runtime.")
        except Exception as exc:
            self._show_error("Restart Slicer failed", exc)

    def _show_imaging_status(self, lines: list[str]) -> None:
        text = "\n".join(lines)
        if hasattr(self, "imagingStatus"):
            self.imagingStatus.setPlainText(text)
        else:
            self.datasetStatus.setPlainText(text)

    def _refresh_validation_status(self) -> None:
        if not hasattr(self, "validationStatus"):
            return
        payload = self.logic.describe_workflow(
            entry_node=self.entrySelector.currentNode(),
            target_node=self.targetSelector.currentNode(),
            pivot_node=self.pivotSelector.currentNode(),
            a_node=self.aSelector.currentNode(),
            b_node=self.bSelector.currentNode(),
            c_node=self.cSelector.currentNode(),
            polygon_node=self.polygonSelector.currentNode(),
            volume_node=self.volumeSelector.currentNode(),
            reference_node=self.referenceSelector.currentNode(),
            background_volume_node=self.imageVolumeSelector.currentNode(),
            session_mode=self._session_mode,
        )

        lines = [
            f"Reference anatomy: {payload.get('reference_anatomy', '-')}",
            f"Background volume: {payload.get('background_volume', '-')}",
            f"AoA / SF ready: {payload.get('aoa_ready', False)}",
            f"VOM ready: {payload.get('vom_ready', False)}",
            f"AoE ready: {payload.get('aoe_ready', False)}",
            f"Distance ready: {payload.get('distance_ready', False)}",
            f"Area ready: {payload.get('area_ready', False)}",
            f"Volume ready: {payload.get('volume_ready', False)}",
        ]
        warnings = list(payload.get("warnings") or [])
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings)
        else:
            lines.append("Warnings: none")
        self.validationStatus.setPlainText("\n".join(lines))

    def _clear_session_observers(self) -> None:
        for node in self._session_observed_nodes:
            try:
                node.RemoveObservers(vtk.vtkCommand.ModifiedEvent)
            except Exception:
                pass
        self._session_observed_nodes = []

    def _observe_session_nodes(self, nodes: list[Any]) -> None:
        self._clear_session_observers()
        for node in nodes:
            if node is None:
                continue
            try:
                node.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_session_node_modified)
                self._session_observed_nodes.append(node)
            except Exception:
                pass

    def _on_session_node_modified(self, caller=None, event=None) -> None:
        self._refresh_all_status()

    def _guided_target_state(self) -> tuple[Optional[str], str]:
        if self._session_mode == "aoa":
            entry_node = self.entrySelector.currentNode()
            pivot_node = self.pivotSelector.currentNode()
            entry_count = self.logic.bridge.markups_count(entry_node)
            pivot_count = self.logic.bridge.markups_count(pivot_node)
            if entry_count < 4:
                role = self.logic.bridge.AOA_ENTRY_ROLES[entry_count]
                return "entry", f"Entry point {entry_count + 1} / 4 ({role})"
            if pivot_count < 1:
                return "pivot", "Pivot point 1 / 1"
            return None, "AoA / SF acquisition ready"

        if self._session_mode == "vom":
            entry_node = self.entrySelector.currentNode()
            target_node = self.targetSelector.currentNode()
            entry_count = self.logic.bridge.markups_count(entry_node)
            target_count = self.logic.bridge.markups_count(target_node)
            if entry_count < 3:
                return "entry", f"Entry contour point {entry_count + 1} (minimum 3)"
            if target_count < 3:
                return "target", f"Target contour point {target_count + 1} (minimum 3)"
            return "target", "Additional target contour point"

        return None, "No guided session active"

    def _refresh_session_status(self) -> None:
        if not hasattr(self, "sessionStatus"):
            return
        if not self._session_mode:
            self.sessionStatus.setPlainText("No guided session active.")
            return

        lines: list[str] = []
        if self._session_mode == "aoa":
            entry_node = self.entrySelector.currentNode()
            pivot_node = self.pivotSelector.currentNode()
            entry_count = self.logic.bridge.markups_count(entry_node)
            pivot_count = self.logic.bridge.markups_count(pivot_node)
            next_role = self.logic.bridge.AOA_ENTRY_ROLES[entry_count] if entry_count < 4 else None
            lines.append("Active session: AoA / SF")
            lines.append(f"Entry node: {entry_node.GetName() if entry_node else '-'}")
            lines.append(f"Pivot node: {pivot_node.GetName() if pivot_node else '-'}")
            lines.append(f"Entry points: {entry_count} / 4")
            lines.append(f"Pivot points: {pivot_count} / 1")
            guided_target, guided_label = self._guided_target_state()
            if next_role is not None:
                lines.append(f"Next expected entry role: {next_role}")
            lines.append(f"Guided target: {guided_target or 'none'}")
            lines.append(f"Next expected step: {guided_label}.")

        elif self._session_mode == "vom":
            entry_node = self.entrySelector.currentNode()
            target_node = self.targetSelector.currentNode()
            entry_count = self.logic.bridge.markups_count(entry_node)
            target_count = self.logic.bridge.markups_count(target_node)
            lines.append("Active session: VOM / sVOM / VoA")
            lines.append(f"Entry node: {entry_node.GetName() if entry_node else '-'}")
            lines.append(f"Target node: {target_node.GetName() if target_node else '-'}")
            lines.append(f"Entry contour points: {entry_count} (minimum 3)")
            lines.append(f"Target contour points: {target_count} (minimum 3)")
            guided_target, guided_label = self._guided_target_state()
            lines.append(f"Guided target: {guided_target or 'none'}")
            if entry_count >= 3 and target_count >= 3:
                lines.append("VOM acquisition has the minimum required points and is ready for computation.")
            lines.append(f"Next expected step: {guided_label}.")

        self.sessionStatus.setPlainText("\n".join(lines))

    def _node_for_guided_target(self, guided_target: Optional[str]):
        if guided_target == "entry":
            return self.entrySelector.currentNode()
        if guided_target == "target":
            return self.targetSelector.currentNode()
        if guided_target == "pivot":
            return self.pivotSelector.currentNode()
        return None

    def _refresh_aoa_role_status(self, *args) -> None:
        if not hasattr(self, "aoaRoleStatus"):
            return
        try:
            status = self.logic.describe_aoa_entry(self.entrySelector.currentNode())
        except Exception as exc:
            self.aoaRoleStatus.setPlainText(f"Role inspection failed:\n{exc}")
            return

        lines = [
            f"Valid AoA/SF entry: {status.get('valid', False)}",
            f"Control-point count: {status.get('point_count', 0)}",
            f"Ordering strategy: {status.get('ordering_strategy', 'unknown')}",
            f"Expected role order: {', '.join(status.get('role_order', [])) or '-'}",
            f"Current labels: {', '.join(lb or '-' for lb in status.get('labels', [])) or '-'}",
        ]
        ordered_labels = status.get("ordered_labels")
        if ordered_labels:
            lines.append(f"Resolved order: {', '.join(lb or '-' for lb in ordered_labels)}")
        warnings = status.get("warnings") or []
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings)
        else:
            lines.append("Warnings: none")
        self.aoaRoleStatus.setPlainText("\n".join(lines))

    def onAssignAoaRoles(self) -> None:
        try:
            payload = self.logic.assign_aoa_roles_by_order(self.entrySelector.currentNode())
            self._refresh_all_status()
            self._show_results(payload)
        except Exception as exc:
            self._show_error("AoA / SF role assignment failed", exc)

    def onStartAoaSession(self) -> None:
        try:
            entry_node, entry_payload = self.logic.create_entry_markups_node(reuse_existing=True)
            pivot_node, pivot_payload = self.logic.create_pivot_markups_node(reuse_existing=True)
            self.entrySelector.setCurrentNode(entry_node)
            self.pivotSelector.setCurrentNode(pivot_node)
            self._session_mode = "aoa"
            self._observe_session_nodes([entry_node, pivot_node])
            self._refresh_all_status()
            self._show_imaging_status(
                [
                    "Started guided AoA / SF session.",
                    f"Entry node: {entry_payload.get('node_name', '-')}",
                    f"Pivot node: {pivot_payload.get('node_name', '-')}",
                    "Collect 4 entry points, then 1 pivot point.",
                ]
            )
            self._show_results(
                {
                    "action": "start_aoa_session",
                    "entry_node": entry_payload.get("node_name"),
                    "pivot_node": pivot_payload.get("node_name"),
                }
            )
        except Exception as exc:
            self._show_error("Start AoA / SF session failed", exc)

    def onStartVomSession(self) -> None:
        try:
            entry_node, entry_payload = self.logic.create_entry_markups_node(reuse_existing=True)
            target_node, target_payload = self.logic.create_target_markups_node(reuse_existing=True)
            self.entrySelector.setCurrentNode(entry_node)
            self.targetSelector.setCurrentNode(target_node)
            self._session_mode = "vom"
            self._observe_session_nodes([entry_node, target_node])
            self._refresh_all_status()
            self._show_imaging_status(
                [
                    "Started guided VOM / sVOM / VoA session.",
                    f"Entry node: {entry_payload.get('node_name', '-')}",
                    f"Target node: {target_payload.get('node_name', '-')}",
                    "Collect at least 3 entry contour points, then at least 3 target contour points.",
                ]
            )
            self._show_results(
                {
                    "action": "start_vom_session",
                    "entry_node": entry_payload.get("node_name"),
                    "target_node": target_payload.get("node_name"),
                }
            )
        except Exception as exc:
            self._show_error("Start VOM session failed", exc)

    def onStopSession(self) -> None:
        self._session_mode = None
        self._clear_session_observers()
        self._refresh_all_status()
        self._show_imaging_status(["Guided session stopped."])

    def onValidateInputs(self) -> None:
        payload = self.logic.describe_workflow(
            entry_node=self.entrySelector.currentNode(),
            target_node=self.targetSelector.currentNode(),
            pivot_node=self.pivotSelector.currentNode(),
            a_node=self.aSelector.currentNode(),
            b_node=self.bSelector.currentNode(),
            c_node=self.cSelector.currentNode(),
            polygon_node=self.polygonSelector.currentNode(),
            volume_node=self.volumeSelector.currentNode(),
            reference_node=self.referenceSelector.currentNode(),
            background_volume_node=self.imageVolumeSelector.currentNode(),
            session_mode=self._session_mode,
        )
        self._refresh_validation_status()
        self._show_results(payload)

    def onClearResults(self) -> None:
        payload = self.logic.clear_result_nodes(reference_node=self.referenceSelector.currentNode())
        self._show_results(payload)

    def onCreateLandmarkDatasetNode(self) -> None:
        node, payload = self.logic.create_landmark_dataset_node(reuse_existing=True)
        self._landmark_node = node
        if self._landmark_node_observer is None and node is not None:
            self._landmark_node_observer = node.AddObserver(node.PointPositionDefinedEvent, self._on_landmark_node_modified)
        self._landmark_last_count = self._defined_markups_count(node)
        self._refresh_all_status()
        self._show_results(payload)

    def onCreateScaleNode(self) -> None:
        node, payload = self.logic.create_scale_markups_node(reuse_existing=True)
        self._scale_node = node
        if self._scale_node_observer is None and node is not None:
            self._scale_node_observer = node.AddObserver(node.PointPositionDefinedEvent, self._on_scale_node_modified)
        self._scale_last_count = self._defined_markups_count(node)
        self._pending_scale_factor_to_mm = None
        self.applyScaleButton.setEnabled(False)
        self._refresh_all_status()
        self._show_results(payload)

    def onPickScalePoints(self) -> None:
        node = self._ensure_scale_node()
        try:
            self._arm_place_click_gate()
            payload = self.logic.activate_place_mode(node, place_label="scale", persistent=False)
            self.datasetStatus.setPlainText(
                "Scale mode active. Place 2 points on the model with the known distance between them, then click Apply Rescaling."
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Scale placement activation failed", exc)

    def onApplyScaleCalibration(self) -> None:
        try:
            if self._pending_scale_factor_to_mm is None or self._pending_scale_factor_to_mm <= 0:
                raise ValueError("Pick 2 valid scale points first.")
            factor = float(self._pending_scale_factor_to_mm)
            reference_node = self.referenceSelector.currentNode() if hasattr(self, "referenceSelector") else None
            payload = self.logic.apply_uniform_scene_scale(
                factor,
                reference_node=reference_node,
                related_nodes=[
                    self._scale_node,
                    self._landmark_node,
                    self.entrySelector.currentNode(),
                    self.targetSelector.currentNode(),
                    self.pivotSelector.currentNode(),
                    self.aSelector.currentNode(),
                    self.bSelector.currentNode(),
                    self.cSelector.currentNode(),
                    self.polygonSelector.currentNode(),
                    self.volumeSelector.currentNode(),
                ],
            )
            if self._scale_node is not None:
                self.logic.normalize_scale_node(self._scale_node)
            self._scene_scale_to_mm = 1.0
            self._pending_scale_factor_to_mm = None
            self.applyScaleButton.setEnabled(False)
            self._refresh_all_status()
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Apply scale calibration failed", exc)

    def onCollectLabeledLandmark(self, spec: dict[str, str]) -> None:
        node = self._ensure_landmark_node()
        self._pending_collect_spec = dict(spec)
        if str(spec.get("repeat", "")).strip().lower() == "true":
            self._active_multicollect_spec = dict(spec)
        self._prepare_metric_role_selection(str(spec.get("mirror_role", "")).strip().lower())
        try:
            self._arm_place_click_gate()
            payload = self.logic.activate_place_mode(node, place_label=json.dumps(spec, sort_keys=True), persistent=False)
            self._refresh_landmark_dataset_status()
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Landmark placement activation failed", exc)

    def onStartSequentialCollection(self, spec: dict[str, str]) -> None:
        self.onCollectLabeledLandmark(spec)
        prefix = str(spec.get("prefix", spec.get("label", "points"))).strip()
        self.datasetStatus.setPlainText(
            f"Sequential picking active for {prefix}. Keep placing points in the view, then click Stop Picking when finished."
        )

    def onStopSequentialCollection(self) -> None:
        self._pending_collect_spec = None
        self._active_multicollect_spec = None
        self._clear_place_click_observers()
        try:
            self.logic.deactivate_place_mode()
        except Exception:
            pass
        self._refresh_all_status()

    def _prepare_metric_role_selection(self, role: str) -> None:
        if role == "entry":
            node, _ = self.logic.create_entry_markups_node(reuse_existing=True)
            self.entrySelector.setCurrentNode(node)
        elif role == "target":
            node, _ = self.logic.create_target_markups_node(reuse_existing=True)
            self.targetSelector.setCurrentNode(node)
        elif role == "pivot":
            node, _ = self.logic.create_pivot_markups_node(reuse_existing=True)
            self.pivotSelector.setCurrentNode(node)
        elif role == "a":
            node, _ = self.logic.create_single_markups_node("Surgiplot A", "a", reuse_existing=True)
            self.aSelector.setCurrentNode(node)
        elif role == "b":
            node, _ = self.logic.create_single_markups_node("Surgiplot B", "b", reuse_existing=True)
            self.bSelector.setCurrentNode(node)
        elif role == "c":
            node, _ = self.logic.create_single_markups_node("Surgiplot C", "c", reuse_existing=True)
            self.cSelector.setCurrentNode(node)
        elif role == "polygon":
            node, _ = self.logic.create_multi_markups_node("Surgiplot Polygon", "polygon", reuse_existing=True)
            self.polygonSelector.setCurrentNode(node)
        elif role == "volume":
            node, _ = self.logic.create_multi_markups_node("Surgiplot Volume", "volume", reuse_existing=True)
            self.volumeSelector.setCurrentNode(node)

    def _bind_selector_for_role(self, role: str, node) -> None:
        if node is None:
            return
        role_key = str(role or "").strip().lower()
        if role_key == "entry":
            self.entrySelector.setCurrentNode(node)
        elif role_key == "target":
            self.targetSelector.setCurrentNode(node)
        elif role_key == "pivot":
            self.pivotSelector.setCurrentNode(node)
        elif role_key == "a":
            self.aSelector.setCurrentNode(node)
        elif role_key == "b":
            self.bSelector.setCurrentNode(node)
        elif role_key == "c":
            self.cSelector.setCurrentNode(node)
        elif role_key == "polygon":
            self.polygonSelector.setCurrentNode(node)
        elif role_key == "volume":
            self.volumeSelector.setCurrentNode(node)

    def _ensure_aoa_inputs(self) -> None:
        try:
            node = self._ensure_landmark_node()
            labels = self.logic.bridge.markups_labels(node)
            points = self.logic.bridge.markups_to_points(node, minimum=1)
        except Exception:
            return
        label_to_point = {str(label).strip().lower(): self._scaled_point(points[idx]) for idx, label in enumerate(labels)}
        required = ["cranial", "caudal", "medial", "lateral"]
        if all(label in label_to_point for label in required):
            entry_labels = required
            entry_points = np.vstack([label_to_point[label] for label in entry_labels]).astype(float)
            entry_node, _ = self.logic.populate_named_markups_node("Surgiplot Entry", "entry", entry_labels, entry_points)
            self.entrySelector.setCurrentNode(entry_node)
        if "pivot" in label_to_point:
            pivot_point = np.asarray([label_to_point["pivot"]], dtype=float)
            pivot_node, _ = self.logic.populate_named_markups_node("Surgiplot Pivot", "pivot", ["pivot"], pivot_point)
            self.pivotSelector.setCurrentNode(pivot_node)
        self._refresh_all_status()

    def onPickCustomLabel(self) -> None:
        label = self._lineedit_text(self.customLabelEdit).strip()
        if not label:
            self._show_error("Custom label missing", ValueError("Enter a custom landmark label first."))
            return
        self.onCollectLabeledLandmark({"mode": "fixed", "label": label})

    def onSaveLandmarkLabelEdits(self) -> None:
        if self._updating_landmark_table:
            return
        node = self._ensure_landmark_node()
        count = self.logic.bridge.markups_count(node)
        labels_seen: set[str] = set()
        for row in range(count):
            item = self.landmarkTable.item(row, 0)
            label = str(item.text() if item is not None else "").strip()
            if not label:
                self._show_error("Save label edits failed", ValueError("Landmark labels cannot be empty."))
                return
            lowered = label.lower()
            if lowered in labels_seen:
                self._show_error("Save label edits failed", ValueError(f"Duplicate landmark label: {label}"))
                return
            labels_seen.add(lowered)
        for row in range(count):
            item = self.landmarkTable.item(row, 0)
            node.SetNthControlPointLabel(row, str(item.text()).strip())
        self._refresh_all_status()
        self._show_results({"action": "save_landmark_label_edits", "point_count": count})

    def _selected_landmark_rows(self) -> list[int]:
        rows = sorted({index.row() for index in self.landmarkTable.selectionModel().selectedRows()})
        return rows

    def _landmark_rows_payload(self) -> tuple[list[str], np.ndarray]:
        node = self._ensure_landmark_node()
        count = self.logic.bridge.markups_count(node)
        labels = self.logic.bridge.markups_labels(node) if count > 0 else []
        points = self.logic.bridge.markups_to_points(node, minimum=1) if count > 0 else np.empty((0, 3), dtype=float)
        return labels, np.asarray(points, dtype=float)

    def _rewrite_landmark_node(self, labels: list[str], points: np.ndarray) -> None:
        node = self._ensure_landmark_node()
        count = self.logic.bridge.markups_count(node)
        for row in reversed(range(count)):
            node.RemoveNthControlPoint(row)
        pts = np.asarray(points, dtype=float)
        if len(labels) != len(pts):
            raise ValueError("labels and points must have the same length.")
        for label, point in zip(labels, pts):
            idx = node.AddControlPointWorld(vtk.vtkVector3d(float(point[0]), float(point[1]), float(point[2])))
            node.SetNthControlPointLabel(idx, str(label))
        self._landmark_last_count = self.logic.bridge.markups_count(node)
        self._refresh_all_status()

    def onDeleteSelectedLandmarks(self) -> None:
        node = self._ensure_landmark_node()
        rows = self._selected_landmark_rows()
        if not rows:
            return
        for row in reversed(rows):
            node.RemoveNthControlPoint(row)
        self._landmark_last_count = self.logic.bridge.markups_count(node)
        self._refresh_all_status()

    def onClearLandmarkDataset(self) -> None:
        node = self._ensure_landmark_node()
        count = self.logic.bridge.markups_count(node)
        for row in reversed(range(count)):
            node.RemoveNthControlPoint(row)
        self._landmark_last_count = 0
        self._refresh_all_status()

    def onSetDatasetVisibility(self, visible: bool) -> None:
        try:
            payload = self.logic.set_nodes_visibility([self._ensure_landmark_node()], visible)
            self._show_results(payload)
        except Exception as exc:
            self._show_error(f"{'Show' if visible else 'Hide'} dataset points failed", exc)

    def onLandmarkSelectionChanged(self) -> None:
        if self._updating_landmark_table:
            return
        node = self._ensure_landmark_node()
        rows = self._selected_landmark_rows()
        count = self.logic.bridge.markups_count(node)
        for index in range(count):
            try:
                node.SetNthControlPointSelected(index, index in rows)
            except Exception:
                pass
        if not rows:
            return
        try:
            points = self.logic.bridge.markups_to_points(node, minimum=1)
            point = points[rows[0]]
            self.logic.jump_to_world_point(point)
        except Exception:
            pass

    def onRenameSelectedLandmark(self) -> None:
        rows = self._selected_landmark_rows()
        if len(rows) != 1:
            self._show_error("Rename landmark failed", ValueError("Select exactly one landmark row to rename."))
            return
        labels, points = self._landmark_rows_payload()
        row = rows[0]
        old = str(labels[row] or "").strip() or f"point_{row + 1}"
        new, ok = qt.QInputDialog.getText(self.parent or None, "Rename landmark", f"New label for '{old}':")
        if not ok:
            return
        new = str(new or "").strip()
        if not new or new == old:
            return
        lowered = {idx: str(label).strip().lower() for idx, label in enumerate(labels)}
        if any(idx != row and value == new.lower() for idx, value in lowered.items()):
            self._show_error("Rename landmark failed", ValueError(f"A landmark labeled '{new}' already exists."))
            return
        labels[row] = new
        self._rewrite_landmark_node(labels, points)
        self.landmarkTable.selectRow(row)

    def onMoveSelectedLandmark(self, direction: int) -> None:
        rows = self._selected_landmark_rows()
        if len(rows) != 1:
            self._show_error("Reorder landmark failed", ValueError("Select exactly one landmark row to move."))
            return
        row = rows[0]
        labels, points = self._landmark_rows_payload()
        new_row = row + int(direction)
        if new_row < 0 or new_row >= len(labels):
            return
        labels[row], labels[new_row] = labels[new_row], labels[row]
        points[[row, new_row]] = points[[new_row, row]]
        self._rewrite_landmark_node(labels, points)
        self.landmarkTable.selectRow(new_row)

    def _selected_landmark_points_and_labels(self) -> tuple[list[str], np.ndarray]:
        node = self._ensure_landmark_node()
        rows = self._selected_landmark_rows()
        if not rows:
            raise ValueError("Select one or more collected landmarks first.")
        points = self.logic.bridge.markups_to_points(node, minimum=1)
        labels = self.logic.bridge.markups_labels(node)
        chosen_labels = [(labels[row] or f"point_{row + 1}") for row in rows]
        chosen_points = np.vstack([self._scaled_point(points[row]) for row in rows]).astype(float)
        return chosen_labels, chosen_points

    def onExportSelectedLandmarks(self, role: str) -> None:
        try:
            labels, points = self._selected_landmark_points_and_labels()
            if role == "entry":
                node, payload = self.logic.populate_named_markups_node("Surgiplot Entry", "entry", labels, points)
                self.entrySelector.setCurrentNode(node)
            elif role == "target":
                node, payload = self.logic.populate_named_markups_node("Surgiplot Target", "target", labels, points)
                self.targetSelector.setCurrentNode(node)
            elif role == "pivot":
                if len(points) != 1:
                    raise ValueError("Pivot export requires exactly one selected landmark.")
                node, payload = self.logic.populate_named_markups_node("Surgiplot Pivot", "pivot", labels, points)
                self.pivotSelector.setCurrentNode(node)
            elif role == "a":
                if len(points) != 1:
                    raise ValueError("A export requires exactly one selected landmark.")
                node, payload = self.logic.populate_named_markups_node("Surgiplot A", "a", labels, points)
                self.aSelector.setCurrentNode(node)
            elif role == "b":
                if len(points) != 1:
                    raise ValueError("B export requires exactly one selected landmark.")
                node, payload = self.logic.populate_named_markups_node("Surgiplot B", "b", labels, points)
                self.bSelector.setCurrentNode(node)
            elif role == "c":
                if len(points) != 1:
                    raise ValueError("C export requires exactly one selected landmark.")
                node, payload = self.logic.populate_named_markups_node("Surgiplot C", "c", labels, points)
                self.cSelector.setCurrentNode(node)
            elif role == "polygon":
                node, payload = self.logic.populate_named_markups_node("Surgiplot Polygon", "polygon", labels, points)
                self.polygonSelector.setCurrentNode(node)
            elif role == "volume":
                node, payload = self.logic.populate_named_markups_node("Surgiplot Volume", "volume", labels, points)
                self.volumeSelector.setCurrentNode(node)
            else:
                raise ValueError(f"Unsupported export role: {role}")
            self._refresh_all_status()
            self._show_results(payload)
        except Exception as exc:
            self._show_error(f"Export selected landmarks to {role} failed", exc)

    def onAutoFillAoaFromDataset(self) -> None:
        node = self._ensure_landmark_node()
        try:
            labels = self.logic.bridge.markups_labels(node)
            points = self.logic.bridge.markups_to_points(node, minimum=1)
            label_to_point = {str(label).strip().lower(): self._scaled_point(points[idx]) for idx, label in enumerate(labels)}
            required = ["cranial", "caudal", "medial", "lateral"]
            missing = [label for label in required if label not in label_to_point]
            if missing:
                raise ValueError(f"Missing AoA labels in landmark dataset: {', '.join(missing)}")
            pivot_point = label_to_point.get("pivot")
            if pivot_point is None:
                raise ValueError("Missing 'pivot' label in landmark dataset.")
            entry_labels = required
            entry_points = np.vstack([label_to_point[label] for label in entry_labels]).astype(float)
            entry_node, entry_payload = self.logic.populate_named_markups_node("Surgiplot Entry", "entry", entry_labels, entry_points)
            pivot_node, pivot_payload = self.logic.populate_named_markups_node("Surgiplot Pivot", "pivot", ["pivot"], np.asarray([pivot_point], dtype=float))
            self.entrySelector.setCurrentNode(entry_node)
            self.pivotSelector.setCurrentNode(pivot_node)
            self._refresh_all_status()
            self._show_results({"action": "auto_fill_aoa_from_dataset", "entry": entry_payload, "pivot": pivot_payload})
        except Exception as exc:
            self._show_error("Auto-fill AoA from dataset failed", exc)

    def _populate_single_role_from_dataset_label(self, role: str, label_text: str):
        node = self._ensure_landmark_node()
        labels = self.logic.bridge.markups_labels(node)
        points = self.logic.bridge.markups_to_points(node, minimum=1)
        for idx, label in enumerate(labels):
            if str(label or "").strip().lower() == str(label_text).strip().lower():
                point = self._scaled_point(points[idx]).reshape(1, 3)
                out_node, _payload = self.logic.populate_named_markups_node(f"Surgiplot {role.upper()}", role, [label_text], point)
                return out_node
        raise ValueError(f"Dataset does not contain a point labeled '{label_text}'.")

    def _ensure_aoe_inputs(self) -> None:
        current = {
            "a": self.aSelector.currentNode(),
            "b": self.bSelector.currentNode(),
            "c": self.cSelector.currentNode(),
        }
        needs = []
        for role, node in current.items():
            if node is None or self.logic.bridge.markups_count(node) != 1:
                needs.append(role)
        if not needs:
            return
        if "a" in needs:
            self.aSelector.setCurrentNode(self._populate_single_role_from_dataset_label("a", "A"))
        if "b" in needs:
            self.bSelector.setCurrentNode(self._populate_single_role_from_dataset_label("b", "B"))
        if "c" in needs:
            self.cSelector.setCurrentNode(self._populate_single_role_from_dataset_label("c", "C"))
        self._refresh_all_status()

    def onSetBackgroundVolume(self) -> None:
        try:
            payload = self.logic.set_background_volume(self.imageVolumeSelector.currentNode())
            self._show_imaging_status(
                [
                    f"Background volume: {payload.get('background_volume', 'none')}",
                    "Slice viewers updated for CT/MRI-linked markup placement.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Set background volume failed", exc)

    def onCreateEntryNode(self) -> None:
        try:
            node, payload = self.logic.create_entry_markups_node()
            self.entrySelector.setCurrentNode(node)
            self._refresh_all_status()
            self._show_imaging_status(
                [
                    f"Created entry node: {payload.get('node_name', '-')}",
                    "Add 4 control points on CT/MRI or model anatomy, then assign or verify AoA/SF roles.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Create entry node failed", exc)

    def onCreateTargetNode(self) -> None:
        try:
            node, payload = self.logic.create_target_markups_node()
            self.targetSelector.setCurrentNode(node)
            self._refresh_all_status()
            self._show_imaging_status(
                [
                    f"Created target node: {payload.get('node_name', '-')}",
                    "Add target contour points from the active CT/MRI/model context.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Create target node failed", exc)

    def onCreatePivotNode(self) -> None:
        try:
            node, payload = self.logic.create_pivot_markups_node()
            self.pivotSelector.setCurrentNode(node)
            self._refresh_all_status()
            self._show_imaging_status(
                [
                    f"Created pivot node: {payload.get('node_name', '-')}",
                    "Add a single pivot point from the active CT/MRI/model context.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Create pivot node failed", exc)

    def onJumpToEntry(self) -> None:
        try:
            payload = self.logic.jump_slices_to_markups(self.entrySelector.currentNode(), mode="centroid")
            self._show_imaging_status(
                [
                    f"Jumped slices to entry node: {payload.get('node_name', '-')}",
                    "Use this to reposition CT/MRI slice views around the entry anatomy.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Jump to entry failed", exc)

    def onJumpToTarget(self) -> None:
        try:
            payload = self.logic.jump_slices_to_markups(self.targetSelector.currentNode(), mode="centroid")
            self._show_imaging_status(
                [
                    f"Jumped slices to target node: {payload.get('node_name', '-')}",
                    "Use this to reposition CT/MRI slice views around the target anatomy.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Jump to target failed", exc)

    def onJumpToPivot(self) -> None:
        try:
            payload = self.logic.jump_slices_to_markups(self.pivotSelector.currentNode(), mode="first")
            self._show_imaging_status(
                [
                    f"Jumped slices to pivot node: {payload.get('node_name', '-')}",
                    "Use this to center slice views exactly on the pivot point.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Jump to pivot failed", exc)

    def onPlaceEntry(self) -> None:
        try:
            payload = self.logic.activate_place_mode(self.entrySelector.currentNode(), place_label="entry", persistent=False)
            self._show_imaging_status(
                [
                    f"Placement armed for entry node: {payload.get('node_name', '-')}",
                    "Click in a slice or 3D view to add the next entry point.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Entry placement activation failed", exc)

    def onPlaceTarget(self) -> None:
        try:
            payload = self.logic.activate_place_mode(self.targetSelector.currentNode(), place_label="target", persistent=False)
            self._show_imaging_status(
                [
                    f"Placement armed for target node: {payload.get('node_name', '-')}",
                    "Click in a slice or 3D view to add the next target point.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Target placement activation failed", exc)

    def onPlacePivot(self) -> None:
        try:
            payload = self.logic.activate_place_mode(self.pivotSelector.currentNode(), place_label="pivot", persistent=False)
            self._show_imaging_status(
                [
                    f"Placement armed for pivot node: {payload.get('node_name', '-')}",
                    "Click in a slice or 3D view to add the pivot point.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Pivot placement activation failed", exc)

    def onPlaceNextGuidedPoint(self) -> None:
        guided_target, guided_label = self._guided_target_state()
        node = self._node_for_guided_target(guided_target)
        if node is None:
            self._show_imaging_status(
                [
                    "No guided target is available.",
                    "Start a guided session or finish selecting the required markups nodes first.",
                ]
            )
            return
        try:
            payload = self.logic.activate_place_mode(node, place_label=guided_label, persistent=False)
            self._show_imaging_status(
                [
                    f"Placement armed for guided target: {guided_target}",
                    f"Next expected point: {guided_label}",
                    "Click in a slice or 3D view to add the point.",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Guided placement activation failed", exc)

    def onJumpToGuidedTarget(self) -> None:
        guided_target, guided_label = self._guided_target_state()
        node = self._node_for_guided_target(guided_target)
        if node is None:
            self._show_imaging_status(["No guided target is available to jump to."])
            return
        mode = "first" if guided_target == "pivot" else "centroid"
        try:
            payload = self.logic.jump_slices_to_markups(node, mode=mode)
            self._show_imaging_status(
                [
                    f"Jumped slices to guided target: {guided_target}",
                    f"Context for next expected point: {guided_label}",
                ]
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Jump to guided target failed", exc)

    def onComputeVom(self) -> None:
        try:
            state = self.logic.describe_workflow(
                entry_node=self.entrySelector.currentNode(),
                target_node=self.targetSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
            )
            if not state.get("vom_ready", False):
                raise ValueError("VOM / sVOM / VoA inputs are not ready. Review the workflow validation panel.")
            payload = self.logic.compute_vom_family(
                self.entrySelector.currentNode(),
                self.targetSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
                stand_dist=self._spinbox_value(self.standDistSpin),
            )
            self._show_results(payload)
            self._show_metric_results("vom", payload)
        except Exception as exc:
            self._show_error("VOM / sVOM / VoA failed", exc)

    def onComputeAoa(self) -> None:
        self._compute_aoa_impl(standardized=False)

    def onComputeAoaStandardized(self) -> None:
        self._compute_aoa_impl(standardized=True)

    def _compute_aoa_impl(self, *, standardized: bool) -> None:
        try:
            self._ensure_aoa_inputs()
            state = self.logic.describe_workflow(
                entry_node=self.entrySelector.currentNode(),
                pivot_node=self.pivotSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
            )
            if not state.get("aoa_ready", False):
                raise ValueError("AoA / SF inputs are not ready. Review the workflow validation panel.")
            if standardized and not self.sfStandardizedCheckBox.isChecked():
                raise ValueError("Enable standardized SF first, then set the desired radius.")
            sf_radius = self._spinbox_value(self.sfRadiusSpin) if standardized else None
            payload = self.logic.compute_aoa_family(
                self.entrySelector.currentNode(),
                self.pivotSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
                sf_rescale_radius_mm=sf_radius or None,
                standardized_requested=standardized,
            )
            payload["standardized_sf_enabled"] = bool(standardized)
            payload["standardized_sf_radius_mm"] = float(sf_radius) if sf_radius else None
            self._show_results(payload)
            self._show_metric_results("aoa", payload)
        except Exception as exc:
            self._show_error("AoA / SF failed", exc)

    def onComputeAoe(self) -> None:
        try:
            self._ensure_aoe_inputs()
            state = self.logic.describe_workflow(
                a_node=self.aSelector.currentNode(),
                b_node=self.bSelector.currentNode(),
                c_node=self.cSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
            )
            if not state.get("aoe_ready", False):
                raise ValueError("AoE inputs are not ready. Review the workflow validation panel.")
            payload = self.logic.compute_aoe(
                self.aSelector.currentNode(),
                self.bSelector.currentNode(),
                self.cSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
            )
            self._show_results(payload)
            self._show_metric_results("aoe", payload)
        except Exception as exc:
            self._show_error("AoE failed", exc)

    def onComputeDistance(self) -> None:
        try:
            state = self.logic.describe_workflow(
                a_node=self.aSelector.currentNode(),
                b_node=self.bSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
            )
            if not state.get("distance_ready", False):
                raise ValueError("Distance inputs are not ready. Review the workflow validation panel.")
            payload = self.logic.compute_distance(
                self.aSelector.currentNode(),
                self.bSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
            )
            self._show_results(payload)
            self._show_metric_results("distance", payload)
        except Exception as exc:
            self._show_error("Distance failed", exc)

    def onComputeArea(self) -> None:
        try:
            state = self.logic.describe_workflow(
                polygon_node=self.polygonSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
            )
            if not state.get("area_ready", False):
                raise ValueError("Area inputs are not ready. Review the workflow validation panel.")
            payload = self.logic.compute_area(self.polygonSelector.currentNode(), reference_node=self.referenceSelector.currentNode())
            self._show_results(payload)
            self._show_metric_results("area", payload)
        except Exception as exc:
            self._show_error("Area failed", exc)

    def onComputeVolume(self) -> None:
        try:
            state = self.logic.describe_workflow(
                volume_node=self.volumeSelector.currentNode(),
                reference_node=self.referenceSelector.currentNode(),
            )
            if not state.get("volume_ready", False):
                raise ValueError("Volume inputs are not ready. Review the workflow validation panel.")
            payload = self.logic.compute_volume(self.volumeSelector.currentNode(), reference_node=self.referenceSelector.currentNode())
            self._show_results(payload)
            self._show_metric_results("volume", payload)
        except Exception as exc:
            self._show_error("Volume failed", exc)

    def onApplyModelOpacity(self) -> None:
        try:
            payload = self.logic.set_reference_model_opacity(
                self.referenceSelector.currentNode(),
                float(self._slider_value(self.modelOpacitySlider)) / 100.0,
            )
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Apply model opacity failed", exc)

    def onApplyMetricOpacity(self) -> None:
        try:
            payload = self.logic.set_result_nodes_opacity(float(self._slider_value(self.metricOpacitySlider)) / 100.0)
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Apply metric opacity failed", exc)

    def onSetMetricOverlayVisibility(self, metric: str, visible: bool) -> None:
        try:
            payload = self.logic.set_metric_group_visibility(metric, visible)
            self._show_results(payload)
        except Exception as exc:
            self._show_error(f"{'Show' if visible else 'Hide'} metric overlay failed", exc)

    def onSetMetricInputVisibility(self, roles: list[str], visible: bool) -> None:
        try:
            role_to_node = {
                "entry": self.entrySelector.currentNode(),
                "target": self.targetSelector.currentNode(),
                "pivot": self.pivotSelector.currentNode(),
                "a": self.aSelector.currentNode(),
                "b": self.bSelector.currentNode(),
                "c": self.cSelector.currentNode(),
                "polygon": self.polygonSelector.currentNode(),
                "volume": self.volumeSelector.currentNode(),
            }
            nodes = [role_to_node[role] for role in roles if role_to_node.get(role) is not None]
            payload = self.logic.set_nodes_visibility(nodes, visible)
            self._show_results(payload)
        except Exception as exc:
            self._show_error(f"{'Show' if visible else 'Hide'} points failed", exc)

    def _all_point_nodes(self) -> list[Any]:
        nodes = [
            self._landmark_node,
            self._scale_node,
            self.entrySelector.currentNode(),
            self.targetSelector.currentNode(),
            self.pivotSelector.currentNode(),
            self.aSelector.currentNode(),
            self.bSelector.currentNode(),
            self.cSelector.currentNode(),
            self.polygonSelector.currentNode(),
            self.volumeSelector.currentNode(),
        ]
        result = []
        seen: set[str] = set()
        for node in nodes:
            if node is None or not hasattr(node, "GetID"):
                continue
            node_id = node.GetID()
            if node_id in seen:
                continue
            seen.add(node_id)
            result.append(node)
        return result

    def onSetAllPointsVisibility(self, visible: bool) -> None:
        try:
            payload = self.logic.set_nodes_visibility(self._all_point_nodes(), visible)
            self._show_results(payload)
        except Exception as exc:
            self._show_error(f"{'Show' if visible else 'Hide'} all points failed", exc)

    def onExportMetricScene(self, metric: str) -> None:
        try:
            directory = qt.QFileDialog.getExistingDirectory(
                slicer.util.mainWindow(),
                "Choose export folder for metric 3D scene",
                str(Path.home()),
            )
            if not directory:
                return
            payload = self.logic.export_metric_scene(metric, Path(str(directory)))
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Export metric 3D scene failed", exc)

    def _export_subject_and_target(self) -> tuple[str, str]:
        subject_id = self._lineedit_text(self.subjectIdEdit).strip()
        anatomical_target = self._lineedit_text(self.anatomicalTargetEdit).strip()
        if not subject_id:
            raise ValueError("Enter a Subject ID first.")
        if not anatomical_target:
            raise ValueError("Enter an Anatomical target first.")
        return subject_id, anatomical_target

    def _current_metric_points_snapshot(self) -> list[dict[str, Any]]:
        payload = self._last_metric_payload or {}
        metric_family = str(payload.get("metric_family", "")).strip()
        rows: list[dict[str, Any]] = []
        mapping_by_metric = {
            "VOM / sVOM / VoA": [("entry", self.entrySelector.currentNode()), ("target", self.targetSelector.currentNode())],
            "AoA / SF": [("entry", self.entrySelector.currentNode()), ("pivot", self.pivotSelector.currentNode())],
            "AoE": [("A", self.aSelector.currentNode()), ("B", self.bSelector.currentNode()), ("C", self.cSelector.currentNode())],
            "Distance": [("A", self.aSelector.currentNode()), ("B", self.bSelector.currentNode())],
            "Area": [("polygon", self.polygonSelector.currentNode())],
            "Volume": [("volume", self.volumeSelector.currentNode())],
        }
        mapping = mapping_by_metric.get(metric_family, [])
        for point_group, node in mapping:
            if node is None:
                continue
            try:
                pts = self.logic.bridge.markups_to_points(node, minimum=1)
                labels = self.logic.bridge.markups_labels(node)
            except Exception:
                continue
            for idx, pt in enumerate(np.asarray(pts, dtype=float)):
                rows.append(
                    {
                        "metric_family": metric_family,
                        "point_group": point_group,
                        "point_label": labels[idx] if idx < len(labels) else f"{point_group}_{idx + 1}",
                        "x": float(pt[0]),
                        "y": float(pt[1]),
                        "z": float(pt[2]),
                    }
                )
        return rows

    def onAppendCurrentResultToDataset(self) -> None:
        try:
            subject_id, anatomical_target = self._export_subject_and_target()
            payload = self._last_metric_payload
            if not payload or not payload.get("metric_family"):
                raise ValueError("Compute a metric first.")
            export_payload = self.logic.append_case_result_record(subject_id, anatomical_target, payload)
            self._show_results(export_payload)
            self.onRefreshExportPreview()
        except Exception as exc:
            self._show_error("Append current result failed", exc)

    def onAppendCurrentPointsToDataset(self) -> None:
        try:
            subject_id, anatomical_target = self._export_subject_and_target()
            rows = self._current_metric_points_snapshot()
            if not rows:
                raise ValueError("No current metric points are available to append.")
            export_payload = self.logic.append_points_record(subject_id, anatomical_target, rows)
            self._show_results(export_payload)
            self.onRefreshExportPreview()
        except Exception as exc:
            self._show_error("Append current points failed", exc)

    def onExportDatasetFile(self, dataset_kind: str, file_kind: str) -> None:
        try:
            if dataset_kind == "results":
                default_name = "surgiplot_case_results"
            else:
                default_name = "surgiplot_points_reproducibility"
            suffix = ".xlsx" if file_kind == "xlsx" else ".csv"
            dialog_result = qt.QFileDialog.getSaveFileName(
                slicer.util.mainWindow(),
                "Export Dataset",
                str(Path.home() / f"{default_name}{suffix}"),
                f"{file_kind.upper()} files (*{suffix})",
            )
            if isinstance(dialog_result, (tuple, list)):
                path = str(dialog_result[0]) if dialog_result else ""
            else:
                path = str(dialog_result or "")
            if not path.strip():
                return
            payload = self.logic.export_dataset_file(dataset_kind, Path(path), file_kind=file_kind)
            self._show_results(payload)
        except Exception as exc:
            self._show_error("Export dataset failed", exc)

    def _populate_export_preview_table(self, table_widget, dataset_kind: str) -> None:
        node = self.logic.get_dataset_table_node(dataset_kind)
        table_widget.blockSignals(True)
        try:
            if node is None:
                table_widget.clear()
                table_widget.setRowCount(0)
                table_widget.setColumnCount(0)
                return
            vtk_table = node.GetTable()
            headers = self.logic._table_headers(node)
            table_widget.clear()
            table_widget.setColumnCount(len(headers))
            table_widget.setHorizontalHeaderLabels(headers)
            row_count = vtk_table.GetNumberOfRows()
            col_count = vtk_table.GetNumberOfColumns()
            table_widget.setRowCount(row_count)
            for row in range(row_count):
                for col in range(col_count):
                    value = str(vtk_table.GetValue(row, col).ToString())
                    item = qt.QTableWidgetItem(value)
                    table_widget.setItem(row, col, item)
            header = table_widget.horizontalHeader()
            if hasattr(header, "setSectionResizeMode"):
                for col in range(len(headers)):
                    mode = qt.QHeaderView.ResizeToContents if col < 2 else qt.QHeaderView.Interactive
                    header.setSectionResizeMode(col, mode)
        finally:
            table_widget.blockSignals(False)

    def onRefreshExportPreview(self) -> None:
        self._populate_export_preview_table(self.resultsPreviewTable, "results")
        self._populate_export_preview_table(self.pointsPreviewTable, "points")

    def _save_export_preview_table(self, table_widget, dataset_kind: str) -> None:
        headers = []
        for col in range(table_widget.columnCount()):
            header_item = table_widget.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item is not None else f"Column {col + 1}")
        rows: list[list[str]] = []
        for row in range(table_widget.rowCount()):
            values: list[str] = []
            for col in range(table_widget.columnCount()):
                item = table_widget.item(row, col)
                values.append(item.text() if item is not None else "")
            if any(str(value).strip() for value in values):
                rows.append(values)
        self.logic.replace_dataset_table_contents(dataset_kind, headers, rows)

    def onSaveExportPreviewEdits(self) -> None:
        try:
            self._save_export_preview_table(self.resultsPreviewTable, "results")
            self._save_export_preview_table(self.pointsPreviewTable, "points")
            self.onRefreshExportPreview()
            self._show_results({"action": "save_export_preview_edits", "status": "ok"})
        except Exception as exc:
            self._show_error("Save preview edits failed", exc)


class SurgiplotToolkitLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = SurgiplotMetricBridge()
        self._result_folder_name = "SurgiplotToolkit Results"
        self._result_node_names = [
            "Surgiplot Entry Polygon",
            "Surgiplot Target Polygon",
            "Surgiplot Entry Ellipse",
            "Surgiplot Target Ellipse",
            "Surgiplot sVOM Cut Ellipse",
            "Surgiplot Corridor Axis",
            "Surgiplot Target Approach",
            "Surgiplot Corridor Centroids",
            "Surgiplot VOM Surface",
            "Surgiplot sVOM Surface",
            "Surgiplot VOM Results",
            "Surgiplot Vertical AoA",
            "Surgiplot Horizontal AoA",
            "Surgiplot SF Surface",
            "Surgiplot Standardized SF Surface",
            "Surgiplot Standardized SF Guide cranial",
            "Surgiplot Standardized SF Guide caudal",
            "Surgiplot Standardized SF Guide medial",
            "Surgiplot Standardized SF Guide lateral",
            "Surgiplot Entry Quad",
            "Surgiplot AoA Landmarks",
            "Surgiplot AoA Results",
            "Surgiplot AoE Arc",
            "Surgiplot AoE Results",
            "Surgiplot Distance Results",
            "Surgiplot Area Results",
            "Surgiplot Volume Results",
            "Surgiplot Case Results Dataset",
            "Surgiplot Points Reproducibility Dataset",
            "Surgiplot Latest Report",
        ]
        self._metric_result_groups = {
            "vom": [
                "Surgiplot Entry Polygon",
                "Surgiplot Target Polygon",
                "Surgiplot Entry Ellipse",
                "Surgiplot Target Ellipse",
                "Surgiplot sVOM Cut Ellipse",
                "Surgiplot Corridor Axis",
                "Surgiplot Target Approach",
                "Surgiplot Corridor Side 1",
                "Surgiplot Corridor Side 2",
                "Surgiplot Corridor Side 3",
                "Surgiplot Corridor Side 4",
                "Surgiplot Corridor Centroids",
                "Surgiplot VOM Surface",
                "Surgiplot sVOM Surface",
                "Surgiplot VOM Results",
            ],
            "aoa": [
                "Surgiplot Vertical AoA",
                "Surgiplot Horizontal AoA",
                "Surgiplot SF Surface",
                "Surgiplot Standardized SF Surface",
                "Surgiplot Standardized SF Guide cranial",
                "Surgiplot Standardized SF Guide caudal",
                "Surgiplot Standardized SF Guide medial",
                "Surgiplot Standardized SF Guide lateral",
                "Surgiplot Entry Quad",
                "Surgiplot AoA Landmarks",
                "Surgiplot AoA Results",
            ],
            "aoe": ["Surgiplot AoE Triangle", "Surgiplot AoE Arc", "Surgiplot AoE Landmarks", "Surgiplot AoE Results"],
            "distance": ["Surgiplot Distance Line", "Surgiplot Distance Landmarks", "Surgiplot Distance Results"],
            "area": ["Surgiplot Area Polygon", "Surgiplot Area Surface", "Surgiplot Area Landmarks", "Surgiplot Area Results"],
            "volume": ["Surgiplot Volume Surface", "Surgiplot Volume Landmarks", "Surgiplot Volume Results"],
        }

    def _reference_name(self, reference_node) -> str:
        return str(reference_node.GetName()) if reference_node is not None else "Unlinked"

    def _jump_all_slices(self, point_xyz: Any) -> None:
        point = np.asarray(point_xyz, dtype=float).reshape(3,)
        slicer.vtkMRMLSliceNode.JumpAllSlices(
            slicer.mrmlScene,
            float(point[0]),
            float(point[1]),
            float(point[2]),
            slicer.vtkMRMLSliceNode.CenteredJumpSlice,
        )

    def jump_to_world_point(self, point_xyz: Any) -> dict[str, Any]:
        point = np.asarray(point_xyz, dtype=float).reshape(3,)
        self._jump_all_slices(point)
        return {
            "action": "jump_to_world_point",
            "point_xyz": [float(point[0]), float(point[1]), float(point[2])],
        }

    def _fit_all_slice_views_to_background(self) -> None:
        layout_manager = slicer.app.layoutManager()
        if layout_manager is None:
            return
        for name in layout_manager.sliceViewNames():
            widget = layout_manager.sliceWidget(name)
            if widget is None:
                continue
            try:
                widget.sliceLogic().FitSliceToBackground()
            except Exception:
                pass

    def snap_point_to_reference_model(self, point_xyz: Any, *, reference_node=None) -> Optional[np.ndarray]:
        if reference_node is None or not hasattr(reference_node, "IsA") or not reference_node.IsA("vtkMRMLModelNode"):
            return None
        poly_data = reference_node.GetPolyData()
        if poly_data is None or poly_data.GetNumberOfPoints() == 0:
            return None
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(poly_data)
        locator.BuildLocator()
        point = np.asarray(point_xyz, dtype=float).reshape(3,)
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        locator.FindClosestPoint(point.tolist(), closest, cell_id, sub_id, dist2)
        return np.asarray(closest, dtype=float)

    def _configure_markups_display(self, node, *, color: tuple[float, float, float]) -> None:
        display_node = node.GetDisplayNode()
        if display_node is None:
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
        if display_node is None:
            return
        display_node.SetSelectedColor(*color)
        display_node.SetColor(*color)
        display_node.SetTextScale(1.6)
        display_node.SetGlyphScale(1.8)
        display_node.SetVisibility2D(True)
        display_node.SetVisibility3D(True)
        display_node.SetSliceProjection(True)
        display_node.SetSliceProjectionUseFiducialColor(True)

    def _configure_metric_input_display(self, node, *, visible: bool) -> None:
        if node is None:
            return
        display_node = node.GetDisplayNode()
        if display_node is None:
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
        if display_node is None:
            return
        node.SetDisplayVisibility(bool(visible))
        if hasattr(display_node, "SetVisibility"):
            display_node.SetVisibility(bool(visible))
        if hasattr(display_node, "SetVisibility2D"):
            display_node.SetVisibility2D(bool(visible))
        if hasattr(display_node, "SetVisibility3D"):
            display_node.SetVisibility3D(bool(visible))

    def _configure_scale_markups_display(self, node) -> None:
        if node is None:
            return
        display_node = node.GetDisplayNode()
        if display_node is None:
            node.CreateDefaultDisplayNodes()
            display_node = node.GetDisplayNode()
        if display_node is None:
            return
        display_node.SetVisibility2D(True)
        display_node.SetVisibility3D(True)
        if hasattr(display_node, "SetSliceProjection"):
            display_node.SetSliceProjection(False)

    def _get_subject_hierarchy_node(self):
        return slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)

    def _get_or_create_results_folder(self, reference_node=None):
        sh_node = self._get_subject_hierarchy_node()
        parent_item = sh_node.GetSceneItemID()
        if reference_node is not None:
            ref_item = sh_node.GetItemByDataNode(reference_node)
            if ref_item:
                subject_item = sh_node.GetItemParent(ref_item)
                if subject_item:
                    parent_item = subject_item
        folder_name = self._result_folder_name
        if reference_node is not None:
            folder_name = f"{self._result_folder_name} - {reference_node.GetName()}"
        folder_item = sh_node.GetItemByName(folder_name)
        if not folder_item:
            folder_item = sh_node.CreateFolderItem(parent_item, folder_name)
        return sh_node, folder_item

    def _register_result_node(self, node, reference_node=None) -> None:
        if node is None:
            return
        sh_node, folder_item = self._get_or_create_results_folder(reference_node)
        item_id = sh_node.GetItemByDataNode(node)
        if item_id:
            sh_node.SetItemParent(item_id, folder_item)

    def _remove_prior_result_nodes(self, names: list[str]) -> None:
        for name in names:
            existing = self._get_node_by_name(name)
            if existing is not None:
                slicer.mrmlScene.RemoveNode(existing)

    def _get_node_by_name(self, name: str):
        try:
            return slicer.util.getFirstNodeByName(name)
        except Exception:
            return None

    def _update_results_report(self, payload: dict[str, Any], *, reference_node=None) -> Optional[Any]:
        name = "Surgiplot Latest Report"
        existing = self._get_node_by_name(name)
        if existing is not None:
            slicer.mrmlScene.RemoveNode(existing)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", name)
        node.SetText(json.dumps(payload, indent=2, sort_keys=True))
        self._register_result_node(node, reference_node)
        return node

    def _set_display_opacity(self, node, opacity: float) -> bool:
        if node is None or not hasattr(node, "GetDisplayNode"):
            return False
        display_node = node.GetDisplayNode()
        if display_node is None:
            return False
        if hasattr(display_node, "SetOpacity"):
            display_node.SetOpacity(float(opacity))
            return True
        return False

    def _set_display_visibility(self, node, visible: bool) -> bool:
        if node is None or not hasattr(node, "GetDisplayNode"):
            return False
        display_node = node.GetDisplayNode()
        if display_node is None:
            if hasattr(node, "CreateDefaultDisplayNodes"):
                node.CreateDefaultDisplayNodes()
                display_node = node.GetDisplayNode()
        if display_node is None:
            return False
        changed = False
        if hasattr(node, "SetDisplayVisibility"):
            node.SetDisplayVisibility(bool(visible))
            changed = True
        if hasattr(display_node, "SetVisibility"):
            display_node.SetVisibility(bool(visible))
            changed = True
        if hasattr(display_node, "SetVisibility2D"):
            display_node.SetVisibility2D(bool(visible))
            changed = True
        if hasattr(display_node, "SetVisibility3D"):
            display_node.SetVisibility3D(bool(visible))
            changed = True
        if hasattr(display_node, "SetSliceProjection"):
            display_node.SetSliceProjection(bool(visible))
            changed = True
        return changed

    def clear_result_nodes(self, *, reference_node=None) -> dict[str, Any]:
        removed = 0
        for name in self._result_node_names:
            node = self._get_node_by_name(name)
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)
                removed += 1
        return {
            "action": "clear_result_nodes",
            "reference_anatomy": self._reference_name(reference_node),
            "removed_node_count": removed,
        }

    def set_reference_model_opacity(self, reference_node, opacity: float) -> dict[str, Any]:
        if reference_node is None or not hasattr(reference_node, "IsA") or not reference_node.IsA("vtkMRMLModelNode"):
            raise ValueError("Select a model node as Reference anatomy first.")
        applied = self._set_display_opacity(reference_node, opacity)
        if not applied:
            raise RuntimeError("Reference model does not have an editable display node.")
        return {
            "action": "set_reference_model_opacity",
            "reference_anatomy": self._reference_name(reference_node),
            "opacity": float(opacity),
        }

    def set_result_nodes_opacity(self, opacity: float) -> dict[str, Any]:
        changed = 0
        for name in self._result_node_names:
            node = self._get_node_by_name(name)
            if node is None:
                continue
            if self._set_display_opacity(node, opacity):
                changed += 1
        return {
            "action": "set_result_nodes_opacity",
            "opacity": float(opacity),
            "changed_node_count": int(changed),
        }

    def set_nodes_visibility(self, nodes: list[Any], visible: bool) -> dict[str, Any]:
        changed = 0
        seen: set[str] = set()
        for node in nodes:
            if node is None or not hasattr(node, "GetID"):
                continue
            node_id = node.GetID()
            if node_id in seen:
                continue
            seen.add(node_id)
            if self._set_display_visibility(node, visible):
                changed += 1
        return {
            "action": "set_nodes_visibility",
            "visible": bool(visible),
            "changed_node_count": int(changed),
        }

    def set_metric_group_visibility(self, metric: str, visible: bool) -> dict[str, Any]:
        metric_key = str(metric or "").strip().lower()
        names = self._metric_result_groups.get(metric_key)
        if not names:
            raise ValueError(f"Unknown metric group: {metric}")
        changed = 0
        for name in names:
            node = self._get_node_by_name(name)
            if node is None:
                continue
            if self._set_display_visibility(node, visible):
                changed += 1
        return {
            "action": "set_metric_group_visibility",
            "metric": metric_key,
            "visible": bool(visible),
            "changed_node_count": int(changed),
        }

    def _safe_export_stem(self, text: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
        stem = re.sub(r"_+", "_", stem).strip("._")
        return stem or "node"

    def _node_export_suffix(self, node) -> str:
        if node is None or not hasattr(node, "IsA"):
            return ".dat"
        if node.IsA("vtkMRMLModelNode"):
            return ".vtp"
        if node.IsA("vtkMRMLMarkupsNode"):
            return ".mrk.json"
        if node.IsA("vtkMRMLTableNode"):
            return ".tsv"
        if node.IsA("vtkMRMLTextNode"):
            return ".json"
        return ".dat"

    def export_metric_scene(self, metric: str, directory: Path) -> dict[str, Any]:
        metric_key = str(metric or "").strip().lower()
        names = self._metric_result_groups.get(metric_key)
        if not names:
            raise ValueError(f"Unknown metric group: {metric}")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        exported: list[str] = []
        for name in names:
            node = self._get_node_by_name(name)
            if node is None:
                continue
            suffix = self._node_export_suffix(node)
            path = directory / f"{self._safe_export_stem(name)}{suffix}"
            if suffix == ".json" and node.IsA("vtkMRMLTextNode"):
                path.write_text(str(node.GetText() or ""), encoding="utf-8")
                exported.append(str(path))
                continue
            if suffix == ".tsv" and node.IsA("vtkMRMLTableNode"):
                headers = self._table_headers(node)
                table = node.GetTable()
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, delimiter="\t")
                    writer.writerow(headers)
                    for r in range(table.GetNumberOfRows()):
                        writer.writerow([str(table.GetValue(r, c).ToString()) for c in range(table.GetNumberOfColumns())])
                exported.append(str(path))
                continue
            if slicer.util.saveNode(node, str(path)):
                exported.append(str(path))
        if not exported:
            raise ValueError("No plotted nodes were available to export for this metric.")
        return {
            "action": "export_metric_scene",
            "metric": metric_key,
            "directory": str(directory),
            "exported_count": len(exported),
            "files": exported,
        }

    def describe_workflow(
        self,
        *,
        entry_node=None,
        target_node=None,
        pivot_node=None,
        a_node=None,
        b_node=None,
        c_node=None,
        polygon_node=None,
        volume_node=None,
        reference_node=None,
        background_volume_node=None,
        session_mode: Optional[str] = None,
    ) -> dict[str, Any]:
        aoa_entry = self.bridge.describe_aoa_entry(entry_node)
        pivot_summary = self.bridge.count_summary(pivot_node, expected=1)
        vom_entry = self.bridge.count_summary(entry_node, minimum=3)
        vom_target = self.bridge.count_summary(target_node, minimum=3)
        a_summary = self.bridge.count_summary(a_node, expected=1)
        b_summary = self.bridge.count_summary(b_node, expected=1)
        c_summary = self.bridge.count_summary(c_node, expected=1)
        polygon_summary = self.bridge.count_summary(polygon_node, minimum=3)
        volume_summary = self.bridge.count_summary(volume_node, minimum=4)

        warnings: list[str] = []
        warnings.extend(list(aoa_entry.get("warnings") or []))
        warnings.extend(list(pivot_summary.get("warnings") or []))

        payload = {
            "reference_anatomy": self._reference_name(reference_node),
            "background_volume": background_volume_node.GetName() if background_volume_node is not None else "Unselected",
            "session_mode": session_mode or "none",
            "aoa_ready": bool(aoa_entry.get("valid")) and bool(pivot_summary.get("valid")),
            "vom_ready": bool(vom_entry.get("valid")) and bool(vom_target.get("valid")),
            "aoe_ready": bool(a_summary.get("valid")) and bool(b_summary.get("valid")) and bool(c_summary.get("valid")),
            "distance_ready": bool(a_summary.get("valid")) and bool(b_summary.get("valid")),
            "area_ready": bool(polygon_summary.get("valid")),
            "volume_ready": bool(volume_summary.get("valid")),
            "entry_point_count": self.bridge.markups_count(entry_node),
            "target_point_count": self.bridge.markups_count(target_node),
            "pivot_point_count": self.bridge.markups_count(pivot_node),
            "warnings": warnings,
        }
        if not vom_entry.get("valid"):
            payload["warnings"].extend(vom_entry.get("warnings") or [])
        if not vom_target.get("valid"):
            payload["warnings"].extend(vom_target.get("warnings") or [])
        if not a_summary.get("valid"):
            payload["warnings"].extend(a_summary.get("warnings") or [])
        if not b_summary.get("valid"):
            payload["warnings"].extend(b_summary.get("warnings") or [])
        if not c_summary.get("valid"):
            payload["warnings"].extend(c_summary.get("warnings") or [])
        if not polygon_summary.get("valid"):
            payload["warnings"].extend(polygon_summary.get("warnings") or [])
        if not volume_summary.get("valid"):
            payload["warnings"].extend(volume_summary.get("warnings") or [])
        return payload

    def _add_triangles_model(self, name: str, triangles: Any, color: tuple[float, float, float], opacity: float, *, reference_node=None) -> Optional[Any]:
        if triangles is None:
            return None
        existing = self._get_node_by_name(name)
        if existing is not None:
            slicer.mrmlScene.RemoveNode(existing)
        try:
            poly_data = self.bridge.triangles_to_polydata(triangles)
        except Exception:
            return None
        if poly_data is None or poly_data.GetNumberOfPoints() == 0:
            return None
        model_node = slicer.modules.models.logic().AddModel(poly_data)
        model_node.SetName(name)
        display_node = model_node.GetDisplayNode()
        if display_node is not None:
            display_node.SetColor(*color)
            display_node.SetOpacity(opacity)
            display_node.SetVisibility2D(True)
            display_node.SetVisibility3D(True)
        self._register_result_node(model_node, reference_node)
        return model_node

    def _add_polyline_model(
        self,
        name: str,
        points: Any,
        *,
        color: tuple[float, float, float],
        line_width: float = 3.0,
        closed: bool = False,
        opacity: float = 1.0,
        dotted: bool = False,
        reference_node=None,
    ) -> Optional[Any]:
        if points is None:
            return None
        existing = self._get_node_by_name(name)
        if existing is not None:
            slicer.mrmlScene.RemoveNode(existing)
        try:
            poly_data = self.bridge.polyline_to_polydata(points, closed=closed)
        except Exception:
            return None
        if poly_data is None or poly_data.GetNumberOfPoints() == 0:
            return None
        model_node = slicer.modules.models.logic().AddModel(poly_data)
        model_node.SetName(name)
        display_node = model_node.GetDisplayNode()
        if display_node is not None:
            display_node.SetColor(*color)
            display_node.SetOpacity(opacity)
            display_node.SetLineWidth(line_width)
            display_node.SetVisibility2D(True)
            display_node.SetVisibility3D(True)
            if dotted:
                try:
                    if hasattr(display_node, "SetLineStipplePattern"):
                        display_node.SetLineStipplePattern(0x0F0F)
                    if hasattr(display_node, "SetLineStippleRepeatPattern"):
                        display_node.SetLineStippleRepeatPattern(1)
                except Exception:
                    pass
        self._register_result_node(model_node, reference_node)
        return model_node

    def _add_fiducials_node(self, name: str, labels_and_points: list[tuple[str, Any]], *, color: tuple[float, float, float], reference_node=None) -> Optional[Any]:
        if not labels_and_points:
            return None
        existing = self._get_node_by_name(name)
        if existing is not None:
            slicer.mrmlScene.RemoveNode(existing)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        display_node = node.GetDisplayNode()
        if display_node is not None:
            display_node.SetSelectedColor(*color)
            display_node.SetColor(*color)
            display_node.SetTextScale(1.2)
            display_node.SetGlyphScale(1.8)
        for label, point in labels_and_points:
            idx = node.AddControlPointWorld(vtk.vtkVector3d(float(point[0]), float(point[1]), float(point[2])))
            node.SetNthControlPointLabel(idx, label)
        self._register_result_node(node, reference_node)
        return node

    def _add_arrow_model(
        self,
        name: str,
        start: Any,
        direction: Any,
        *,
        scale: float,
        color: tuple[float, float, float],
        opacity: float = 1.0,
        reference_node=None,
    ) -> Optional[Any]:
        if start is None or direction is None:
            return None
        start_arr = np.asarray(start, dtype=float).reshape(3,)
        dir_arr = np.asarray(direction, dtype=float).reshape(3,)
        norm = float(np.linalg.norm(dir_arr))
        if norm <= 1e-12 or scale <= 0:
            return None
        existing = self._get_node_by_name(name)
        if existing is not None:
            slicer.mrmlScene.RemoveNode(existing)

        x_axis = dir_arr / norm
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(x_axis, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        y_axis = np.cross(helper, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(24)
        arrow.SetShaftResolution(24)
        arrow.SetTipLength(0.22)
        arrow.SetTipRadius(0.08)
        arrow.SetShaftRadius(0.028)

        matrix = vtk.vtkMatrix4x4()
        for row in range(3):
            matrix.SetElement(row, 0, float(x_axis[row] * scale))
            matrix.SetElement(row, 1, float(y_axis[row] * scale))
            matrix.SetElement(row, 2, float(z_axis[row] * scale))
            matrix.SetElement(row, 3, float(start_arr[row]))
        matrix.SetElement(3, 0, 0.0)
        matrix.SetElement(3, 1, 0.0)
        matrix.SetElement(3, 2, 0.0)
        matrix.SetElement(3, 3, 1.0)

        transform = vtk.vtkTransform()
        transform.SetMatrix(matrix)
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(arrow.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        poly_data = transform_filter.GetOutput()
        if poly_data is None or poly_data.GetNumberOfPoints() == 0:
            return None
        model_node = slicer.modules.models.logic().AddModel(poly_data)
        model_node.SetName(name)
        display_node = model_node.GetDisplayNode()
        if display_node is not None:
            display_node.SetColor(*color)
            display_node.SetOpacity(opacity)
            display_node.SetVisibility2D(True)
            display_node.SetVisibility3D(True)
        self._register_result_node(model_node, reference_node)
        return model_node

    def _update_results_table(self, name: str, rows: list[tuple[str, Any]], *, reference_node=None) -> Optional[Any]:
        existing = self._get_node_by_name(name)
        if existing is not None:
            slicer.mrmlScene.RemoveNode(existing)
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        table = table_node.GetTable()
        metric_col = vtk.vtkStringArray()
        metric_col.SetName("Metric")
        value_col = vtk.vtkStringArray()
        value_col.SetName("Value")
        table.AddColumn(metric_col)
        table.AddColumn(value_col)
        for metric, value in rows:
            row = table.InsertNextBlankRow()
            table.SetValue(row, 0, str(metric))
            table.SetValue(row, 1, str(value))
        self._register_result_node(table_node, reference_node)
        return table_node

    def _get_or_create_dataset_table(self, name: str, columns: list[str]):
        node = self._get_node_by_name(name)
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", name)
        table = node.GetTable()
        existing_names = self._table_headers(node)
        for column_name in columns:
            if column_name in existing_names:
                continue
            arr = vtk.vtkStringArray()
            arr.SetName(column_name)
            table.AddColumn(arr)
            existing_names.append(column_name)
        self._register_result_node(node, None)
        return node

    def get_dataset_table_node(self, dataset_kind: str):
        dataset_key = str(dataset_kind).strip().lower()
        if dataset_key == "results":
            return self._get_node_by_name("Surgiplot Case Results Dataset")
        if dataset_key == "points":
            return self._get_node_by_name("Surgiplot Points Reproducibility Dataset")
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")

    def _table_headers(self, node) -> list[str]:
        table = node.GetTable()
        headers: list[str] = []
        for i in range(table.GetNumberOfColumns()):
            name = ""
            try:
                col = table.GetColumn(i)
                if col is not None and hasattr(col, "GetName"):
                    name = str(col.GetName() or "")
            except Exception:
                name = ""
            if not name:
                try:
                    name = str(table.GetColumnName(i) or "")
                except Exception:
                    name = ""
            headers.append(name)
        return headers

    def _ensure_table_column(self, node, column_name: str) -> int:
        table = node.GetTable()
        headers = self._table_headers(node)
        for idx, existing in enumerate(headers):
            if str(existing) == str(column_name):
                try:
                    col = table.GetColumn(idx)
                    if col is not None and hasattr(col, "GetNumberOfValues") and hasattr(col, "SetValue"):
                        row_count = table.GetNumberOfRows()
                        current_count = int(col.GetNumberOfValues())
                        if current_count < row_count:
                            col.SetNumberOfValues(row_count)
                            for fill_idx in range(current_count, row_count):
                                col.SetValue(fill_idx, "")
                except Exception:
                    pass
                return idx
        arr = vtk.vtkStringArray()
        arr.SetName(str(column_name))
        row_count = table.GetNumberOfRows()
        if row_count > 0:
            arr.SetNumberOfValues(row_count)
            for fill_idx in range(row_count):
                arr.SetValue(fill_idx, "")
        table.AddColumn(arr)
        return table.GetNumberOfColumns() - 1

    def _find_subject_row(self, node, subject_id: str) -> Optional[int]:
        table = node.GetTable()
        try:
            col = self._ensure_table_column(node, "Subject ID")
        except Exception:
            return None
        wanted = str(subject_id).strip()
        for row in range(table.GetNumberOfRows()):
            if str(table.GetValue(row, col).ToString()).strip() == wanted:
                return row
        return None

    def _set_table_cell(self, node, row: int, column_name: str, value: Any) -> None:
        table = node.GetTable()
        col = self._ensure_table_column(node, str(column_name))
        while row >= table.GetNumberOfRows():
            table.InsertNextBlankRow()
        table.SetValue(row, col, str(value))

    def _normalize_target_key(self, anatomical_target: str) -> str:
        text = str(anatomical_target or "").strip().upper()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^A-Z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text or "TARGET"

    def _read_table_dataset(self, node) -> tuple[list[str], list[list[str]]]:
        table = node.GetTable()
        headers = self._table_headers(node)
        rows = [
            [str(table.GetValue(r, c).ToString()) for c in range(table.GetNumberOfColumns())]
            for r in range(table.GetNumberOfRows())
        ]
        return headers, rows

    def _write_table_dataset(self, node, headers: list[str], rows: list[list[str]]) -> None:
        table = node.GetTable()
        while table.GetNumberOfColumns() > 0:
            table.RemoveColumn(0)
        while table.GetNumberOfRows() > 0:
            table.RemoveRow(0)
        for header in headers:
            arr = vtk.vtkStringArray()
            arr.SetName(str(header))
            table.AddColumn(arr)
        for row_values in rows:
            row = table.InsertNextBlankRow()
            padded = list(row_values) + [""] * max(0, len(headers) - len(row_values))
            for col, value in enumerate(padded[: len(headers)]):
                table.SetValue(row, col, str(value))

    def _is_case_results_column_allowed(self, column_name: str) -> bool:
        name = str(column_name or "").strip()
        if not name:
            return False
        if name == "Subject ID":
            return True
        blocked_suffixes = (
            "_action",
            "_table_node",
            "_subject_id",
            "_anatomical_target",
            "_added_column_count",
            "_row_count",
            "_column_count",
            "_dataset_kind",
            "_file_kind",
            "_path",
        )
        return not any(name.endswith(suffix) for suffix in blocked_suffixes)

    def append_case_result_record(self, subject_id: str, anatomical_target: str, payload: dict[str, Any]) -> dict[str, Any]:
        metric_family = str(payload.get("metric_family", "")).strip()
        if not metric_family:
            raise ValueError("Current payload does not contain a metric family.")
        table_node = self._get_or_create_dataset_table("Surgiplot Case Results Dataset", ["Subject ID"])
        headers, existing_rows = self._read_table_dataset(table_node)
        filtered_indices = [idx for idx, header in enumerate(headers) if self._is_case_results_column_allowed(header)]
        headers = [headers[idx] for idx in filtered_indices] if filtered_indices else ["Subject ID"]
        cleaned_rows: list[list[str]] = []
        for row_values in existing_rows:
            if filtered_indices:
                cleaned = [row_values[idx] if idx < len(row_values) else "" for idx in filtered_indices]
            else:
                cleaned = row_values[:1] if row_values else [""]
            cleaned_rows.append(cleaned)
        if "Subject ID" not in headers:
            headers = ["Subject ID"] + headers
            cleaned_rows = [[row[0] if row else ""] + row for row in cleaned_rows]

        row_by_subject: dict[str, dict[str, str]] = {}
        ordered_subjects: list[str] = []
        for row_values in cleaned_rows:
            padded = list(row_values) + [""] * max(0, len(headers) - len(row_values))
            row_map = {headers[idx]: padded[idx] for idx in range(len(headers))}
            subject_key = str(row_map.get("Subject ID", "")).strip()
            if not subject_key:
                continue
            row_by_subject[subject_key] = row_map
            ordered_subjects.append(subject_key)
        if str(subject_id).strip() not in row_by_subject:
            row_by_subject[str(subject_id).strip()] = {"Subject ID": str(subject_id).strip()}
            ordered_subjects.append(str(subject_id).strip())

        skip_keys = {
            "metric_family",
            "reference_anatomy",
            "warnings",
            "entry_role_order",
            "ordered_entry_labels",
            "action",
            "table_node",
            "subject_id",
            "anatomical_target",
            "added_column_count",
            "row_count",
            "column_count",
            "dataset_kind",
            "file_kind",
            "path",
        }
        added_columns: list[str] = []
        target_key = self._normalize_target_key(anatomical_target)
        subject_row = row_by_subject[str(subject_id).strip()]
        for key, value in payload.items():
            if key in skip_keys:
                continue
            if isinstance(value, (list, dict)):
                value = json.dumps(value, sort_keys=True)
            column_name = f"{target_key}_{key}"
            if column_name not in headers:
                headers.append(column_name)
                added_columns.append(column_name)
            subject_row[column_name] = str(value)
        rebuilt_rows = []
        for subject_key in ordered_subjects:
            row_map = row_by_subject[subject_key]
            rebuilt_rows.append([row_map.get(header, "") for header in headers])
        self._write_table_dataset(table_node, headers, rebuilt_rows)
        self._register_result_node(table_node, None)
        return {
            "action": "append_case_result_record",
            "table_node": table_node.GetName(),
            "subject_id": subject_id,
            "anatomical_target": anatomical_target,
            "appended_metric_family": metric_family,
            "added_column_count": len(added_columns),
        }

    def append_points_record(self, subject_id: str, anatomical_target: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        table_node = self._get_or_create_dataset_table(
            "Surgiplot Points Reproducibility Dataset",
            ["Subject ID", "Anatomical Target", "Metric Family", "Point Group", "Point Label", "X", "Y", "Z"],
        )
        table = table_node.GetTable()
        inserted = 0
        for row_payload in rows:
            row = table.InsertNextBlankRow()
            self._set_table_cell(table_node, row, "Subject ID", subject_id)
            self._set_table_cell(table_node, row, "Anatomical Target", anatomical_target)
            self._set_table_cell(table_node, row, "Metric Family", row_payload.get("metric_family", ""))
            self._set_table_cell(table_node, row, "Point Group", row_payload.get("point_group", ""))
            self._set_table_cell(table_node, row, "Point Label", row_payload.get("point_label", ""))
            self._set_table_cell(table_node, row, "X", f"{float(row_payload.get('x', 0.0)):.6f}")
            self._set_table_cell(table_node, row, "Y", f"{float(row_payload.get('y', 0.0)):.6f}")
            self._set_table_cell(table_node, row, "Z", f"{float(row_payload.get('z', 0.0)):.6f}")
            inserted += 1
        return {
            "action": "append_points_record",
            "table_node": table_node.GetName(),
            "subject_id": subject_id,
            "anatomical_target": anatomical_target,
            "row_count_added": inserted,
        }

    def replace_dataset_table_contents(self, dataset_kind: str, headers: list[str], rows: list[list[str]]) -> dict[str, Any]:
        dataset_key = str(dataset_kind).strip().lower()
        if dataset_key == "results":
            node_name = "Surgiplot Case Results Dataset"
        elif dataset_key == "points":
            node_name = "Surgiplot Points Reproducibility Dataset"
        else:
            raise ValueError(f"Unknown dataset kind: {dataset_kind}")
        node = self._get_or_create_dataset_table(node_name, [])
        table = node.GetTable()
        while table.GetNumberOfColumns() > 0:
            table.RemoveColumn(0)
        while table.GetNumberOfRows() > 0:
            table.RemoveRow(0)
        for header in headers:
            arr = vtk.vtkStringArray()
            arr.SetName(str(header))
            table.AddColumn(arr)
        for row_values in rows:
            row = table.InsertNextBlankRow()
            for col, value in enumerate(row_values[: table.GetNumberOfColumns()]):
                table.SetValue(row, col, str(value))
        self._register_result_node(node, None)
        return {
            "action": "replace_dataset_table_contents",
            "dataset_kind": dataset_key,
            "table_node": node.GetName(),
            "row_count": len(rows),
            "column_count": len(headers),
        }

    def _xlsx_cell_ref(self, row_index: int, col_index: int) -> str:
        col_num = col_index + 1
        letters = []
        while col_num > 0:
            col_num, remainder = divmod(col_num - 1, 26)
            letters.append(chr(65 + remainder))
        return f"{''.join(reversed(letters))}{row_index + 1}"

    def _write_simple_xlsx(self, path: Path, headers: list[str], rows: list[list[str]]) -> None:
        all_rows = [headers] + rows
        sheet_rows: list[str] = []
        for row_idx, row_values in enumerate(all_rows):
            cell_xml: list[str] = []
            for col_idx, value in enumerate(row_values):
                text = "" if value is None else str(value)
                ref = self._xlsx_cell_ref(row_idx, col_idx)
                cell_xml.append(
                    f'<c r="{ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
                )
            sheet_rows.append(f'<row r="{row_idx + 1}">{"".join(cell_xml)}</row>')
        worksheet_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(sheet_rows)}</sheetData>'
            '</worksheet>'
        )
        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Surgiplot" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>'
        )
        workbook_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            '</Relationships>'
        )
        root_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '</Relationships>'
        )
        content_types_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>'
        )
        styles_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>'
        )
        with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types_xml)
            archive.writestr("_rels/.rels", root_rels_xml)
            archive.writestr("xl/workbook.xml", workbook_xml)
            archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
            archive.writestr("xl/worksheets/sheet1.xml", worksheet_xml)
            archive.writestr("xl/styles.xml", styles_xml)

    def export_dataset_file(self, dataset_kind: str, path: Path, *, file_kind: str) -> dict[str, Any]:
        dataset_key = str(dataset_kind).strip().lower()
        node = self.get_dataset_table_node(dataset_key)
        if node is None:
            raise ValueError("No dataset table is available yet.")
        headers = self._table_headers(node)
        table = node.GetTable()
        rows = [
            [str(table.GetValue(r, c).ToString()) for c in range(table.GetNumberOfColumns())]
            for r in range(table.GetNumberOfRows())
        ]
        path = Path(path)
        if file_kind == "csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                writer.writerows(rows)
        elif file_kind == "xlsx":
            self._write_simple_xlsx(path, headers, rows)
        else:
            raise ValueError(f"Unsupported export format: {file_kind}")
        return {
            "action": "export_dataset_file",
            "dataset_kind": dataset_key,
            "file_kind": file_kind,
            "path": str(path),
            "row_count": len(rows),
            "column_count": len(headers),
        }

    def compute_vom_family(self, entry_node, target_node, *, reference_node=None, stand_dist: float) -> dict[str, Any]:
        result = self.bridge.compute_vom_family(entry_node, target_node, stand_dist=stand_dist, return_debug=True)
        debug = result.debug or {}
        entry_polygon_area_mm2 = float(debug.get("entry_polygon_area_mm2", 0.0) or 0.0)
        target_polygon_area_mm2 = float(debug.get("target_polygon_area_mm2", 0.0) or 0.0)
        entry_ellipse_area_mm2 = float(debug.get("entry_ellipse_area_mm2", 0.0) or 0.0)
        target_ellipse_area_mm2 = float(debug.get("target_ellipse_area_mm2", 0.0) or 0.0)
        cut_ellipse_a_mm = float(debug.get("cut_ellipse_a_mm", 0.0) or 0.0)
        cut_ellipse_b_mm = float(debug.get("cut_ellipse_b_mm", 0.0) or 0.0)
        svom_cut_ellipse_area_mm2 = float(np.pi * cut_ellipse_a_mm * cut_ellipse_b_mm) if cut_ellipse_a_mm > 0 and cut_ellipse_b_mm > 0 else 0.0
        target_distance_mm = float(debug.get("distance_h", 0.0) or 0.0)
        ordered_entry_polygon = self._ordered_polygon_points(debug.get("entry_polygon"))
        ordered_target_polygon = self._ordered_polygon_points(debug.get("target_polygon"))
        self._add_polyline_model("Surgiplot Entry Polygon", ordered_entry_polygon, color=(0.2, 0.45, 0.95), line_width=2.0, closed=True, opacity=0.9, reference_node=reference_node)
        self._add_polyline_model("Surgiplot Target Polygon", ordered_target_polygon, color=(0.9, 0.25, 0.2), line_width=2.0, closed=True, opacity=0.9, reference_node=reference_node)
        self._add_polyline_model("Surgiplot Entry Ellipse", debug.get("entry_ellipse_3d"), color=(0.2, 0.45, 0.95), line_width=3.0, closed=True, opacity=0.9, reference_node=reference_node)
        self._add_polyline_model("Surgiplot Target Ellipse", debug.get("target_ellipse_3d"), color=(0.9, 0.25, 0.2), line_width=3.0, closed=True, opacity=0.9, reference_node=reference_node)
        self._add_polyline_model("Surgiplot sVOM Cut Ellipse", debug.get("svom_cut_ellipse_3d"), color=(0.2, 0.8, 0.35), line_width=3.0, closed=True, opacity=0.95, reference_node=reference_node)
        c_entry = debug.get("centroid_entry")
        c_target = debug.get("centroid_target")
        if c_entry is not None and c_target is not None:
            self._add_polyline_model("Surgiplot Corridor Axis", [c_entry, c_target], color=(0.1, 0.1, 0.1), line_width=4.0, opacity=0.95, reference_node=reference_node)
            self._add_fiducials_node(
                "Surgiplot Corridor Centroids",
                [("Entry centroid", c_entry), ("Target centroid", c_target)],
                color=(0.95, 0.9, 0.2),
                reference_node=reference_node,
            )
        entry_ellipse = np.asarray(debug.get("entry_ellipse_3d"), dtype=float) if debug.get("entry_ellipse_3d") is not None else None
        target_ellipse = np.asarray(debug.get("target_ellipse_3d"), dtype=float) if debug.get("target_ellipse_3d") is not None else None
        if entry_ellipse is not None and target_ellipse is not None and entry_ellipse.ndim == 2 and target_ellipse.ndim == 2 and len(entry_ellipse) == len(target_ellipse) and len(entry_ellipse) >= 4:
            guide_indices = [0, len(entry_ellipse) // 4, len(entry_ellipse) // 2, (3 * len(entry_ellipse)) // 4]
            for idx, guide_index in enumerate(guide_indices, start=1):
                self._add_polyline_model(
                    f"Surgiplot Corridor Side {idx}",
                    [target_ellipse[guide_index], entry_ellipse[guide_index]],
                    color=(0.55, 0.8, 0.98),
                    line_width=2.0,
                    opacity=0.92,
                    reference_node=reference_node,
                )
        self._add_triangles_model("Surgiplot VOM Surface", debug.get("frustum_surface"), (0.55, 0.8, 0.98), 0.16, reference_node=reference_node)
        self._add_triangles_model("Surgiplot sVOM Surface", debug.get("svom_surface"), (0.25, 0.78, 0.35), 0.34, reference_node=reference_node)
        if c_entry is not None and c_target is not None:
            approach = np.asarray(c_target, dtype=float).reshape(3,) - np.asarray(c_entry, dtype=float).reshape(3,)
            norm = float(np.linalg.norm(approach))
            if norm > 1e-9:
                scale = max(float(debug.get("distance_h", 0.0) or 0.0) * 0.25, 1.0)
                direction = approach / norm
                start = np.asarray(c_target, dtype=float).reshape(3,) - direction * scale
                self._add_arrow_model(
                    "Surgiplot Target Approach",
                    start,
                    direction,
                    scale=scale,
                    color=(0.88, 0.72, 0.18),
                    opacity=0.95,
                    reference_node=reference_node,
                )
        self._update_results_table(
            "Surgiplot VOM Results",
            [
                ("Metric family", "VOM / sVOM / VoA"),
                ("Reference anatomy", self._reference_name(reference_node)),
                ("VoA_deg", f"{result.voa_deg:.6f}"),
                ("VOM_mm3", f"{result.vom_mm3:.6f}"),
                ("sVOM_mm3", f"{result.svom_mm3:.6f}"),
                ("Target_distance_mm", f"{target_distance_mm:.6f}"),
                ("Entry_polygon_area_mm2", f"{entry_polygon_area_mm2:.6f}"),
                ("Target_polygon_area_mm2", f"{target_polygon_area_mm2:.6f}"),
                ("Entry_ellipse_area_mm2", f"{entry_ellipse_area_mm2:.6f}"),
                ("Target_ellipse_area_mm2", f"{target_ellipse_area_mm2:.6f}"),
                ("sVOM_cut_ellipse_area_mm2", f"{svom_cut_ellipse_area_mm2:.6f}"),
                ("Entry points", self.bridge.markups_count(entry_node)),
                ("Target points", self.bridge.markups_count(target_node)),
                ("Stand distance mm", f"{stand_dist:.6f}"),
            ],
            reference_node=reference_node,
        )
        payload = {
            "metric_family": "VOM / sVOM / VoA",
            "reference_anatomy": self._reference_name(reference_node),
            "VoA_deg": result.voa_deg,
            "VOM_mm3": result.vom_mm3,
            "sVOM_mm3": result.svom_mm3,
            "target_distance_mm": target_distance_mm,
            "entry_polygon_area_mm2": entry_polygon_area_mm2,
            "target_polygon_area_mm2": target_polygon_area_mm2,
            "entry_ellipse_area_mm2": entry_ellipse_area_mm2,
            "target_ellipse_area_mm2": target_ellipse_area_mm2,
            "svom_cut_ellipse_area_mm2": svom_cut_ellipse_area_mm2,
            "stand_distance_mm": float(stand_dist),
            "entry_point_count": self.bridge.markups_count(entry_node),
            "target_point_count": self.bridge.markups_count(target_node),
        }
        self._update_results_report(payload, reference_node=reference_node)
        return payload

    def describe_aoa_entry(self, entry_node) -> dict[str, Any]:
        return self.bridge.describe_aoa_entry(entry_node)

    def assign_aoa_roles_by_order(self, entry_node) -> dict[str, Any]:
        if entry_node is None:
            raise ValueError("Select an entry markups node first.")
        count = self.bridge.markups_count(entry_node)
        if count != 4:
            raise ValueError(f"AoA / SF role assignment expects exactly 4 control points, got {count}.")
        assigned = []
        for index, role in enumerate(self.bridge.AOA_ENTRY_ROLES):
            entry_node.SetNthControlPointLabel(index, role)
            assigned.append(role)
        status = self.describe_aoa_entry(entry_node)
        return {
            "action": "assign_aoa_roles_by_order",
            "assigned_roles": assigned,
            "status": status,
        }

    def set_background_volume(self, volume_node) -> dict[str, Any]:
        if volume_node is None:
            raise ValueError("Select a scalar volume node first.")
        slicer.util.setSliceViewerLayers(background=volume_node)
        self._fit_all_slice_views_to_background()
        return {
            "action": "set_background_volume",
            "background_volume": volume_node.GetName(),
        }

    def _get_or_create_markups_node(self, name: str, role: str, *, reuse_existing: bool):
        if reuse_existing:
            existing = self._get_node_by_name(name)
            if existing is not None:
                existing.SetAttribute("SurgiplotRole", role)
                self._configure_markups_display(existing, color=self._role_color(role))
                if role == "scale":
                    self._configure_scale_markups_display(existing)
                elif role != "landmarks":
                    self._configure_metric_input_display(existing, visible=False)
                return existing
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.SetAttribute("SurgiplotRole", role)
        self._configure_markups_display(node, color=self._role_color(role))
        if role == "scale":
            self._configure_scale_markups_display(node)
        elif role != "landmarks":
            self._configure_metric_input_display(node, visible=False)
        return node

    def _ordered_polygon_points(self, points_like: Any) -> Optional[np.ndarray]:
        if points_like is None:
            return None
        pts = np.asarray(points_like, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
            return pts
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2].T
        coords2 = centered @ basis
        angles = np.arctan2(coords2[:, 1], coords2[:, 0])
        order = np.argsort(angles)
        return pts[order]

    def _aoe_arc_points(self, A: Any, B: Any, C: Any, *, n_samples: int = 48) -> Optional[np.ndarray]:
        A_pt = np.asarray(A, dtype=float).reshape(3,)
        B_pt = np.asarray(B, dtype=float).reshape(3,)
        C_pt = np.asarray(C, dtype=float).reshape(3,)
        u = A_pt - B_pt
        v = C_pt - B_pt
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu <= 1e-12 or nv <= 1e-12:
            return None
        u_hat = u / nu
        v_hat = v / nv
        normal = np.cross(u_hat, v_hat)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            return None
        normal = normal / normal_norm
        tangent = np.cross(normal, u_hat)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-12:
            return None
        tangent = tangent / tangent_norm
        theta = float(np.arctan2(np.dot(v_hat, tangent), np.dot(v_hat, u_hat)))
        arc_angles = np.linspace(0.0, theta, int(max(3, n_samples)))
        radius = 0.28 * min(nu, nv)
        arc = np.vstack([
            B_pt + radius * (np.cos(angle) * u_hat + np.sin(angle) * tangent)
            for angle in arc_angles
        ])
        return arc

    def _role_color(self, role: str) -> tuple[float, float, float]:
        mapping = {
            "entry": (0.2, 0.45, 0.95),
            "target": (0.9, 0.25, 0.2),
            "pivot": (0.95, 0.8, 0.2),
            "landmarks": (0.85, 0.55, 0.15),
            "scale": (0.45, 0.2, 0.85),
        }
        return mapping.get(role, (0.8, 0.8, 0.8))

    def activate_place_mode(self, node, *, place_label: str = "point", persistent: bool = False) -> dict[str, Any]:
        if node is None:
            raise ValueError("Select or create a markups node first.")
        app_logic = slicer.app.applicationLogic()
        if app_logic is None:
            raise RuntimeError("Slicer application logic is not available.")
        selection_node = app_logic.GetSelectionNode()
        interaction_node = app_logic.GetInteractionNode()
        if selection_node is None or interaction_node is None:
            raise RuntimeError("Slicer interaction state is not available.")

        selection_node.SetReferenceActivePlaceNodeClassName("vtkMRMLMarkupsFiducialNode")
        selection_node.SetReferenceActivePlaceNodeID(node.GetID())
        if hasattr(interaction_node, "SetPlaceModePersistence"):
            interaction_node.SetPlaceModePersistence(1 if persistent else 0)
        if not persistent and hasattr(interaction_node, "SwitchToSinglePlaceMode"):
            interaction_node.SwitchToSinglePlaceMode()
        elif persistent and hasattr(interaction_node, "SwitchToPersistentPlaceMode"):
            interaction_node.SwitchToPersistentPlaceMode()
        elif hasattr(interaction_node, "SwitchToSinglePlaceMode"):
            interaction_node.SwitchToSinglePlaceMode()

        return {
            "action": "activate_place_mode",
            "node_name": node.GetName(),
            "node_id": node.GetID(),
            "place_label": place_label,
            "persistent": bool(persistent),
        }

    def deactivate_place_mode(self) -> dict[str, Any]:
        app_logic = slicer.app.applicationLogic()
        if app_logic is None:
            raise RuntimeError("Slicer application logic is not available.")
        selection_node = app_logic.GetSelectionNode()
        interaction_node = app_logic.GetInteractionNode()
        if interaction_node is None or selection_node is None:
            raise RuntimeError("Slicer interaction state is not available.")
        if hasattr(interaction_node, "SetPlaceModePersistence"):
            interaction_node.SetPlaceModePersistence(0)
        if hasattr(interaction_node, "SwitchToViewTransformMode"):
            interaction_node.SwitchToViewTransformMode()
        elif hasattr(interaction_node, "SetCurrentInteractionMode") and hasattr(interaction_node, "ViewTransform"):
            interaction_node.SetCurrentInteractionMode(interaction_node.ViewTransform)
        if hasattr(selection_node, "SetReferenceActivePlaceNodeID"):
            selection_node.SetReferenceActivePlaceNodeID(None)
        return {"action": "deactivate_place_mode"}

    def create_entry_markups_node(self, *, reuse_existing: bool = False):
        node = self._get_or_create_markups_node("Surgiplot Entry", "entry", reuse_existing=reuse_existing)
        return node, {
            "action": "create_entry_markups_node",
            "node_name": node.GetName(),
            "expected_usage": "Add 4 entry control points for AoA/SF or 3+ contour points for VOM.",
        }

    def create_landmark_dataset_node(self, *, reuse_existing: bool = False):
        node = self._get_or_create_markups_node("Surgiplot Landmarks", "landmarks", reuse_existing=reuse_existing)
        return node, {
            "action": "create_landmark_dataset_node",
            "node_name": node.GetName(),
            "expected_usage": "Collect reusable labeled landmarks here before assigning them to a metric-specific role.",
        }

    def create_scale_markups_node(self, *, reuse_existing: bool = False):
        node = self._get_or_create_markups_node("Surgiplot Scale", "scale", reuse_existing=reuse_existing)
        return node, {
            "action": "create_scale_markups_node",
            "node_name": node.GetName(),
            "expected_usage": "Pick 2 scale points on a loaded 3D model to calibrate the scene to mm.",
        }

    def create_single_markups_node(self, name: str, role: str, *, reuse_existing: bool = False):
        node = self._get_or_create_markups_node(name, role, reuse_existing=reuse_existing)
        return node, {
            "action": "create_single_markups_node",
            "node_name": node.GetName(),
            "role": role,
            "expected_usage": "Single-point metric input.",
        }

    def create_multi_markups_node(self, name: str, role: str, *, reuse_existing: bool = False):
        node = self._get_or_create_markups_node(name, role, reuse_existing=reuse_existing)
        return node, {
            "action": "create_multi_markups_node",
            "node_name": node.GetName(),
            "role": role,
            "expected_usage": "Multi-point metric input.",
        }

    def create_target_markups_node(self, *, reuse_existing: bool = False):
        node = self._get_or_create_markups_node("Surgiplot Target", "target", reuse_existing=reuse_existing)
        return node, {
            "action": "create_target_markups_node",
            "node_name": node.GetName(),
            "expected_usage": "Add 3+ target contour points for VOM / sVOM / VoA.",
        }

    def create_pivot_markups_node(self, *, reuse_existing: bool = False):
        node = self._get_or_create_markups_node("Surgiplot Pivot", "pivot", reuse_existing=reuse_existing)
        return node, {
            "action": "create_pivot_markups_node",
            "node_name": node.GetName(),
            "expected_usage": "Add exactly 1 pivot point for AoA/SF.",
        }

    def append_metric_point(self, role: str, label: str, point: np.ndarray):
        role = str(role or "").strip().lower()
        label = str(label or "").strip() or role
        point = np.asarray(point, dtype=float).reshape(3,)
        label_key = label.strip().lower()

        if role == "entry":
            node, _ = self.create_entry_markups_node(reuse_existing=True)
        elif role == "target":
            node, _ = self.create_target_markups_node(reuse_existing=True)
        elif role == "pivot":
            node, _ = self.create_pivot_markups_node(reuse_existing=True)
            count = self.bridge.markups_count(node)
            for row in reversed(range(count)):
                node.RemoveNthControlPoint(row)
        elif role == "a":
            node, _ = self.create_single_markups_node("Surgiplot A", "a", reuse_existing=True)
        elif role == "b":
            node, _ = self.create_single_markups_node("Surgiplot B", "b", reuse_existing=True)
        elif role == "c":
            node, _ = self.create_single_markups_node("Surgiplot C", "c", reuse_existing=True)
        elif role == "polygon":
            node, _ = self.create_multi_markups_node("Surgiplot Polygon", "polygon", reuse_existing=True)
        elif role == "volume":
            node, _ = self.create_multi_markups_node("Surgiplot Volume", "volume", reuse_existing=True)
        else:
            return None

        if role in {"a", "b", "c"}:
            count = self.bridge.markups_count(node)
            for row in reversed(range(count)):
                node.RemoveNthControlPoint(row)
        elif role == "entry":
            existing_count = self.bridge.markups_count(node)
            if existing_count > 0:
                existing_labels = [str(lb or "").strip().lower() for lb in self.bridge.markups_labels(node)]
                if label_key in {"cranial", "caudal", "medial", "lateral"} and label_key in existing_labels:
                    idx_existing = existing_labels.index(label_key)
                    node.SetNthControlPointPositionWorld(
                        idx_existing,
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                    )
                    node.SetNthControlPointLabel(idx_existing, label)
                    return node

        idx = node.AddControlPointWorld(vtk.vtkVector3d(float(point[0]), float(point[1]), float(point[2])))
        node.SetNthControlPointLabel(idx, label)
        if role == "entry":
            try:
                labels = self.bridge.markups_labels(node)
                points = self.bridge.markups_to_points(node, minimum=1)
                preferred = ["cranial", "caudal", "medial", "lateral"]
                paired = list(zip(labels, points))
                ordered: list[tuple[str, np.ndarray]] = []
                used: set[int] = set()
                for pref in preferred:
                    for pos, (lb, pt) in enumerate(paired):
                        if pos in used:
                            continue
                        if str(lb or "").strip().lower() == pref:
                            ordered.append((str(lb), np.asarray(pt, dtype=float)))
                            used.add(pos)
                            break
                for pos, (lb, pt) in enumerate(paired):
                    if pos not in used:
                        ordered.append((str(lb), np.asarray(pt, dtype=float)))
                count = self.bridge.markups_count(node)
                for row in reversed(range(count)):
                    node.RemoveNthControlPoint(row)
                for lb, pt in ordered:
                    new_idx = node.AddControlPointWorld(vtk.vtkVector3d(float(pt[0]), float(pt[1]), float(pt[2])))
                    node.SetNthControlPointLabel(new_idx, lb)
            except Exception:
                pass
        return node

    def _scale_markups_node(self, node, factor: float) -> int:
        if node is None:
            return 0
        count = self.bridge.markups_count(node)
        for index in range(count):
            point = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(index, point)
            scaled = np.asarray(point, dtype=float) * float(factor)
            node.SetNthControlPointPositionWorld(index, float(scaled[0]), float(scaled[1]), float(scaled[2]))
        return count

    def normalize_scale_node(self, node) -> dict[str, Any]:
        if node is None:
            raise ValueError("Scale node is not available.")
        defined_indices = []
        defined_status = getattr(node, "PositionDefined", None)
        total = self.bridge.markups_count(node)
        for index in range(total):
            status = node.GetNthControlPointPositionStatus(index)
            if defined_status is None or status == defined_status:
                defined_indices.append(index)
        if len(defined_indices) < 2:
            raise ValueError("Scale node requires at least 2 defined points.")
        keep = defined_indices[:2]
        for remove_index in reversed(range(total)):
            if remove_index not in keep:
                node.RemoveNthControlPoint(remove_index)
        for idx in range(min(2, self.bridge.markups_count(node))):
            node.SetNthControlPointLabel(idx, f"scale_{idx + 1}")
        self._configure_scale_markups_display(node)
        return {
            "action": "normalize_scale_node",
            "kept_point_count": min(2, self.bridge.markups_count(node)),
        }

    def _scale_model_node(self, node, factor: float) -> int:
        if node is None or not node.IsA("vtkMRMLModelNode"):
            return 0
        poly_data = node.GetPolyData()
        if poly_data is None or poly_data.GetPoints() is None:
            return 0
        vtk_points = poly_data.GetPoints()
        for index in range(vtk_points.GetNumberOfPoints()):
            x, y, z = vtk_points.GetPoint(index)
            vtk_points.SetPoint(index, float(x) * factor, float(y) * factor, float(z) * factor)
        vtk_points.Modified()
        poly_data.Modified()
        node.Modified()
        return int(vtk_points.GetNumberOfPoints())

    def apply_uniform_scene_scale(self, factor: float, *, reference_node=None, related_nodes: Optional[list[Any]] = None) -> dict[str, Any]:
        if factor <= 0:
            raise ValueError("Scale factor must be positive.")
        scaled_model_points = 0
        scaled_markups_points = 0
        if reference_node is not None and hasattr(reference_node, "IsA") and reference_node.IsA("vtkMRMLModelNode"):
            scaled_model_points = self._scale_model_node(reference_node, factor)
        else:
            raise ValueError("Select a model node as Reference anatomy before applying rescaling.")
        if related_nodes:
            seen: set[str] = set()
            for node in related_nodes:
                if node is None or not hasattr(node, "GetID") or not hasattr(node, "IsA"):
                    continue
                node_id = node.GetID()
                if node_id in seen:
                    continue
                seen.add(node_id)
                if node.IsA("vtkMRMLMarkupsFiducialNode"):
                    scaled_markups_points += self._scale_markups_node(node, factor)
        return {
            "action": "apply_uniform_scene_scale",
            "reference_anatomy": self._reference_name(reference_node),
            "scale_factor_to_mm": float(factor),
            "scaled_model_points": int(scaled_model_points),
            "scaled_markups_points": int(scaled_markups_points),
        }

    def populate_named_markups_node(self, name: str, role: str, labels: list[str], points: np.ndarray):
        node = self._get_or_create_markups_node(name, role, reuse_existing=True)
        count = self.bridge.markups_count(node)
        for row in reversed(range(count)):
            node.RemoveNthControlPoint(row)
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(labels) != pts.shape[0]:
            raise ValueError("labels and points must describe an (N, 3) set with matching lengths.")
        for label, point in zip(labels, pts):
            idx = node.AddControlPointWorld(vtk.vtkVector3d(float(point[0]), float(point[1]), float(point[2])))
            node.SetNthControlPointLabel(idx, str(label))
        if role != "landmarks":
            self._configure_metric_input_display(node, visible=False)
        return node, {
            "action": "populate_named_markups_node",
            "node_name": node.GetName(),
            "role": role,
            "labels": [str(label) for label in labels],
            "point_count": int(pts.shape[0]),
        }

    def jump_slices_to_markups(self, node, *, mode: str = "centroid") -> dict[str, Any]:
        if node is None:
            raise ValueError("Select a markups node first.")
        points = self.bridge.markups_to_points(node, minimum=1)
        if mode == "first":
            point = points[0]
            jump_mode = "first"
        else:
            point = np.mean(points, axis=0)
            jump_mode = "centroid"
        self._jump_all_slices(point)
        return {
            "action": "jump_slices_to_markups",
            "node_name": node.GetName(),
            "jump_mode": jump_mode,
            "point_xyz": [float(point[0]), float(point[1]), float(point[2])],
            "point_count": int(points.shape[0]),
        }

    def compute_aoa_family(
        self,
        entry_node,
        pivot_node,
        *,
        reference_node=None,
        sf_rescale_radius_mm: Optional[float],
        standardized_requested: bool = False,
    ) -> dict[str, Any]:
        result = self.bridge.compute_aoa_family(
            entry_node,
            pivot_node,
            sf_rescale_radius_mm=sf_rescale_radius_mm,
            return_debug=True,
        )
        debug = result.debug or {}
        ordered_labels = list(debug.get("ordered_entry_labels") or [])
        role_order = list(debug.get("entry_role_order") or ["cranial", "caudal", "medial", "lateral"])
        warnings = list(debug.get("warnings") or [])
        pivot = debug.get("pivot")
        entry_points = debug.get("entry_points")
        rescaled_entry_points = debug.get("rescaled_entry_points")
        vertical_triangle = [debug.get("vertical_triangle")] if debug.get("vertical_triangle") is not None else None
        horizontal_triangle = [debug.get("horizontal_triangle")] if debug.get("horizontal_triangle") is not None else None
        sf_surface = debug.get("sf_surface_3d")
        sf_surface_rescaled = debug.get("sf_surface_rescaled_3d")
        self._add_triangles_model("Surgiplot Vertical AoA", vertical_triangle, (0.2, 0.45, 0.95), 0.35, reference_node=reference_node)
        self._add_triangles_model("Surgiplot Horizontal AoA", horizontal_triangle, (0.2, 0.75, 0.3), 0.35, reference_node=reference_node)
        self._add_triangles_model("Surgiplot SF Surface", sf_surface, (0.55, 0.62, 0.72), 0.22, reference_node=reference_node)
        standardized_node = self._add_triangles_model(
            "Surgiplot Standardized SF Surface",
            sf_surface_rescaled,
            (0.95, 0.6, 0.15),
            0.18,
            reference_node=reference_node,
        )
        if standardized_node is not None:
            self._set_display_visibility(standardized_node, standardized_requested)
        if pivot is not None and rescaled_entry_points is not None:
            rescaled_arr = np.asarray(rescaled_entry_points, dtype=float)
            if rescaled_arr.ndim == 2 and rescaled_arr.shape[1] == 3:
                for idx, projected_point in enumerate(rescaled_arr[:4]):
                    role = role_order[idx] if idx < len(role_order) else f"role_{idx + 1}"
                    guide_node = self._add_polyline_model(
                        f"Surgiplot Standardized SF Guide {role}",
                        [pivot, projected_point],
                        color=(0.1, 0.1, 0.1),
                        line_width=2.0,
                        opacity=0.85,
                        dotted=True,
                        reference_node=reference_node,
                    )
                    if guide_node is not None:
                        self._set_display_visibility(guide_node, standardized_requested)
        if entry_points is not None:
            self._add_polyline_model(
                "Surgiplot Entry Quad",
                np.vstack([entry_points[0], entry_points[3], entry_points[1], entry_points[2]]),
                color=(0.12, 0.12, 0.12),
                line_width=3.0,
                closed=True,
                opacity=0.95,
                reference_node=reference_node,
            )
            labels_and_points = []
            for idx, point in enumerate(entry_points):
                role = role_order[idx] if idx < len(role_order) else f"entry_{idx + 1}"
                label = ordered_labels[idx] if idx < len(ordered_labels) and ordered_labels[idx] else role
                labels_and_points.append((f"{role}: {label}", point))
            if pivot is not None:
                labels_and_points.append(("pivot", pivot))
            self._add_fiducials_node("Surgiplot AoA Landmarks", labels_and_points, color=(0.95, 0.8, 0.2), reference_node=reference_node)
        self._update_results_table(
            "Surgiplot AoA Results",
            [
                ("Metric family", "AoA / SF"),
                ("Reference anatomy", self._reference_name(reference_node)),
                ("AoA_vertical_deg", f"{result.aoa_vertical_deg:.6f}"),
                ("AoA_horizontal_deg", f"{result.aoa_horizontal_deg:.6f}"),
                ("SF_area_mm2", f"{result.sf_entry_area_mm2:.6f}"),
                ("SF_area_standardized_mm2", f"{result.sf_entry_area_rescaled_mm2:.6f}"),
                ("Entry points", self.bridge.markups_count(entry_node)),
                ("Ordering strategy", str(debug.get("ordering_strategy", "unknown"))),
                ("Expected role order", ", ".join(role_order)),
                ("Ordered labels", ", ".join(lb or "-" for lb in ordered_labels) if ordered_labels else "-"),
                ("Warnings", " | ".join(warnings) if warnings else "none"),
            ],
            reference_node=reference_node,
        )
        payload = {
            "metric_family": "AoA / SF",
            "reference_anatomy": self._reference_name(reference_node),
            "AoA_vertical_deg": result.aoa_vertical_deg,
            "AoA_horizontal_deg": result.aoa_horizontal_deg,
            "SF_area_mm2": result.sf_entry_area_mm2,
            "SF_area_standardized_mm2": result.sf_entry_area_rescaled_mm2,
            "standardized_sf_radius_mm": float(debug.get("sf_rescale_radius_mm", 0.0) or 0.0),
            "entry_point_count": self.bridge.markups_count(entry_node),
            "ordering_strategy": debug.get("ordering_strategy", "unknown"),
            "entry_role_order": role_order,
            "ordered_entry_labels": ordered_labels,
            "warnings": warnings,
        }
        self._update_results_report(payload, reference_node=reference_node)
        return payload

    def compute_aoe(self, a_node, b_node, c_node, *, reference_node=None) -> dict[str, Any]:
        result = self.bridge.compute_aoe(a_node, b_node, c_node)
        debug = result.debug or {}
        A = debug.get("A")
        B = debug.get("B")
        C = debug.get("C")
        triangle = None
        arc = None
        labels_and_points = []
        if A is not None and B is not None and C is not None:
            triangle = [np.vstack([A, B, C])]
            arc = self._aoe_arc_points(A, B, C)
            labels_and_points = [("A", A), ("B", B), ("C", C)]
        self._add_triangles_model("Surgiplot AoE Triangle", triangle, (0.25, 0.7, 0.35), 0.28, reference_node=reference_node)
        self._add_polyline_model("Surgiplot AoE Arc", arc, color=(0.95, 0.7, 0.15), line_width=3.2, opacity=0.96, reference_node=reference_node)
        self._add_fiducials_node("Surgiplot AoE Landmarks", labels_and_points, color=(0.95, 0.8, 0.2), reference_node=reference_node)
        self._update_results_table(
            "Surgiplot AoE Results",
            [
                ("Metric family", "AoE"),
                ("Reference anatomy", self._reference_name(reference_node)),
                ("AoE_deg", f"{result.aoe_deg:.6f}"),
            ],
            reference_node=reference_node,
        )
        payload = {"metric_family": "AoE", "reference_anatomy": self._reference_name(reference_node), "AoE_deg": result.aoe_deg}
        self._update_results_report(payload, reference_node=reference_node)
        return payload

    def compute_distance(self, a_node, b_node, *, reference_node=None) -> dict[str, Any]:
        result = self.bridge.compute_distance(a_node, b_node)
        debug = result.debug or {}
        A = debug.get("A")
        B = debug.get("B")
        line = None
        labels_and_points = []
        if A is not None and B is not None:
            line = [A, B]
            labels_and_points = [("A", A), ("B", B)]
        self._add_polyline_model("Surgiplot Distance Line", line, color=(0.15, 0.75, 0.9), line_width=4.0, opacity=0.95, reference_node=reference_node)
        self._add_fiducials_node("Surgiplot Distance Landmarks", labels_and_points, color=(0.15, 0.75, 0.9), reference_node=reference_node)
        self._update_results_table(
            "Surgiplot Distance Results",
            [
                ("Metric family", "Distance"),
                ("Reference anatomy", self._reference_name(reference_node)),
                ("distance_mm", f"{result.distance_mm:.6f}"),
            ],
            reference_node=reference_node,
        )
        payload = {"metric_family": "Distance", "reference_anatomy": self._reference_name(reference_node), "distance_mm": result.distance_mm}
        self._update_results_report(payload, reference_node=reference_node)
        return payload

    def compute_area(self, polygon_node, *, reference_node=None) -> dict[str, Any]:
        result = self.bridge.compute_area(polygon_node)
        debug = result.debug or {}
        polygon = self._ordered_polygon_points(debug.get("polygon"))
        coplanar = debug.get("coplanar_polygon_3d")
        labels_and_points = []
        if polygon is not None:
            polygon_arr = np.asarray(polygon, dtype=float)
            labels_and_points = [(f"P{idx + 1}", point) for idx, point in enumerate(polygon_arr)]
            self._add_polyline_model("Surgiplot Area Polygon", polygon_arr, color=(0.95, 0.55, 0.15), line_width=3.0, closed=True, opacity=0.95, reference_node=reference_node)
        if coplanar is not None:
            coplanar_arr = np.asarray(coplanar, dtype=float)
            self._add_triangles_model("Surgiplot Area Surface", coplanar_arr, (0.55, 0.62, 0.72), 0.20, reference_node=reference_node)
        self._add_fiducials_node("Surgiplot Area Landmarks", labels_and_points, color=(0.95, 0.55, 0.15), reference_node=reference_node)
        self._update_results_table(
            "Surgiplot Area Results",
            [
                ("Metric family", "Area"),
                ("Reference anatomy", self._reference_name(reference_node)),
                ("area_mm2", f"{result.area_mm2:.6f}"),
            ],
            reference_node=reference_node,
        )
        payload = {"metric_family": "Area", "reference_anatomy": self._reference_name(reference_node), "area_mm2": result.area_mm2}
        self._update_results_report(payload, reference_node=reference_node)
        return payload

    def compute_volume(self, volume_node, *, reference_node=None) -> dict[str, Any]:
        result = self.bridge.compute_volume(volume_node)
        debug = result.debug or {}
        triangles = debug.get("surface_triangles")
        points = debug.get("points")
        labels_and_points = []
        if points is not None:
            points_arr = np.asarray(points, dtype=float)
            labels_and_points = [(f"P{idx + 1}", point) for idx, point in enumerate(points_arr)]
        self._add_triangles_model("Surgiplot Volume Surface", triangles, (0.65, 0.65, 0.78), 0.22, reference_node=reference_node)
        self._add_fiducials_node("Surgiplot Volume Landmarks", labels_and_points, color=(0.85, 0.45, 0.2), reference_node=reference_node)
        self._update_results_table(
            "Surgiplot Volume Results",
            [
                ("Metric family", "Volume"),
                ("Reference anatomy", self._reference_name(reference_node)),
                ("volume_mm3", f"{result.volume_mm3:.6f}"),
            ],
            reference_node=reference_node,
        )
        payload = {"metric_family": "Volume", "reference_anatomy": self._reference_name(reference_node), "volume_mm3": result.volume_mm3}
        self._update_results_report(payload, reference_node=reference_node)
        return payload


class SurgiplotToolkitTest(ScriptedLoadableModuleTest):
    def runTest(self) -> None:
        self.setUp()
        self.test_placeholder()

    def test_placeholder(self) -> None:
        self.delayDisplay("SurgiplotToolkit scaffold loaded successfully.")
