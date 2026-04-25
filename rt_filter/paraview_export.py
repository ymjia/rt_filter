from __future__ import annotations

from pathlib import Path


def write_paraview_comparison_script(
    vtk_files: list[str | Path],
    labels: list[str],
    output_script: str | Path,
    *,
    normal_scale: float = 0.03,
) -> Path:
    if not vtk_files:
        raise ValueError("at least one VTK file is required")
    if len(vtk_files) != len(labels):
        raise ValueError("vtk_files and labels must have the same length")

    script_path = Path(output_script)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "path": str(Path(path).resolve()).replace("\\", "/"),
            "label": label.replace("'", "\\'"),
        }
        for path, label in zip(vtk_files, labels, strict=True)
    ]

    lines = [
        "import paraview",
        "from paraview.simple import *",
        "",
        "DATASETS = [",
    ]
    for item in entries:
        lines.append(f"    ('{item['label']}', r'{item['path']}'),")
    lines.extend(
        [
            "]",
            f"NORMAL_SCALE = {normal_scale!r}",
            "",
            "paraview.simple._DisableFirstRenderCameraReset()",
            "layout = CreateLayout('Trajectory Filter Comparison')",
            "views = []",
            "",
            "for index, (label, filename) in enumerate(DATASETS):",
            "    view = CreateView('RenderView')",
            "    view.ViewSize = [720, 520]",
            "    AssignViewToLayout(view=view, layout=layout, hint=index)",
            "    reader = XMLUnstructuredGridReader(registrationName=label, FileName=[filename])",
            "    display = Show(reader, view)",
            "    display.Representation = 'Points'",
            "    display.PointSize = 5",
            "    ColorBy(display, ('POINTS', 'SampleIndex'))",
            "    display.RescaleTransferFunctionToDataRange(True, False)",
            "    glyph = Glyph(registrationName=label + '_Normals', Input=reader, GlyphType='Arrow')",
            "    glyph.OrientationArray = ['POINTS', 'Normals']",
            "    glyph.ScaleArray = ['POINTS', 'No scale array']",
            "    glyph.ScaleFactor = NORMAL_SCALE",
            "    glyph_display = Show(glyph, view)",
            "    glyph_display.DiffuseColor = [0.1, 0.1, 0.1]",
            "    view.Update()",
            "    ResetCamera(view)",
            "    views.append(view)",
            "",
            "if views:",
            "    active = views[0]",
            "    for view in views[1:]:",
            "        try:",
            "            AddCameraLink(active, view, 'camera_link_' + str(len(views)))",
            "        except Exception:",
            "            pass",
            "    SetActiveView(active)",
            "",
            "# Run with: pvpython this_script.py",
            "# Or open it from ParaView: File > Load State/Tools > Python Shell depending on version.",
            "RenderAllViews()",
        ]
    )
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path
