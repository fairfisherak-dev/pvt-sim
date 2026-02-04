"""Report generation for PVT calculations.

This module generates formatted reports from PVT calculation results
in various formats:
- Plain text (terminal/console output)
- Markdown (documentation, GitHub)
- HTML (web display)
- CSV tables (data export)

Report types:
- Fluid composition summary
- Flash calculation results
- Phase envelope data
- CCE/DL/CVD experiment results
- Separator train results
- Regression quality summary

References
----------
SPE report formatting guidelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import io

import numpy as np
from numpy.typing import NDArray


@dataclass
class ReportSection:
    """A section within a report.

    Attributes:
        title: Section title
        content: Section content (text or table data)
        level: Heading level (1=main, 2=sub, etc.)
    """
    title: str
    content: str
    level: int = 2


class PVTReport:
    """Generate formatted PVT reports.

    Parameters
    ----------
    title : str
        Report title.
    description : str, optional
        Report description/summary.

    Examples
    --------
    >>> report = PVTReport("Flash Calculation Results")
    >>> report.add_section("Conditions", "P = 10 MPa, T = 350 K")
    >>> report.add_table("Phase Properties", headers, data)
    >>> print(report.to_text())
    """

    def __init__(self, title: str, description: str = ""):
        self.title = title
        self.description = description
        self.sections: List[ReportSection] = []
        self.tables: List[Dict[str, Any]] = []
        self.metadata: Dict[str, str] = {
            'generated_at': datetime.now().isoformat(),
            'generator': 'pvtcore',
        }

    def add_section(self, title: str, content: str, level: int = 2) -> None:
        """Add a text section to the report.

        Parameters
        ----------
        title : str
            Section title.
        content : str
            Section content.
        level : int
            Heading level.
        """
        self.sections.append(ReportSection(title, content, level))

    def add_table(
        self,
        title: str,
        headers: List[str],
        data: List[List[Any]],
        formats: Optional[List[str]] = None,
    ) -> None:
        """Add a data table to the report.

        Parameters
        ----------
        title : str
            Table title.
        headers : list of str
            Column headers.
        data : list of list
            Table data (rows).
        formats : list of str, optional
            Format strings for each column (e.g., '.3f', '.2e').
        """
        self.tables.append({
            'title': title,
            'headers': headers,
            'data': data,
            'formats': formats,
        })

    def to_text(self, width: int = 80) -> str:
        """Generate plain text report.

        Parameters
        ----------
        width : int
            Maximum line width.

        Returns
        -------
        str
            Formatted text report.
        """
        lines = []

        # Title
        lines.append("=" * width)
        lines.append(self.title.center(width))
        lines.append("=" * width)

        if self.description:
            lines.append("")
            lines.append(self.description)

        lines.append("")
        lines.append(f"Generated: {self.metadata['generated_at']}")
        lines.append("-" * width)

        # Sections
        for section in self.sections:
            lines.append("")
            if section.level == 1:
                lines.append(section.title.upper())
                lines.append("-" * len(section.title))
            else:
                lines.append(f"{section.title}")
                lines.append("-" * len(section.title))
            lines.append(section.content)

        # Tables
        for table in self.tables:
            lines.append("")
            lines.append(table['title'])
            lines.append("-" * len(table['title']))
            lines.append(_format_text_table(
                table['headers'],
                table['data'],
                table.get('formats'),
            ))

        lines.append("")
        lines.append("=" * width)

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate Markdown report.

        Returns
        -------
        str
            Markdown formatted report.
        """
        lines = []

        # Title
        lines.append(f"# {self.title}")
        lines.append("")

        if self.description:
            lines.append(self.description)
            lines.append("")

        lines.append(f"*Generated: {self.metadata['generated_at']}*")
        lines.append("")

        # Sections
        for section in self.sections:
            prefix = "#" * (section.level + 1)
            lines.append(f"{prefix} {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")

        # Tables
        for table in self.tables:
            lines.append(f"### {table['title']}")
            lines.append("")
            lines.append(_format_markdown_table(
                table['headers'],
                table['data'],
                table.get('formats'),
            ))
            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate HTML report.

        Returns
        -------
        str
            HTML formatted report.
        """
        lines = []

        lines.append("<!DOCTYPE html>")
        lines.append("<html>")
        lines.append("<head>")
        lines.append(f"<title>{self.title}</title>")
        lines.append("<style>")
        lines.append(_get_html_style())
        lines.append("</style>")
        lines.append("</head>")
        lines.append("<body>")

        lines.append(f"<h1>{self.title}</h1>")

        if self.description:
            lines.append(f"<p class='description'>{self.description}</p>")

        lines.append(f"<p class='metadata'>Generated: {self.metadata['generated_at']}</p>")

        # Sections
        for section in self.sections:
            tag = f"h{section.level + 1}"
            lines.append(f"<{tag}>{section.title}</{tag}>")
            lines.append(f"<p>{section.content}</p>")

        # Tables
        for table in self.tables:
            lines.append(f"<h3>{table['title']}</h3>")
            lines.append(_format_html_table(
                table['headers'],
                table['data'],
                table.get('formats'),
            ))

        lines.append("</body>")
        lines.append("</html>")

        return "\n".join(lines)

    def save(self, filepath: Union[str, Path], format: str = 'auto') -> None:
        """Save report to file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        format : str
            Output format ('text', 'markdown', 'html', or 'auto').
            'auto' determines format from file extension.
        """
        filepath = Path(filepath)

        if format == 'auto':
            ext = filepath.suffix.lower()
            if ext in ('.md', '.markdown'):
                format = 'markdown'
            elif ext in ('.html', '.htm'):
                format = 'html'
            else:
                format = 'text'

        if format == 'markdown':
            content = self.to_markdown()
        elif format == 'html':
            content = self.to_html()
        else:
            content = self.to_text()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


def _format_text_table(
    headers: List[str],
    data: List[List[Any]],
    formats: Optional[List[str]] = None,
) -> str:
    """Format a table as plain text."""
    if not data:
        return "(no data)"

    # Calculate column widths
    n_cols = len(headers)
    widths = [len(h) for h in headers]

    formatted_data = []
    for row in data:
        formatted_row = []
        for i, val in enumerate(row):
            if formats and i < len(formats) and formats[i]:
                try:
                    s = format(val, formats[i])
                except (ValueError, TypeError):
                    s = str(val)
            else:
                if isinstance(val, float):
                    s = f"{val:.6g}"
                else:
                    s = str(val)
            formatted_row.append(s)
            if i < n_cols:
                widths[i] = max(widths[i], len(s))
        formatted_data.append(formatted_row)

    # Build table
    lines = []

    # Header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-+-".join("-" * w for w in widths))

    # Data
    for row in formatted_data:
        row_line = " | ".join(
            row[i].rjust(widths[i]) if i < len(row) else " " * widths[i]
            for i in range(n_cols)
        )
        lines.append(row_line)

    return "\n".join(lines)


def _format_markdown_table(
    headers: List[str],
    data: List[List[Any]],
    formats: Optional[List[str]] = None,
) -> str:
    """Format a table as Markdown."""
    if not data:
        return "*No data*"

    lines = []

    # Header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    # Data
    for row in data:
        formatted_row = []
        for i, val in enumerate(row):
            if formats and i < len(formats) and formats[i]:
                try:
                    s = format(val, formats[i])
                except (ValueError, TypeError):
                    s = str(val)
            else:
                if isinstance(val, float):
                    s = f"{val:.6g}"
                else:
                    s = str(val)
            formatted_row.append(s)
        lines.append("| " + " | ".join(formatted_row) + " |")

    return "\n".join(lines)


def _format_html_table(
    headers: List[str],
    data: List[List[Any]],
    formats: Optional[List[str]] = None,
) -> str:
    """Format a table as HTML."""
    if not data:
        return "<p><em>No data</em></p>"

    lines = []
    lines.append("<table>")

    # Header
    lines.append("<thead><tr>")
    for h in headers:
        lines.append(f"<th>{h}</th>")
    lines.append("</tr></thead>")

    # Body
    lines.append("<tbody>")
    for row in data:
        lines.append("<tr>")
        for i, val in enumerate(row):
            if formats and i < len(formats) and formats[i]:
                try:
                    s = format(val, formats[i])
                except (ValueError, TypeError):
                    s = str(val)
            else:
                if isinstance(val, float):
                    s = f"{val:.6g}"
                else:
                    s = str(val)
            lines.append(f"<td>{s}</td>")
        lines.append("</tr>")
    lines.append("</tbody>")

    lines.append("</table>")
    return "\n".join(lines)


def _get_html_style() -> str:
    """Return CSS styles for HTML reports."""
    return """
body {
    font-family: Arial, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    line-height: 1.6;
}
h1 {
    color: #333;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
}
h2, h3 {
    color: #555;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: right;
}
th {
    background-color: #f4f4f4;
    text-align: center;
}
tr:nth-child(even) {
    background-color: #f9f9f9;
}
.description {
    font-style: italic;
    color: #666;
}
.metadata {
    font-size: 0.9em;
    color: #888;
}
"""


# =============================================================================
# Specialized Report Generators
# =============================================================================

def generate_flash_report(
    pressure: float,
    temperature: float,
    composition: NDArray[np.float64],
    component_names: List[str],
    flash_result: Any,
    title: str = "PT Flash Calculation Results",
) -> PVTReport:
    """Generate report for PT flash calculation.

    Parameters
    ----------
    pressure : float
        Pressure (Pa).
    temperature : float
        Temperature (K).
    composition : ndarray
        Feed composition.
    component_names : list of str
        Component names.
    flash_result : FlashResult
        Flash calculation result.
    title : str
        Report title.

    Returns
    -------
    PVTReport
        Formatted report.
    """
    report = PVTReport(title)

    # Conditions
    report.add_section(
        "Conditions",
        f"Pressure: {pressure/1e6:.3f} MPa ({pressure/1e5:.2f} bar)\n"
        f"Temperature: {temperature:.2f} K ({temperature-273.15:.2f} C)"
    )

    # Phase state
    report.add_section("Phase State", f"Result: {flash_result.phase}")

    if flash_result.phase == 'two-phase':
        report.add_section(
            "Split",
            f"Vapor fraction: {flash_result.vapor_fraction:.6f}\n"
            f"Liquid fraction: {1-flash_result.vapor_fraction:.6f}"
        )

    # Composition table
    headers = ['Component', 'Feed (zi)', 'Liquid (xi)', 'Vapor (yi)', 'Ki']
    data = []
    for i, name in enumerate(component_names):
        xi = flash_result.liquid_composition[i] if flash_result.liquid_composition is not None else 0
        yi = flash_result.vapor_composition[i] if flash_result.vapor_composition is not None else 0
        Ki = flash_result.K_values[i] if flash_result.K_values is not None else 0
        data.append([name, composition[i], xi, yi, Ki])

    report.add_table("Phase Compositions", headers, data, [None, '.6f', '.6f', '.6f', '.4f'])

    return report


def generate_cce_report(
    cce_result: Any,
    component_names: List[str],
    title: str = "Constant Composition Expansion Results",
) -> PVTReport:
    """Generate report for CCE simulation.

    Parameters
    ----------
    cce_result : CCEResult
        CCE simulation result.
    component_names : list of str
        Component names.
    title : str
        Report title.

    Returns
    -------
    PVTReport
        Formatted report.
    """
    report = PVTReport(title)

    # Conditions
    report.add_section(
        "Test Conditions",
        f"Temperature: {cce_result.temperature:.2f} K ({cce_result.temperature-273.15:.2f} C)\n"
        f"Saturation pressure: {cce_result.saturation_pressure/1e6:.3f} MPa\n"
        f"Saturation type: {cce_result.saturation_type}"
    )

    # Results table
    headers = ['P (MPa)', 'V/Vsat', 'Liquid Dropout', 'Phase']
    data = []
    for step in cce_result.steps:
        data.append([
            step.pressure / 1e6,
            step.relative_volume,
            step.liquid_volume_fraction,
            step.phase,
        ])

    report.add_table("Expansion Data", headers, data, ['.3f', '.4f', '.4f', None])

    return report


def generate_separator_report(
    sep_result: Any,
    component_names: List[str],
    title: str = "Separator Train Results",
) -> PVTReport:
    """Generate report for separator calculations.

    Parameters
    ----------
    sep_result : SeparatorTrainResult
        Separator calculation result.
    component_names : list of str
        Component names.
    title : str
        Report title.

    Returns
    -------
    PVTReport
        Formatted report.
    """
    report = PVTReport(title)

    # Summary
    report.add_section(
        "Summary",
        f"Oil Formation Volume Factor (Bo): {sep_result.Bo:.4f}\n"
        f"Solution GOR (Rs): {sep_result.Rs:.1f} m3/m3 ({sep_result.Rs_scf_stb:.0f} scf/STB)\n"
        f"API Gravity: {sep_result.API_gravity:.1f}\n"
        f"Stock Tank Oil SG: {sep_result.stock_tank_oil_SG:.4f}"
    )

    # Stage details
    headers = ['Stage', 'P (MPa)', 'T (K)', 'Vapor Fraction', 'Liquid Out (mol)']
    data = []
    for stage in sep_result.stages:
        data.append([
            stage.conditions.name or f"Stage {stage.stage_number}",
            stage.conditions.pressure / 1e6,
            stage.conditions.temperature,
            stage.vapor_fraction,
            stage.liquid_moles,
        ])

    report.add_table("Stage Details", headers, data, [None, '.3f', '.1f', '.4f', '.6f'])

    # Stock tank oil composition
    headers = ['Component', 'Mole Fraction']
    data = [[name, sep_result.stock_tank_oil_composition[i]]
            for i, name in enumerate(component_names)]

    report.add_table("Stock Tank Oil Composition", headers, data, [None, '.6f'])

    return report
