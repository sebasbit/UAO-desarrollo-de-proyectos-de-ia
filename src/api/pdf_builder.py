"""
Generación del reporte PDF

Construye el PDF a partir de los datos de predicción.
Define su propio DTO (ReportData) para no depender de src.api.schemas,
que pertenece a T05 (aún pendiente). Cuando T05 esté listo, report.py
simplemente mapeará PredictionResponse → ReportData antes de llamar aquí.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle


# DTO propio — sin dependencia de schemas.py

@dataclass
class ReportData:
    """
    Datos necesarios para generar el PDF de triage.

    Este DTO es propio de pdf_builder y no depende de PredictionResponse
    (schemas.py / T05). Cuando T05 esté listo, report.py hará el mapeo:
        ReportData.from_prediction(prediction_response)
    """
    category: str
    category_key: str
    score: float
    team: str
    timestamp: datetime
    human_review_required: bool
    image_filename: str = "imagen_incidente"

    @classmethod
    def from_prediction(
        cls,
        category: str,
        category_key: str,
        score: float,
        team: str,
        human_review_required: bool,
        image_filename: str = "imagen_incidente",
        timestamp: datetime | None = None,
    ) -> ReportData:
        return cls(
            category=category,
            category_key=category_key,
            score=score,
            team=team,
            timestamp=timestamp or datetime.now(UTC),
            human_review_required=human_review_required,
            image_filename=image_filename,
        )


# Paleta

class _P:
    NAVY       = colors.HexColor("#0F2645")   # azul oscuro — header/títulos
    BLUE       = colors.HexColor("#1A4A8A")   # azul medio — acentos
    STEEL      = colors.HexColor("#4A7FC1")   # azul claro — detalles
    RED        = colors.HexColor("#C0392B")   # rojo alerta — revisión humana
    GREEN      = colors.HexColor("#1E7E4A")   # verde — OK
    OFF_WHITE  = colors.HexColor("#F7F9FC")   # fondo filas alternas
    RULE       = colors.HexColor("#D0DAE8")   # líneas separadoras
    BODY       = colors.HexColor("#1A1A2E")   # texto principal
    MUTED      = colors.HexColor("#6B7A99")   # texto secundario
    WHITE      = colors.white


# Builder

class _PDFBuilder:
    PAGE   = A4
    MARGIN = 2.2 * cm

    def __init__(self, data: ReportData) -> None:
        self._d     = data
        self._st    = getSampleStyleSheet()
        self._buf   = io.BytesIO()
        self._story: list = []

    # Public

    def build(self) -> bytes:
        doc = SimpleDocTemplate(
            self._buf,
            pagesize=self.PAGE,
            leftMargin=self.MARGIN, rightMargin=self.MARGIN,
            topMargin=3.2 * cm,    bottomMargin=2.0 * cm,
            title=f"Reporte Triage TI — {self._d.timestamp:%Y-%m-%d}",
            author="Sistema de Triage de Soporte TI — UAO",
        )
        self._build_story()
        doc.build(self._story,
                  onFirstPage=self._draw_chrome,
                  onLaterPages=self._draw_chrome)
        return self._buf.getvalue()

    # Story Sections

    def _build_story(self) -> None:
        self._story.clear()
        self._add_section_title("Datos del análisis")
        self._story.append(Spacer(1, .3 * cm))
        self._add_meta_table()
        self._story.append(Spacer(1, .6 * cm))
        self._add_section_title("Resultado de clasificación")
        self._story.append(Spacer(1, .3 * cm))
        self._add_classification_table()
        self._story.append(Spacer(1, .6 * cm))
        self._add_action_banner()
        self._story.append(Spacer(1, .8 * cm))
        self._add_footer_note()

    def _add_section_title(self, text: str) -> None:
        self._story.append(Paragraph(
            text.upper(),
            ParagraphStyle(
                "SecTitle",
                parent=self._st["Normal"],
                fontName="Helvetica-Bold",
                fontSize=7,
                textColor=_P.STEEL,
                spaceAfter=2,
                letterSpacing=1.5,
            ),
        ))
        self._story.append(
            HRFlowable(width="100%", thickness=0.75, color=_P.RULE, spaceAfter=0)
        )

    def _add_meta_table(self) -> None:
        """Tabla de 2 columnas con metadatos del análisis."""
        d = self._d
        label_style = ParagraphStyle(
            "ML", parent=self._st["Normal"],
            fontSize=8, textColor=_P.MUTED, fontName="Helvetica",
        )
        value_style = ParagraphStyle(
            "MV", parent=self._st["Normal"],
            fontSize=8.5, textColor=_P.BODY, fontName="Helvetica-Bold",
        )

        rows = [
            [Paragraph("Archivo analizado", label_style),
             Paragraph(d.image_filename, value_style),
             Paragraph("Fecha y hora", label_style),
             Paragraph(f"{d.timestamp:%d %b %Y  •  %H:%M:%S} UTC", value_style)],
            [Paragraph("Modelo", label_style),
             Paragraph("DeiT-Tiny · facebook/deit-tiny-patch16-224", value_style),
             Paragraph("Identificador", label_style),
             Paragraph(d.category_key, value_style)],
        ]

        col_w = [3 * cm, 6.5 * cm, 3 * cm, 4.5 * cm]
        tbl = Table(rows, colWidths=col_w)
        tbl.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 0),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("LINEBELOW",     (0, 0), (-1, -2), 0.3, _P.RULE),
        ]))
        self._story.append(tbl)

    def _add_classification_table(self) -> None:
        """Tabla principal con el resultado de la clasificación."""
        d     = self._d
        score = d.score
        pct   = f"{score * 100:.1f} %"
        bar   = self._score_bar(score, width=14)

        # Colores de confianza
        if score >= 0.70:
            score_color = _P.GREEN
        elif score >= 0.40:
            score_color = colors.HexColor("#D4860A")
        else:
            score_color = _P.RED

        review_text  = "Revisión humana requerida" if d.human_review_required else "Enrutamiento automático"
        review_color = _P.RED if d.human_review_required else _P.GREEN
        review_icon  = "⚠" if d.human_review_required else "✓"

        hdr = ParagraphStyle("H", parent=self._st["Normal"],
                             fontSize=8, fontName="Helvetica-Bold",
                             textColor=_P.WHITE)
        cell = ParagraphStyle("C", parent=self._st["Normal"],
                              fontSize=9, fontName="Helvetica", textColor=_P.BODY)
        score_st = ParagraphStyle("SC", parent=self._st["Normal"],
                                  fontSize=9, fontName="Helvetica-Bold",
                                  textColor=score_color)
        review_st = ParagraphStyle("RC", parent=self._st["Normal"],
                                   fontSize=9, fontName="Helvetica-Bold",
                                   textColor=review_color)

        data = [
            # Encabezado
            [Paragraph("CATEGORÍA", hdr),
             Paragraph("EQUIPO DESTINATARIO", hdr),
             Paragraph("CONFIANZA", hdr),
             Paragraph("ACCIÓN", hdr)],
            # Datos
            [Paragraph(d.category, cell),
             Paragraph(d.team, cell),
             Paragraph(f"{pct}\n{bar}", score_st),
             Paragraph(f"{review_icon}  {review_text}", review_st)],
        ]

        col_w = [4.5 * cm, 4.5 * cm, 4 * cm, 4 * cm]
        tbl = Table(data, colWidths=col_w, rowHeights=[None, 1.4 * cm])
        tbl.setStyle(TableStyle([
            # Header
            ("BACKGROUND",    (0, 0), (-1, 0), _P.NAVY),
            ("TOPPADDING",    (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            # Data row
            ("BACKGROUND",    (0, 1), (-1, 1), _P.OFF_WHITE),
            ("TOPPADDING",    (0, 1), (-1, 1), 10),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            # Borders
            ("BOX",           (0, 0), (-1, -1), 0.5, _P.RULE),
            ("LINEAFTER",     (0, 0), (-2, -1), 0.3, _P.RULE),
            # Left accent bar on data row
            ("LINEAFTER",     (0, 1), (0, 1), 0, _P.RULE),
        ]))
        self._story.append(tbl)

    def _add_action_banner(self) -> None:
        d = self._d

        if d.human_review_required:
            bg     = colors.HexColor("#FDF2F2")
            border = _P.RED
            icon   = "⚠"
            title  = "Requiere revisión humana"
            detail = (
                f"El score de confianza ({d.score:.0%}) es inferior al umbral mínimo (40 %). "
                f"Este ticket debe ser revisado manualmente antes de ser asignado."
            )
        else:
            bg     = colors.HexColor("#F2FAF5")
            border = _P.GREEN
            icon   = "✓"
            title  = f"Enrutar a: {d.team}"
            detail = (
                f"Clasificación aceptada con una confianza del {d.score:.0%}. "
                f"El ticket puede asignarse automáticamente al equipo indicado."
            )

        title_st = ParagraphStyle(
            "BT", parent=self._st["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=border, leading=14,
        )
        detail_st = ParagraphStyle(
            "BD", parent=self._st["Normal"],
            fontSize=8.5, textColor=_P.MUTED, leading=13,
        )

        content = [
            Paragraph(f"{icon}  {title}", title_st),
            Spacer(1, 4),
            Paragraph(detail, detail_st),
        ]

        banner = Table([[content]], colWidths=[17 * cm])
        banner.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("LINEBEFORE",    (0, 0), (0, -1), 4, border),
            ("BOX",           (0, 0), (-1, -1), 0.4, _P.RULE),
            ("TOPPADDING",    (0, 0), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ("LEFTPADDING",   (0, 0), (-1, -1), 14),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ]))
        self._story.append(banner)

    def _add_footer_note(self) -> None:
        self._story.append(
            HRFlowable(width="100%", thickness=0.4, color=_P.RULE)
        )
        self._story.append(Spacer(1, .2 * cm))
        self._story.append(Paragraph(
            "Este reporte es generado automáticamente con carácter orientativo. ",
            ParagraphStyle("FN", parent=self._st["Normal"],
                           fontSize=7, textColor=_P.MUTED, leading=10),
        ))

    # Canvas chrome

    @staticmethod
    def _draw_chrome(canvas, doc) -> None:  # noqa: ARG004
        canvas.saveState()
        w, h = A4
        m    = 2.2 * cm

        # Header band
        band_h = 1.8 * cm
        canvas.setFillColor(_P.NAVY)
        canvas.rect(0, h - band_h, w, band_h, fill=1, stroke=0)

        # Accent stripe
        canvas.setFillColor(_P.STEEL)
        canvas.rect(0, h - band_h - 0.18 * cm, w, 0.18 * cm, fill=1, stroke=0)

        # Title in header
        canvas.setFont("Helvetica-Bold", 11)
        canvas.setFillColor(_P.WHITE)
        canvas.drawString(m, h - 1.25 * cm, "Reporte de Triage de Soporte TI")

        # Subtitle in header
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#A8C4E0"))
        canvas.drawRightString(
            w - m, h - 1.25 * cm,
            f"UAO · {datetime.now():%d/%m/%Y}",
        )

        # Footer
        canvas.setStrokeColor(_P.RULE)
        canvas.setLineWidth(0.4)
        canvas.line(m, 1.5 * cm, w - m, 1.5 * cm)

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(_P.MUTED)
        canvas.drawString(m, 1.1 * cm,
                          "Triage de Soporte TI — Desarrollo de Proyectos de IA")
        canvas.drawCentredString(w / 2, 1.1 * cm, f"Página {doc.page}")
        canvas.drawRightString(w - m, 1.1 * cm,
                               f"Generado: {datetime.now():%H:%M:%S}")

        canvas.restoreState()

    # Util

    @staticmethod
    def _score_bar(score: float, width: int = 14) -> str:
        filled = round(score * width)
        return "█" * filled + "░" * (width - filled)


# Función pública

def build_report_pdf(data: ReportData, image_filename: str | None = None) -> bytes:
    """
    Genera el PDF de reporte y retorna sus bytes.

    Args:
        data           : DTO con los datos de la predicción.
        image_filename : si se indica, sobreescribe data.image_filename.
    """
    if image_filename is not None:
        from dataclasses import replace
        data = replace(data, image_filename=image_filename)
    return _PDFBuilder(data).build()
