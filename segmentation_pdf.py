"""Build a segmentation PDF in memory; no uploaded images are saved."""
import base64
from io import BytesIO

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def build_segmentation_pdf(prediction, segmentation):
    buffer = BytesIO()
    page_width, page_height = landscape(A4)
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    pdf.setTitle("KidneyVision - Segmentation result")
    pdf.setAuthor("KidneyVision")
    pdf.setFillColorRGB(.06, .18, .31)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(36, page_height - 48, "KidneyVision | Segmentation result")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(36, page_height - 76,
                   f"Predicted class: {prediction['label'].title()}    "
                   f"Model score: {prediction['confidence'] * 100:.2f}%")
    image_width = (page_width - 90) / 2
    image_bottom, image_height = 116, page_height - 244
    for index, (key, caption) in enumerate((("overlay", "Highlighted CT"),
                                             ("mask", "Segmentation mask"))):
        left = 36 + index * (image_width + 18)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(left, page_height - 108, caption)
        raw = base64.b64decode(segmentation[key].split(",", 1)[1], validate=True)
        image = ImageReader(BytesIO(raw))
        width, height = image.getSize()
        scale = min(image_width / width, image_height / height)
        draw_width, draw_height = width * scale, height * scale
        pdf.setFillColorRGB(0, 0, 0)
        pdf.rect(left, image_bottom, image_width, image_height, fill=1, stroke=0)
        pdf.drawImage(image, left + (image_width - draw_width) / 2,
                      image_bottom + (image_height - draw_height) / 2,
                      draw_width, draw_height, mask="auto")
    for index, (name, color) in enumerate((("Stone", (1, .314, .314)),
                                          ("Tumor", (1, .745, .196)),
                                          ("Cyst", (.275, .667, 1)))):
        left = 36 + index * 105
        pdf.setFillColorRGB(*color)
        pdf.rect(left, 88, 10, 10, fill=1, stroke=0)
        pdf.setFillColorRGB(.06, .18, .31)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(left + 16, 89, name)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(36, 62, "Research model output. Model scores are uncalibrated; clinical review is required.")
    pdf.drawString(36, 47, "Independent masks may overlap. Physical size requires verified pixel spacing.")
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()
