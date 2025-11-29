"""
Generate PDF report from Markdown file.
"""

import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md, output_pdf):
    # 1. Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        md_text = f.read()
        
    # 2. Convert to HTML
    # Enable extensions for tables and math (math support is limited in basic markdown)
    html_text = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    
    # Add some basic CSS for styling
    css = """
    <style>
        body { font-family: Helvetica, sans-serif; font-size: 12pt; }
        h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 20px; }
        h3 { color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: left; }
        th { background-color: #ecf0f1; }
        code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 4px; font-family: monospace; }
        pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; overflow-x: auto; }
    </style>
    """
    
    full_html = f"<html><head>{css}</head><body>{html_text}</body></html>"
    
    # 3. Convert HTML to PDF
    with open(output_pdf, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
        
    if pisa_status.err:
        print(f"Error generating PDF: {pisa_status.err}")
    else:
        print(f"Successfully generated PDF: {output_pdf}")

if __name__ == "__main__":
    convert_md_to_pdf("report.md", "report.pdf")
