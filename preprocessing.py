import pdfplumber

def extract_text_without_header_footer(pdf_path, top_margin=72, bottom_margin=72):
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content_area = page.crop((0, top_margin, page.width, page.height))
            full_text.append(content_area.extract_text())
    return "\n".join(full_text)


# Testing
if __name__ == "__main__":
    pdf_file = "file.pdf"
    extracted_content = extract_text_without_header_footer(pdf_file)
    print(extracted_content)