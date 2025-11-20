import math
import fitz
import base64
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path

class APAPsycTestDocument():

    def __init__(self, pdf_path, max_megapixels=None):

        self.pdf_path = pdf_path
        self.pdf = None
        self.image_count = 0
        self.image_rescale_factor = 1
        self.max_megapixels = max_megapixels

        # content
        self.cover_text = ""
        self.text = ""
        self.image = None
        self.images = []
        self.base4images = []

        with fitz.open(self.pdf_path) as pdf:
            self.pdf = pdf
            self._extract_content_as_text()
            self._count_images_in_pdf()

        self._extract_images_from_content()
        self._convert_images_to_base64()

    def _convert_images_to_base64(self):
        for image in self.images:
            buffered = BytesIO()
            image.save(buffered, format=image.format or "PNG")
            base64image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            self.base4images.append(base64image)

    def _extract_images_from_content(self):
        image_pages = convert_from_path(pdf_path=self.pdf_path, dpi=300, fmt="png")
        image_pages = image_pages if len(image_pages) == 1 else image_pages[1:]

        widths, heights = zip(*(page.size for page in image_pages))
        max_width = max(widths)
        total_height = sum(heights)

        single_image = Image.new('RGB', (max_width, total_height), color='white')
        y_offset = 0

        for page in image_pages:

            x_offset = (max_width - page.width) // 2
            single_image.paste(page, (x_offset, y_offset))
            y_offset += page.height

        if self.max_megapixels:
            self.image, self.image_rescale_factor = self._resize_bitmap_content(single_image)
            self.images = [self._resize_bitmap_content(x)[0] for x in image_pages]
        else:
            self.image = single_image
            self.images = image_pages

    def _resize_bitmap_content(self, image):
        megapixels = (image.size[0] * image.size[1] / 1e6)
        if megapixels > self.max_megapixels:
            rescale_factor = math.sqrt(self.max_megapixels / megapixels)
            new_size = [int(x * rescale_factor) for x in image.size]
            return image.resize(new_size, Image.LANCZOS), rescale_factor
        return image, 1.0

    def _extract_content_as_text(self):
        for page_num in range(len(self.pdf)):
            page = self.pdf.load_page(page_num)
            if page_num == 0:
                self.cover_text += page.get_text()
            else:
                self.text += page.get_text()

    def _count_images_in_pdf(self):
        for page_num in range(len(self.pdf)):
            page = self.pdf.load_page(page_num)
            image_list = page.get_images()
            self.image_count += len(image_list)