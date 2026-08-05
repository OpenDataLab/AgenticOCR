import re
import os
import uuid
import json
import time
import mimetypes
import base64
import logging
from math import cos, sin, radians
from typing import Any, List, Tuple

from PIL import Image, ImageOps
from mineru_vl_utils import MinerUClient

from src.utils.common import smart_resize


logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def local_image_to_data_url(path: str) -> str:
    """ Base64 Data URL"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


class ImageZoomOCRTool:
    description = 'Zoom in on a specific region of an image by cropping it based on a bounding box (bbox), optionally rotate it or perform OCR.'

    def __init__(self,work_dir ,mineru_server_url="http://localhost:8000/",mineru_model_path="/root/checkpoints/MinerU2.5-2509-1.2B/"):
        self.mineru_model_path = mineru_model_path
        self.mineru_server_url = mineru_server_url
        self.mineru_client = None
        self.work_dir=work_dir

    def _get_mineru_client(self):
        return MinerUClient(
            model_name=self.mineru_model_path,
            backend="http-client",
            server_url=self.mineru_server_url.rstrip("/"),
        )

    def _smart_resize(self, height: int, width: int, **kwargs) -> Tuple[int, int]:
        return smart_resize(height, width, **kwargs)

    def map_point_back(self, x, y, 
                       final_size: Tuple[int, int], 
                       rotated_size: Tuple[int, int], 
                       crop_size: Tuple[int, int], 
                       crop_offset: Tuple[int, int], 
                       rotation_angle: float,
                       original_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        MinerU(final_image0-1000)
        (original_image)0-1000
        """
        # 1. Convert relative (0-1000) on final image to absolute pixels on final image
        abs_x = x / 1000.0 * final_size[0]
        abs_y = y / 1000.0 * final_size[1]

        # 2. Undo Resize (Mapping from final_size back to rotated_size)
        scale_x = final_size[0] / rotated_size[0]
        scale_y = final_size[1] / rotated_size[1]
        
        abs_x = abs_x / scale_x
        abs_y = abs_y / scale_y

        # 3. Undo Rotation (Mapping from rotated_size back to crop_size)
        # 
        cx_rot, cy_rot = rotated_size[0] / 2.0, rotated_size[1] / 2.0
        cx_crop, cy_crop = crop_size[0] / 2.0, crop_size[1] / 2.0

        # 
        dx = abs_x - cx_rot
        dy = abs_y - cy_rot

        #  (PIL rotate is counter-clockwise, so inverse is clockwise / negative angle)
        # : x' = x cos(-θ) - y sin(-θ)
        #       y' = x sin(-θ) + y cos(-θ)
        rad = radians(rotation_angle)
        cos_a = cos(rad)
        sin_a = sin(rad)

        rot_x = dx * cos_a - dy * sin_a
        rot_y = dx * sin_a + dy * cos_a

        #  Crop 
        orig_crop_x = rot_x + cx_crop
        orig_crop_y = rot_y + cy_crop

        # 4. Undo Crop (Add offset)
        final_abs_x = orig_crop_x + crop_offset[0]
        final_abs_y = orig_crop_y + crop_offset[1]

        # 5. Normalize to 0-1000 relative to original image
        norm_x = min(1000, max(0, int(final_abs_x / original_size[0] * 1000)))
        norm_y = min(1000, max(0, int(final_abs_y / original_size[1] * 1000)))

        return norm_x, norm_y

    def safe_crop_bbox(self, left, top, right, bottom, img_width, img_height):
        """Only clamp bbox to image bounds, without resizing or expanding it."""
        left = max(0, min(left, img_width))
        top = max(0, min(top, img_height))
        right = max(0, min(right, img_width))
        bottom = max(0, min(bottom, img_height))
        # Ensure valid order
        if left >= right:
            right = left + 1
        if top >= bottom:
            bottom = top + 1
        # Clamp again in case of degenerate
        right = min(right, img_width)
        bottom = min(bottom, img_height)
        return [left, top, right, bottom]

    def call(self, tool_call: dict, image_path) -> List[Any]:
        try:
            params = tool_call["arguments"]
            bbox = params['bbox']
            angle = params.get('angle', 0)
            type_ = params.get('type', "region")
        except Exception:
            logger.exception("Invalid tool call params: %s", tool_call)
            return [False, f'Error: Invalid tool_call params']

        os.makedirs(self.work_dir, exist_ok=True)

        try:
            image = Image.open(image_path)
        except Exception:
            logger.exception("Invalid input image: %s", image_path)
            return [False, f'Error: Invalid input image']

        try:
            # 1. Convert relative bbox (0-1000) to absolute pixels
            img_width, img_height = image.size
            rel_x1, rel_y1, rel_x2, rel_y2 = bbox
            abs_x1 = rel_x1 / 1000.0 * img_width
            abs_y1 = rel_y1 / 1000.0 * img_height
            abs_x2 = rel_x2 / 1000.0 * img_width
            abs_y2 = rel_y2 / 1000.0 * img_height

            # 2. ONLY clamp to image bounds — no resizing or padding!
            validated_bbox = self.safe_crop_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)
            left, top, right, bottom = validated_bbox

            # Record crop info for coordinate mapping
            crop_offset = (left, top)
            crop_size = (right - left, bottom - top)

            # 3. Crop the image (even if very small)
            cropped_image = image.crop((left, top, right, bottom))

            # 4. Rotate the image
            rotated_image = cropped_image.rotate(angle, expand=True)
            rotated_size = rotated_image.size

            new_h, new_w = self._smart_resize(
                height=rotated_size[1],
                width=rotated_size[0],
                factor=32,
                min_pixels=32 * 32,
                max_pixels=12845056,
            )
            final_image = rotated_image.resize((new_w, new_h), resample=Image.BICUBIC)
            final_size = final_image.size

            # Save processed image
            output_filename = f'{uuid.uuid4()}.png'
            output_path = os.path.abspath(os.path.join(self.work_dir, output_filename))
            final_image.save(output_path)

            if type_ == "image":
                return [True,output_path]

            # 6. OCR with MinerU
            mineru_result = []
            ocr_text_output = ""

            if type_ == "region":
                client = self._get_mineru_client()
                img_for_ocr = Image.open(output_path).convert("RGB")

                max_retries = 3
                raw_mineru_result = None
                for attempt in range(max_retries):
                    try:
                        raw_mineru_result = client.two_step_extract(img_for_ocr)
                        if raw_mineru_result:
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.error(f"MinerU failed after {max_retries} attempts: {e}")
                        time.sleep(1)

                if raw_mineru_result:
                    transformed_results = []
                    for item in raw_mineru_result:
                        raw_bbox = item.get('bbox', [])
                        if not raw_bbox or len(raw_bbox) != 4:
                            continue

                        # Normalize MinerU bbox to 0-1000 on final_image
                        if all(0 <= x <= 1.0 for x in raw_bbox):
                            # Assume [x1, y1, x2, y2] in 0-1
                            x1, y1, x2, y2 = [c * 1000 for c in raw_bbox]
                        else:
                            # Assume already in 0-1000
                            x1, y1, x2, y2 = raw_bbox

                        # Map back to original image coordinates
                        orig_x1, orig_y1 = self.map_point_back(
                            x1, y1, final_size, rotated_size, crop_size, crop_offset, angle, image.size
                        )
                        orig_x2, orig_y2 = self.map_point_back(
                            x2, y2, final_size, rotated_size, crop_size, crop_offset, angle, image.size
                        )

                        new_item = item.copy()
                        new_item['bbox'] = [
                            min(orig_x1, orig_x2),
                            min(orig_y1, orig_y2),
                            max(orig_x1, orig_x2),
                            max(orig_y1, orig_y2),
                            
                        ]
                        new_item['angle'] = (new_item['angle'] + angle) % 360
                        transformed_results.append(new_item)

                    mineru_result = transformed_results
                    ocr_text_output = json.dumps(mineru_result, ensure_ascii=False)
                else:
                    ocr_text_output = "MinerU returned empty result."

                return [
                    True,
                    output_path,
                    f"Region OCR Result (Mapped to original coords): {ocr_text_output}"
                ]
            else:
                client = self._get_mineru_client()
                img_for_ocr = Image.open(output_path).convert("RGB")

                max_retries = 3
                raw_mineru_result = None
                for attempt in range(max_retries):
                    try:
                        raw_mineru_result = client.content_extract(image=img_for_ocr,type=type_)
                        if raw_mineru_result:
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.error(f"MinerU failed after {max_retries} attempts: {e}")
                        time.sleep(1)

                ocr_text_output = raw_mineru_result

                return [
                    True,
                    output_path,
                    f"{type_.capitalize()} OCR Result: {ocr_text_output}"
                ]
        except Exception as e:
            obs = f'Tool Execution Error: {str(e)}'
            return [False,obs]


class MinerUBboxExtractor:
    def __init__(self, mineru_server_url: str = "http://localhost:8000/", 
                 mineru_model_path: str = "/root/checkpoints/MinerU2.5-2509-1.2B/"):
        self.mineru_server_url = mineru_server_url
        self.mineru_model_path = mineru_model_path

    def _get_mineru_client(self) -> MinerUClient:
        return MinerUClient(
            model_name=self.mineru_model_path,
            backend="http-client",
            server_url=self.mineru_server_url.rstrip('/')
        )

    def _smart_resize(self, height: int, width: int, **kwargs) -> Tuple[int, int]:
        return smart_resize(height, width, **kwargs)

    def extract_content_str(self, image_path: str, bbox: List[float], padding_ratio: float = 0.15) -> str:
        """
        Extracts content from a BBox, crops in memory, adds padding, and returns concatenated text.
        
        Args:
            image_path (str): Path to the original image.
            bbox (List[float]): [x1, y1, x2, y2] in relative coordinates (0-1000).
            padding_ratio (float): Padding ratio to fix edge recognition issues.

        Returns:
            str: The concatenated content text (joined by newlines). Returns empty string on failure.
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Error: Image file not found at {image_path}")
                return ""
            
            original_image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = original_image.size

            # 1. Coordinate Conversion (Relative 0-1000 -> Absolute)
            rel_x1, rel_y1, rel_x2, rel_y2 = bbox
            
            x1 = min(rel_x1, rel_x2)
            y1 = min(rel_y1, rel_y2)
            x2 = max(rel_x1, rel_x2)
            y2 = max(rel_y1, rel_y2)

            abs_x1 = int((x1 / 1000.0) * orig_w)
            abs_y1 = int((y1 / 1000.0) * orig_h)
            abs_x2 = int((x2 / 1000.0) * orig_w)
            abs_y2 = int((y2 / 1000.0) * orig_h)

            # Clamp to bounds
            abs_x1 = max(0, abs_x1)
            abs_y1 = max(0, abs_y1)
            abs_x2 = min(orig_w, abs_x2)
            abs_y2 = min(orig_h, abs_y2)

            crop_w = abs_x2 - abs_x1
            crop_h = abs_y2 - abs_y1

            if crop_w <= 0 or crop_h <= 0:
                logger.error("Error: Invalid bbox dimensions resulting in 0 size crop.")
                return ""

            # 2. Crop Image (In Memory)
            cropped_image = original_image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
            
            # 3. Add White Padding (Crucial for edge recognition)
            pad_size_x = int(crop_w * padding_ratio)
            pad_size_y = int(crop_h * padding_ratio)
            
            # Fill with white background
            padded_image = ImageOps.expand(
                cropped_image, 
                border=(pad_size_x, pad_size_y, pad_size_x, pad_size_y), 
                fill='white'
            )
            
            # 4. Smart Resize
            new_h, new_w = self._smart_resize(padded_image.height, padded_image.width)
            final_image = padded_image.resize((new_w, new_h), resample=Image.BICUBIC)

            # 5. MinerU Inference
            client = self._get_mineru_client()
            
            max_retries = 3
            raw_result = None
            
            for attempt in range(max_retries):
                try:
                    # Pass the PIL object directly, no disk saving
                    raw_result = client.two_step_extract(final_image)
                    if raw_result:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"MinerU extraction failed: {e}")
                    time.sleep(1)

            # 6. Extract and Concatenate Content
            extracted_texts = []
            if raw_result:
                for item in raw_result:
                    # MinerU returns blocks with 'content'
                    content = item.get('content', '')
                    if content is None:
                        content = ''
                    content = content.strip()
                    if content:
                        extracted_texts.append(content)
                
                # Join with newlines to separate distinct text blocks or formulas
                return "\n".join(extracted_texts)
            else:
                return ""

        except Exception:
            logger.exception("Processing error for image: %s", image_path)
            return ""

# Usage Example
if __name__ == "__main__":
    extractor = MinerUBboxExtractor()
    
    # 
    img_path = "/path/to/test_image.png" 
    target_bbox = [
        93,
        165,
        914,
        218
    ]
    
    # 
    content_str = extractor.extract_content_str(img_path, target_bbox)
    
    logger.info("-----  -----")
    logger.info(content_str)
    logger.info("-------------------")
