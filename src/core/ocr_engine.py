import pymupdf
import numpy as np
import re
import PIL
import paddle
import gc
from pathlib import Path
from paddleocr import PaddleOCR
from typing import Union, Any
from loguru import logger

class OCREngine:

    def __init__(
        self,
        use_doc_orientation_classify: bool = True,
        use_doc_unwarping: bool = True, 
        use_textline_orientation: bool = True,
        device: str = "gpu",
        precision: str = "fp16",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        textline_orientation_batch_size=4,
        text_recognition_batch_size=4,
        text_det_limit_side_len=960,
        text_det_limit_type="max",
        text_det_box_thresh=0.5,
        text_rec_score_thresh=0.3,
        post_processing_config={"y_threshold": 10, "column_threshold": 0.3}
    ):
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.device = device
        self.precision = precision
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.textline_orientation_batch_size = textline_orientation_batch_size
        self.text_recognition_batch_size = text_recognition_batch_size
        self.text_det_limit_side_len = text_det_limit_side_len
        self.text_det_limit_type = text_det_limit_type
        self.text_det_box_thresh = text_det_box_thresh
        self.text_rec_score_thresh = text_rec_score_thresh
        self.post_processing_config = post_processing_config

        try:
            logger.info(f"Initializing OCR pipeline with device={device}, precision={precision}")
            self.pipeline = PaddleOCR(
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping, 
                use_textline_orientation=self.use_textline_orientation,
                device=self.device,
                precision=self.precision,
                text_detection_model_name=self.text_detection_model_name,
                text_recognition_model_name=self.text_recognition_model_name,
                textline_orientation_batch_size=self.textline_orientation_batch_size,
                text_recognition_batch_size=self.text_recognition_batch_size,
                text_det_limit_side_len=self.text_det_limit_side_len,
                text_det_limit_type=self.text_det_limit_type,
                text_det_box_thresh=self.text_det_box_thresh,
                text_rec_score_thresh=self.text_rec_score_thresh,
            )
            logger.info("OCR pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OCR pipeline: {str(e)}")
            raise e

    def _detect_layout(self, bboxes, page_width, column_threshold=0.3):
        """
        Detect if layout is single or multi-column
        
        Args:
            bboxes: numpy array of shape (n, 4) with [x_min, y_min, x_max, y_max]
            page_width: x maximum of bboxes
            column_threshold: Controls detection sensitivity (0.0-1.0)
                - Lower (0.1-0.3): More sensitive, detects columns easier
                - Higher (0.4-0.6): Less sensitive, requires stronger column evidence
        
        Strategy: Check if boxes are concentrated in left/right regions
        """
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        widths = bboxes[:, 2] - bboxes[:, 0]
        
        # Count boxes in left half vs right half
        left_half = np.sum(centers_x < page_width / 2)
        right_half = np.sum(centers_x >= page_width / 2)
        total = len(bboxes)
        
        # Calculate distribution ratios
        left_ratio = left_half / total
        right_ratio = right_half / total
        
        # Count narrow boxes (likely in columns)
        narrow_boxes = np.sum(widths < page_width * 0.6)
        narrow_ratio = narrow_boxes / total
        
        # Use column_threshold to determine sensitivity
        min_side_ratio = column_threshold  # Minimum boxes needed on each side
        min_narrow_ratio = 0.5 + column_threshold  # Threshold for narrow boxes
        
        # Multi-column if:
        # 1. Most boxes are narrow (not spanning full width)
        # 2. Boxes are distributed on both left and right sides
        if (narrow_ratio > min_narrow_ratio and 
            left_ratio > min_side_ratio and 
            right_ratio > min_side_ratio):
            return 'multi'
        else:
            return 'single'

    def _sort_single_column(self, texts, bboxes, y_threshold=10):
        """
        Sort for single column layout (top to bottom, left to right within rows)
        """
        items = list(zip(texts, bboxes, range(len(texts))))
        
        # Sort by y_min first
        items.sort(key=lambda x: x[1][1])
        
        # Group into rows
        rows = []
        current_row = [items[0]]
        
        for item in items[1:]:
            avg_y = np.mean([i[1][1] for i in current_row])
            if abs(item[1][1] - avg_y) <= y_threshold:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]
        rows.append(current_row)
        
        # Sort each row by x_min (left to right)
        sorted_items = []
        for row in rows:
            row.sort(key=lambda x: x[1][0])
            sorted_items.extend(row)
        
        sorted_texts = [item[0] for item in sorted_items]
        sorted_bboxes = np.array([item[1] for item in sorted_items])
        sorted_indices = [item[2] for item in sorted_items]
        
        return sorted_texts, sorted_bboxes, sorted_indices

    def _sort_multi_column_research(self, texts, bboxes, page_width, y_threshold=10):
        """
        Sort for research paper layout (handles mixed single/multi-column sections)
        
        Strategy:
        1. Divide page into vertical sections
        2. Detect if section is full-width or multi-column
        3. Sort accordingly
        """
        items = list(zip(texts, bboxes, range(len(texts))))
        
        # Group by vertical position (y-coordinate)
        items.sort(key=lambda x: x[1][1])
        
        # Find column boundary (usually around middle)
        column_boundary = page_width / 2
        
        # Segment document into blocks
        blocks = self._segment_into_blocks(items, page_width, y_threshold)
        
        sorted_items = []
        for block in blocks:
            if self._is_full_width_block(block, page_width):
                # Full-width block (title, abstract, etc.)
                block.sort(key=lambda x: (x[1][1], x[1][0]))  # Top to bottom, left to right
                sorted_items.extend(block)
            else:
                # Multi-column block
                sorted_block = self._sort_two_column_block(block, column_boundary, y_threshold)
                sorted_items.extend(sorted_block)
        
        sorted_texts = [item[0] for item in sorted_items]
        sorted_bboxes = np.array([item[1] for item in sorted_items])
        sorted_indices = [item[2] for item in sorted_items]
        
        return sorted_texts, sorted_bboxes, sorted_indices

    def _segment_into_blocks(self, items, page_width, y_threshold=30):
        """
        Segment document into blocks (groups of text with similar layout)
        """
        if not items:
            return []
        
        blocks = []
        current_block = [items[0]]
        
        for item in items[1:]:
            prev_item = current_block[-1]
            y_gap = item[1][1] - prev_item[1][3]  # Vertical gap
            
            # If large vertical gap, start new block
            if y_gap > y_threshold:
                blocks.append(current_block)
                current_block = [item]
            else:
                current_block.append(item)
        
        blocks.append(current_block)
        return blocks

    def _is_full_width_block(self, block, page_width, width_threshold=0.7):
        """
        Determine if a block spans full width (like title, abstract)
        """
        widths = [item[1][2] - item[1][0] for item in block]
        avg_width = np.mean(widths)
        
        # If average width is > 70% of page width, consider it full-width
        return avg_width > page_width * width_threshold

    def _sort_two_column_block(self, block, column_boundary, y_threshold=10):
        """
        Sort a two-column block (read left column top-to-bottom, then right column)
        """
        # Separate into left and right columns
        left_column = [item for item in block if (item[1][0] + item[1][2]) / 2 < column_boundary]
        right_column = [item for item in block if (item[1][0] + item[1][2]) / 2 >= column_boundary]
        
        # Sort each column independently (top to bottom, left to right)
        def sort_column(column):
            column.sort(key=lambda x: x[1][1])  # Sort by y first
            
            # Group into rows within column
            rows = []
            if column:
                current_row = [column[0]]
                for item in column[1:]:
                    avg_y = np.mean([i[1][1] for i in current_row])
                    if abs(item[1][1] - avg_y) <= y_threshold:
                        current_row.append(item)
                    else:
                        rows.append(current_row)
                        current_row = [item]
                rows.append(current_row)
            
            # Sort each row by x
            sorted_col = []
            for row in rows:
                row.sort(key=lambda x: x[1][0])
                sorted_col.extend(row)
            return sorted_col
        
        left_sorted = sort_column(left_column)
        right_sorted = sort_column(right_column)
        
        # Combine: left column first, then right column
        return left_sorted + right_sorted

    def _sort_boxes_smart(
        self,
        texts: list,
        bboxes: list,
        mode='auto',
        y_threshold=10,
        column_threshold=0.3
    ):
        """
        Smart sorting for both single-column and multi-column layouts (like research papers)
        
        Args:
            texts: List of recognized text strings
            bboxes: numpy array of shape (n, 4) with [x_min, y_min, x_max, y_max]
            mode: 'auto', 'single', or 'multi'
                - 'auto': Automatically detect layout
                - 'single': Force single column reading order
                - 'multi': Force multi-column (research paper) layout
            y_threshold: Threshold to group boxes into same row (pixels)
            column_threshold: Ratio threshold to detect column boundaries (0.0-1.0)
        
        Returns:
            sorted_texts: List of texts in reading order
            sorted_bboxes: Array of bboxes in reading order
            sorted_indices: Original indices for reference
        """
        if len(texts) == 0:
            logger.warning("No texts to sort")
            return [], np.array([]), []
        
        # Convert to numpy array if needed
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)
        
        # Get page dimensions
        page_width = np.max(bboxes[:, 2])
        page_height = np.max(bboxes[:, 3])
        
        # Detect layout if mode is auto
        if mode == 'auto':
            mode = self._detect_layout(bboxes, page_width, column_threshold)
            logger.info(f"Layout detected: {mode}-column (page_width={page_width:.0f}px)")
        
        if mode == 'single':
            sorted_texts, sorted_bboxes, sorted_indices = self._sort_single_column(
                texts, bboxes, y_threshold
            )
        else:  # multi-column
            sorted_texts, sorted_bboxes, sorted_indices = self._sort_multi_column_research(
                texts, bboxes, page_width, y_threshold
            )
        
        logger.info(f"Sorted {len(texts)} text boxes in {mode}-column mode")
        return sorted_texts, sorted_bboxes, sorted_indices

    def _postprocess_text(self, text: str) -> str:
        """
        Post-process OCR text to fix common issues
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return text
        
        # Remove excessive whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Apply selective corrections (only for clearly wrong cases)
        text = text.replace('\x00', '')
                
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)  # Add space after punctuation
        
        # Fix hyphenation at line breaks (common in PDFs)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)  # "exam- ple" -> "example"
        
        # Trim leading/trailing whitespace
        text = text.strip()
        
        return text

    def run_ocr(
        self, 
        images: Union[
            str,
            np.ndarray, 
            PIL.Image.Image, 
            list[np.ndarray], 
            list[PIL.Image.Image],
            list[str]
        ],
        page_indices: list[int]
    ) -> list[dict[str, Any]]:
        """
        Run OCR on single image or list of images and return cleaned text in reading order
        
        Args:
            images: Input image(s) as:
                - Single image file path (str)
                - Single numpy array
                - Single PIL Image
                - List of numpy arrays
                - List of PIL Images
                - List of image file paths (str)

            page_indices: List of page indices corresponding to images
            
        Returns:
            List of strings, one per image/page
        """

        imgs_array = []
        results = []
        final_output = []

        try:
            # Normalize input to list
            if isinstance(images, (str, np.ndarray, PIL.Image.Image)):
                images = [images]
                logger.info("Single image provided, wrapping in list")
            
            if not images:
                logger.warning("Empty image list provided")
                return []
            
            logger.info(f"Processing {len(images)} image(s)")
            
            # Convert all images to numpy arrays
            for idx, image in enumerate(images):
                if isinstance(image, PIL.Image.Image):
                    imgs_array.append(np.array(image))
                elif isinstance(image, np.ndarray):
                    imgs_array.append(image)
                elif isinstance(image, str):
                    # Load image from file path
                    imgs_array.append(image)
                else:
                    raise ValueError(f"Image {idx} should be numpy array or PIL Image, got {type(image)}")
            
            logger.debug(f"Prepared {len(imgs_array)} images for OCR")
            
            # Run OCR on all images at once (batch processing)
            logger.info(f"Running batch OCR on {len(imgs_array)} images")

            results = self.pipeline.predict(imgs_array)
            
            # Process results for each image
            for idx, (result, page_idx) in enumerate(zip(results, page_indices)):
                texts = result.get("rec_texts", [])
                bboxes = result.get("rec_boxes", [])
                
                if len(texts) == 0:
                    logger.warning(f"No text detected in image {idx + 1}")
                    final_output.append({"page_index": page_idx, "text": ""})
                    continue
                
                logger.debug(f"Image {idx + 1}: detected {len(texts)} text regions")
                
                # Sort texts in reading order
                sorted_texts, _, _ = self._sort_boxes_smart(
                    texts=texts,
                    bboxes=bboxes,
                    mode='auto',
                    **self.post_processing_config
                )
                
                # Join and clean text
                full_text = " ".join(sorted_texts)
                cleaned_text = self._postprocess_text(full_text)
                output = {
                    "page_index": page_idx,
                    "text": cleaned_text
                }
                final_output.append(output)
                logger.info(f"Image {idx + 1}: extracted {len(cleaned_text)} characters")
            
            logger.info(f"OCR completed. Processed {len(final_output)} pages")
            return final_output
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
            raise

        finally:
            # delete cache
            logger.warning("Deleting cache memory")

            del imgs_array
            del results

            gc.collect()
            
            if "gpu" in self.device:
                paddle.device.cuda.empty_cache()
