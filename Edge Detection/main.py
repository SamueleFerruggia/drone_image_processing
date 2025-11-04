#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PowerLine Image Processor for YOLO Training
Enhanced .TIF support for macOS/PyCharm

Handles various .TIF formats including 16-bit, multi-channel, and compressed TIF files
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import sys
from typing import Tuple, Optional, List
import warnings

# Import additional libraries for robust TIF support
try:
    import tifffile

    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    warnings.warn("tifffile not installed. Some .TIF formats may not be supported.")

try:
    from skimage import io as skio

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not installed. Fallback TIF reading may be limited.")

try:
    import imageio.v3 as imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    warnings.warn("imageio not installed. Alternative TIF reading not available.")


class PowerLineImageProcessor:
    def __init__(self, input_folder: str, output_base_folder: str = "processed_images"):
        """
        Initialize the PowerLine Image Processor with enhanced TIF support

        Args:
            input_folder (str): Path to folder containing input images
            output_base_folder (str): Base path for output folders
        """
        self.input_folder = Path(input_folder)
        self.output_base_folder = Path(output_base_folder)
        self.create_output_folders()
        self.check_tif_support()

    def check_tif_support(self) -> None:
        """Check available libraries for TIF support"""
        print("🔍 Checking TIF support libraries:")
        print(f"   tifffile: {'✅ Available' if TIFFFILE_AVAILABLE else '❌ Not installed'}")
        print(f"   scikit-image: {'✅ Available' if SKIMAGE_AVAILABLE else '❌ Not installed'}")
        print(f"   imageio: {'✅ Available' if IMAGEIO_AVAILABLE else '❌ Not installed'}")
        print()

    def load_tif_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load TIF images using multiple fallback methods

        Args:
            image_path (Path): Path to TIF image

        Returns:
            Optional[np.ndarray]: Loaded image array or None if failed
        """
        print(f"   📂 Loading TIF: {image_path.name}")

        # Method 1: Try OpenCV with IMREAD_UNCHANGED flag (handles most TIFs)
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                print(f"   ✅ Loaded with OpenCV: shape={image.shape}, dtype={image.dtype}")
                return self.normalize_tif_image(image)
        except Exception as e:
            print(f"   ⚠️ OpenCV failed: {str(e)}")

        # Method 2: Try tifffile (best for complex TIF formats)
        if TIFFFILE_AVAILABLE:
            try:
                image = tifffile.imread(str(image_path))
                if image is not None:
                    print(f"   ✅ Loaded with tifffile: shape={image.shape}, dtype={image.dtype}")
                    return self.normalize_tif_image(image)
            except Exception as e:
                print(f"   ⚠️ tifffile failed: {str(e)}")

        # Method 3: Try scikit-image with tifffile plugin
        if SKIMAGE_AVAILABLE:
            try:
                image = skio.imread(str(image_path), plugin='tifffile' if TIFFFILE_AVAILABLE else None)
                if image is not None:
                    print(f"   ✅ Loaded with scikit-image: shape={image.shape}, dtype={image.dtype}")
                    return self.normalize_tif_image(image)
            except Exception as e:
                print(f"   ⚠️ scikit-image failed: {str(e)}")

        # Method 4: Try imageio
        if IMAGEIO_AVAILABLE:
            try:
                image = imageio.imread(str(image_path))
                if image is not None:
                    print(f"   ✅ Loaded with imageio: shape={image.shape}, dtype={image.dtype}")
                    return self.normalize_tif_image(image)
            except Exception as e:
                print(f"   ⚠️ imageio failed: {str(e)}")

        print(f"   ❌ All methods failed to load {image_path.name}")
        return None

    def normalize_tif_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize TIF images to 8-bit BGR format for OpenCV processing

        Args:
            image (np.ndarray): Input TIF image array

        Returns:
            np.ndarray: Normalized BGR image
        """
        # Handle different bit depths
        if image.dtype == np.uint16:
            # Convert 16-bit to 8-bit
            image = (image / 256).astype(np.uint8)
            print(f"   🔄 Converted 16-bit to 8-bit")
        elif image.dtype == np.float32 or image.dtype == np.float64:
            # Normalize float images to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            print(f"   🔄 Converted float to 8-bit")
        elif image.dtype != np.uint8:
            # Convert any other format to uint8
            image = cv2.convertScaleAbs(image)
            print(f"   🔄 Converted {image.dtype} to 8-bit")

        # Handle different channel configurations
        if len(image.shape) == 2:
            # Grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            print(f"   🔄 Converted grayscale to BGR")
        elif len(image.shape) == 3:
            if image.shape[2] == 1:
                # Single channel to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                print(f"   🔄 Converted single channel to BGR")
            elif image.shape[2] == 4:
                # RGBA to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                print(f"   🔄 Converted RGBA to BGR")
            elif image.shape[2] > 4:
                # Multi-channel: take first 3 channels
                image = image[:, :, :3]
                print(f"   🔄 Used first 3 channels from {image.shape[2]}-channel image")

        return image

    def create_output_folders(self) -> None:
        """Create output folders for each filter stage"""
        self.filter_folders = [
            "OUTPUT_0_TIFF_TO_JPG",
            "FILTER_1_WHITE_MASK",
            "FILTER_2_MORPHOLOGY",
            "FILTER_3_GAUSSIAN_BLUR",
            "FILTER_4_EDGE_DETECTION",
            "FILTER_5_LINE_ENHANCEMENT",
            "FILTER_6_FINAL_EDGES"
        ]

        for folder in self.filter_folders:
            folder_path = self.output_base_folder / folder
            folder_path.mkdir(parents=True, exist_ok=True)

    def filter_1_white_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter 1: Extract white regions (power lines)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Adjusted for TIF images which might have different dynamic ranges
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 25, 255])

        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        result = cv2.bitwise_and(image, image, mask=white_mask)

        return result, white_mask

    def filter_2_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Filter 2: Morphological operations to enhance thin lines"""
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        kernel_diagonal1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_small = np.ones((3, 3), np.uint8)

        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        horizontal_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_horizontal, iterations=1)
        vertical_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_vertical, iterations=1)
        diagonal_lines = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_diagonal1, iterations=1)

        enhanced_mask = cv2.addWeighted(horizontal_lines, 0.4, vertical_lines, 0.4, 0)
        enhanced_mask = cv2.addWeighted(enhanced_mask, 0.8, diagonal_lines, 0.2, 0)

        final_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        return final_mask

    def filter_3_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Filter 3: Apply Gaussian blur to reduce noise"""
        blurred = cv2.GaussianBlur(image, (3, 3), 0.8)
        return blurred

    def filter_4_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Filter 4: Canny edge detection optimized for thin lines"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        edges = cv2.Canny(gray, 25, 75, apertureSize=3, L2gradient=True)
        return edges

    def filter_5_line_enhancement(self, edges: np.ndarray) -> np.ndarray:
        """Filter 5: Enhance detected lines using HoughLines"""
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=25, maxLineGap=8)

        enhanced_lines = np.zeros_like(edges)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_length > 20:
                    cv2.line(enhanced_lines, (x1, y1), (x2, y2), 255, 2)

        return enhanced_lines

    def filter_6_final_edge_refinement(self, enhanced_lines: np.ndarray,
                                       original_edges: np.ndarray) -> np.ndarray:
        """Filter 6: Combine enhanced lines with original edges"""
        final_edges = cv2.addWeighted(enhanced_lines, 0.7, original_edges, 0.3, 0)
        kernel = np.ones((2, 2), np.uint8)
        final_result = cv2.morphologyEx(final_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        return final_result

    def save_image(self, image: np.ndarray, folder_name: str, filename: str) -> bool:
        """Save image to specified folder"""
        try:
            folder_path = self.output_base_folder / folder_name
            file_path = folder_path / filename
            folder_path.mkdir(parents=True, exist_ok=True)

            success = cv2.imwrite(str(file_path), image)
            if success:
                print(f"✅ Saved: {file_path}")
                return True
            else:
                print(f"❌ Failed to save: {file_path}")
                return False
        except Exception as e:
            print(f"❌ Error saving {filename}: {str(e)}")
            return False

    def process_single_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Process a single image through all filter stages"""
        print(f"🔄 Processing: {image_path.name}")

        # Load image with TIF support
        original = self.load_tif_image(image_path)
        if original is None:
            print(f"❌ Error: Could not load image {image_path}")
            return None

        base_name = image_path.stem

        # Step 0: Save original as JPG (OUTPUT_0) for reference
        print("   📍 Output 0: TIFF converted in JPG")
        self.save_image(original, "OUTPUT_0_TIFF_TO_JPG", f"{base_name}_original.jpg")

        try:
            # Apply all filters
            print("   📍 Filter 1: White mask...")
            filtered1, white_mask = self.filter_1_white_mask(original)
            self.save_image(filtered1, "FILTER_1_WHITE_MASK", f"{base_name}_white_mask.jpg")

            print("   📍 Filter 2: Morphological operations...")
            enhanced_mask = self.filter_2_morphology(white_mask)
            filtered2 = cv2.bitwise_and(original, original, mask=enhanced_mask)
            self.save_image(filtered2, "FILTER_2_MORPHOLOGY", f"{base_name}_morphology.jpg")

            print("   📍 Filter 3: Gaussian blur...")
            filtered3 = self.filter_3_gaussian_blur(filtered2)
            self.save_image(filtered3, "FILTER_3_GAUSSIAN_BLUR", f"{base_name}_blurred.jpg")

            print("   📍 Filter 4: Edge detection...")
            edges = self.filter_4_edge_detection(filtered3)
            self.save_image(edges, "FILTER_4_EDGE_DETECTION", f"{base_name}_edges.jpg")

            print("   📍 Filter 5: Line enhancement...")
            enhanced_lines = self.filter_5_line_enhancement(edges)
            self.save_image(enhanced_lines, "FILTER_5_LINE_ENHANCEMENT", f"{base_name}_enhanced.jpg")

            print("   📍 Filter 6: Final refinement...")
            final_result = self.filter_6_final_edge_refinement(enhanced_lines, edges)
            self.save_image(final_result, "FILTER_6_FINAL_EDGES", f"{base_name}_final.jpg")

            print(f"   ✅ Completed: {image_path.name}\n")
            return final_result

        except Exception as e:
            print(f"   ❌ Error processing {image_path.name}: {str(e)}\n")
            return None

    def process_all_images(self, image_extensions: Tuple[str, ...] = ('.tif', '.tiff', '.TIF', '.TIFF', '.jpg', '.jpeg',
                                                                      '.png', '.bmp')) -> None:
        """Process all images in the input folder with enhanced TIF support"""
        if not self.input_folder.exists():
            print(f"❌ Input folder does not exist: {self.input_folder}")
            return

        all_images = []
        for extension in image_extensions:
            all_images.extend(list(self.input_folder.glob(f"*{extension}")))

        if not all_images:
            print(f"❌ No images found in {self.input_folder}")
            print(f"   Supported formats: {', '.join(image_extensions)}")
            return

        print(f"🎯 Found {len(all_images)} images to process")
        print("=" * 60)

        successful_count = 0
        for i, image_path in enumerate(all_images, 1):
            print(f"[{i}/{len(all_images)}]")
            result = self.process_single_image(image_path)
            if result is not None:
                successful_count += 1

        print("=" * 60)
        print(f"🎉 Processing completed!")
        print(f"   Successfully processed: {successful_count}/{len(all_images)} images")
        print(f"   Results saved in: {self.output_base_folder.absolute()}")


def setup_opencv_macos():
    """Setup OpenCV for optimal performance on macOS"""
    cv2.setUseOptimized(True)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"OpenCV optimizations enabled: {cv2.useOptimized()}")


def main():
    """Main execution function"""
    print("🚀 PowerLine Image Processor with Enhanced TIF Support")
    print("=" * 60)

    setup_opencv_macos()

    INPUT_FOLDER = "immagini-IPCV/train"
    OUTPUT_FOLDER = "output/train"

    try:
        processor = PowerLineImageProcessor(INPUT_FOLDER, OUTPUT_FOLDER)
        processor.process_all_images()

    except Exception as e:
        print(f"❌ Error initializing processor: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
