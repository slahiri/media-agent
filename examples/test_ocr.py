#!/usr/bin/env python3
"""Simple test script for DeepSeek-OCR."""

from pathlib import Path


def test_ocr():
    """Test DeepSeek-OCR text extraction."""
    from media_utils import DeepSeekOCR

    # Initialize with 4-bit quantization for lower memory usage
    print("Initializing DeepSeek-OCR...")
    ocr = DeepSeekOCR(quantization="4bit")

    # Check for test image
    test_images = list(Path(".").glob("*.png")) + list(Path(".").glob("*.jpg"))
    if not test_images:
        print("No test images found. Please provide a PNG or JPG file.")
        print("Usage: Place an image file in the current directory and run again.")
        return

    image_path = test_images[0]
    print(f"Processing: {image_path}")

    # Extract text
    text = ocr.extract(
        image_path,
        mode="markdown",  # Convert to markdown format
        resolution="gundam",  # Recommended for documents
    )

    print("\n" + "=" * 50)
    print("Extracted Text:")
    print("=" * 50)
    print(text)

    # Cleanup
    ocr.unload()
    print("\nDone!")


if __name__ == "__main__":
    test_ocr()
