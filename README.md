# Indian Number Plate Recognition (ANPR)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is a powerful, Python-based Automatic Number Plate Recognition (ANPR) system specifically tailored for Indian license plates. It uses a multi-stage computer vision pipeline to detect, read, and intelligently correct vehicle registration numbers from images, even under challenging conditions.

## Demonstration: Before and After

The system can take a complex, real-world image and isolate the number plate for accurate reading.

<table>
  <tr>
    <td align="center"><strong>Original Image (`Images/1.jpeg`)</strong></td>
    <td align="center"><strong>Processed Plate for OCR (`segmented_plates/1_plate.png`)</strong></td>
  </tr>
  <tr>
    <td><img src="Images/1.jpeg" alt="Original Car Image" width="400"></td>
    <td><img src="segmented_plates/1_plate.png" alt="Segmented Plate" width="400"></td>
  </tr>
</table>

## Core Features

-   **Robust Plate Detection:** Accurately locates number plates in complex images.
-   **Rotation Invariant:** Successfully detects plates that are rotated at 0, 90, 180, or 270 degrees.
-   **Multi-Color Support:** Includes a specialized color segmentation pipeline to effectively detect yellow commercial plates in addition to standard white plates.
-   **Advanced Image Pre-processing:** Applies a "smooth and sharpen" filter pipeline to significantly enhance image quality for OCR, increasing accuracy.
-   **High-Accuracy OCR:** Utilizes the **EasyOCR** deep learning library, which consistently outperforms traditional OCR engines.
-   **Intelligent Text Correction:** Implements a rule-based post-processing validator that understands the official Indian RTO format (e.g., `LLDDLLNNNN`). It automatically corrects common OCR errors (like `I` -> `1`, `G` -> `6`, `HH` -> `MH`) based on character similarity and position.
-   **Detailed Output:** Saves the segmented plate image (white text on a black background) for verification and generates a clean `detected_number_plates.txt` file with the final results.
-   **Performance Tracking:** Includes counters to provide a final summary of successfully processed and failed images.

## How It Works

The script follows a sophisticated, multi-pass pipeline to ensure high accuracy:

1.  **Rotation Handling:** The system first loads an image and iterates through four rotations (0°, 90°, 180°, 270°), stopping as soon as a valid plate is found.
2.  **Plate Localization (Dual Approach):**
    *   **Color Segmentation:** It first converts the image to the HSV color space to specifically look for yellow-colored regions, making it highly effective for commercial vehicle plates.
    *   **Edge Detection Fallback:** If no yellow plate is found, it falls back to a robust grayscale pipeline using a bilateral filter and Canny edge detection to find rectangular contours.
3.  **Image Pre-processing:** Once a plate is located, the cropped region undergoes an advanced cleaning process:
    *   The image is upscaled to increase pixel density.
    *   A bilateral filter is applied to smooth noise while preserving sharp character edges.
    *   A sharpening kernel is used to make the characters highly distinct.
    *   Otsu's thresholding creates a crisp, high-contrast binary image (black text on white background).
4.  **OCR with EasyOCR:** The cleaned image is **inverted** (to white text on a black background, which is optimal for EasyOCR) and passed to the OCR engine. A character whitelist (`A-Z`, `0-9`) ensures the engine only outputs valid characters.
5.  **Intelligent Post-Processing:** The raw OCR text is passed to a custom validator that:
    *   Enforces the standard plate structure by correcting character *types* based on their position.
    *   Validates the two-letter state code against an official list and attempts to fix common misreadings (e.g., `HH` -> `MH`).

## Requirements

-   Python 3.8 or newer
-   The Python libraries listed in `requirements.txt`.

## Installation

1.  **Clone or download the repository:**
    ```bash
    git clone https://your-repo-url.git
    cd your-repo-folder-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    ```bash
    # On Windows
    .\venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:** Create a file named `requirements.txt` with the content below and run the `pip` command.

    **`requirements.txt`:**
    ```text
    opencv-python
    easyocr
    numpy
    ```

    **Installation command:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Place your images:** Create a folder named `Images` in the project's root directory. Place all the `.jpeg`, `.jpg`, or `.png` images you want to process inside this folder.

2.  **Run the script:** Open your terminal in the project directory and run:
    ```bash
    python your_script_name.py
    ```

3.  **Check the output:** The script will generate two outputs:
    -   A folder named `segmented_plates`, which will contain the cropped, processed images (white text on black background) of the detected number plates.
    -   A file named `detected_number_plates.txt`, which will contain the final, corrected list of all recognized number plates.

### Example Terminal Output

```
Loading EasyOCR model into memory... (This may take a moment)

HELLO!!
Starting the Number Plate Detection System.

Processing image: car_rotated_90.jpeg...
  -> Raw OCR: 'HH140T8831' | Corrected: 'MH14DT8831' (found at 90 degrees)
  -> SUCCESS: Stored Number Plate: MH14DT8831
Processing image: truck_yellow_plate.jpeg...
  -> Raw OCR: 'RJ11GB8850' | Corrected: 'RJ118850'
  -> SUCCESS: Stored Number Plate: RJ118850

--- Processing Summary ---
Total images found:    2
Successfully processed: 2
Failed to process:      0
--------------------------

--- Processing Complete ---
All valid vehicle numbers have been saved to: detected_number_plates.txt
 - MH14DT8831
 - RJ118850
```

## License

This project is licensed under the MIT License.
