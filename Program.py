import cv2
import numpy as np
import easyocr
import os
import glob
import re

# --- Initialize the EasyOCR model once when the script starts ---
print("Loading EasyOCR model into memory... (This may take a moment)")
reader = easyocr.Reader(['en'], gpu=False)


def extract_number_plate(image_path, debug=False):
    """
    Locates the plate, performs advanced image cleaning, and extracts text using EasyOCR.
    """
    try:
        img = cv2.imread(image_path)
        if img is None: return None, None
    except Exception as e:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None: return None, None

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0, 255, -1)
    (x, y) = np.where(mask == 255)
    (x1, y1), (x2, y2) = (np.min(x), np.min(y)), (np.max(x), np.max(y))
    cropped_plate_gray = gray[x1:x2 + 1, y1:y2 + 1]

    upscaled_plate = cv2.resize(cropped_plate_gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    smoothed_plate = cv2.bilateralFilter(upscaled_plate, 9, 75, 75)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_plate = cv2.filter2D(smoothed_plate, -1, sharpen_kernel)
    _, final_plate_image = cv2.threshold(sharpened_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inverted_plate = cv2.bitwise_not(final_plate_image)

    try:
        char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        result = reader.readtext(inverted_plate, allowlist=char_whitelist)
        if not result: return None, inverted_plate

        detected_text = ' '.join([res[1] for res in result])
        cleaned_text = "".join(re.split(r'[^A-Z0-9]', detected_text)).upper()

        return cleaned_text, inverted_plate
    except Exception as e:
        return None, None


def post_process_and_validate(raw_text):
    """
    Applies a multi-pass correction and validation logic based on Indian license plate rules.
    """
    if not raw_text: return ""

    cleaned_text = "".join(re.split(r'[^A-Z0-9]', raw_text)).upper()
    if len(cleaned_text) != 10:
        return cleaned_text

    VALID_STATE_CODES = {
        "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL", "MP",
        "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS", "TR", "UP",
        "UK", "WB", "AN", "CH", "DN", "DD", "DL", "JK", "LA", "LD", "PY"
    }

    digit_to_letter = {'0': 'D', '1': 'I', '5': 'S', '6': 'G', '8': 'B', '4': 'A'}
    letter_to_digit = {'D': '0', 'I': '1', 'S': '5', 'G': '6', 'B': '8', 'O': '0', 'A': '4', 'L': '1', 'Z': '2'}
    letter_to_letter = {'H': 'M', 'N': 'M', 'B': 'R', 'Y': 'T', 'C': 'G', 'Q': 'O'}

    correct_pattern = "LLDDLLDDDD"

    structurally_corrected = list(cleaned_text)
    for i, char_type in enumerate(correct_pattern):
        char = structurally_corrected[i]
        if char_type == 'L' and char.isdigit():
            structurally_corrected[i] = digit_to_letter.get(char, char)
        elif char_type == 'D' and char.isalpha():
            structurally_corrected[i] = letter_to_digit.get(char, char)

    structurally_corrected_str = "".join(structurally_corrected)

    state_code = structurally_corrected_str[:2]
    if state_code in VALID_STATE_CODES:
        return structurally_corrected_str

    else:
        c1, c2 = state_code[0], state_code[1]

        c1_corr = letter_to_letter.get(c1, c1)
        if c1_corr + c2 in VALID_STATE_CODES:
            return c1_corr + structurally_corrected_str[2:]

        c2_corr = letter_to_letter.get(c2, c2)
        if c1 + c2_corr in VALID_STATE_CODES:
            return c1 + c2_corr + structurally_corrected_str[2:]

        if c1_corr + c2_corr in VALID_STATE_CODES:
            return c1_corr + c2_corr + structurally_corrected_str[2:]

    return structurally_corrected_str


# --- Main Execution Block ---
if __name__ == "__main__":
    print("\nHELLO!!")
    print("Starting the Number Plate Detection System.\n")

    VISUAL_DEBUG_MODE = False

    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_dir = os.path.join(dir_path, "Images")
    output_dir_segmented = os.path.join(dir_path, "segmented_plates")
    os.makedirs(output_dir_segmented, exist_ok=True)

    image_paths = glob.glob(os.path.join(image_dir, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(image_dir, "*.[jJ][pP][eE][gG]")) + \
                  glob.glob(os.path.join(image_dir, "*.[pP][nN][gG]"))

    # ###############################################################
    # ## NEW: Initialize counters                                  ##
    # ###############################################################
    total_images = len(image_paths)
    processed_count = 0
    failed_count = 0
    # ###############################################################

    detected_plates_text = []

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing image: {filename}...")

        raw_ocr_text, segmented_image = extract_number_plate(img_path, debug=VISUAL_DEBUG_MODE)

        if raw_ocr_text and segmented_image is not None:
            validated_text = post_process_and_validate(raw_ocr_text)

            print(f"  -> Raw OCR: '{raw_ocr_text}' | Corrected: '{validated_text}'")
            detected_plates_text.append(validated_text)
            print(f"  -> SUCCESS: Stored Number Plate: {validated_text}")

            output_path = os.path.join(output_dir_segmented, f"{os.path.splitext(filename)[0]}_plate.png")
            cv2.imwrite(output_path, segmented_image)

            # Increment the success counter
            processed_count += 1
        else:
            print("  -> FAILED: No valid number plate was detected or extracted.")
            # Increment the failure counter
            failed_count += 1

    # --- Save the final list to a text file ---
    output_file_path = os.path.join(dir_path, "detected_number_plates.txt")
    with open(output_file_path, "w") as f:
        for plate in detected_plates_text:
            f.write(f"{plate}\n")

    # ###############################################################
    # ## NEW: Print the final summary report                       ##
    # ###############################################################
    print("\n--- Processing Summary ---")
    print(f"Total images found:    {total_images}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed to process:      {failed_count}")
    print("--------------------------")
    # ###############################################################

    print("\n--- Processing Complete ---")
    if detected_plates_text:
        print(f"All valid vehicle numbers have been saved to: {output_file_path}")
        for plate in detected_plates_text:
            print(f" - {plate}")
    else:
        print("No number plates were successfully extracted.")