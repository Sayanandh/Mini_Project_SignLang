import os
import cv2
import random
import numpy as np

def augment_images(input_dir, output_dir):
    """
    Augment all images in the input directory and save them to the output directory.

    Args:
        input_dir (str): Path to the directory containing normalized images.
        output_dir (str): Path to save the augmented images.
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                # Save original image
                cv2.imwrite(os.path.join(output_class_path, img_name), img)

                # Flip image horizontally
                flipped = cv2.flip(img, 1)
                flipped_name = img_name.split('.')[0] + '_flipped.jpg'
                cv2.imwrite(os.path.join(output_class_path, flipped_name), flipped)

                # Rotate image
                rows, cols, _ = img.shape
                angle = random.choice([-15, 15])
                rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                rotated = cv2.warpAffine(img, rotation_matrix, (cols, rows))
                rotated_name = img_name.split('.')[0] + '_rotated.jpg'
                cv2.imwrite(os.path.join(output_class_path, rotated_name), rotated)

                # Adjust brightness randomly
                brightness_factor = random.uniform(0.7, 1.3)
                brightened = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
                brightened_name = img_name.split('.')[0] + '_brightened.jpg'
                cv2.imwrite(os.path.join(output_class_path, brightened_name), brightened)

                # Add random noise
                noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
                noisy = cv2.add(img, noise)
                noisy_name = img_name.split('.')[0] + '_noisy.jpg'
                cv2.imwrite(os.path.join(output_class_path, noisy_name), noisy)

                print(f"Augmented and saved: {img_name} in {output_class_path}")

# Example usage
if __name__ == "__main__":
    augment_images('A:/Sign-to-Speech-main/data', 'A:/Sign-to-Speech-main/data/augmented')
