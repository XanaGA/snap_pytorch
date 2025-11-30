import os
import time
import numpy as np
import cv2

from snap.data.loader import get_dataset
from snap.data import types


def show_image(window_name, img):
    """Helper to display an RGB numpy image."""
    if img is None:
        print("[No image provided]")
        return
    # Convert RGB → BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, img_bgr)
    cv2.waitKey(1)  # small delay so image updates


def print_dict_recursive(d, indent=0):
    """Pretty-print nested dictionaries."""
    pad = "  " * indent
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{pad}{k}: ", end="")
            if isinstance(v, (dict, list)):
                print()
                print_dict_recursive(v, indent + 1)
            else:
                print(type(v), np.shape(v))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            print(f"{pad}[{i}]:")
            print_dict_recursive(v, indent + 1)
    else:
        print(f"{pad}{d}")


def main():
    # ----------------------------
    # CONFIGURE DATASET
    # ----------------------------
    config = types.ProcessingConfig(
        data_path="/local/home/xanadon/libdrive_mount/Gibson Floorplan Localization Dataset",  # <-- CHANGE THIS
        mode=types.DataMode.SINGLE_SCENE,
    )

    # Required by loader (minimal fill)
    class Locations:
        training = "gibson_f"
        evaluation = "gibson_f"  # can also use gibson_g or gibson_t

    config.locations = Locations()

    # ----------------------------
    # LOAD DATASET
    # ----------------------------
    dataset = get_dataset(
        batch_size=4,
        eval_batch_size=2,
        num_shards=1,
        dataset_configs=config,
    )

    train_iter = dataset.train

    if train_iter is None:
        print("No training iterator available.")
        return

    # ----------------------------
    # FETCH ONE BATCH
    # ----------------------------
    batch = next(train_iter)

    print("\n========================")
    print("BATCH KEYS:", batch.keys())
    print("========================\n")

    # ----------------------------
    # SHOW CONTENTS OF EXAMPLES
    # ----------------------------
    for i in range(len(batch["scene_id"])):
        print(f"\n===== SAMPLE {i} =====")
        sample = {k: (v[i] if isinstance(v, np.ndarray) else v[i] if isinstance(v, list) else v)
                  for k, v in batch.items()}

        # Print full nested structure
        print_dict_recursive(sample)

        # -------------------------
        # DISPLAY RGB IMAGE
        # -------------------------
        if "images" in sample:
            print("Showing image… (press ESC to continue)")
            img = sample["images"]
            show_image("sample_image", img)

            key = cv2.waitKey(0)
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                break

        print("========================\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
