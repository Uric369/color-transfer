# -*- coding: utf-8 -*-

import os
import time

import cv2
import numpy as np

from python_color_transfer.color_transfer import ColorTransfer


def demo():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    img_folder = os.path.join(cur_dir, "leaves")

    img_names = [
        # "brk1.png",
        "lef1.tif",
    ]
    ref_names = [
        # "bbb.png",
        "a.tif",
    ]
    out_names = [
        # "1_brk.tif",
        "result2",
    ]

    img_paths = [os.path.join(img_folder, x) for x in img_names]
    ref_paths = [os.path.join(img_folder, x) for x in ref_names]
    out_paths = [os.path.join(img_folder, x) for x in out_names]

    # cls init
    PT = ColorTransfer()

    for img_path, ref_path, out_path in zip(img_paths, ref_paths, out_paths):
        # read source tif
        image_src_with_alpha = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        alpha_mask_src = image_src_with_alpha[:, :, 3]
        img_arr_in = image_src_with_alpha[:, :, :3]
        [h, w, c] = img_arr_in.shape

        # read reference img, convert to png and resize to input img size
        img_arr_ref = cv2.imread(ref_path)
        img_arr_ref = cv2.resize(img_arr_ref, (w, h), interpolation=cv2.INTER_AREA)

        print(f"{img_path}: {h}x{w}x{c}")
        print(f"{ref_path} resized to: {h}x{w}x{c}")

        # pdf transfer
        t0 = time.time()
        img_arr_reg = PT.pdf_transfer(img_arr_in=img_arr_in,
                                      img_arr_ref=img_arr_ref,
                                      regrain=True)
        print(f"Pdf transfer time: {time.time() - t0:.2f}s")

        # mean transfer
        t0 = time.time()
        img_arr_mt = PT.mean_std_transfer(img_arr_in=img_arr_in,
                                          img_arr_ref=img_arr_ref)
        print(f"Mean std transfer time: {time.time() - t0:.2f}s")

        # lab transfer
        t0 = time.time()
        img_arr_lt = PT.lab_transfer(img_arr_in=img_arr_in,
                                     img_arr_ref=img_arr_ref)
        print(f"Lab mean std transfer time: {time.time() - t0:.2f}s")

        # apply alpha mask
        # img_arr_in = cv2.merge((img_arr_in[:, :, :3], alpha_mask_src))
        img_arr_mt = cv2.merge((img_arr_mt[:, :, :3], alpha_mask_src))
        img_arr_lt = cv2.merge((img_arr_lt[:, :, :3], alpha_mask_src))
        img_arr_reg = cv2.merge((img_arr_reg[:, :, :3], alpha_mask_src))
        # img_arr_ref = cv2.merge((img_arr_ref, alpha_mask_ref))

        # concatenate and save
        # img_arr_out = np.concatenate(
        #     (image_src_with_alpha, img_arr_ref, img_arr_mt, img_arr_lt, img_arr_reg),
        #     axis=1)

        # Save the output as a TIF file
        # cv2.imwrite(out_path, img_arr_out)
        if not os.path.exists(out_path):
            # If it doesn't exist, create it
            os.makedirs(out_path)
            print("Create directory: " + out_path)

        cv2.imwrite(out_path + "/mt.tif", img_arr_mt)
        cv2.imwrite(out_path + "/lt.tif", img_arr_lt)
        cv2.imwrite(out_path + "/reg.tif", img_arr_reg)
        print(f"Saved to {out_path}\n")


if __name__ == "__main__":
    demo()
