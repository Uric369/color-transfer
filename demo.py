# -*- coding: utf-8 -*-

import os
import time

import cv2
import numpy as np

from python_color_transfer.color_transfer import ColorTransfer

# cover模式：输出的同一张图片需要覆盖同个路径下多个texture
# 需覆盖的图片路径为out_img_folder（文件夹路径） + covered_image（数组，定义图片名称）
mode = "cover"

# ouput模式：供对比三种方法的输出结果
# 输出路径为out_img_folder + out_names 路径下的lt.tif mt.tif pdf.tif文件
#mode = "output"


covered_image = [
    # "LS01lef1.tif",
    # "LS01lef2.tif",
    # "LS01lef3.tif",
    # "LS01lef4.tif",
    # "LS01brk1.tif",
    # "LS01brk2.tif",
    # "LS01brn1.tif",
    # "LS01brn2.tif",
    "lef1.tif"
    # "brk1.tif"
]

# 因处理树叶、树干时发现数据格式不同
# 若处理树叶则设置为leaf
# 处理树干设置为trunk
target = "leaf"
# target = "trunk"

# 仅在mode为cover时有效，选择输出方法
method = "lt"
# method = "mt"

def demo():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    src_img_folder = os.path.join(cur_dir, "leaves\\1\\src")
    ref_img_folder = os.path.join(cur_dir, "leaves\\1\\ref")
    out_img_folder = os.path.join(cur_dir, "leaves\\1\\example_1\\reconstructed_model")

    src_names = [
        "lef1.tif",
        # "brk1.tif",
    ]
    ref_names = [
        "lef1.png",
        # "brk1.png",
    ]
    out_names = [
        "",
    ]

    src_paths = [os.path.join(src_img_folder, x) for x in src_names]
    ref_paths = [os.path.join(ref_img_folder, x) for x in ref_names]
    out_paths = [os.path.join(out_img_folder, x) for x in out_names]

    # cls init
    PT = ColorTransfer()

    for img_path, ref_path, out_path in zip(src_paths, ref_paths, out_paths):
        # read source tif
        print(img_path)
        image_src_with_alpha = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Check if the image has an alpha channel
        if image_src_with_alpha.shape[2] == 4:
            # Image has an alpha channel
            alpha_mask_src = image_src_with_alpha[:, :, 3]
            img_arr_in = image_src_with_alpha[:, :, :3]
        else:
            # Image does not have an alpha channel
            alpha_mask_src = None  # No alpha mask needed
            img_arr_in = image_src_with_alpha  # The image is already without alpha

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
                                     img_arr_ref=img_arr_ref,
                                     target=target)
        print(f"Lab mean std transfer time: {time.time() - t0:.2f}s")

        # apply alpha mask
        # img_arr_in = cv2.merge((img_arr_in[:, :, :3], alpha_mask_src))
        if alpha_mask_src is not None and alpha_mask_src.size > 0:
            img_arr_mt = cv2.merge((img_arr_mt[:, :, :3], alpha_mask_src))
            img_arr_lt = cv2.merge((img_arr_lt[:, :, :3], alpha_mask_src))
            img_arr_reg = cv2.merge((img_arr_reg[:, :, :3], alpha_mask_src))

        if not os.path.exists(out_path):
            # If it doesn't exist, create it
            os.makedirs(out_path)
            print("Create directory: " + out_path)


        if mode == "cover":
            for img_name in covered_image:
                # 根据method变量确定使用哪个图像数组
                if method == "lt":
                    img_to_save = img_arr_lt
                elif method == "mt":
                    img_to_save = img_arr_mt
                else:
                    raise ValueError(f"Unrecognized method: {method}")

                # 保存图像到指定的out_path路径，确保out_path是目录
                cv2.imwrite(out_path + img_name, img_to_save)
                print(f"Saved to {out_path + img_name}\n")
        elif mode == "output":
            # 直接保存图像到out_path路径，这里假设out_path是文件路径
            cv2.imwrite(out_path + "lt.tif", img_arr_lt)
            cv2.imwrite(out_path + "mt.tif", img_arr_mt)
            cv2.imwrite(out_path + "pdf.tif", img_arr_reg)
            print(f"Saved to {out_path + "lt.tif"}\n")
            print(f"Saved to {out_path + "mt.tif"}\n")
            print(f"Saved to {out_path + "pdf.tif"}\n")
        else:
            raise ValueError(f"Unrecognized mode: {mode}")


if __name__ == "__main__":
    demo()
