"""
Usage:
python  scripts/augment_vl_data/fisheye.py -i $image-dir -o $output-dir -s $start-img-idx -e $end-img-idx

Output:
    $output-dir
    |-- images_fisheye: contains fisheye distorted images
    |-- images_full: contains full augmented images
    |-- images_gripper: contains images augmented with a robot gripper composited with adaptive brightness
    `-- images_resize: contains resized images without augmentation
"""

import copy
import enum
from pathlib import Path

import click
import cv2
import numpy as np
import scipy.interpolate
import tqdm
from PIL import Image, ImageEnhance

OUT_RES = (224, 224)

json_data = {
    "final_reproj_error": 0.2916398582648,
    "fps": 59.94005994005994,
    "image_height": 2028,
    "image_width": 2704,
    "intrinsic_type": "FISHEYE",
    "intrinsics": {
      "aspect_ratio": 1.0029788958491257,
      "focal_length": 796.8544625226342,
      "principal_pt_x": 1354.4265245977356,
      "principal_pt_y": 1011.4847310011687,
      "radial_distortion_1": -0.02196117964405394,
      "radial_distortion_2": -0.018959717016668237,
      "radial_distortion_3": 0.001693880829392453,
      "radial_distortion_4": -0.00016807228608000285,
      "skew": 0.0
    },
    "nr_calib_images": 59,
    "stabelized": False,
}


def parse_fisheye_intrinsics(json_data: dict=json_data) -> dict[str, np.ndarray]:
    """
    Reads camera intrinsics from OpenCameraImuCalibration to opencv format.
    """
    assert json_data['intrinsic_type'] == 'FISHEYE'
    intr_data = json_data['intrinsics']
    
    # img size
    h = json_data['image_height']
    w = json_data['image_width']

    # pinhole parameters
    f = intr_data['focal_length']
    px = intr_data['principal_pt_x']
    py = intr_data['principal_pt_y']
    
    # Kannala-Brandt non-linear parameters for distortion
    kb8 = [
        intr_data['radial_distortion_1'],
        intr_data['radial_distortion_2'],
        intr_data['radial_distortion_3'],
        intr_data['radial_distortion_4']
    ]

    opencv_intr_dict = {
        'DIM': np.array([w, h], dtype=np.int64),
        'K': np.array([
            [f, 0, px],
            [0, f, py],
            [0, 0, 1]
        ], dtype=np.float64),
        'D': np.array([kb8]).T
    }
    return opencv_intr_dict

# copied from https://github.com/Synthesis-AI-Dev/fisheye-distortion

def convert_fisheye_intrinsics_resolution(
        opencv_intr_dict: dict[str, np.ndarray], 
        target_resolution: tuple[int, int]
        ) -> dict[str, np.ndarray]:
    """
    Convert fisheye intrinsics parameter to a different resolution,
    assuming that images are not cropped in the vertical dimension,
    and only symmetrically cropped/padded in horizontal dimension.
    """
    iw, ih = opencv_intr_dict['DIM']
    iK = opencv_intr_dict['K']
    ifx = iK[0,0]
    ify = iK[1,1]
    ipx = iK[0,2]
    ipy = iK[1,2]

    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    oK = np.array([
        [ofx, 0, opx],
        [0, ofy, opy],
        [0, 0, 1]
    ], dtype=np.float64)

    out_intr_dict = copy.deepcopy(opencv_intr_dict)
    out_intr_dict['DIM'] = np.array([ow, oh], dtype=np.int64)
    out_intr_dict['K'] = oK
    return out_intr_dict


def get_image_transform(in_res, out_res, crop_ratio:float = 1.0, bgr_to_rgb: bool=False):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # crop
        img = img[h_slice, w_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img
    
    return transform


class DistortMode(enum.Enum):
    LINEAR = 'linear'
    NEAREST = 'nearest'


def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
                  mode: DistortMode = DistortMode.LINEAR, crop_output: bool = False,
                  crop_type: str = "corner") -> np.ndarray:
    """Apply fisheye distortion to an image

    Args:
        img (numpy.ndarray): BGR image. Shape: (H, W, 3)
        cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                            Shape: (3, 3)
        dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                            Shape: (1, 4)
        mode (DistortMode): For distortion, whether to use nearest neighbour or linear interpolation.
                            RGB images = linear, Mask/Surface Normals/Depth = nearest
        crop_output (bool): Whether to crop the output distorted image into a rectangle. The 4 corners of the input
                            image will be mapped to 4 corners of the distorted image for cropping.
        crop_type (str): How to crop.
            "corner": We crop to the corner points of the original image, maintaining FOV at the top edge of image.
            "middle": We take the widest points along the middle of the image (height and width). There will be black
                      pixels on the corners. To counter this, original image has to be higher FOV than the desired output.

    Returns:
        numpy.ndarray: The distorted image, same resolution as input image. Unmapped pixels will be black in color.
    """
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    imshape = img.shape
    if len(imshape) == 3:
        h, w, chan = imshape
    elif len(imshape) == 2:
        h, w = imshape
        chan = 1
    else:
        raise RuntimeError(f'Image has unsupported shape: {imshape}. Valid shapes: (H, W), (H, W, N)')

    imdtype = img.dtype

    # Get array of pixel co-ords
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2)

    # Get the mapping from distorted pixels to undistorted pixels
    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff)  # shape: (N, 1, 2)
    undistorted_px = cv2.convertPointsToHomogeneous(undistorted_px)  # Shape: (N, 1, 3)
    undistorted_px = np.tensordot(undistorted_px, cam_intr, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
    undistorted_px = cv2.convertPointsFromHomogeneous(undistorted_px)  # Shape: (N, 1, 2)
    undistorted_px = undistorted_px.reshape((h, w, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

    # Map RGB values from input img using distorted pixel co-ordinates
    if chan == 1:
        img = np.expand_dims(img, 2)
    interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], method=mode.value,
                                                               bounds_error=False, fill_value=0)
                     for channel in range(chan)]
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

    if imdtype == np.uint8:
        # RGB Image
        img_dist = img_dist.round().clip(0, 255).astype(np.uint8)
    elif imdtype == np.uint16:
        # Mask
        img_dist = img_dist.round().clip(0, 65535).astype(np.uint16)
    elif imdtype == np.float16 or imdtype == np.float32 or imdtype == np.float64:
        img_dist = img_dist.astype(imdtype)
    else:
        raise RuntimeError(f'Unsupported dtype for image: {imdtype}')

    if crop_output:
        # Crop rectangle from resulting distorted image
        # Get mapping from undistorted to distorted
        distorted_px = cv2.convertPointsToHomogeneous(img_pts)  # Shape: (N, 1, 3)
        cam_intr_inv = np.linalg.inv(cam_intr)
        distorted_px = np.tensordot(distorted_px, cam_intr_inv, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
        distorted_px = cv2.convertPointsFromHomogeneous(distorted_px)  # Shape: (N, 1, 2)
        distorted_px = cv2.fisheye.distortPoints(distorted_px, cam_intr, dist_coeff)  # shape: (N, 1, 2)
        distorted_px = distorted_px.reshape((h, w, 2))
        if crop_type == "corner":
            # Get the corners of original image. Round values up/down accordingly to avoid invalid pixel selection.
            top_left = np.ceil(distorted_px[0, 0, :]).astype(int)
            bottom_right = np.floor(distorted_px[(h - 1), (w - 1), :]).astype(int)
            img_dist = img_dist[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        elif crop_type == "middle":
            # Get the widest point of original image, then get the corners from that.
            width_min = np.ceil(distorted_px[int(h / 2), 0, 0]).astype(np.int32)
            width_max = np.ceil(distorted_px[int(h / 2), -1, 0]).astype(np.int32)
            height_min = np.ceil(distorted_px[0, int(w / 2), 1]).astype(np.int32)
            height_max = np.ceil(distorted_px[-1, int(w / 2), 1]).astype(np.int32)
            img_dist = img_dist[height_min:height_max, width_min:width_max]
        else:
            raise ValueError

    if chan == 1:
        img_dist = img_dist[:, :, 0]

    return img_dist


def compose_image(
        distorted_img: np.ndarray,
        umi_img: np.ndarray,
        finger_mask: np.ndarray,
        lens_img: np.ndarray,
        gripper_lens_mask: np.ndarray,
    ):
    lens_radius = 1088
    h, w, _ = distorted_img.shape
    # resize to the same height
    orig_h, orig_w, _ = umi_img.shape
    resize_ratio = h / (2 * lens_radius)
    target_h = int(orig_h * resize_ratio)
    target_w = int(orig_w * resize_ratio)

    # resize finger_img, finger_mask, lens_img, gripper_lens_mask to the same size
    umi_img = cv2.resize(umi_img, (target_w, target_h))
    finger_mask = cv2.resize(finger_mask, (target_w, target_h))
    lens_img = cv2.resize(lens_img, (target_w, target_h))
    gripper_lens_mask = cv2.resize(gripper_lens_mask, (target_w, target_h))
    # pad the distorted_img to the same size
    canvas = np.zeros_like(umi_img)
    canvas_left = np.clip(int((target_w - w) / 2), 0, target_w)
    canvas_right = np.clip(int((target_w - w) / 2 + w), 0, target_w)
    canvas_top = np.clip(int((target_h - h) / 2), 0, target_h)
    canvas_bottom = np.clip(int((target_h - h) / 2 + h), 0, target_h)

    img_left = np.clip(int((w - target_w) / 2), 0, w)
    img_right = np.clip(int((w - target_w) / 2 + target_w), 0, w)
    img_top = np.clip(int((h - target_h) / 2), 0, h)
    img_bottom = np.clip(int((h - target_h) / 2 + target_h), 0, h)

    canvas[canvas_top:canvas_bottom, canvas_left:canvas_right] = distorted_img[img_top:img_bottom, img_left:img_right]

    # 1. add lens
    lens_added = cv2.bitwise_and(lens_img, gripper_lens_mask) + cv2.bitwise_and(canvas, 255 - gripper_lens_mask)
    # 2. add finger
    def adjust_lighting(img_part, target_img, mask):
        pil_part = Image.fromarray(img_part)
        pil_target = Image.fromarray(target_img)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype(bool)
        
        target_brightness = np.mean(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)[mask])
        
        part_brightness = np.mean(cv2.cvtColor(img_part, cv2.COLOR_BGR2GRAY)[mask])
        
        brightness_factor = target_brightness / part_brightness
        brightness_factor = max(brightness_factor, 0.9)
        enhancer = ImageEnhance.Brightness(pil_part)
        pil_part = enhancer.enhance(brightness_factor)
        
        target_contrast = np.std(cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY))
        part_contrast = np.std(cv2.cvtColor(np.array(pil_part), cv2.COLOR_BGR2GRAY))
        
        contrast_factor = target_contrast / part_contrast
        contrast_factor = 1
        enhancer = ImageEnhance.Contrast(pil_part)
        pil_part = enhancer.enhance(contrast_factor)
        
        return np.array(pil_part)

    umi_img = adjust_lighting(umi_img, canvas, finger_mask)
    finger_added = cv2.bitwise_and(umi_img, finger_mask) + cv2.bitwise_and(lens_added, 255 - finger_mask)

    trans = get_image_transform(
        in_res=(target_w, target_h),
        out_res=OUT_RES,
        crop_ratio=1.0,
    )

    return trans(finger_added)


@click.command()
@click.option('--input-dir', '-i', required=True, help='Input directory containing images', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option('--output-dir', '-o', required=True, help='Input directory containing images', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option('--start', '-s', default=0, required=True, help='Start image index')
@click.option('--end', '-e', default=30, required=True, help='End image index')
@click.option('--run-again', '-ra', is_flag=True, help='Run again even if output exists')
def main(input_dir: Path, output_dir: Path, start: int, end: int, run_again: bool = False):
    task = 'random_plan'

    full_output_dir = output_dir / 'images_full'
    fisheye_output_dir = output_dir / 'images_fisheye'
    gripper_output_dir = output_dir / 'images_gripper'
    resize_output_dir = output_dir / 'images_resize'


    resource_dir = Path(__file__).parent
    umi_image = cv2.imread(resource_dir / 'inpainted.jpg')
    finger_mask = cv2.imread(resource_dir / 'finger_mask.jpg')
    lens_image = cv2.imread(resource_dir / 'lens.jpg')
    gripper_lens_mask = cv2.imread(resource_dir / 'gripper_lens_mask.jpg')

    full_output_dir.mkdir(parents=True, exist_ok=True)
    fisheye_output_dir.mkdir(parents=True, exist_ok=True)
    gripper_output_dir.mkdir(parents=True, exist_ok=True)
    resize_output_dir.mkdir(parents=True, exist_ok=True)

    for img_id in tqdm.tqdm(range(start, end + 1), dynamic_ncols=True):
        input_image_path = input_dir / f'{img_id}.png'
        # Example usage
        full_output_image_path = full_output_dir / input_image_path.name
        fisheye_output_image_path = fisheye_output_dir / input_image_path.name
        gripper_output_image_path = gripper_output_dir / input_image_path.name
        resize_output_image_path = resize_output_dir / input_image_path.name

        if (
            all(
                [
                    full_output_image_path.exists(),
                    fisheye_output_image_path.exists(),
                    gripper_output_image_path.exists(),
                    resize_output_image_path.exists(),
                ]
            )
            and not run_again
        ):
            continue


        # Load the input image
        input_image = cv2.imread(input_image_path,)
        resize_trans = get_image_transform(
            in_res=(input_image.shape[1], input_image.shape[0]),
            out_res=OUT_RES,
        )
        # save the resized image
        cv2.imwrite(resize_output_image_path, resize_trans(input_image.copy()))
        intrinsics = parse_fisheye_intrinsics(json_data)
        intrinsics = convert_fisheye_intrinsics_resolution(intrinsics, (input_image.shape[1], input_image.shape[0]))
        dist_coeff = intrinsics['D'].reshape(4,)

        # Distort the image
        distorted_image = distort_image(input_image, intrinsics['K'], dist_coeff, mode=DistortMode.LINEAR, crop_output=True, crop_type="corner")
        distorted_resize_trans = get_image_transform(
            in_res=(distorted_image.shape[1], distorted_image.shape[0]),
            out_res=OUT_RES,
        )
        # save the distorted image
        cv2.imwrite(fisheye_output_image_path, distorted_resize_trans(distorted_image.copy()))

        gripper_added_image = compose_image(input_image, umi_image, finger_mask, lens_image, gripper_lens_mask)
        # save the gripper image
        cv2.imwrite(gripper_output_image_path, gripper_added_image)

        full_aug_image = compose_image(distorted_image, umi_image, finger_mask, lens_image, gripper_lens_mask)
        # save the full augmented image
        cv2.imwrite(str(full_output_image_path), full_aug_image)

if __name__ == '__main__':
    main()
