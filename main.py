import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from shutil import move
from os import listdir, path, mkdir
from shutil import rmtree

from model import MyModel
from utils import dense_image_warp, combine_videos

ORIG_WIDTH = 0
ORIG_HEIGHT = 0
TRAIN_EPOCHS = 500

im_sz = 1024
morph_dir = "morph/morph.mp4"


@tf.function
def warp(origins, targets, preds_org, preds_trg):
    """
    The warp function takes in the original images, and the predicted offsets for each image.
    It then applies these offsets to create a warped version of each image.
    The warp function is used to generate training data for our model.

    :param origins: Pass the original image to the warp function
    :param targets: Define the target image
    :param preds_org: Warp the origin image
    :param preds_trg: Warp the target image
    :return: The warped images
    """
    scale_org = tf.maximum(0.1, 1.0 + preds_org[..., :3] * mult_scale)
    scale_trg = tf.maximum(0.1, 1.0 + preds_trg[..., :3] * mult_scale)

    offset_org = preds_org[..., 3:6] * 2.0 * add_scale
    offset_trg = preds_trg[..., 3:6] * 2.0 * add_scale

    warp_org = preds_org[..., 6:8] * im_sz * warp_scale
    warp_trg = preds_trg[..., 6:8] * im_sz * warp_scale

    if add_first:
        res_targets = dense_image_warp((origins + offset_org) * scale_org, warp_org)
        res_origins = dense_image_warp((targets + offset_trg) * scale_trg, warp_trg)
    else:
        res_targets = dense_image_warp(origins * scale_org + offset_org, warp_org)
        res_origins = dense_image_warp(targets * scale_trg + offset_trg, warp_trg)

    return res_targets, res_origins


def create_grid(scale):
    """
    The create_grid function creates a grid of coordinates that can be used to
    sample the output of a convolutional layer.

    :param scale: Determine the size of the grid
    :return: A grid of size (scale, scale) with values between - 1 and 1
    """
    grid = np.mgrid[0:scale, 0:scale] / (scale - 1) * 2 - 1
    grid = np.swapaxes(grid, 0, 2)
    grid = np.expand_dims(grid, axis=0)
    return grid


def produce_warp_maps(origins, targets):
    """
    The produce_warp_maps function is the main function of this script. It takes two images,
    origins and targets, as input and produces a set of warp maps that can be used to transform
    the origins image into the targets image. The produce_warp_maps function uses a custom model
    (MyModel) to train on these two images for TRAIN_EPOCHS epochs (default: 10000). After each
    epoch, it saves an intermediate result in the train/ directory.

    :param origins: Store the original images
    :param targets: Warp the original image to the target image
    :return: The predicted maps
    """
    model = MyModel()

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    train_loss = tf.keras.metrics.Mean(name="train_loss")

    @tf.function
    def train_step(training_maps, training_origins, training_targets):
        """    
        The train_step function is called inside the train_loop function.
        It takes in a batch of training data, and uses that batch to perform one step of model training.
        In this case, using tf.GradientTape() allows us to calculate the gradients used to optimize our model's variables (tf.trainable_variables). 
        These optimized variables are then applied back to the model with an optimizer (in this case Adam). 
        The loss function is also calculated and stored in a history object for later visualization.
        
        :param training_maps: Train the model
        :param training_origins: Calculate the loss of the origin images
        :param training_targets: Calculate the loss of the target images
        :return: The loss, so we can plot it
        """
        with tf.GradientTape() as tape:
            map_pred = model(training_maps)
            map_pred = tf.image.resize(map_pred, [im_sz, im_sz])
            res_targets_, res_origins_ = warp(
                training_origins, training_targets, map_pred[..., :8], map_pred[..., 8:]
            )

            flow_scale = im_sz * warp_scale
            res_map = dense_image_warp(training_maps, map_pred[:, :, :, 6:8] * flow_scale)
            res_map = dense_image_warp(res_map, map_pred[:, :, :, 14:16] * flow_scale)

            loss = (
                    loss_object(training_maps, res_map) * 1
                    + loss_object(res_targets_, training_targets) * 0.3
                    + loss_object(res_origins_, training_origins) * 0.3
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    maps = create_grid(im_sz)
    maps = np.concatenate((maps, origins * 0.1, targets * 0.1), axis=-1).astype(np.float32)

    epoch = 0
    template = "Epoch {}, Loss: {}"

    t = tqdm(range(TRAIN_EPOCHS), desc=template.format(epoch, train_loss.result()))

    for iteration in t:
        epoch = iteration + 1

        t.set_description(template.format(epoch, train_loss.result()))
        t.refresh()

        train_step(maps, origins, targets)

        if (epoch < 100 and epoch % 10 == 0) or (epoch < 1000 and epoch % 100 == 0) or \
                (epoch % 1000 == 0):
            preds = model(maps, training=False)[:1]
            preds = tf.image.resize(preds, [im_sz, im_sz])

            res_targets, res_origins = warp(origins, targets, preds[..., :8], preds[..., 8:])
            np.save("preds.npy", preds.numpy())

            res_targets = tf.clip_by_value(res_targets, -1, 1)[0]
            res_origins = tf.clip_by_value(res_origins, -1, 1)[0]

            for res, prefix in zip([res_targets, res_origins], ["a_to_b", "b_to_a"]):
                res_img = ((res.numpy() + 1) * 127.5).astype(np.uint8)
                res_img = cv2.resize(res_img, (ORIG_WIDTH, ORIG_HEIGHT),
                                     interpolation=cv2.INTER_AREA)
                cv2.imwrite(f"train/{prefix}_{epoch}.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))


def use_warp_maps(origins, targets):
    """
    The use_warp_maps function takes in the original and target images, as well as the frames per second
    and number of steps to be used for morphing. It then loads in the predicted warp maps from a .npy file,
    and uses them to morph between two faces. The function outputs a video of this morphing process.

    :param origins: Get the original image
    :param targets: Store the target image
    :return: A video of the morph between two images
    """
    preds = np.load("preds.npy")
    STEPS = steps

    res_img = np.zeros((im_sz * 2, im_sz * 3, 3), dtype=np.uint8)

    res_img[im_sz * 0: im_sz * 1, im_sz * 0: im_sz * 1] = preds[0, :, :, 0:3]
    res_img[im_sz * 0: im_sz * 1, im_sz * 1: im_sz * 2] = preds[0, :, :, 3:6]
    res_img[im_sz * 0: im_sz * 1, im_sz * 2: im_sz * 3, :2] = preds[0, :, :, 6:8]
    res_img[im_sz * 1: im_sz * 2, im_sz * 0: im_sz * 1] = preds[0, :, :, 8:11]
    res_img[im_sz * 1: im_sz * 2, im_sz * 1: im_sz * 2] = preds[0, :, :, 11:14]
    res_img[im_sz * 1: im_sz * 2, im_sz * 2: im_sz * 3, :2] = preds[0, :, :, 14:16]

    res_img = np.clip(res_img, -1, 1)
    res_img = ((res_img + 1) * 127.5).astype(np.uint8)
    cv2.imwrite("morph/maps.jpg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

    org_strength = tf.reshape(tf.range(STEPS, dtype=tf.float32), [STEPS, 1, 1, 1]) / (STEPS - 1)
    trg_strength = tf.reverse(org_strength, axis=[0])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(morph_dir, fourcc, fps, (ORIG_WIDTH, ORIG_HEIGHT))

    for iterations in tqdm(range(STEPS)):
        preds_org = preds * org_strength[iterations]
        preds_trg = preds * trg_strength[iterations]

        res_targets, res_origins = warp(origins, targets, preds_org[..., :8], preds_trg[..., 8:])
        res_targets = tf.clip_by_value(res_targets, -1, 1)
        res_origins = tf.clip_by_value(res_origins, -1, 1)

        results = (res_targets * trg_strength[iterations]) + \
                  (res_origins * org_strength[iterations])
        res_numpy = results.numpy()

        output_img = ((res_numpy[0] + 1) * 127.5).astype(np.uint8)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        output_img = cv2.resize(output_img, (ORIG_WIDTH, ORIG_HEIGHT), interpolation=cv2.INTER_AREA)

        video.write(output_img)

    cv2.destroyAllWindows()
    video.release()


def match_size(image_one, image_two):
    """
    The matchSize function takes two images as input and returns the same two images, but with one of them resized to match
    the size of the other. The function first checks if image_one is larger than image_two in terms of width. If it is, then
    image_one will be resized to match the dimensions (width and height) of image_two using cv2's resize function with an
    interpolation method set to INTER_AREA. Otherwise, if image one is not larger than image two in terms of width, then
    image two will be resized instead.

    :param image_one: Pass in the first image
    :param image_two: Resize the image_one parameter to match its size
    :return: The two images that have been resized to the same size
    """
    if image_one.shape[1] > image_two.shape[1]:
        image_one = cv2.resize(image_one, (image_two.shape[1], image_two.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        image_two = cv2.resize(image_two, (image_one.shape[1], image_one.shape[0]), interpolation=cv2.INTER_AREA)
    return image_one, image_two


def driver(source, target):
    """
    The driver function takes in the source and target images, as well as the fps and steps.
    It then resizes both images to a square of size im_sz x im_sz, converts them to RGB from BGR,
    and normalizes them by dividing each pixel value by 127.5 and subtracting 1. It then reshapes
    the image arrays into 4D tensors with shape (batch_size=None, height=im_sz, width=im_sz).
    The driver function calls produce warp maps on these two tensors which produces a set of warp maps for each image pair

    :param source: Specify the source image
    :param target: Specify the target image
    :return: The final morphed image
    """
    dom_a = cv2.imread(source, cv2.IMREAD_COLOR)
    dom_b = cv2.imread(target, cv2.IMREAD_COLOR)
    if dom_a.shape[1] != dom_b.shape[1] or dom_a.shape[0] != dom_b.shape[0]:
        dom_a, dom_b = (dom_a, dom_b)
    global ORIG_WIDTH, ORIG_HEIGHT
    ORIG_WIDTH = dom_a.shape[1]
    ORIG_HEIGHT = dom_a.shape[0]
    dom_a = cv2.cvtColor(dom_a, cv2.COLOR_BGR2RGB)
    dom_a = cv2.resize(dom_a, (im_sz, im_sz), interpolation=cv2.INTER_AREA)
    dom_a = dom_a / 127.5 - 1
    dom_b = cv2.cvtColor(dom_b, cv2.COLOR_BGR2RGB)
    dom_b = cv2.resize(dom_b, (im_sz, im_sz), interpolation=cv2.INTER_AREA)
    dom_b = dom_b / 127.5 - 1
    origins = dom_a.reshape(1, im_sz, im_sz, 3).astype(np.float32)
    targets = dom_b.reshape(1, im_sz, im_sz, 3).astype(np.float32)
    produce_warp_maps(origins, targets)
    use_warp_maps(origins, targets)


def main():
    """
    The main function is the driver function for this program. It first checks if there are any existing output folders
    and deletes them if they exist. Then it creates a new output folder and morph folder to store the intermediate files
    in. Next, it gets a list of all images in the input directory and sorts them alphabetically so that they can be morphed
    in order from start to finish (or vice versa). The main loop then iterates through each image pair in order, calling
    the driver function on each pair with specified fps and steps values as well as storing their filenames into an array
    for later use when combining

    :return: The output file name
    """
    if path.exists("output"):
        rmtree("output")
    mkdir("output")
    if path.exists("morph"):
        rmtree("morph")
    mkdir("morph")
    image_list = [file for file in listdir("input") if not file.startswith(".")]
    image_list.sort()
    filenames = []
    for i in range(len(image_list) - 1):
        start = f"input/{image_list[i]}"
        end = f"input/{image_list[i + 1]}"
        driver(start, end)
        filename = f"output/morph{i:03d}.mp4"
        filenames.append(filename)
        move(morph_dir, filename)
    if loop:
        driver(f"input/{image_list[-1]}", f"input/{image_list[0]}")
        filename = f"output/morph{len(image_list):03d}.mp4"
        filenames.append(filename)
        move(morph_dir, filename)
    combine_videos(filenames, "output/Final", fps, loop)


if __name__ == "__main__":
    warp_scale = 0.075
    mult_scale = 0.4
    add_scale = 0.4
    add_first = False
    loop = True
    fps = 60
    steps = 120
    main()
