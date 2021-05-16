import tensorflow as tf
import cv2
import time
import argparse
import os
import csv
import json
import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=0.5)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():
    errors = 0
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        filenames = [f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.30)

            keypoint_coords *= output_scale

            if args.output_dir:
              draw_image = posenet.draw_skel_and_kp(draw_image, pose_scores, keypoint_scores, keypoint_coords,min_pose_score=0.25, min_part_score=0.25)

              cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            if not args.notxt:
                print("Results for image: %s" % f)
                coords = "{"
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))

                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                       print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

                       coords += str(posenet.COORDENADAS[ki] % (c[0], c[1]))
                try:
                    coords += ",\"atividade\"" + str(":") + "\"3\""
                    coords += "}"
                    createFile(coords)
                except:
                  print("Erro file: " + str(filenames))
                  errors +=1

        print('Average FPS:', len(filenames) / (time.time() - start))
        print('ERRROS:', errors)


def createFile(row):
    rows = json.loads(row)

    with open('dataset6.csv', mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #fieldnames = ['emp_name', 'dept', 'birth_month']
        #dialect = csv.Sniffer().sniff(csv_file.readline())

        #csv_file.seek(0)
        #writer = csv.DictWriter(csv_file, fieldnames=posenet.HEADERS)
        #writer.writeheader()
        writer.writerow(rows.values())


if __name__ == "__main__":
    main()
