import os
import argparse
import xml.etree.ElementTree as ET
from utils import *


argparser = argparse.ArgumentParser()

argparser.add_argument(
    '--dataset',
    help='path to train subset')

argparser.add_argument(
    '--for_classes',
    default=None,
    help='calculate centroids only for specified class'
)


def convert_annotation(in_file_path, out_file_path, valid_classes):
    in_file = open(in_file_path, 'r')
    out_file = open(out_file_path, 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    bboxes_converted = 0
    for obj in root.iter('object'):
        cls, cls_id = convert_label(obj.find('name').text)

        if not cls:
            continue

        if valid_classes and cls not in valid_classes:
            continue

        xmlbox = obj.find('bndbox')

        xmin, xmax = float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text)
        ymin, ymax = float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)

        bbox = xmin, xmax, ymin, ymax

        try:
            assert all(val >= 0 for val in bbox), "All values must be >= 0"
            assert xmax > xmin, "Xmax must be >= Xmin"
            assert ymax > ymin, "Ymax must be >= Ymin"

            bb = convert_bbox((w, h), bbox)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

            bboxes_converted += 1
        except AssertionError as exc:
            print(
                "Data consistency error: %s (%.2f, %.2f, %.2f, %.2f) in file: %s" % (
                    exc, xmin, xmax, ymin, ymax, in_file_path
                )
            )

    out_file.flush()
    out_file.close()

    in_file.close()

    if bboxes_converted == 0:
        os.remove(out_file_path)
        raise ValueError('No bboxes converted!')


def main(args):
    outputs = {}

    dataset_path = args.dataset

    labels = CLASSES
    if args.for_classes:
        classes_str = args.for_classes
        labels = list(map(str.strip, classes_str.split(',')))

        print('Generating bboxes for classes: %s' % labels)

    for set_type in SETS:
        subset_path = os.path.join(dataset_path, set_type)

        list_file_path = os.path.join(subset_path, '{}.txt'.format(set_type))
        print('Generating {}'.format(list_file_path))

        images_path = os.path.join(subset_path, 'images')
        annotations_path = os.path.join(subset_path, 'ann')
        labels_path = os.path.join(subset_path, 'labels')

        if not os.path.exists(labels_path):
            os.mkdir(labels_path)

        with open(list_file_path, 'w') as list_file:
            for image_file in os.listdir(images_path):
                try:
                    image_name = os.path.splitext(image_file)[0]
                    annotation_file = '{}.xml'.format(image_name)
                    label_file = '{}.txt'.format(image_name)

                    image_path = os.path.join(images_path, image_file)
                    annotation_path = os.path.join(annotations_path, annotation_file)
                    label_path = os.path.join(labels_path, label_file)

                    assert os.path.exists(annotation_path), 'Annotation does not exists!'

                    convert_annotation(annotation_path, label_path, labels)
                    list_file.write(image_path + '\n')

                    if not os.path.exists(label_path):
                        raise EnvironmentError('Label file %s was not created!' % label_path)

                except (AssertionError, ValueError) as exc:
                    print('Cannot process {}: {}'.format(image_file, exc))

        outputs[set_type] = list_file_path

    print('Generating classes.names file')
    with open(os.path.join(dataset_path, 'classes.names'), 'w') as classes_file:
        for class_name in labels:
            classes_file.write(class_name + '\n')

    print('Generating dataset.data file')
    with open(os.path.join(dataset_path, 'dataset.data'), 'w') as dataset_file:
        dataset_file.write(
            OUTPUT_TEMPLATE.format(len(labels), outputs['train'], outputs['val'], dataset_path)
        )

    print('Done!')


if __name__ == '__main__':
    argv = argparser.parse_args()
    main(argv)
