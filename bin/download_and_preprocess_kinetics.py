import os
import sys
import time
import logging
import threading

import ujson as json


def main(args):
    logging.info('args: %s', args)
    kinetics_filename = os.path.join(args.path, 'kinetics_train.json')
    if not os.path.exists(kinetics_filename):
        raise Exception('not exists kinetic data file at %s', kinetics_filename)
    for foldername in ['original', 'processed']:
        if not os.path.exists(os.path.join(args.path, foldername)):
            os.mkdir(os.path.join(args.path, foldername))

    kinetics = json.load(open(kinetics_filename))
    keys = sorted(kinetics.keys())

    if not args.process:
        for i, key in enumerate(keys):
            value = kinetics[key]
            original_path = os.path.join(args.path, 'original', key + '.mp4')
            if os.path.exists(original_path):
                logging.info('[%04d/%04d] exists video %s', i, len(kinetics), key)
                continue
            try:
                logging.info('[%04d/%04d] download video %s', i, len(kinetics), key)
                command = [
                    'youtube-dl', '--quiet', '--no-warnings', '-f', 'mp4',
                    '-o', '"%s"' % original_path, '"%s"' % value['url'], '&',
                ]
                logging.info(' '.join(command))
                os.system(' '.join(command))
                time.sleep(0.5)
            except Exception as e:
                logging.error('error with %s video', key)
                logging.error('%s: %s', type(e), str(e))

    else:
        for i, key in enumerate(keys):
            value = kinetics[key]
            original_path = os.path.join(args.path, 'original', key + '.mp4')
            processed_path = os.path.join(args.path, 'processed', key + '.mp4')
            if not os.path.exists(original_path):
                logging.info('[%04d/%04d] not exists video %s', i, len(kinetics), key)
                continue
            if os.path.exists(processed_path):
                logging.info('[%04d/%04d] already processed video %s', i, len(kinetics), key)
                continue
            try:
                logging.info('[%04d/%04d] process video %s', i, len(kinetics), key)
                command = [
                    'ffmpeg', '-loglevel panic',
                    '-i', '"%s"' % original_path,
                    '-t', '%f' % value['duration'],
                    '-ss', '%f' % value['annotations']['segment'][0],
                    '-strict', '-2',
                    '"%s"' % processed_path,
                    '&'
                ]
                logging.info(' '.join(command))
                os.system(' '.join(command))
                time.sleep(2)
            except Exception as e:
                logging.error('error with %s video', key)
                logging.error('%s: %s', type(e), str(e))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--process', action='store_true')

    parser.add_argument('--log-filename', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parsed_args = parser.parse_args()
     
    log_format = '[%(asctime)s %(levelname)s] %(message)s'
    level = logging.DEBUG if parsed_args.debug else logging.INFO
    if not parsed_args.log_filename:
        logging.basicConfig(level=level, format=log_format, stream=sys.stderr)
    else:
        logging.basicConfig(level=level, format=log_format, filename=parsed_args.log_filename)

    main(parsed_args)
