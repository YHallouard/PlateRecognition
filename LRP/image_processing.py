import numpy as np
import copy

def ROI(imgcv, net):
    heigth, width = imgcv.shape[:2]
    first_level = [
        {'x': heigth / 2, 'y': width / 2, 'size': [[1., 1.], [0.7, 0.7], [0.8, 0.5], [0.5, 0.8]], 'heigth': heigth,
         'width': width}]
    heigth, width = (np.asarray(imgcv.shape[:2]) / 2).astype(int)
    second_level = []
    for i in range(2):
        for j in range(2):
            second_level.append({'x': heigth * i + heigth / 2, 'y': width * j + width / 2,
                                 'size': [[0.7, 0.7], [0.8, 0.5], [0.5, 0.8]], 'heigth': heigth, 'width': width})

    result = []
    for meta in first_level:
        for i in range(len(meta['size'])):
            xw = int(meta['x'] - meta['size'][i][0] / 2 * meta['heigth'])
            Xw = int(meta['x'] + meta['size'][i][0] / 2 * meta['heigth'])
            yw = int(meta['y'] - meta['size'][i][1] / 2 * meta['width'])
            Yw = int(meta['y'] + meta['size'][i][1] / 2 * meta['width'])

            img = imgcv[xw:Xw, yw:Yw, :]

            pred = net.return_predict(img)
            if len(pred) > 0:
                best_confidence = np.argmax(np.asarray([pred[i]['confidence'] \
                                                        for i in range(len(pred))]))

                data = copy.deepcopy(meta)
                data['size'] = meta['size'][i]
                result.append({'w_topleft': {'x': xw, 'y': yw},
                               'w_bottom': {'x': Xw, 'y': Yw},
                               'confidence': pred[best_confidence]['confidence'],
                               'pred': pred[best_confidence]})

    for meta in second_level:
        for i in range(len(meta['size'])):
            xw = int(meta['x'] - meta['size'][i][0] / 2 * meta['heigth'])
            Xw = int(meta['x'] + meta['size'][i][0] / 2 * meta['heigth'])
            yw = int(meta['y'] - meta['size'][i][1] / 2 * meta['width'])
            Yw = int(meta['y'] + meta['size'][i][1] / 2 * meta['width'])

            img = imgcv[xw:Xw, yw:Yw, :]

            pred = net.return_predict(img)
            if len(pred) > 0:
                best_confidence = np.argmax(np.asarray([pred[i]['confidence'] \
                                                        for i in range(len(pred))]))

                data = copy.deepcopy(meta)
                data['size'] = meta['size'][i]
                result.append({'w_topleft': {'x': xw, 'y': yw},
                               'w_bottom': {'x': Xw, 'y': Yw},
                               'confidence': pred[best_confidence]['confidence'],
                               'pred': pred[best_confidence]})

    if len(result) > 0:
        best_res = np.argmax(np.asarray([result[i]['confidence'] \
                                         for i in range(len(result))]))

        return result[best_res]
    else:
        return None

