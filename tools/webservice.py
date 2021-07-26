import requests
import datetime
import json
import time

import numpy as np
import torch
from tqdm import tqdm
from tsai.all import *


def preprocess(x, f):

    def sample(input: np.ndarray, src_freq, dst_freq=5e5, n_samples=None):
        """Samples input series and scales to specified length."""
        n_samples = int(2400 * dst_freq / 2e6)
        output = input[::int(src_freq / dst_freq)]

        if len(output) > n_samples:
            preamble = np.where(abs(output) > 40)[0]
            preamble = preamble[0] if len(preamble) else 0  # no element > 40
            preamble200 = int(200 * dst_freq / 2e6)  # 200 preamble for 2MHz sample frequency
            if preamble > preamble200:
                preamble = min(preamble - preamble200, len(output) - n_samples)
                output = output[preamble:]
            if len(output) > n_samples:
                output = output[:n_samples]
        if len(output) < n_samples:
            output = np.concatenate((output, np.zeros(n_samples - len(output))))
        return output

    x = sample(x, f)
    x = x / 5000 if np.max(x) < 5000 else x / np.max(x)
    x = torch.Tensor(x).view((1, 1, -1))
    return x


t_pre = '1949-10-01'


def get_data(url, auth):
    global t_pre
    t_cur = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    r = requests.get(url, auth=auth, params={'strtime': t_pre, 'endtime': t_cur})
    if r.status_code != 200:
        print(f'GET returns {r.status_code}')
        return
    t_pre = t_cur
    text = json.loads(r.text)
    if text['status'] != 200:
        print(f'GET returns {text["status"]}: {text["messages"]}')
        return
    return ((d['acciId'], d['waveType'], (d['sampleRate'], d['waveData']))
            for d in text['data'])


def put_pred(url, auth, acciid, pred):
    headers = {"content-type":"application/json"}
    r = requests.put(url, auth=auth, headers=headers, data=json.dumps([{'acciId': acciid, 'inteliDiagnosis': pred}]))
    if r.status_code != 200:
        print(f'PUT returns {r.status_code}')
    elif json.loads(r.text)['status'] != 200:
        print(f'PUT returns data status {json.loads(r.text)['status']}')


def main():
    get_url = 'http://121.89.217.11:30000/api/v1/getAccidentByTime'
    put_url = 'http://121.89.217.11:30000/api/v1/putInteliDiagnosis'
    user = 'huake'
    password = 'Qwert123#'
    auth = requests.models.HTTPBasicAuth(user, password)

    model = InceptionTimeXLPlus(1, 2)
    CLASSES = ('雷击', '非雷击')
    model.load_state_dict(torch.load('save/icptxlp_e60_2cls.pt', map_location='cpu'))
    model.eval()
    print('Model loaded')

    prev_acciid = None
    p = None

    while (True):
        data = get_data(get_url, auth)
        if not data:
            continue
        bar = tqdm(data)
        bar.set_description(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        for acciid, wavetype, d in bar:
            if wavetype != 2:
                continue

            f, x = d[0], np.array(eval(d[1]))
            x = preprocess(x, f)
            if acciid != prev_acciid:
                if p is not None:
                    pred = torch.argmax(p, axis=1).item()
                    put_pred(put_url, auth, prev_acciid, CLASSES[pred])
                p = model(x)
            else:
                p += model(x)
            prev_acciid = acciid
        time.sleep(30)


if __name__ == '__main__':
    main()
