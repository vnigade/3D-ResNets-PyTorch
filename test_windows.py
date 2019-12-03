import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json
from collections import defaultdict
import numpy as np

from utils import AverageMeter


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i]],
            'score': sorted_scores[i]
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    prev_video_id = None
    dump_dir = os.path.join(opt.root_path, opt.scores_dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    for i, (inputs, targets, video_info) in enumerate(data_loader):
        if os.path.exists(dump_dir + "/" + video_info[0][0]):
           continue
        data_time.update(time.time() - end_time)
        if prev_video_id != video_info[0][0]:
            if prev_video_id is not None:
                with open(dump_dir + "/" + prev_video_id, 'w') as outfile:
                    json.dump(output_dict, outfile)
                    print("{0} Scores saved for {1} \n"
                          "Batch Time: {2}".format(len(output_dict), prev_video_id, batch_time.avg))
            prev_video_id = video_info[0][0]
            output_dict = defaultdict(lambda: defaultdict(list))
            idx = 0
         
        inputs = Variable(inputs, volatile=True)
        if opt.window_size != opt.sample_duration:
            inputs = inputs.reshape((-1, 3, opt.sample_duration, opt.sample_size, opt.sample_size))
        start_time = time.time()
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs)
        outputs = torch.mean(outputs, dim=0)
        batch_time.update(time.time() - start_time)
        assert (idx + 1) == int(video_info[1][0]) 
        idx = int(video_info[1][0]) 
        rgb_scores = outputs.cpu().detach().numpy().flatten().tolist()
        # print(idx, np.argmax(np.asarray(rgb_scores, dtype=float)))
        output_dict["window_" + str(idx)]["rgb_scores"] = rgb_scores 

    if prev_video_id is not None:
        with open(dump_dir + "/" + prev_video_id, 'w') as outfile:
            json.dump(output_dict, outfile)
            print("{0} Scores saved for {1}\n"
                  "Batch Time: {2}".format(len(output_dict), prev_video_id, batch_time.avg))
