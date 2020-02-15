import torch
from torch.autograd import Variable
import torch.nn.functional as F
import timeit
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

    end_time = timeit.default_timer()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    prev_video_id = None
    dump_dir = os.path.join(opt.root_path, opt.scores_dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    for i, (inputs, targets, video_info) in enumerate(data_loader):
        video_id = video_info['video_id'][0]
        window_id = video_info['window_id'][0]
        segment = video_info['segment'][0]
        # print(video_id, window_id, segment)
        if os.path.exists(dump_dir + "/" + video_id):
           continue
        data_time.update(timeit.default_timer() - end_time)
        # if prev_video_id != video_info[0][0]:
        if prev_video_id != video_id:
            if prev_video_id is not None:
                with open(dump_dir + "/" + prev_video_id, 'w') as outfile:
                    json.dump(output_dict, outfile)
                    print("{0} Scores saved for {1} \n"
                          "Batch Time: {2}".format(len(output_dict), prev_video_id, batch_time.avg))
            # prev_video_id = video_info[0][0]
            prev_video_id = video_id
            output_dict = defaultdict(lambda: defaultdict(list))
            idx = 0
         
        # inputs = Variable(inputs, volatile=True)
        # inputs = Variable(inputs)
        inputs = inputs.cuda()
        if opt.window_size != opt.sample_duration:
            # If model is trained on lower number of sample duration, then create batches of sample duration
            # size and take mean of the output.
            inputs = inputs.reshape((-1, 3, opt.sample_duration, opt.sample_size, opt.sample_size))
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        outputs = model(inputs)
        torch.cuda.synchronize()
        prediction_time = (timeit.default_timer() - start_time) * 1000
        batch_time.update(prediction_time)
        if not opt.no_softmax_in_test:
           outputs = F.softmax(outputs)
        outputs = torch.mean(outputs, dim=0)
        # batch_time.update((timeit.default_timer() - start_time) * 1000)
        # assert (idx + 1) == int(video_info[1][0]) 
        assert (idx + 1) == int(window_id)
        # idx = int(video_info[1][0]) 
        idx = int(window_id)
        rgb_scores = outputs.cpu().detach().numpy().flatten().tolist()
        target_actions = targets
        # print(video_id, idx, np.argmax(np.asarray(rgb_scores, dtype=float)), target_actions)
        output_dict["window_" + str(idx)]["rgb_scores"] = rgb_scores 

    if prev_video_id is not None:
        with open(dump_dir + "/" + prev_video_id, 'w') as outfile:
            json.dump(output_dict, outfile)
            print("{0} Scores saved for {1}\n"
                  "Batch Time: {2}".format(len(output_dict), prev_video_id, batch_time.avg))
