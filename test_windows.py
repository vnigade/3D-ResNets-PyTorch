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
        if opt.window_size > opt.sample_duration:
            # If model is trained on lower number of sample duration, then create batches of sample duration
            # size and take mean of the output.
            # inputs = inputs.reshape((-1, 3, opt.sample_duration, opt.sample_size, opt.sample_size))
            tick = opt.window_size / float(opt.sample_duration)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(opt.sample_duration)])
            inputs = inputs[:,:,offsets,:,:]
        elif opt.window_size < opt.sample_duration:
            raise NotImplemented()
            
        inputs = inputs.cuda()
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

def get_next_video_index(dataset, start_index, max_index):
    _, _, video_info = dataset[start_index]
    prev_video_id = video_info['video_id']
    index = max_index
    for index in range(start_index + 1, max_index):
        _, _, video_info = dataset[index]
        video_id = video_info['video_id']
        if prev_video_id != video_id:
            break
    return prev_video_id, start_index, index
        
def test_batch(dataset, model, opt, class_names, batch_size):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = timeit.default_timer()
    previous_video_id = ''
    test_results = {'results': {}}
    dump_dir = os.path.join(opt.root_path, opt.scores_dump_path)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    total_windows = len(dataset)
    index = 0
    cur_video_id = None
    output_dict = defaultdict(lambda: defaultdict(list))
    while index < total_windows:
        cur_video_id, start_index, end_index = get_next_video_index(dataset, index, total_windows)
        idx = start_index
        while idx < end_index:
            batch_start = idx
            batch_end = idx + batch_size
            if batch_end > end_index:
                batch_end = end_index
            batch = []
            # Get inputs for the batch
            for j in range(batch_start, batch_end):
                print("Reading data {}/{}".format(j, total_windows))
                inputs, targets, video_info = dataset[j]
                video_id = video_info['video_id']
                inputs = inputs.unsqueeze(dim=0)
                batch.append((inputs, video_info))

            inputs = []
            for (input, _) in batch:
                inputs.append(input)
            inputs = torch.cat(inputs, dim=0)
            inputs.cuda()
            outputs = model(inputs)
            if not opt.no_softmax_in_test:
                outputs = F.softmax(outputs, dim=1)
            outputs = outputs.cpu().detach().numpy()
            for i in range(len(batch)):
                video_info = batch[i][1]
                window_id = int(video_info['window_id'])
                rgb_scores = outputs[i].flatten().tolist()
                output_dict["window_" + str(window_id)]["rgb_scores"] = rgb_scores
            idx += batch_size
                
        with open(dump_dir + "/" + cur_video_id, 'w') as outfile:
            json.dump(output_dict, outfile)
            output_dict = defaultdict(lambda: defaultdict(list))
        index = end_index
