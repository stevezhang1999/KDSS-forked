# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List

# 导入数据
def import_data(file_name: str) -> List[str]:
    res = []
    with open(file_name) as f:
        res = f.readlines()
    res = [i for i in res if i != '\n']
    res = [eval(i) for i in res]
    return res




for (model_name, performance_type) in [('LeNet-5', 'lenet-5_compute_performance'), ('ResNet-50', 'resnet-50_compute_performance'), ('VGG-16', 'vgg-16_compute_performance')]:
    y_compute = np.asarray(import_data('/home/lijiakang/KDSS/worker/performance/%s/default_allocator_compute_time.txt' % performance_type))
    y_compute_kgallocv2 = np.asarray(import_data('/home/lijiakang/KDSS/worker/performance/%s/kgmallocv2_allocator_compute_time.txt'  % performance_type))
    y_compute_stream = np.asarray(import_data('/home/lijiakang/KDSS/worker/performance/%s/default_allocator_compute_with_stream_time.txt' % performance_type))
    y_compute_stream_kgallocv2 = np.asarray(import_data('/home/lijiakang/KDSS/worker/performance/%s/kgmallocv2_allocator_compute_with_stream_time.txt' % performance_type))


    print(model_name + ' information')
    
    print('[default allocator] Compute with linear average time: %.3f ms' % y_compute.mean())
    print('[kgmallocv2] Compute with linear average time: %.3f ms' % y_compute_kgallocv2.mean())
    print('[default allocator] Compute with CUDA stream average time: %.3f ms' % y_compute_stream.mean())
    print('[kgmallocv2] Compute with CUDA stream average time: %.3f ms' % y_compute_stream_kgallocv2.mean())

    print()

    print('[default allocator] Compute with linear median time: %.3f ms' % np.median(y_compute))
    print('[kgmallocv2] Compute with linear median time: %.3f ms' % np.median(y_compute_kgallocv2))
    print('[default allocator] Compute with CUDA stream median time: %.3f ms' % np.median(y_compute_stream))
    print('[kgmallocv2] Compute with CUDA stream median time: %.3f ms' % np.median(y_compute_stream_kgallocv2))

    print()

    print('[default allocator] Compute with linear time variance: %.5f' % np.var(y_compute))
    print('[kgmallocv2] Compute with linear time variance: %.5f' % np.var(y_compute_kgallocv2))
    print('[default allocator] Compute with CUDA stream time variance: %.5f' % np.var(y_compute_stream))
    print('[kgmallocv2] Compute with CUDA stream time variance: %.5f' % np.var(y_compute_stream_kgallocv2))

    print()
    print()

    y_compute = y_compute[1000:1300]
    y_compute_stream = y_compute_stream [1000:1300]
    y_compute_kgallocv2 = y_compute_kgallocv2[1000:1300]
    y_compute_stream_kgallocv2 = y_compute_stream_kgallocv2[1000:1300]
    x_compute = np.asarray([i for i in range(1, len(y_compute) + 1)])
    x_compute_stream = np.asarray([i for i in range(1, len(y_compute_stream) + 1)])
    x_compute_kgallocv2 = np.asarray([i for i in range(1, len(y_compute_kgallocv2) + 1)])
    x_compute_stream_kgallocv2 = np.asarray([i for i in range(1, len(y_compute_stream_kgallocv2) + 1)])

    plt.figure(dpi=300,figsize=(25,5))
    plt.xlabel(u'index',horizontalalignment='right', x=1.0)
    plt.ylabel(u'compute time(ms)',horizontalalignment='right', y=1.0)
    plt.grid(True)

    plt.plot(x_compute, y_compute, color='blue', label='Compute with default allocator')

    plt.plot(x_compute_stream, y_compute_stream, color='red', label='Stream compute with default allocator')

    plt.plot(x_compute_kgallocv2, y_compute_kgallocv2, color='green', label='Compute with kgmalloc V2')

    plt.plot(x_compute_stream_kgallocv2, y_compute_stream_kgallocv2, color='purple', label='Stream compute with kgmalloc V2')

    plt.plot()

    plt.legend()

    plt.title(model_name +' compute time difference after cold start')

    plt.savefig(model_name + '_all_compute_time_perf.png')
