#!/bin/bash
HIP_PLATFORM='amd' hipcc -xhip $1.cu -o $1