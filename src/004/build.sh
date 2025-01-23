#!/bin/bash

g++ "$1".cc -Ofast -march=native -o "$1" -g --std=c++20