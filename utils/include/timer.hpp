#pragma once

#include <time.h>
#include <stdio.h>

double now_sec();

struct CPUTimer {
    // Member variable to store the starting time.
    double startTime;
    const char *function_name;

    // Constructor: Records the start event.
    CPUTimer(const char *name);
    ~CPUTimer();
};
