#pragma once

#include <time.h>
#include <stdio.h>

double now_sec();

struct CPUTimer {
    // Member variable to store the starting time.
    double startTime;
    const char *function_name;
    const bool warm_up;

    // Constructor: Records the start event.
    CPUTimer(const char *name, const bool warm_up_);
    ~CPUTimer();
};
