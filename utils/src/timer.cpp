#include "../include/timer.hpp"

double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

CPUTimer::CPUTimer(
    const char *name, const bool warm_up_
) : function_name(name), warm_up(warm_up_) {
    startTime = now_sec();
}

CPUTimer::~CPUTimer() {
    double endTime = now_sec();
    double elapsedTime = endTime - startTime;
    if (!warm_up) {
        printf("%s CPU time: %.6f seconds\n", function_name, elapsedTime);
    }
}
