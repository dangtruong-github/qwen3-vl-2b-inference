#include "../include/timer.hpp"

double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

CPUTimer::CPUTimer(const char *name) : function_name(name) {
    startTime = now_sec();
}

CPUTimer::~CPUTimer() {
    double endTime = now_sec();
    double elapsedTime = endTime - startTime;
    printf("%s CPU time: %.6f seconds\n", function_name, elapsedTime);
    fflush(stdout);
}
