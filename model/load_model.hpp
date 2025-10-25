#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.hpp"

void init_model_weights(const char* path, QwenConfig* config, QwenWeight* weights);
void init_model_run_state(QwenRunState *run_state);

void free_model_weights(QwenWeight* weights);