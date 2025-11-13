#pragma once
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

extern "C"
{
#include <libavutil/mathematics.h>
#include <libavutil/time.h>
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
};