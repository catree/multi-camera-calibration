#pragma once
#include <vector>
#include <stdint.h>

#include "linalg.h"

using linalg::aliases::float3;
using linalg::aliases::float4x4;

class CalibrationMethod {
public:
    //takes a list of images for depth/color, and their calibration
    void addImages(std::vector<uint16_t*> depths, std::vector<uint8_t*> colors, int w, int h,
        std::vector<float> fxs, std::vector<float> fys, std::vector<float> pxs, std::vector<float> pys);
    bool solvePose();
    float4x4 getPose(int index);
    void setScale(float s) { scale = s; }
private:
    float4x4 computePose(int index1, int index2);
    std::vector< std::vector<float3> > points;
    std::vector< float4x4 > poses;
    int n = 0;
    float scale = 1.0f;
};