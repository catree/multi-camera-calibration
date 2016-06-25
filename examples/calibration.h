#pragma once
#include <vector>
#include <stdint.h>

#include "linalg.h"

using linalg::aliases::float3;
using linalg::aliases::float4x4;

class CalibrationMethod {
public:
    // calibration data for all cameras
    CalibrationMethod(const int w, const int h,
        const std::vector<float> &fxs,
        const std::vector<float> &fys,
        const std::vector<float> &pxs,
        const std::vector<float> &pys)
        : w(w), h(h), fxs(fxs), fys(fys), pxs(pxs), pys(pys) {}
    //takes a list of images for depth/color
    void addImages(std::vector<uint16_t*> depths, std::vector<uint8_t*> colors);
    bool solvePose();
    float4x4 getPose(int index);
    void setScale(float s) { scale = s; }
private:
    float4x4 computePose(int index1, int index2);
    float CalibrationMethod::computeReprojectionError(float3 target, float fx, float fy, float ppx, float ppy, float3 source, float4x4 & pose);
    std::vector< std::vector<float3> > points;
    std::vector< float4x4 > poses;
    int n = 0;
    float scale = 1.0f;
    int w;
    int h;
    std::vector<float> fxs;
    std::vector<float> fys;
    std::vector<float> pxs;
    std::vector<float> pys;
};