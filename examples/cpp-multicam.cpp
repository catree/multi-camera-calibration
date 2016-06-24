// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

#include <librealsense/rs.hpp>

#include "example.hpp"
#include "calibration.h"

#include <iostream>
#include <algorithm>

std::vector<texture_buffer> buffers;

double yaw, pitch, lastX, lastY; int ml;
static void on_mouse_button(GLFWwindow * win, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) ml = action == GLFW_PRESS;
}
static double clamp(double val, double lo, double hi) { return val < lo ? lo : val > hi ? hi : val; }
static void on_cursor_pos(GLFWwindow * win, double x, double y)
{
    if (ml)
    {
        yaw = clamp(yaw - (x - lastX), -120, 120);
        pitch = clamp(pitch + (y - lastY), -80, 80);
    }
    lastX = x;
    lastY = y;
}

int main(int argc, char * argv[]) try
{
    rs::log_to_console(rs::log_severity::warn);
    //rs::log_to_file(rs::log_severity::debug, "librealsense.log");

    rs::context ctx;
    if(ctx.get_device_count() == 0) throw std::runtime_error("No device detected. Is it plugged in?");
    
    // Enumerate all devices
    std::vector<rs::device *> devices;
    for(int i=0; i<ctx.get_device_count(); ++i)
    {
        devices.push_back(ctx.get_device(i));
    }

    // Configure and start our devices
    for(auto dev : devices)
    {
        std::cout << "Starting " << dev->get_name() << "... ";
		dev->enable_stream(rs::stream::depth, 480, 360, rs::format::z16, 30);
		dev->enable_stream(rs::stream::color, 640, 480, rs::format::rgb8, 30);

        dev->start();
        dev->set_option(rs::option::r200_lr_auto_exposure_enabled, 1.0);
        rs_apply_depth_control_preset((rs_device*)dev,4 ); //4 or 5
        std::cout << "done." << std::endl;
    }

    // Depth and color
    buffers.resize(ctx.get_device_count() * 2);

    // Open a GLFW window
    glfwInit();
    std::ostringstream ss; ss << "CPP Multi-Camera Example";
    GLFWwindow * win = glfwCreateWindow(1280, 960, ss.str().c_str(), 0, 0);
    GLFWwindow * winCld = glfwCreateWindow(1280, 960, "PointCloud", nullptr, nullptr);
    glfwSetCursorPosCallback(winCld, on_cursor_pos);
    glfwSetMouseButtonCallback(winCld, on_mouse_button);

    glfwMakeContextCurrent(win);

    int windowWidth, windowHeight;
    glfwGetWindowSize(win, &windowWidth, &windowHeight);

    // Does not account for correct aspect ratios
    auto perTextureWidth = windowWidth / devices.size();
    auto perTextureHeight = 480;

    int h_cam, w_cam;
    std::vector<float> fxs;
    std::vector<float> fys;
    std::vector<float> pxs;
    std::vector<float> pys;
    for (auto dev : devices)
    {
        const auto c = dev->get_stream_intrinsics(rs::stream::rectified_color);
        const auto d = dev->get_stream_intrinsics(rs::stream::depth_aligned_to_rectified_color);

        fxs.push_back(c.fx);
        fys.push_back(c.fy);
        pxs.push_back(c.ppx);
        pys.push_back(c.ppy);
        w_cam = c.width;
        h_cam = c.height;

    }

    CalibrationMethod calib(w_cam, h_cam, fxs, fys, pxs, pys);


    while (!glfwWindowShouldClose(win) && !glfwWindowShouldClose(winCld))
    {
        glfwMakeContextCurrent(win);

        // Wait for new images
        glfwPollEvents();
        
        // Draw the images
        int w,h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glfwGetWindowSize(win, &w, &h);
        glPushMatrix();
        glOrtho(0, w, h, 0, -1, +1);
        glPixelZoom(1, -1);
        int i=0, x=0;
        std::vector<uint16_t*> depths;
        std::vector<uint8_t*> colors;

        for (auto dev : devices)
        {
            dev->wait_for_frames();
            buffers[i++].show(*dev, rs::stream::rectified_color, x, 0, perTextureWidth, perTextureHeight);
            buffers[i++].show(*dev, rs::stream::depth_aligned_to_rectified_color, x, perTextureHeight, perTextureWidth, perTextureHeight);
            x += perTextureWidth;

            depths.push_back((uint16_t*)dev->get_frame_data(rs::stream::depth_aligned_to_rectified_color));
            colors.push_back((uint8_t*)dev->get_frame_data(rs::stream::rectified_color));
        }
        calib.addImages(depths, colors);
        float scale1 = devices[0]->get_depth_scale();
        calib.setScale(scale1);
        auto calibrated = calib.solvePose();
        glPopMatrix();
        glfwSwapBuffers(win);

        {
            glfwMakeContextCurrent(winCld);


            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluPerspective(60, (float)1280 / 960, 0.01f, 20.0f);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);
            glTranslatef(0, 0, +0.5f);
            glRotated(pitch, 1, 0, 0);
            glRotated(yaw, 0, 1, 0);
            glTranslatef(0, 0, -0.5f);

            // We will render our depth data as a set of points in 3D space
            glPointSize(2);
            glEnable(GL_DEPTH_TEST);
            glBegin(GL_POINTS);
            for (size_t i = 0; i < devices.size(); i++)
            {
                auto dev = devices[i];
                rs::intrinsics depth_intrin = dev->get_stream_intrinsics(rs::stream::depth_aligned_to_rectified_color);
                auto depth_image = (uint16_t*)dev->get_frame_data(rs::stream::depth_aligned_to_rectified_color);
                auto color_image = (uint8_t*)dev->get_frame_data(rs::stream::rectified_color);
                float scale = dev->get_depth_scale();

                auto pose = calib.getPose(i);
                
                for (int dy = 0; dy < depth_intrin.height; ++dy)
                {
                    for (int dx = 0; dx < depth_intrin.width; ++dx)
                    {
                        // Retrieve the 16-bit depth value and map it into a depth in meters
                        uint16_t depth_value = depth_image[dy * depth_intrin.width + dx];
                        float depth_in_meters = depth_value * scale;

                        // Skip over pixels with a depth value of zero, which is used to indicate no data
                        if (depth_value == 0) continue;

                        // Map from pixel coordinates in the depth image to pixel coordinates in the color image
                        rs::float2 depth_pixel = { (float)dx, (float)dy };
                        rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
                        linalg::aliases::float4 depth_point_h = { depth_point.x, depth_point.y, depth_point.z, 1.0f };

                        auto depth_point_trans = linalg::mul(pose,depth_point_h);

                        glColor3ubv(color_image + (dy * depth_intrin.width + dx) * 3);
            
                        // Emit a vertex at the 3D location of this depth pixel
                        glVertex3f(depth_point_trans.x, depth_point_trans.y, depth_point_trans.z);
                    }
                }
            }
            glEnd();

            glfwSwapBuffers(winCld);
        }

    }

    glfwDestroyWindow(win);
    glfwDestroyWindow(winCld);
    glfwTerminate();
    return EXIT_SUCCESS;
}
catch(const rs::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch(const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
