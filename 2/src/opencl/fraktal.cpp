#include "fraktal.h"
#include "common.h"
#include "CL/cl.hpp"
#include "fraktal.cls"

void OpenCL_Atraktor(char *image, int width, int height, float s)
{
    std::vector<cl::Platform>   platform;
    std::vector<cl::Device>     device;
    cl::Program::Sources        kernel;
    cl::Platform::get(&platform);
    if(platform.size() == 0)
    {
        puts("No platforms found!");
        return;
    }

    platform[0].getDevices(CL_DEVICE_TYPE_GPU, &device);
    if(device.size() == 0)
    {
        puts("No devices found!");
        return;
    }

    cl::Context context({device[0]});
    kernel.push_back({kernelSource.c_str(), kernelSource.size()});

    cl::Program program(context, kernel);
    if(program.build({device[0]}) != CL_SUCCESS)
    {
        printf("Error building: %s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0]).c_str());
        return;
    }

    int size = sizeof(char) * width * height;
    cl::Buffer gpuImage(context, CL_MEM_READ_WRITE, size);

    cl::CommandQueue queue(context, device[0]);
    queue.enqueueWriteBuffer(gpuImage, CL_TRUE, 0, size, image);

    cl::KernelFunctor fraktal_opencl(cl::Kernel(program, "fraktal_opencl"), queue, cl::NullRange, cl::NDRange(height * std::min(256, width)), cl::NDRange(std::min(256, width)));
    fraktal_opencl(gpuImage, width, height, s);

    queue.enqueueReadBuffer(gpuImage, CL_TRUE, 0, size, image);
}
