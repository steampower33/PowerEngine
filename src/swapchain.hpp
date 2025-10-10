#pragma once

#include "preamble.hpp"

class Context;

class Swapchain
{
public:
    Swapchain(Context* ctx);
    Swapchain(const Swapchain& rhs) = delete;
    Swapchain(Swapchain&& rhs) = delete;
    ~Swapchain();

    Swapchain& operator=(const Swapchain& rhs) = delete;
    Swapchain& operator=(Swapchain&& rhs) = delete;

    bool draw();

    Context* ctx;


private:
};