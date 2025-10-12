#pragma once

#include "preamble.hpp"

class Context;

class Window
{
public:
    Window();
    Window(const Window& rhs) = delete;
    Window(Window&& rhs) = delete;
    ~Window();

    Window& operator=(const Window& rhs) = delete;
    Window& operator=(Window&& rhs) = delete;

    void run();

    GLFWwindow* glfwWindow_ = nullptr;
    bool framebufferResized_ = false;

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);

private:
    Context* ctx_;
};
