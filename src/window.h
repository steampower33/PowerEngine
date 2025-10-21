#pragma once

#include "camera.h"

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

private:
    // GLFW�� �䱸�ϴ� ��Ȯ�� �ñ״�ó(��ȯ�� void, ù ���� GLFWwindow*)
    static void CursorPosCallback(GLFWwindow* w, double x, double y);
    static void FramebufferResizeCallback(GLFWwindow* w, int width, int height);
    static void KeyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);

    void OnFramebufferResize(int width, int height);
    void OnCursorPos(double x, double y);
    void OnKey(int key, int scancode, int action, int mods);

private:
    GLFWwindow* glfw_window_{};
    std::unique_ptr<Context> ctx_;
    Camera camera_;

    bool mouse_enabled_ = false;

    uint32_t init_width_ = 1400;
    uint32_t init_height_ = 900;

    bool framebuffer_resized_ = false;

    // ���콺 ���´� �ν��Ͻ� �����
    bool   first_mouse_ = true;
    double last_x_ = 0.0, last_y_ = 0.0;

    void ProcessKeyboard(float dt);
};
