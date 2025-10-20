#pragma once

class Context;
#include "camera.hpp"

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
    GLFWwindow* glfwWindow_{};
    std::unique_ptr<Context> ctx_;
    Camera camera_;

    bool mouseEnabled_ = false;

    uint32_t initWidth_ = 1400;
    uint32_t initHeight_ = 900;

    bool framebufferResized_ = false;

    // ���콺 ���´� �ν��Ͻ� �����
    bool   firstMouse_ = true;
    double lastX_ = 0.0, lastY_ = 0.0;

    void processKeyboard(float dt);
};
