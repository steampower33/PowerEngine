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
    // GLFW가 요구하는 정확한 시그니처(반환형 void, 첫 인자 GLFWwindow*)
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

    // 마우스 상태는 인스턴스 멤버로
    bool   firstMouse_ = true;
    double lastX_ = 0.0, lastY_ = 0.0;

    void processKeyboard(float dt);
};
