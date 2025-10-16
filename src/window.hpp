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

    // 실제 인스턴스 로직 (this 사용 가능)
    void OnCursorPos(double x, double y);
    void OnFramebufferResize(int width, int height);


private:
    GLFWwindow* glfwWindow_{};
    std::unique_ptr<Context> ctx_;
    Camera camera_;

    bool framebufferResized_ = false;

    // 마우스 상태는 인스턴스 멤버로
    bool   firstMouse_ = true;
    double lastX_ = 0.0, lastY_ = 0.0;

    void processKeyboard(float dt);
};
