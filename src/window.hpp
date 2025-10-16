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

    // ���� �ν��Ͻ� ���� (this ��� ����)
    void OnCursorPos(double x, double y);
    void OnFramebufferResize(int width, int height);


private:
    GLFWwindow* glfwWindow_{};
    std::unique_ptr<Context> ctx_;
    Camera camera_;

    bool framebufferResized_ = false;

    // ���콺 ���´� �ν��Ͻ� �����
    bool   firstMouse_ = true;
    double lastX_ = 0.0, lastY_ = 0.0;

    void processKeyboard(float dt);
};
