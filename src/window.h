#pragma once

#include "camera.h"

class Context;
struct Camera;
class MouseInteractor;

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
    static void MouseButtonCallback(GLFWwindow* w, int button, int action, int mods);

    void OnFramebufferResize(int width, int height);
    void OnCursorPos(double x, double y);
    void OnKey(int key, int scancode, int action, int mods);
    void OnMouseClick(int button, int action, int mods);

private:
    GLFWwindow* glfw_window_{};
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<MouseInteractor> mouse_interactor_{ nullptr };

    bool mouse_enabled_ = false;

    uint32_t init_width_ = 1400;
    uint32_t init_height_ = 900;

    bool framebuffer_resized_ = false;

    bool first_mouse_ = true;

    void ProcessKeyboard(float dt);
};
