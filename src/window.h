#pragma once

#include "camera.h"

class Context;
struct Camera;
class MouseInteractor;
class GUI;

class Window
{
public:
    Window();
    Window(const Window& rhs) = delete;
    Window(Window&& rhs) = delete;
    ~Window();

    Window& operator=(const Window& rhs) = delete;
    Window& operator=(Window&& rhs) = delete;

    void mainloop();

private:
    static void CursorPosCallback(GLFWwindow* w, double x, double y);
    static void FramebufferResizeCallback(GLFWwindow* w, int width, int height);
    static void KeyCallback(GLFWwindow* w, int key, int scancode, int action, int mods);
    static void MouseButtonCallback(GLFWwindow* w, int button, int action, int mods);

    void OnFramebufferResize(int width, int height);
    void OnCursorPos(double x, double y);
    void OnKey(int key, int scancode, int action, int mods);
    void OnMouseClick(int button, int action, int mods);
    void ProcessKeyboard(float dt);

private:
    GLFWwindow* glfw_window_{};
    std::unique_ptr<Context> ctx_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<MouseInteractor> mouse_interactor_;
    std::unique_ptr<GUI> gui_;

    bool mouse_enabled_ = false;

    uint32_t init_width_ = 1600;
    uint32_t init_height_ = 900;

    bool framebuffer_resized_ = false;

    bool first_mouse_ = true;

    bool print_timestamp_ = false;
    float key_timeout_ = 0.2f;

};
