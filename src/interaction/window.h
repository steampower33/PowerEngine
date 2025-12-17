#pragma once

struct Camera;
class MouseInteractor;
class Renderer;

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
    static void ScrollCallback(GLFWwindow* w, double xoffset, double yoffset);

    void OnFramebufferResize(int width, int height);
    void OnCursorPos(double x, double y);
    void OnKey(int key, int scancode, int action, int mods);
    void OnMouseClick(int button, int action, int mods);
    void OnMouseWheel(double xoffset, double yoffset);

    void ProcessKeyboard(float dt);

private:
    GLFWwindow* glfw_window_{};
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<MouseInteractor> mouse_interactor_;
    std::unique_ptr<Renderer> renderer_;

    uint32_t init_width_ = 1600;
    uint32_t init_height_ = 900;

    bool framebuffer_resized_ = false;

    bool first_mouse_ = true;

    bool paused_ = false;
    bool pause_eachframe_ = false;
    float key_timeout_ = 0.2f;

    float target_sim_fps_ = 144.0f;
    double sim_dt_ = 1.0 / target_sim_fps_;

    double sim_accum_ = 0.0;
};
