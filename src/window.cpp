#include "window.h"
#include "context.h"

Window::Window()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfw_window_ = glfwCreateWindow(init_width_, init_height_, "Vulkan", nullptr, nullptr);

    // 이 윈도우에 this를 매달아둔다
    glfwSetWindowUserPointer(glfw_window_, this);

    // 콜백 등록: static 트램펄린
    glfwSetFramebufferSizeCallback(glfw_window_, &Window::FramebufferResizeCallback);
    glfwSetCursorPosCallback(glfw_window_, &Window::CursorPosCallback);
    glfwSetKeyCallback(glfw_window_, &Window::KeyCallback); // <-- 이 줄을 추가!

    if (!glfw_window_)
    {
        std::cerr << "Failure creating glfw window " << std::endl;
    }

    ctx_ = std::make_unique<Context>(glfw_window_, init_width_, init_height_);

}

// --- static 트램펄린들 ---
void Window::FramebufferResizeCallback(GLFWwindow* w, int width, int height) {
    if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
        self->OnFramebufferResize(width, height);
    }
}

void Window::CursorPosCallback(GLFWwindow* w, double x, double y) {
    if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
        self->OnCursorPos(x, y);
    }
}

void Window::KeyCallback(GLFWwindow* w, int key, int scancode, int action, int mods) {
    if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
        self->OnKey(key, scancode, action, mods);
    }
}

// --- 실제 인스턴스 핸들러들 ---
void Window::OnFramebufferResize(int, int) 
{
    framebuffer_resized_ = true;
}

void Window::OnCursorPos(double xpos, double ypos)
{
    // 이 가드(guard) 코드를 추가합니다!
    if (!mouse_enabled_) {
        return;
    }

    if (first_mouse_) { last_x_ = xpos; last_y_ = ypos; first_mouse_ = false; return; }
    double xoffset = xpos - last_x_;
    double yoffset = last_y_ - ypos; // 상하 반전
    last_x_ = xpos; last_y_ = ypos;

    camera_.yaw += static_cast<float>(xoffset) * camera_.sensitivity;
    camera_.pitch += static_cast<float>(yoffset) * camera_.sensitivity;
    camera_.pitch = glm::clamp(camera_.pitch, -89.0f, 89.0f);
}

void Window::OnKey(int key, int scancode, int action, int mods) {
    // 키가 '눌리는 순간(PRESS)'에만 반응하도록 합니다.
    if (key == GLFW_KEY_F && action == GLFW_PRESS) {
        mouse_enabled_ = !mouse_enabled_;

        if (mouse_enabled_) {
            glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            first_mouse_ = true; // 모드 전환 시 카메라 튀는 현상 방지
        }
        else {
            glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

Window::~Window()
{
    glfwDestroyWindow(glfw_window_);

    glfwTerminate();
}

void Window::run()
{
    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(glfw_window_)) {
        double current = glfwGetTime();
        float dt = static_cast<float>(current - lastTime);
        lastTime = current;

        glfwPollEvents();
        ProcessKeyboard(dt);

        ctx_->Draw(camera_, dt);
    }

    ctx_->WaitIdle();
}

void Window::ProcessKeyboard(float dt) {
    float v = camera_.move_speed * dt;
    
    // ASDW
    if (glfwGetKey(glfw_window_, GLFW_KEY_W) == GLFW_PRESS)
        camera_.position += camera_.Front() * v;
    if (glfwGetKey(glfw_window_, GLFW_KEY_S) == GLFW_PRESS) 
        camera_.position -= camera_.Front() * v;
    if (glfwGetKey(glfw_window_, GLFW_KEY_A) == GLFW_PRESS) 
        camera_.position -= camera_.Right() * v;
    if (glfwGetKey(glfw_window_, GLFW_KEY_D) == GLFW_PRESS) 
        camera_.position += camera_.Right() * v;

    // QE
    if (glfwGetKey(glfw_window_, GLFW_KEY_Q) == GLFW_PRESS)
        camera_.position -= glm::vec3(0, 1, 0) * v;
    if (glfwGetKey(glfw_window_, GLFW_KEY_E) == GLFW_PRESS)
        camera_.position += glm::vec3(0, 1, 0) * v;

    // ← → 로 고개 좌/우 (yaw)
    const float yawSpeed = 120.0f; // deg/sec. 취향대로 60~180 사이로
    if (glfwGetKey(glfw_window_, GLFW_KEY_LEFT) == GLFW_PRESS) camera_.yaw -= yawSpeed * dt;
    if (glfwGetKey(glfw_window_, GLFW_KEY_RIGHT) == GLFW_PRESS) camera_.yaw += yawSpeed * dt;

    // ↑ ↓ 로 고개 위/아래 (pitch)도 원하면
    const float pitchSpeed = 90.0f; // deg/sec
    if (glfwGetKey(glfw_window_, GLFW_KEY_UP) == GLFW_PRESS) camera_.pitch = glm::clamp(camera_.pitch + pitchSpeed * dt, -89.0f, 89.0f);
    if (glfwGetKey(glfw_window_, GLFW_KEY_DOWN) == GLFW_PRESS) camera_.pitch = glm::clamp(camera_.pitch - pitchSpeed * dt, -89.0f, 89.0f);

    if (glfwGetKey(glfw_window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(glfw_window_, GLFW_TRUE);
    }
}