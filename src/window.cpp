#include "window.hpp"
#include "context.hpp"

Window::Window()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfwWindow_ = glfwCreateWindow(initWidth_, initHeight_, "Vulkan", nullptr, nullptr);

    // 이 윈도우에 this를 매달아둔다
    glfwSetWindowUserPointer(glfwWindow_, this);

    // 콜백 등록: static 트램펄린
    glfwSetFramebufferSizeCallback(glfwWindow_, &Window::FramebufferResizeCallback);
    glfwSetCursorPosCallback(glfwWindow_, &Window::CursorPosCallback);
    glfwSetKeyCallback(glfwWindow_, &Window::KeyCallback); // <-- 이 줄을 추가!

    if (!glfwWindow_)
    {
        std::cerr << "Failure creating glfw window " << std::endl;
    }

    ctx_ = std::make_unique<Context>(glfwWindow_, initWidth_, initHeight_);

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
    framebufferResized_ = true;
}

void Window::OnCursorPos(double xpos, double ypos)
{
    // 이 가드(guard) 코드를 추가합니다!
    if (!mouseEnabled_) {
        return;
    }

    if (firstMouse_) { lastX_ = xpos; lastY_ = ypos; firstMouse_ = false; return; }
    double xoffset = xpos - lastX_;
    double yoffset = lastY_ - ypos; // 상하 반전
    lastX_ = xpos; lastY_ = ypos;

    camera_.yaw += static_cast<float>(xoffset) * camera_.sensitivity;
    camera_.pitch += static_cast<float>(yoffset) * camera_.sensitivity;
    camera_.pitch = glm::clamp(camera_.pitch, -89.0f, 89.0f);
}

void Window::OnKey(int key, int scancode, int action, int mods) {
    // 키가 '눌리는 순간(PRESS)'에만 반응하도록 합니다.
    if (key == GLFW_KEY_F && action == GLFW_PRESS) {
        mouseEnabled_ = !mouseEnabled_;

        if (mouseEnabled_) {
            glfwSetInputMode(glfwWindow_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            firstMouse_ = true; // 모드 전환 시 카메라 튀는 현상 방지
        }
        else {
            glfwSetInputMode(glfwWindow_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
    }
}

Window::~Window()
{
    glfwDestroyWindow(glfwWindow_);

    glfwTerminate();
}

void Window::run()
{
    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(glfwWindow_)) {
        double current = glfwGetTime();
        float dt = static_cast<float>(current - lastTime);
        lastTime = current;

        glfwPollEvents();
        processKeyboard(dt);

        ctx_->draw(camera_, dt);
    }

    ctx_->device_.waitIdle();
}

void Window::processKeyboard(float dt) {
    float v = camera_.moveSpeed * dt;
    
    // ASDW
    if (glfwGetKey(glfwWindow_, GLFW_KEY_W) == GLFW_PRESS)
        camera_.position += camera_.front() * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_S) == GLFW_PRESS) 
        camera_.position -= camera_.front() * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_A) == GLFW_PRESS) 
        camera_.position -= camera_.right() * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_D) == GLFW_PRESS) 
        camera_.position += camera_.right() * v;

    // QE
    if (glfwGetKey(glfwWindow_, GLFW_KEY_Q) == GLFW_PRESS)
        camera_.position -= glm::vec3(0, 1, 0) * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_E) == GLFW_PRESS)
        camera_.position += glm::vec3(0, 1, 0) * v;

    // ← → 로 고개 좌/우 (yaw)
    const float yawSpeed = 120.0f; // deg/sec. 취향대로 60~180 사이로
    if (glfwGetKey(glfwWindow_, GLFW_KEY_LEFT) == GLFW_PRESS) camera_.yaw -= yawSpeed * dt;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_RIGHT) == GLFW_PRESS) camera_.yaw += yawSpeed * dt;

    // ↑ ↓ 로 고개 위/아래 (pitch)도 원하면
    const float pitchSpeed = 90.0f; // deg/sec
    if (glfwGetKey(glfwWindow_, GLFW_KEY_UP) == GLFW_PRESS) camera_.pitch = glm::clamp(camera_.pitch + pitchSpeed * dt, -89.0f, 89.0f);
    if (glfwGetKey(glfwWindow_, GLFW_KEY_DOWN) == GLFW_PRESS) camera_.pitch = glm::clamp(camera_.pitch - pitchSpeed * dt, -89.0f, 89.0f);

    if (glfwGetKey(glfwWindow_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(glfwWindow_, GLFW_TRUE);
    }
}