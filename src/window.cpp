#include "window.hpp"
#include "context.hpp"

Window::Window()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfwWindow_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    // 이 윈도우에 this를 매달아둔다
    glfwSetWindowUserPointer(glfwWindow_, this);

    // 콜백 등록: static 트램펄린
    glfwSetFramebufferSizeCallback(glfwWindow_, &Window::FramebufferResizeCallback);
    glfwSetInputMode(glfwWindow_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetCursorPosCallback(glfwWindow_, &Window::CursorPosCallback);

    if (!glfwWindow_)
    {
        std::cerr << "Failure creating glfw window " << std::endl;
    }

    ctx_ = new Context(glfwWindow_);
    camera_ = new Camera();
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

// --- 실제 인스턴스 핸들러들 ---
void Window::OnFramebufferResize(int, int) {
    framebufferResized_ = true;
}

void Window::OnCursorPos(double xpos, double ypos) {
    if (firstMouse_) { lastX_ = xpos; lastY_ = ypos; firstMouse_ = false; return; }
    double xoffset = xpos - lastX_;
    double yoffset = lastY_ - ypos; // 상하 반전
    lastX_ = xpos; lastY_ = ypos;

    camera_->yaw += static_cast<float>(xoffset) * camera_->sensitivity;
    camera_->pitch += static_cast<float>(yoffset) * camera_->sensitivity;
    camera_->pitch = glm::clamp(camera_->pitch, -89.0f, 89.0f);
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

        ctx_->draw();
    }

    ctx_->device_.waitIdle();
}

void Window::processKeyboard(float dt) {
    float v = camera_->moveSpeed * dt;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_W) == GLFW_PRESS)
        camera_->position += camera_->front() * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_S) == GLFW_PRESS) 
        camera_->position -= camera_->front() * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_A) == GLFW_PRESS) 
        camera_->position -= camera_->right() * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_D) == GLFW_PRESS) 
        camera_->position += camera_->right() * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_SPACE) == GLFW_PRESS) 
        camera_->position += glm::vec3(0, 1, 0) * v;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) 
        camera_->position -= glm::vec3(0, 1, 0) * v;

    if (glfwGetKey(glfwWindow_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(glfwWindow_, GLFW_TRUE);
    }

    //// FOV 줌(선택)
    //if (glfwGetKey(glfwWindow_, GLFW_KEY_Q) == GLFW_PRESS) 
    //    camera_->fov = glm::max(20.0f, camera_->fov - 40.0f * dt);
    //if (glfwGetKey(glfwWindow_, GLFW_KEY_E) == GLFW_PRESS) 
    //    camera_->fov = glm::min(90.0f, camera_->fov + 40.0f * dt);
}