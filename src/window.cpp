#include "window.hpp"
#include "context.hpp"

Window::Window()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfwWindow_ = glfwCreateWindow(width_, height_, "Vulkan", nullptr, nullptr);

    // �� �����쿡 this�� �Ŵ޾Ƶд�
    glfwSetWindowUserPointer(glfwWindow_, this);

    // �ݹ� ���: static Ʈ���޸�
    glfwSetFramebufferSizeCallback(glfwWindow_, &Window::FramebufferResizeCallback);
    glfwSetInputMode(glfwWindow_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    //glfwSetCursorPosCallback(glfwWindow_, &Window::CursorPosCallback);

    if (!glfwWindow_)
    {
        std::cerr << "Failure creating glfw window " << std::endl;
    }

    ctx_ = std::make_unique<Context>(glfwWindow_, width_, height_);

}

// --- static Ʈ���޸��� ---
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

// --- ���� �ν��Ͻ� �ڵ鷯�� ---
void Window::OnFramebufferResize(int, int) {
    framebufferResized_ = true;
}

void Window::OnCursorPos(double xpos, double ypos) {
    if (firstMouse_) { lastX_ = xpos; lastY_ = ypos; firstMouse_ = false; return; }
    double xoffset = xpos - lastX_;
    double yoffset = lastY_ - ypos; // ���� ����
    lastX_ = xpos; lastY_ = ypos;

    camera_.yaw += static_cast<float>(xoffset) * camera_.sensitivity;
    camera_.pitch += static_cast<float>(yoffset) * camera_.sensitivity;
    camera_.pitch = glm::clamp(camera_.pitch, -89.0f, 89.0f);
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

        ctx_->draw(camera_);
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

    // �� �� �� �� ��/�� (yaw)
    const float yawSpeed = 120.0f; // deg/sec. ������ 60~180 ���̷�
    if (glfwGetKey(glfwWindow_, GLFW_KEY_LEFT) == GLFW_PRESS) camera_.yaw -= yawSpeed * dt;
    if (glfwGetKey(glfwWindow_, GLFW_KEY_RIGHT) == GLFW_PRESS) camera_.yaw += yawSpeed * dt;

    // �� �� �� �� ��/�Ʒ� (pitch)�� ���ϸ�
    const float pitchSpeed = 90.0f; // deg/sec
    if (glfwGetKey(glfwWindow_, GLFW_KEY_UP) == GLFW_PRESS) camera_.pitch = glm::clamp(camera_.pitch + pitchSpeed * dt, -89.0f, 89.0f);
    if (glfwGetKey(glfwWindow_, GLFW_KEY_DOWN) == GLFW_PRESS) camera_.pitch = glm::clamp(camera_.pitch - pitchSpeed * dt, -89.0f, 89.0f);

    if (glfwGetKey(glfwWindow_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(glfwWindow_, GLFW_TRUE);
    }
}