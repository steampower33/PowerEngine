#include "context.hpp"
#include "window.hpp"

Window::Window()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    glfwWindow_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(glfwWindow_, this);
    glfwSetFramebufferSizeCallback(glfwWindow_, framebufferResizeCallback);

    if (!glfwWindow_)
    {
        std::cerr << "Failure creating glfw window " << std::endl;
    }

    ctx_ = new Context(glfwWindow_);
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = static_cast<Window*>(glfwGetWindowUserPointer(window));
    app->framebufferResized_ = true;
}

Window::~Window()
{
    glfwDestroyWindow(glfwWindow_);

    glfwTerminate();
}

void Window::run()
{
    while (!glfwWindowShouldClose(glfwWindow_)) {
        glfwPollEvents();
        ctx_->draw();
    }

    ctx_->device_.waitIdle();
}
