#include "window.hpp"

Window::Window()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    if (!window)
    {
        std::cerr << "Failure creating glfw window " << std::endl;
    }

    ctx = new Context(this);
}

void Window::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = static_cast<Window*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

Window::~Window()
{
    glfwDestroyWindow(window);

    glfwTerminate();
}

void Window::run()
{
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ctx->draw();
    }

    ctx->device.waitIdle();
}
