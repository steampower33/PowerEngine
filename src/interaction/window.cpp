
#include "camera.h"
#include "mouse_interactor.h"
#include "renderer.h"
#include "vulkan_utils.h"

#include "window.h"

Window::Window()
{
	glfwInit();

	GLFWmonitor* primary = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(primary);

	int screenWidth = mode->width;
	int screenHeight = mode->height;

	float ratio = 0.9f;

	init_width_ = static_cast<int>(screenWidth * ratio);
	init_height_ = static_cast<int>(init_width_ * 9.0f / 16.0f);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfw_window_ = glfwCreateWindow(init_width_, init_height_, "Vulkan", nullptr, nullptr);

	glfwSetWindowUserPointer(glfw_window_, this);

	glfwSetFramebufferSizeCallback(glfw_window_, &Window::FramebufferResizeCallback);
	glfwSetCursorPosCallback(glfw_window_, &Window::CursorPosCallback);
	glfwSetKeyCallback(glfw_window_, &Window::KeyCallback);
	glfwSetMouseButtonCallback(glfw_window_, &Window::MouseButtonCallback);
	glfwSetScrollCallback(glfw_window_, &Window::ScrollCallback);

	if (!glfw_window_)
	{
		std::cerr << "Failure creating glfw window " << std::endl;
	}

	camera_ = std::make_unique<Camera>();
	mouse_interactor_ = std::make_unique<MouseInteractor>();
	renderer_ = std::make_unique<Renderer>(glfw_window_, init_width_, init_height_);
}

void Window::mainloop()
{
	double lastTime = glfwGetTime();

	while (!glfwWindowShouldClose(glfw_window_)) {
		glfwPollEvents();

		double current = glfwGetTime();
		float dt = static_cast<float>(current - lastTime);
		lastTime = current;

		ProcessKeyboard(dt);

		key_timeout_ -= dt;

		renderer_->Update(*camera_, *mouse_interactor_, dt);
		renderer_->Draw();
	}

	renderer_->WaitIdle();
}

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

void Window::MouseButtonCallback(GLFWwindow* w, int button, int action, int mods)
{
	if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
		self->OnMouseClick(button, action, mods);
	}
}

void Window::ScrollCallback(GLFWwindow* w, double xoffset, double yoffset)
{
	if (auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w))) {
		self->OnMouseWheel(xoffset, yoffset);
	}
}

void Window::OnFramebufferResize(int, int)
{
	framebuffer_resized_ = true;
}

void Window::OnCursorPos(double xpos, double ypos)
{
	static double lastX = xpos;
	static double lastY = ypos;

	if (mouse_enabled_) {
		if (first_mouse_) {
			lastX = xpos;
			lastY = ypos;
			first_mouse_ = false;
		}

		double xoffset = xpos - lastX;
		double yoffset = lastY - ypos;

		camera_->yaw += static_cast<float>(xoffset) * camera_->sensitivity;
		camera_->pitch += static_cast<float>(yoffset) * camera_->sensitivity;
		camera_->pitch = glm::clamp(camera_->pitch, -89.0f, 89.0f);
	}

	lastX = xpos;
	lastY = ypos;

	mouse_interactor_->mouse_pos_ = glm::vec2(static_cast<float>(xpos), static_cast<float>(ypos));
}

void Window::OnKey(int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_F && action == GLFW_PRESS) {
		mouse_enabled_ = !mouse_enabled_;

		if (mouse_enabled_) {
			//glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			first_mouse_ = true;
		}
		else {
			glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
}

void Window::OnMouseClick(int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			mouse_interactor_->is_left_button_down_event = true;
			mouse_interactor_->is_left_down = true;
		}
		else if (action == GLFW_RELEASE) {
			mouse_interactor_->is_left_button_up_event = true;
			mouse_interactor_->is_left_down = false;
		}
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		if (action == GLFW_PRESS)   mouse_interactor_->is_right_button_down_event = true;
		else if (action == GLFW_RELEASE) mouse_interactor_->is_right_button_up_event = true;
	}
}

void Window::OnMouseWheel(double xoffset, double yoffset)
{
	float sensitivity = 0.2f;

	if (yoffset > 0) {
		//std::cout << "DEPTH IN" << std::endl;
		//std::cout << yoffset << std::endl;
		mouse_interactor_->depth_state = vku::DepthState::MOUSE_DEPTH_IN;
		mouse_interactor_->depth_delta = sensitivity;
	}
	else if (yoffset < 0) {
		//std::cout << "DEPTH OUT" << std::endl;
		//std::cout << yoffset << std::endl;
		mouse_interactor_->depth_state = vku::DepthState::MOUSE_DEPTH_OUT;
		mouse_interactor_->depth_delta = sensitivity;
	}
}

Window::~Window()
{
	glfwDestroyWindow(glfw_window_);

	glfwTerminate();
}

void Window::ProcessKeyboard(float dt) {
	float v = camera_->move_speed * dt;

	// ASDW
	if (glfwGetKey(glfw_window_, GLFW_KEY_W) == GLFW_PRESS)
		camera_->position += camera_->Front() * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_S) == GLFW_PRESS)
		camera_->position -= camera_->Front() * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_A) == GLFW_PRESS)
		camera_->position -= camera_->Right() * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_D) == GLFW_PRESS)
		camera_->position += camera_->Right() * v;

	// Space
	if (glfwGetKey(glfw_window_, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		if (key_timeout_ < 0.0f)
		{
			print_timestamp_ = true;
			key_timeout_ = 0.1f;
		}
	}

	// QE
	if (glfwGetKey(glfw_window_, GLFW_KEY_Q) == GLFW_PRESS)
		camera_->position -= glm::vec3(0, 1, 0) * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_E) == GLFW_PRESS)
		camera_->position += glm::vec3(0, 1, 0) * v;

	const float yawSpeed = 120.0f;
	if (glfwGetKey(glfw_window_, GLFW_KEY_LEFT) == GLFW_PRESS) camera_->yaw -= yawSpeed * dt;
	if (glfwGetKey(glfw_window_, GLFW_KEY_RIGHT) == GLFW_PRESS) camera_->yaw += yawSpeed * dt;

	const float pitchSpeed = 90.0f;
	if (glfwGetKey(glfw_window_, GLFW_KEY_UP) == GLFW_PRESS) camera_->pitch = glm::clamp(camera_->pitch + pitchSpeed * dt, -89.0f, 89.0f);
	if (glfwGetKey(glfw_window_, GLFW_KEY_DOWN) == GLFW_PRESS) camera_->pitch = glm::clamp(camera_->pitch - pitchSpeed * dt, -89.0f, 89.0f);

	if (glfwGetKey(glfw_window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(glfw_window_, GLFW_TRUE);
	}
}