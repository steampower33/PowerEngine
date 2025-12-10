
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
	
	//init_width_ = static_cast<int>(1440);
	//init_height_ = static_cast<int>(init_width_ * 9.0f / 16.0f);


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
		float frameDt = static_cast<float>(current - lastTime);
		lastTime = current;

		sim_accum_ += frameDt;

		ProcessKeyboard(frameDt);

		key_timeout_ -= frameDt;

		if (sim_accum_ >= sim_dt_) {
			sim_accum_ -= sim_dt_;

			renderer_->Update(*camera_, *mouse_interactor_, frameDt, target_sim_fps_, sim_dt_, paused_);

			renderer_->Draw(paused_);

		}
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

	if (camera_->mouse_move_enabled) {
		if (first_mouse_) {
			lastX = xpos;
			lastY = ypos;
			first_mouse_ = false;
		}

		double xoffset = xpos - lastX;
		double yoffset = lastY - ypos;

		camera_->yaw += static_cast<float>(xoffset) * camera_->mouse_move_speed;
		camera_->pitch += static_cast<float>(yoffset) * camera_->mouse_move_speed;
		camera_->pitch = glm::clamp(camera_->pitch, -89.0f, 89.0f);
	}

	lastX = xpos;
	lastY = ypos;

	mouse_interactor_->mouse_pos_ = glm::vec2(static_cast<float>(xpos), static_cast<float>(ypos));
}

void Window::OnKey(int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_F && action == GLFW_PRESS) {
		camera_->focus_enabled = !camera_->focus_enabled;

		if (camera_->focus_enabled) {
			float radius = glm::length(camera_->position);
			if (radius < 0.001f) radius = 10.0f;

			glm::vec3 dir = camera_->Front();
			camera_->position = -dir * radius;
		}
	}

	if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		camera_->mouse_move_enabled = !camera_->mouse_move_enabled;

		if (camera_->mouse_move_enabled) {
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
	if (yoffset > 0) {
		//std::cout << "DEPTH OUT" << std::endl;
		//std::cout << yoffset << std::endl;
		mouse_interactor_->depth_state = vku::DepthState::MOUSE_DEPTH_OUT;
	}
	else if (yoffset < 0) {
		//std::cout << "DEPTH IN" << std::endl;
		//std::cout << yoffset << std::endl;
		mouse_interactor_->depth_state = vku::DepthState::MOUSE_DEPTH_IN;
	}
}

Window::~Window()
{
	glfwDestroyWindow(glfw_window_);

	glfwTerminate();
}

void Window::ProcessKeyboard(float dt) {
	float v = camera_->camera_move_speed * dt;

	if (camera_->focus_enabled)
	{
		const float orbitYawSpeed = camera_->orbit_yaw_speed;
		const float orbitPitchSpeed = camera_->orbit_pitch_speed;

		// ASDW
		if (glfwGetKey(glfw_window_, GLFW_KEY_W) == GLFW_PRESS)
			camera_->SetCircularPos(0.0f, -orbitPitchSpeed * dt);
		if (glfwGetKey(glfw_window_, GLFW_KEY_S) == GLFW_PRESS)
			camera_->SetCircularPos(0.0f, +orbitPitchSpeed * dt);
		if (glfwGetKey(glfw_window_, GLFW_KEY_A) == GLFW_PRESS)
			camera_->SetCircularPos(+orbitYawSpeed * dt, 0.0f);
		if (glfwGetKey(glfw_window_, GLFW_KEY_D) == GLFW_PRESS)
			camera_->SetCircularPos(-orbitYawSpeed * dt, 0.0f);

		// QE
		if (glfwGetKey(glfw_window_, GLFW_KEY_Q) == GLFW_PRESS)
		{
			glm::vec3 dirToOrigin = glm::normalize(-camera_->position);
			camera_->position += dirToOrigin * dt * camera_->zoom_speed;
		}
		if (glfwGetKey(glfw_window_, GLFW_KEY_E) == GLFW_PRESS)
		{
			glm::vec3 dirToOrigin = glm::normalize(+camera_->position);
			camera_->position += dirToOrigin * dt * camera_->zoom_speed;
		}
	}
	else
	{
		// ASDW
		if (glfwGetKey(glfw_window_, GLFW_KEY_W) == GLFW_PRESS)
			camera_->position += camera_->Front() * v;
		if (glfwGetKey(glfw_window_, GLFW_KEY_S) == GLFW_PRESS)
			camera_->position -= camera_->Front() * v;
		if (glfwGetKey(glfw_window_, GLFW_KEY_A) == GLFW_PRESS)
			camera_->position -= camera_->Right() * v;
		if (glfwGetKey(glfw_window_, GLFW_KEY_D) == GLFW_PRESS)
			camera_->position += camera_->Right() * v;

		// QE
		if (glfwGetKey(glfw_window_, GLFW_KEY_Q) == GLFW_PRESS)
			camera_->position -= glm::vec3(0, 1, 0) * v;
		if (glfwGetKey(glfw_window_, GLFW_KEY_E) == GLFW_PRESS)
			camera_->position += glm::vec3(0, 1, 0) * v;

		//const float yawSpeed = 120.0f;
		//if (glfwGetKey(glfw_window_, GLFW_KEY_LEFT) == GLFW_PRESS) camera_->yaw -= yawSpeed * dt;
		//if (glfwGetKey(glfw_window_, GLFW_KEY_RIGHT) == GLFW_PRESS) camera_->yaw += yawSpeed * dt;

		//const float pitchSpeed = 90.0f;
		//if (glfwGetKey(glfw_window_, GLFW_KEY_UP) == GLFW_PRESS) camera_->pitch = glm::clamp(camera_->pitch + pitchSpeed * dt, -89.0f, 89.0f);
		//if (glfwGetKey(glfw_window_, GLFW_KEY_DOWN) == GLFW_PRESS) camera_->pitch = glm::clamp(camera_->pitch - pitchSpeed * dt, -89.0f, 89.0f);
	}

	// Space
	if (glfwGetKey(glfw_window_, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		if (key_timeout_ < 0.0f)
		{
			paused_ = !paused_;
			key_timeout_ = 0.2f;
		}
	}

	if (glfwGetKey(glfw_window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(glfw_window_, GLFW_TRUE);
	}
}