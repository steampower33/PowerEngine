#include "context.h"
#include "camera.h"
#include "mouse_interactor.h"

#include "window.h"

Window::Window()
{
	glfwInit();

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	glfw_window_ = glfwCreateWindow(init_width_, init_height_, "Vulkan", nullptr, nullptr);

	// �� �����쿡 this�� �Ŵ޾Ƶд�
	glfwSetWindowUserPointer(glfw_window_, this);

	// �ݹ� ���: static Ʈ���޸�
	glfwSetFramebufferSizeCallback(glfw_window_, &Window::FramebufferResizeCallback);
	glfwSetCursorPosCallback(glfw_window_, &Window::CursorPosCallback);
	glfwSetKeyCallback(glfw_window_, &Window::KeyCallback);
	glfwSetMouseButtonCallback(glfw_window_, &Window::MouseButtonCallback);

	if (!glfw_window_)
	{
		std::cerr << "Failure creating glfw window " << std::endl;
	}

	ctx_ = std::make_unique<Context>(glfw_window_, init_width_, init_height_);
	camera_ = std::make_unique<Camera>();
	mouse_interactor_ = std::make_unique<MouseInteractor>();

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

// --- ���� �ν��Ͻ� �ڵ鷯�� ---
void Window::OnFramebufferResize(int, int)
{
	framebuffer_resized_ = true;
}

void Window::OnCursorPos(double xpos, double ypos)
{
	static double lastX = xpos;
	static double lastY = ypos;

	if (mouse_enabled_) {
		// ī�޶� ���� ó�� ������ ���, ���� ��ġ�� ���� ��ġ�� �ʱ�ȭ�ϰ� ������.
		// �̷��� �ϸ� ī�޶� Ƣ�� ������ ������ �� �ִ�.
		if (first_mouse_) {
			lastX = xpos;
			lastY = ypos;
			first_mouse_ = false;
		}

		// ���� �����Ӱ��� ����(offset)�� ����Ѵ�.
		double xoffset = xpos - lastX;
		double yoffset = lastY - ypos; // Y�� ���� ���� (y���� �۾������� ���� ��)

		// ī�޶��� ������ ������Ʈ�Ѵ�.
		camera_->yaw += static_cast<float>(xoffset) * camera_->sensitivity;
		camera_->pitch += static_cast<float>(yoffset) * camera_->sensitivity;
		camera_->pitch = glm::clamp(camera_->pitch, -89.0f, 89.0f);
	}

	// ���� �������� ���� ���� ��ġ�� '���� ��ġ'�� �����Ѵ�.
	lastX = xpos;
	lastY = ypos;

	// --- ���콺 ���ͷ��� ������Ʈ ���� ---
	// ���콺 ���ͷ��Ϳ��Դ� �������� ���� ������ ��ǥ�� �Ѱ��ִ� ���Ҹ� �Ѵ�.
	// ī�޶� ��尡 Ȱ��ȭ�Ǿ� �ֵ� �ƴϵ�, ���콺�� ���� ��ġ�� �׻� �˷���� �Ѵ�.
	mouse_interactor_->mouse_pos_ = glm::vec2(static_cast<float>(xpos), static_cast<float>(ypos));
}

void Window::OnKey(int key, int scancode, int action, int mods) {
	// Ű�� '������ ����(PRESS)'���� �����ϵ��� �մϴ�.
	if (key == GLFW_KEY_F && action == GLFW_PRESS) {
		mouse_enabled_ = !mouse_enabled_;

		if (mouse_enabled_) {
			//glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			first_mouse_ = true; // ��� ��ȯ �� ī�޶� Ƣ�� ���� ����
		}
		else {
			glfwSetInputMode(glfw_window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
}

void Window::OnMouseClick(int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS)   mouse_interactor_->is_left_button_down_event = true;
		else if (action == GLFW_RELEASE) mouse_interactor_->is_left_button_up_event = true;
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		if (action == GLFW_PRESS)   mouse_interactor_->is_right_button_down_event = true;
		else if (action == GLFW_RELEASE) mouse_interactor_->is_right_button_up_event = true;
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

		ctx_->Update(*camera_, *mouse_interactor_, dt);
		ctx_->Draw();
	}

	ctx_->WaitIdle();
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

	// QE
	if (glfwGetKey(glfw_window_, GLFW_KEY_Q) == GLFW_PRESS)
		camera_->position -= glm::vec3(0, 1, 0) * v;
	if (glfwGetKey(glfw_window_, GLFW_KEY_E) == GLFW_PRESS)
		camera_->position += glm::vec3(0, 1, 0) * v;

	// �� �� �� �� ��/�� (yaw)
	const float yawSpeed = 120.0f; // deg/sec. ������ 60~180 ���̷�
	if (glfwGetKey(glfw_window_, GLFW_KEY_LEFT) == GLFW_PRESS) camera_->yaw -= yawSpeed * dt;
	if (glfwGetKey(glfw_window_, GLFW_KEY_RIGHT) == GLFW_PRESS) camera_->yaw += yawSpeed * dt;

	// �� �� �� �� ��/�Ʒ� (pitch)�� ���ϸ�
	const float pitchSpeed = 90.0f; // deg/sec
	if (glfwGetKey(glfw_window_, GLFW_KEY_UP) == GLFW_PRESS) camera_->pitch = glm::clamp(camera_->pitch + pitchSpeed * dt, -89.0f, 89.0f);
	if (glfwGetKey(glfw_window_, GLFW_KEY_DOWN) == GLFW_PRESS) camera_->pitch = glm::clamp(camera_->pitch - pitchSpeed * dt, -89.0f, 89.0f);

	if (glfwGetKey(glfw_window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(glfw_window_, GLFW_TRUE);
	}
}